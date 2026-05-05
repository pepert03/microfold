"""Validation sweep + macro HTML report builder.

Runs the model over a val subset, generating per-sample 3D viewers and a single macro
HTML that flips between them with sidebar metrics + aggregate summary.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from microfold.dataset import collate
from microfold.losses import total_loss
from microfold.stamping import stamp_backbone
from microfold.visualization import calculate_backbone_rmsd, generate_html_report, kabsch_superimpose


@dataclass
class SampleRecord:
    id: int
    sequence: str
    length: int
    intermediate: float
    final: float
    bond: float
    clash: float
    total: float
    rmsd: float
    html: str  # relative path to per-sample html


def _aggregate(records: list[SampleRecord]) -> dict[str, float | int]:
    rmsds = [r.rmsd for r in records]
    totals = [r.total for r in records]
    finals = [r.final for r in records]
    inters = [r.intermediate for r in records]
    bonds = [r.bond for r in records]
    clashes = [r.clash for r in records]
    return {
        "n": len(records),
        "mean_total": float(statistics.fmean(totals)),
        "mean_intermediate": float(statistics.fmean(inters)),
        "mean_final": float(statistics.fmean(finals)),
        "mean_bond": float(statistics.fmean(bonds)),
        "mean_clash": float(statistics.fmean(clashes)),
        "mean_rmsd": float(statistics.fmean(rmsds)),
        "median_rmsd": float(statistics.median(rmsds)),
        "min_rmsd": float(min(rmsds)),
        "max_rmsd": float(max(rmsds)),
    }


@torch.no_grad()
def run_val(
    model: torch.nn.Module,
    val_set: Dataset,
    device: torch.device,
    out_dir: Path,
    epoch: int,
    w_bond: float = 0.1,
    w_clash: float = 0.05,
    bond_tol: float = 0.02,
    clash_limit: float = 2.0,
    use_clash: bool = False,
) -> dict[str, Any]:
    """Evaluate model on every sample; write per-sample html + macro html into out_dir."""
    out_dir = Path(out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    records: list[SampleRecord] = []

    dl = DataLoader(val_set, batch_size=1, collate_fn=collate, shuffle=False)
    for batch in dl:
        seqs = batch["sequence"]
        mask = batch["mask"].to(device)
        true_R = batch["R"].to(device)
        true_t = batch["t"].to(device)
        true_all = batch["true_all_backbone_coords"].to(device)

        out = model(seqs, mask)
        losses = total_loss(
            out["intermediate"], out["R"], out["t"], true_R, true_t, true_all, mask,
            w_bond=w_bond, w_clash=w_clash, bond_tol=bond_tol,
            clash_limit=clash_limit, use_clash=use_clash,
        )
        pred_all = stamp_backbone(out["R"], out["t"]).cpu()
        truth_all = batch["true_all_backbone_coords"]
        m_cpu = batch["mask"]

        rmsd = calculate_backbone_rmsd(truth_all[0], pred_all[0], m_cpu[0])

        dbid = int(batch["id"][0])
        seq = batch["sequence"][0]
        n = int(m_cpu[0].sum().item())

        # Kabsch-align prediction onto truth so the viewer shows maximal overlap;
        # the underlying RMSD is unchanged because Kabsch is the alignment used to compute it.
        aligned_pred = kabsch_superimpose(pred_all[0], truth_all[0], m_cpu[0])

        html_rel = f"samples/{dbid}.html"
        generate_html_report(
            truth_all[0], aligned_pred, seq[:n], m_cpu[0], samples_dir / f"{dbid}.html"
        )
        records.append(
            SampleRecord(
                id=dbid,
                sequence=seq,
                length=n,
                intermediate=float(losses["intermediate"].item()),
                final=float(losses["final"].item()),
                bond=float(losses["bond"].item()),
                clash=float(losses["clash"].item()),
                total=float(losses["total"].item()),
                rmsd=float(rmsd),
                html=html_rel,
            )
        )

    records.sort(key=lambda r: r.rmsd)
    agg = _aggregate(records)
    _write_macro(records, agg, epoch, out_dir / "macro.html")
    (out_dir / "metrics.json").write_text(
        json.dumps({"epoch": epoch, "aggregate": agg, "samples": [asdict(r) for r in records]}, indent=2)
    )
    model.train()
    return {"records": records, "aggregate": agg}


_MACRO_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>microfold val · epoch {epoch}</title>
<style>
  :root { --bg:#0b0d10; --panel:#15181d; --fg:#e6e6e6; --muted:#8a93a3; --accent:#7cf; --good:#7cf6a0; --bad:#ff8c8c; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace; background: var(--bg); color: var(--fg); height:100vh; overflow:hidden; }
  .layout { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  .sidebar { background: var(--panel); padding: 18px 16px; overflow-y: auto; border-right: 1px solid #222; }
  .sidebar h2 { margin: 0 0 10px 0; font-size: 14px; letter-spacing: 0.08em; color: var(--muted); text-transform: uppercase; }
  .sidebar h3 { margin: 14px 0 6px 0; font-size: 12px; letter-spacing: 0.06em; color: var(--muted); text-transform: uppercase; }
  .metric { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed #222; font-size: 13px; }
  .metric .v { color: var(--accent); }
  .seq { font-size: 12px; word-break: break-all; line-height: 1.5; color: #cfd6e0; background: #0e1116; padding: 8px; border-radius: 4px; }
  .main { position: relative; }
  iframe { width: 100%; height: 100%; border: 0; background: #fff; }
  .controls { position: absolute; top: 12px; left: 12px; z-index: 10; display: flex; gap: 6px; align-items: center; background: rgba(0,0,0,0.55); padding: 6px 8px; border-radius: 6px; }
  .controls button, .controls select { background: #1f242c; color: var(--fg); border: 1px solid #2a3038; padding: 6px 10px; border-radius: 4px; font-family: inherit; font-size: 12px; cursor: pointer; }
  .controls button:hover { background: #283038; }
  .summary { position: absolute; top: 12px; right: 12px; background: rgba(15,18,24,0.92); padding: 12px 16px; border-radius: 8px; z-index: 10; font-size: 12px; min-width: 220px; border: 1px solid #2a3038; }
  .summary h3 { margin: 0 0 8px 0; font-size: 11px; letter-spacing: 0.08em; color: var(--muted); text-transform: uppercase; }
  .epoch-pill { display: inline-block; padding: 2px 8px; background: var(--accent); color: #001218; border-radius: 99px; font-weight: bold; margin-left: 6px; }
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <h2>Sample <span class="epoch-pill">epoch {epoch}</span></h2>
    <div id="info"></div>
    <h3>Sequence</h3>
    <div id="seq" class="seq"></div>
  </aside>
  <main class="main">
    <div class="controls">
      <button id="prev">‹ prev</button>
      <select id="picker"></select>
      <button id="next">next ›</button>
      <button id="sort">sort: rmsd↑</button>
    </div>
    <div class="summary">
      <h3>Validation aggregate</h3>
      <div id="agg"></div>
    </div>
    <iframe id="viewer" src=""></iframe>
  </main>
</div>
<script>
const SAMPLES = __SAMPLES__;
const AGG = __AGG__;
let idx = 0;
let sortMode = "rmsd_asc";

function fmt(x, d=3) { return Number(x).toFixed(d); }

function sorted() {
  const arr = SAMPLES.slice();
  if (sortMode === "rmsd_asc") arr.sort((a,b)=>a.rmsd-b.rmsd);
  else if (sortMode === "rmsd_desc") arr.sort((a,b)=>b.rmsd-a.rmsd);
  else if (sortMode === "id") arr.sort((a,b)=>a.id-b.id);
  return arr;
}

function render() {
  const list = sorted();
  const s = list[idx];
  document.getElementById("viewer").src = s.html;
  document.getElementById("info").innerHTML = `
    <div class="metric"><span>ID</span><span class="v">${s.id}</span></div>
    <div class="metric"><span>Length</span><span class="v">${s.length}</span></div>
    <div class="metric"><span>RMSD (Å)</span><span class="v">${fmt(s.rmsd)}</span></div>
    <div class="metric"><span>Total FAPE</span><span class="v">${fmt(s.total, 4)}</span></div>
    <div class="metric"><span>Intermediate</span><span class="v">${fmt(s.intermediate, 4)}</span></div>
    <div class="metric"><span>Final</span><span class="v">${fmt(s.final, 4)}</span></div>
  `;
  document.getElementById("seq").textContent = s.sequence;
  document.getElementById("picker").value = String(idx);
}

function refillPicker() {
  const list = sorted();
  const sel = document.getElementById("picker");
  sel.innerHTML = "";
  list.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.text = `#${s.id}  RMSD ${fmt(s.rmsd, 2)}Å  len ${s.length}`;
    sel.appendChild(o);
  });
}

function renderAgg() {
  document.getElementById("agg").innerHTML = `
    <div class="metric"><span>Samples</span><span class="v">${AGG.n}</span></div>
    <div class="metric"><span>Mean total</span><span class="v">${fmt(AGG.mean_total, 4)}</span></div>
    <div class="metric"><span>Mean intermed.</span><span class="v">${fmt(AGG.mean_intermediate, 4)}</span></div>
    <div class="metric"><span>Mean final</span><span class="v">${fmt(AGG.mean_final, 4)}</span></div>
    <div class="metric"><span>Mean RMSD</span><span class="v">${fmt(AGG.mean_rmsd)} Å</span></div>
    <div class="metric"><span>Median RMSD</span><span class="v">${fmt(AGG.median_rmsd)} Å</span></div>
    <div class="metric"><span>Best RMSD</span><span class="v">${fmt(AGG.min_rmsd)} Å</span></div>
    <div class="metric"><span>Worst RMSD</span><span class="v">${fmt(AGG.max_rmsd)} Å</span></div>
  `;
}

document.getElementById("prev").onclick = () => { idx = (idx - 1 + SAMPLES.length) % SAMPLES.length; render(); };
document.getElementById("next").onclick = () => { idx = (idx + 1) % SAMPLES.length; render(); };
document.getElementById("picker").onchange = e => { idx = parseInt(e.target.value); render(); };
document.getElementById("sort").onclick = e => {
  sortMode = sortMode === "rmsd_asc" ? "rmsd_desc" : (sortMode === "rmsd_desc" ? "id" : "rmsd_asc");
  e.target.textContent = "sort: " + (sortMode === "rmsd_asc" ? "rmsd↑" : sortMode === "rmsd_desc" ? "rmsd↓" : "id");
  idx = 0;
  refillPicker();
  render();
};

document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
});

refillPicker();
renderAgg();
render();
</script>
</body>
</html>
"""


def _write_macro(records: list[SampleRecord], agg: dict[str, Any], epoch: int, out_path: Path) -> None:
    payload = json.dumps([asdict(r) for r in records])
    agg_json = json.dumps(agg)
    html = (
        _MACRO_TEMPLATE.replace("{epoch}", str(epoch))
        .replace("__SAMPLES__", payload)
        .replace("__AGG__", agg_json)
    )
    out_path.write_text(html, encoding="utf-8")
