import copy

import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdb
import numpy as np
import py3Dmol

# --- 1. LOAD THE GROUND TRUTH ---
# We will use the first 3 amino acids of 4ONK (TYR-VAL-VAL)
file_path = rcsb.fetch("4ONK", "pdb", ".")
atoms = pdb.PDBFile.read(file_path).get_structure(model=1)

# Clean it: Keep only Chain A, remove Hydrogen, keep only residues 1 to 3
truth = atoms[(atoms.chain_id == "A") & (atoms.element != "H") & (atoms.res_id <= 3)]

# --- 2. GENERATE THE "GUESS" (Random Frames) ---
# We create a copy to act as the AI's Epoch 0 output
prediction = copy.deepcopy(truth)

# For every independent amino acid frame...
for res_id in np.unique(prediction.res_id):
    mask = prediction.res_id == res_id

    # Get the center of the frame (Alpha-Carbon)
    ca_pos = prediction[mask & (prediction.atom_name == "CA")].coord[0]

    # 1. Randomize the Translation (Move it randomly within a 15 Angstrom box)
    random_translation = np.random.uniform(-5, 5, 3)

    # 2. Randomize the Rotation (Generate a random 3x3 orthogonal matrix)
    random_matrix = np.random.randn(3, 3)
    q, r = np.linalg.qr(
        random_matrix
    )  # QR decomposition ensures it's a valid 3D rotation
    random_rotation = q

    # Apply the math: Shift to origin -> Rotate -> Translate to random spot
    shifted_coords = prediction.coord[mask] - ca_pos
    rotated_coords = np.dot(shifted_coords, random_rotation)
    prediction.coord[mask] = rotated_coords + random_translation

# --- 3. CONVERT BACK TO PDB STRINGS ---
# Biotite easily writes the raw arrays back into PDB format for the visualizer
truth_pdb = pdb.PDBFile()
pdb.set_structure(truth_pdb, truth)
truth_str = str(truth_pdb)

pred_pdb = pdb.PDBFile()
pdb.set_structure(pred_pdb, prediction)
pred_str = str(pred_pdb)

# --- 4. VISUALIZE: TRUTH VS PREDICTION ---
view = py3Dmol.view(width=800, height=600)

# Add Ground Truth (Green/Cyan, solid)
view.addModel(truth_str, "pdb")
view.setStyle(
    {"model": 0},
    {"stick": {"colorscheme": "greenCarbon", "radius": 0.15}},
)
# view.addStyle({"model": 0}, {"sphere": {"radius": 0.3, "opacity": 1.0}})


# Add AI Guess (Red, slightly transparent to compare)
view.addModel(pred_str, "pdb")
view.setStyle(
    {"model": 1},
    {"stick": {"colorscheme": "redCarbon", "radius": 0.15, "opacity": 0.7}},
)
view.addStyle({"model": 1}, {"sphere": {"radius": 0.3, "opacity": 0.5}})

view.zoomTo()

# 2. Save to a standalone HTML file
with open("molecule_view.html", "w") as f:
    f.write(view.write_html())
