# based on https://github.com/ikalvet/heme_binder_diffusion/blob/main/pipeline.ipynb

import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import getpass
import subprocess
import time
import importlib
from shutil import copy2

### Path to this cloned GitHub repo:
SCRIPT_DIR = os.path.dirname(__file__)  # edit this to the GitHub repo path. Throws an error by default.
assert os.path.exists(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR+"/scripts/utils")
import utils

ligandMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"  # from submodule

# Path where the jobs will be run and outputs dumped
WDIR = "/data/jpc/code/heme_binder_diffusion/output_test"

# Python and/or Apptainer executables needed for running the jobs
# Please provide paths to executables that are able to run the different tasks.
# They can all be the same if you have an environment with all of the necessary Python modules in one

CONDAPATH = "/data/jpc/code/miniconda3"   # edit this depending on where your Conda environments live
PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
          "pymol-env": f"{CONDAPATH}/envs/pymol-env/bin/python",
          "ligandMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
          "general": f"{CONDAPATH}/envs/diffusion/bin/python"}

username = getpass.getuser()  # your username on the running system

# Ligand information
params = [f"{SCRIPT_DIR}/theozyme/PA6_L5_mod/LIG.params"]  # Rosetta params file
LIGAND = "LIG"

diffusion_inputs = glob.glob(f"{SCRIPT_DIR}/input_test/nyl12_jmp.pdb")
print(f"Found {len(diffusion_inputs)} PDB files")

DIFFUSION_DIR = f"{WDIR}/0_diffusion"

## Diffusion jobs are run in separate directories for each input PDB

diffusion_rundirs = []
for p in diffusion_inputs:
    pdbname = os.path.basename(p).replace(".pdb", "")
    diffusion_rundirs.append(pdbname)

print('diffusion_rundirs', diffusion_rundirs)

diffused_backbones_good = glob.glob(f"{DIFFUSION_DIR}/nyl12_jmp/out/*.pdb")
assert len(diffused_backbones_good) > 0, "No good backbones found!"

os.chdir(WDIR)

MPNN_DIR = f"{WDIR}/1_ligandmpnn"
os.makedirs(MPNN_DIR, exist_ok=True)
os.chdir(MPNN_DIR)

### Parse diffusion output TRB files to extract fixed motif residues
## These residues will not be redesigned with ligandMPNN
mask_json_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/make_maskdict_from_trb.py --out masked_pos.jsonl --trb"
for d in diffused_backbones_good:
    mask_json_cmd += " " + d.replace(".pdb", ".trb")
p = subprocess.Popen(mask_json_cmd, shell=True)
(output, err) = p.communicate()

assert os.path.exists("masked_pos.jsonl"), "Failed to create masked positions JSONL file"

### Set up ligandMPNN run commands
## We're doing design with 3 temperatures, and 5 sequences each.
## This usually gives decent success with designable backbones.
## For more complicated cases consider doing >100 sequences.

MPNN_temperatures = [0.1, 0.2, 0.3]
MPNN_outputs_per_temperature = 5
MPNN_omit_AAs = "C"

commands_mpnn = []
cmds_filename_mpnn = "commands_mpnn"
with open(cmds_filename_mpnn, "w") as file:
    for T in MPNN_temperatures:
        for f in diffused_backbones_good:
            # get LigandMPNN model parameters: https://github.com/dauparas/LigandMPNN/get_model_params.sh
            commands_mpnn.append(f"{PYTHON['ligandMPNN']} {ligandMPNN_script} "
                                 f"--model_type ligand_mpnn --ligand_mpnn_use_atom_context 1 "
                                 f"--ligand_mpnn_use_side_chain_context 1 "
                                 f"--fixed_residues_multi masked_pos.jsonl --out_folder ./ "
                                 f"--number_of_batches {MPNN_outputs_per_temperature} --temperature {T} "
                                 f"--file_ending _{T} "
                                 f"--omit_AA {MPNN_omit_AAs} --pdb_path {f} "
                                 f"--checkpoint_ligand_mpnn {SCRIPT_DIR}/lib/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt\n")
            file.write(commands_mpnn[-1])

print("Example MPNN command:")
print(commands_mpnn[-1])

### Run ligandMPNN 

if not os.path.exists(MPNN_DIR+"/.done"):
    p = subprocess.Popen(['bash', cmds_filename_mpnn], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

## If you're done with diffusion and happy with the outputs then mark it as done
MPNN_DIR = f"{WDIR}/1_ligandmpnn"
os.chdir(MPNN_DIR)

if not os.path.exists(MPNN_DIR+"/.done"):
    with open(f"{MPNN_DIR}/.done", "w") as file:
        file.write(f"Run user: {username}\n")
