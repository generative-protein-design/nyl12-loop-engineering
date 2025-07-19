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

diffusion_script = "/data/jpc/code/rf_diffusion_all_atom/run_inference.py"  # edit this

### Python and/or Apptainer executables needed for running the jobs
### Please provide paths to executables that are able to run the different tasks.
### They can all be the same if you have an environment with all of the ncessary Python modules in one

# If your added Apptainer does not execute scripts directly,
# try adding 'apptainer run' or 'apptainer run --nv' (for GPU) in front of the command

CONDAPATH = "/data/jpc/code/miniconda3"   # edit this depending on where your Conda environments live
PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
          "ligandMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
          "pymol-env": f"{CONDAPATH}/envs/pymol-env/bin/python",
          "general": f"{CONDAPATH}/envs/diffusion/bin/python"}

### Path where the jobs will be run and outputs dumped
WDIR = "/data/jpc/code/heme_binder_diffusion/output_test"

if not os.path.exists(WDIR):
    os.makedirs(WDIR, exist_ok=True)

print(f"Working directory: {WDIR}")

# Ligand information
params = [f"{SCRIPT_DIR}/theozyme/PA6_L5_mod/LIG.params"]  # Rosetta params file
LIGAND = "LIG"

# Set up diffusion run
diffusion_inputs = glob.glob(f"{SCRIPT_DIR}/input_test/nyl12_jmp.pdb")
print(f"Found {len(diffusion_inputs)} PDB files")

DIFFUSION_DIR = f"{WDIR}/0_diffusion"
if not os.path.exists(DIFFUSION_DIR):
    os.makedirs(DIFFUSION_DIR, exist_ok=False)

os.chdir(DIFFUSION_DIR)

N_designs = 10 # JMP. Should ideally make hundreds/thousands of models.

## Set up diffusion commands based on the input PDB file(s)
## Diffusion jobs are run in separate directories for each input PDB

commands_diffusion = []
cmds_filename = "commands_diffusion"
diffusion_rundirs = []
with open(cmds_filename, "w") as file:
    for p in diffusion_inputs:
        pdbname = os.path.basename(p).replace(".pdb", "")
        os.makedirs(pdbname, exist_ok=True)
        cmd = f"cd {pdbname} ; {PYTHON['diffusion']} {diffusion_script} "\
              f"inference.input_pdb={p} "\
              f"inference.output_prefix='./out/{pdbname}_diff' "\
              f"inference.model_runner=NRBStyleSelfCond "\
              f"inference.ligand=\\'LIG\\' "\
              f"inference.num_designs={N_designs} "\
              f"model.freeze_track_motif=True "\
              f"potentials.guiding_potentials=[\\'type:ligand_ncontacts,weight:1\\'] "\
              f"potentials.guide_scale=2 "\
              f"contigmap.contigs=[\\'A1-5,A18-114,7-7,A122-150,14-14,A165-221,A233-328,B1-5,B286-292,8-8,B301-308,C1-5,D1-5,D29-35,5-5,D41-95,9-9,D105-110\\'] "\
              f"potentials.guide_decay=cubic "\
              f"diffuser.T=50 ; cd ..\n"
        commands_diffusion.append(cmd)
        diffusion_rundirs.append(pdbname)
        file.write(cmd)

print(f"Example diffusion command:\n {cmd}")

print(f"Wrote diffusion commands to {cmds_filename}")
print(f"{len(commands_diffusion)} diffusion jobs to run")

log = f"{DIFFUSION_DIR}/output.log"

if not os.path.exists(DIFFUSION_DIR+"/.done"):
    with open(log, "w") as diff_log: 
        p = subprocess.Popen(['bash', cmds_filename], stdout=diff_log, stderr=diff_log)
        p.wait()

## If you're done with diffusion and happy with the outputs then mark it as done
DIFFUSION_DIR = f"{WDIR}/0_diffusion"
os.chdir(DIFFUSION_DIR)

if not os.path.exists(DIFFUSION_DIR+"/.done"):
    with open(f"{DIFFUSION_DIR}/.done", "w") as file:
        file.write(f"done\n")
