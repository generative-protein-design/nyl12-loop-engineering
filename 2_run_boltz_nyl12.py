# based on https://github.com/ikalvet/heme_binder_diffusion/blob/main/pipeline.ipynb

import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import getpass
import subprocess
import time
import importlib
import shutil
import re
import tempfile
from Bio import AlignIO
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

### Path to this cloned GitHub repo:
SCRIPT_DIR = os.path.dirname(__file__)  # edit this to the GitHub repo path. Throws an error by default.
assert os.path.exists(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR+"/scripts/utils")
import utils

### Path where the jobs will be run and outputs written 
WDIR = "/data/jpc/code/heme_binder_diffusion/output_test"

CONDAPATH = "/data/jpc/code/miniconda3"   # edit this depending on where your Conda environments live
PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
          "boltz": f"{CONDAPATH}/envs/boltz2.1.1/bin/boltz",
          "ligandMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
          "pymol-env": f"{CONDAPATH}/envs/pymol-env/bin/python",
          "general": f"{CONDAPATH}/envs/diffusion/bin/python"}

username = getpass.getuser()  # your username on the running system

os.chdir(WDIR)

MPNN_DIR = f"{WDIR}/1_ligandmpnn"
BOLTZ_DIR = f"{WDIR}/2_boltz"
BOLTZ_CACHE_DIR = "/data/jpc/code/boltz" # directory where Boltz weights are stored
os.makedirs(BOLTZ_DIR, exist_ok=True)
os.chdir(BOLTZ_DIR)

def parse_ranges(input_str):
    elements = input_str.split(',')
    result = []
    prev_value = None

    for elem in elements:
        if re.match(r'^[A-Za-z]', elem):
            prev_value = int(elem.split('-')[1])
        else:
            range_values = list(map(int, elem.split('-')))
            size = max(range_values) + 1  # Use the higher value of the two
            result.append((prev_value, size))
            prev_value = None  # Reset prev_value after using it

    return result

def split_by_repeating_substring(input_str):
    if len(input_str) < 5:
        return [(0, len(input_str), input_str)] # Handle small strings

    pattern = input_str[:5] # First 5 characters
    segments = []
    start_idx = 0
    search_idx = 5  # Start searching after the first 5 characters

    while True:
        next_idx = input_str.find(pattern, search_idx) # Find next occurrence
        if next_idx == -1:
            break # No more occurrences

        segments.append((start_idx, next_idx - start_idx, input_str[start_idx:next_idx]))
        start_idx = next_idx
        search_idx = next_idx + 5 # Move forward

    # Add final segment
    segments.append((start_idx, len(input_str) - start_idx, input_str[start_idx:]))

    return segments

def generate_msa(input_sequences):
    with open(f"{BOLTZ_DIR}/temp_sequences.fasta", "w") as f:
        for i, (_, _, seq) in enumerate(input_sequences):
            f.write(f">chain_{chr(i + 97)}\n{seq}\n")

    # Run MAFFT alignment (adjust paths if necessary)
    # Download MAFFT here: https://mafft.cbrc.jp/alignment/software/mafft-7.526-linux.tgz
    # and follow instructions: https://mafft.cbrc.jp/alignment/software/linuxportable.html

    os.system(f"/data/jpc/code/mafft-linux64/mafft.bat {BOLTZ_DIR}/temp_sequences.fasta > out.fa 2>/dev/null") # 2>/dev/null silences stderr if you want
    alignment = AlignIO.read("out.fa", "fasta")
    os.system("rm temp_sequences.fasta out.fa")

    return alignment

def correct_alignment(aligned_sequences, contig):
    seq_length = len(aligned_sequences[0].seq)
    num_sequences = len(aligned_sequences)

    for i in range(seq_length):
        column_chars = [seq.seq[i] for seq in aligned_sequences]
        unique_chars = set(column_chars)

        if len(unique_chars) > 1:  # Found a difference
            for seq_index, char in enumerate(column_chars):
                if column_chars.count(char) == 1:  # Identify differing sequence
                    adjusted_index = (seq_index * seq_length) + i

                    if any(start <= adjusted_index < start + length for start, length in contig):
                        for seq in aligned_sequences:
                            seq.seq = Seq(seq.seq[:i] + char + seq.seq[i+1:])  # Correct others

    return aligned_sequences

def remove_gaps(aligned_sequences):
    for seq in aligned_sequences:
        seq.seq = Seq(str(seq.seq).replace("-", ""))

    return aligned_sequences

def extract_beta_strings(final_sequences):
    marker = "TTLTIVIT" # Nyl12 first few residues of the beta chain
    new_sequences = []

    for name, record in enumerate(final_sequences):
        seq_str = str(record.seq)

        if marker in seq_str:
            alpha_part, beta_part = seq_str.split(marker, 1)  # Split at first occurrence of marker
            beta_part = marker + beta_part
            # Create new sequence records with modified names
            alpha_record = SeqRecord(Seq(alpha_part), id=f"{record.id}_alpha", description="")
            beta_record = SeqRecord(Seq(beta_part), id=f"{record.id}_beta", description="")

            new_sequences.extend([alpha_record, beta_record])
        else:
            # If the marker isn't found, keep the sequence with a modified name
            unchanged_record = SeqRecord(Seq(seq_str), id=f"{record.id}_unchanged", description="")
            new_sequences.append(unchanged_record)

    return new_sequences

def write_yaml(sequence_dict):
    """
    Split protein sequences into separate chains, append the ligand SMILES string, 
    and write out YAML files in Boltz format.

    Parameters:
    - sequence_dict (dict): Keys are FASTA headers and values are merged sequences
      of all chains.
    """

    ligand_data = """
 - ligand:
     id: I
      smiles: '[NH3+]CCCCCCC(=O)NCCCCCC(O)NCCCCCC(=O)NCCCCCC(=O)[O-]'
""".strip()

    template_data = """
  templates:
    -cif: /data/jpc/projects/nyl12/boltz2/cif/Nyl12_refine13.cif
""".strip()

    contig = "A1-5,A18-114,7-7,A122-150,14-14,A165-221,A233-328,B1-5,B286-292,8-8,B301-308,C1-5,D1-5,D29-35,5-5,D41-95,9-9,D105-110"

    for key, sequence in sequence_dict.items():
        filename = f"{key.lstrip('>').strip()}.fasta"

        #JMM
        segments = split_by_repeating_substring(sequence)
        print('segments', segments)
        aligned_sequences = generate_msa(segments)
        print('aligned_sequences', aligned_sequences)
        sys.exit()
        corrected_sequences = correct_alignment(aligned_sequences, parse_ranges(contig))
        final_sequences = remove_gaps(corrected_sequences)
        split_sequences = extract_beta_strings(final_sequences)

        for record in split_sequences:
            print(record.id, record.seq)

        subsequences = [str(record.seq) for record in split_sequences]

        with open(filename, 'w') as y:
            for idx, subseq in enumerate(subsequences):
                #print('idx', idx, 'subseq', subseq)
                chain_label = chr(97 + idx)  # 'a', 'b', 'c', 'd', ...
                header = f">protein|name=chain_{chain_label}"
                y.write(f"{header}\n{subseq}\n")

            # Append ligand data at the end
            f.write(f"{ligand_data}\n")
            f.write(f"{template_data}\n")

# Collect MPNN outputs and create FASTA files
mpnn_fasta = utils.parse_fasta_files(glob.glob(f"{MPNN_DIR}/seqs/*.fa"))
mpnn_fasta = {k: seq.strip() for k, seq in mpnn_fasta.items() if "model_path" not in k}

# Give sequences unique names based on input PDB name, temperature, and sequence identifier
mpnn_fasta = {k.split(",")[0]+"_"+k.split(",")[2].replace(" T=", "T")+"_0_"+k.split(",")[1].replace(" id=", ""): seq for k, seq in mpnn_fasta.items()} # original code

# Split sequences and write individual YAML files for Boltz 
#print('mpnn_fasta', mpnn_fasta)
write_yaml(mpnn_fasta)

print(f"A total of {len(mpnn_fasta)} sequences will be predicted.")

sys.exit()

commands_boltz = []
cmds_filename_boltz = "commands_boltz"

with open(cmds_filename_boltz, "w") as file:
    for ff in glob.glob("*.fasta"):
        #output_dir, extension = os.path.splitext(ff)
        #if os.path.exists(output_dir):
        #    shutil.rmtree(output_dir)
        ##os.mkdir(output_dir) # Not needed.

        # Run Boltz, rename/move .npz files, and clean up
        commands_boltz.append(f"{PYTHON['boltz']} predict {YAML_FILE} "
                             f"--model boltz2 "
                             f"--output_format pdb "
                             f"--use_msa_server "
                             f"--msa_pairing_strategy greedy "
                             f"--use_potentials "
                             f"--cache {BOLTZ_CACHE_DIR}    ; \n")
                             #f"{ff} {CHAI_DIR}/{output_dir} ; "
                             #f"mv {CHAI_DIR}/{output_dir}/pred.model_idx_0.cif {CHAI_DIR}/{output_dir}.cif ; "
                             #f"mv {CHAI_DIR}/{output_dir}/scores.model_idx_0.npz {CHAI_DIR}/scores.{output_dir}.npz ; \n")

        file.write(commands_boltz[-1])

print("Example Boltz command:")
print(commands_boltz[-1])
sys.exit()

log = f"{BOLTZ_DIR}/output.log"

if not os.path.exists(BOLTZ_DIR+"/.done"):
    with open(log, "w") as chai_log:
        #p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #p = subprocess.Popen(['bash', cmds_filename_chai], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #(output, err) = p.communicate()
        p = subprocess.Popen(['bash', cmds_filename_boltz], stdout=boltz_log, stderr=boltz_log)
        p.wait()
else:
    print('Skipping Boltz calcs...')

# Postprocess boltz models 
#BETA_OFFSET = 232 # number of residues in the alpha subunit
#NUM_RES_PER_CHAIN = 328 # total number of residues in both the alpha and beta subunit

commands_convert = []
cmds_filename_convert = "commands_convert"
with open(cmds_filename_convert, "w") as conv:
    for cif in glob.glob('*.cif'):
        base_name, extension = os.path.splitext(cif)
        #commands_convert.append(f"{PYTHON['pymol-env']} {SCRIPT_DIR}/scripts/utils/cif2pdb.py {CHAI_DIR}/{base_name}.cif {BETA_OFFSET} {NUM_RES_PER_CHAIN} {CHAI_DIR}/{base_name}_chai.pdb ; \n")
        commands_convert.append(f"{PYTHON['pymol-env']} {BOLTZ_DIR}/test.py --model {BOLTZ_DIR}/{base_name}.cif --output_dir {BOLTZ_DIR} ; \n")

        conv.write(commands_convert[-1])

conversion_log = f"{BOLTZ_DIR}/convert.log"
with open(conversion_log, "w") as conv_log:
    p = subprocess.Popen(['bash', cmds_filename_convert], stdout=conv_log, stderr=conv_log)
    p.wait()

## If you're done with Boltz and happy with the outputs then mark it as done
os.chdir(BOLTZ_DIR)

if not os.path.exists(BOLTZ_DIR+"/.done"):
    with open(f"{BOLTZ_DIR}/.done", "w") as file:
        file.write(f"Run user: {username}\n")
