#!/usr/bin/env bash

python prepare_boltz_input_nyl12.py \
--input-dir output_test/1_ligandmpnn/seqs \
--output-dir output_test/2_boltz/input \
--cif-file /path/to/cif/file.cif


/data/jpc/code/miniconda3/envs/boltz2.1.1/bin/boltz predict output_test/2_boltz/input \
                             --model boltz2  \
                             --output_format pdb \
                             --use_msa_server \
                             --msa_pairing_strategy greedy \
                             --use_potentials \
                             --out_dir output_test/2_boltz/output \
                             --cache /data/jpc/code/boltz
