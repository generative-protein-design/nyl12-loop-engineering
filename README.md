# Pipeline to engineer a hydrolase to degrade nylon

## Installation

Before running the pipeline, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running Boltz-2 

- Prepare input files

Run script from the root directory:

```bash
python prepare_boltz_input_nyl12.py \
--input-dir output_test/1_ligandmpnn/seqs \
--output-dir output_test/2_boltz/input \
--cif-file /path/to/cif/file.cif
```

for a full list of available options, type:
```bash
python prepare_boltz_input_nyl12.py -h
```


example to run inference:
```bash
pixi run run-inference inference.deterministic=True diffuser.T=20 inference.output_prefix=output/ligand_only/sample inference.input_pdb=input_test/nyl12_jmp.pdb contigmap.contigs=[\'150-150\'] inference.ligand=LIG inference.num_designs=1 inference.design_startnum=0 inference.ckpt_path=/data/jpc/code/rf_diffusion_all_atom/RFDiffusionAA_paper_weights.pt
```


