# Pipeline to engineer a hydrolase to degrade nylon

## Installation

 Before running the pipeline, install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Set pixi cache dir (if needed):
```bash
export PIXI_CACHE_DIR=/global/cfs/cdirs/xxxx/protein-design/pixi/cache # on NERSC, use project name
```

then install dependencies (will take some time)

```bash
pixi install --all
```

then download the model weights (if not already present)

```bash
wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFDiffusionAA_paper_weights.pt
```


then install Ligand model params (if not already present)

curl -fsSL https://raw.githubusercontent.com/generative-protein-design/LigandMPNN/refs/heads/main/get_model_params.sh | sh -s -- model_params 


## Running Inference 


example to run inference (set correct path to downloaded model weights file):
```bash
pixi run run-inference inference.deterministic=True diffuser.T=20 inference.output_prefix=output/ligand_only/sample inference.input_pdb=input_test/nyl12_jmp.pdb contigmap.contigs=[\'150-150\'] inference.ligand=LIG inference.num_designs=1 inference.design_startnum=0 inference.ckpt_path=../RFDiffusionAA_paper_weights.pt
```


## Running Boltz-2 

- Prepare input files

Run script from the root directory:

```bash
pixi run python prepare_boltz_input_nyl12.py \
--input-dir output_test/1_ligandmpnn/seqs \
--output-dir output_test/2_boltz/input \
--cif-file /path/to/cif/file.cif
```

for a full list of available options, type:
```bash
pixi run python prepare_boltz_input_nyl12.py -h
```


