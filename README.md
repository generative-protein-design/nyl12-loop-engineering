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

then install LigandMPNN model params (if not already present)

curl -fsSL https://raw.githubusercontent.com/generative-protein-design/LigandMPNN/refs/heads/main/get_model_params.sh |
sh -s -- model_params

then install colabfold database (if not already present)

```bash
bash setup_colabfold_databases.sh <path where to install database>
```

### On Perlmutter

All steps above, plus build Podman image

```bash
podman-hpc build -t boltz2:0.1.0 -f dockerfiles/Dockerfile .
podman-hpc migrate
```

## Running Workflows

### On Aster

- cd `/data/35y/nyl12-loop-engineering`
- Modify site config file `config/site/aster.yaml` - set paths to installed files
- Copy config file `config/config.yaml` to `config/<your_name>.yaml` or modify it in place
- Run workflow `start_worlflow_in_bg.sh <config_name (without .yaml)>`. Alternatively, look at
  `run_all_aster.sh` file and run relevant commands manually.

### On Perlmutter

- cd `/global/cfs/cdirs/m906/protein-design/nyl12-loop-engineering`
- Modify site config file `config/site/perlmutter.yaml` - set paths to installed files
- Copy config file `config/config.yaml` to `config/<your_name>.yaml` or modify it in place
- Submit SLURM job `sbatch run_all_perlmutter.sh <config_name (without .yaml)>`.
  Alternatively, start interactive job
  `salloc --nodes 1 --qos interactive --time 00:30:00 --constraint "gpu&hbm80g"  --account m906 --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-task=4`,
  look at `run_all_perlmutter.sh` file and run relevant commands manually.
 



