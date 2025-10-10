# based on https://github.com/ikalvet/heme_binder_diffusion/blob/main/pipeline.ipynb

import os, glob
import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig

def escape_commas(str):
    return str.replace(",","\\\\,")


def run_diffusion(conf):

    conf.base_dir = os.path.abspath(conf.base_dir)

    if not os.path.exists(conf.work_dir):
        os.makedirs(conf.work_dir, exist_ok=True)

    print(f"Working directory: {conf.work_dir}")

   # Set up diffusion run
    diffusion_inputs = glob.glob(f"{conf.diffusion.input_dir}/*.pdb")
    print(f"Found {len(diffusion_inputs)} PDB files")

    diffusion_dir = conf.diffusion.output_dir
    if not os.path.exists(diffusion_dir):
        os.makedirs(diffusion_dir, exist_ok=False)

    ## Set up diffusion commands based on the input PDB file(s)
    ## Diffusion jobs are run in separate directories for each input PDB

    commands_diffusion = []
    cmds_filename = os.path.join(diffusion_dir, "commands_diffusion.sh")
    diffusion_rundirs = []



    with open(cmds_filename, "w") as file:
        for p in diffusion_inputs:
            pdbname = os.path.basename(p).replace(".pdb", "")
            pdb_dir = os.path.join(diffusion_dir, pdbname)
            cmd = f"{conf.diffusion.command} " \
                  f"inference.input_pdb={p} " \
                  f"inference.output_prefix={pdb_dir}/out/{pdbname} " \
                  f"inference.model_runner={conf.diffusion.inference.model_runner} " \
                  f"inference.ligand={conf.diffusion.inference.ligand} " \
                  f"inference.num_designs={conf.diffusion.inference.num_designs} " \
                  f"inference.ckpt_path={conf.diffusion.inference.ckpt_path} " \
                  f"model.freeze_track_motif={conf.diffusion.model.freeze_track_motif} " \
                  f"potentials.guiding_potentials=[{escape_commas(conf.diffusion.potentials.guiding_potential)}] " \
                  f"potentials.guide_scale={conf.diffusion.potentials.guide_scale} " \
                  f"contigmap.contigs=[{escape_commas(conf.contig_map)}] " \
                  f"potentials.guide_decay={conf.diffusion.potentials.guide_decay} " \
                  f"diffuser.T={conf.diffusion.diffuser.T} " \
                  "hydra.run.dir="+pdb_dir+"/outputs/\\${now:%Y-%m-%d}/\\${now:%H-%M-%S}"

            commands_diffusion.append(cmd)
            diffusion_rundirs.append(pdbname)
            file.write(cmd)

    print(f"Example diffusion command:\n {cmd}")

    print(f"Wrote diffusion commands to {cmds_filename}")
    print(f"{len(commands_diffusion)} diffusion jobs to run")


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf: HydraConfig) -> None:
    run_diffusion(conf)


if __name__ == '__main__':
    main()
