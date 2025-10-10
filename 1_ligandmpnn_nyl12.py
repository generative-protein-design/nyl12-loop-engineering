# based on https://github.com/ikalvet/heme_binder_diffusion/blob/main/pipeline.ipynb

import os, glob
import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig


def run_ligand(conf):
    conf.base_dir = os.path.abspath(conf.base_dir)

    if not os.path.exists(conf.work_dir):
        os.makedirs(conf.work_dir, exist_ok=True)

    print(f"Working directory: {conf.work_dir}")

    diffusion_inputs = glob.glob(f"{conf.diffusion.input_dir}/*.pdb")
    print(f"Found {len(diffusion_inputs)} PDB files")

    diffusion_rundirs = []
    for p in diffusion_inputs:
        pdbname = os.path.basename(p).replace(".pdb", "")
        diffusion_rundirs.append(pdbname)

    print('diffusion_rundirs', diffusion_rundirs)

    commands_mpnn = []
    for diffusion_rundir in diffusion_rundirs:
        diffused_backbones_good = glob.glob(f"{conf.diffusion.output_dir}/{diffusion_rundir}/out/*.pdb")
        assert len(diffused_backbones_good) > 0, "No good backbones found!"

        output_dir = os.path.join(conf.ligand_mpnn.output_dir, diffusion_rundir)
        os.makedirs(output_dir, exist_ok=True)

        ### Parse diffusion output TRB files to extract fixed motif residues
        ## These residues will not be redesigned with ligandMPNN
        mask_json_cmd = f"{conf.ligand_mpnn.make_maskdict_command} --out {output_dir}/masked_pos.jsonl --trb"
        for d in diffused_backbones_good:
            mask_json_cmd += " " + d.replace(".pdb", ".trb")
        p = subprocess.Popen(mask_json_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()

        assert not err and os.path.exists(
            f"{output_dir}/masked_pos.jsonl"), "Failed to create masked positions JSONL file"

        for T in conf.ligand_mpnn.mpnn_temperatures:
            for f in diffused_backbones_good:
                commands_mpnn.append(f"{conf.ligand_mpnn.command} "
                                     f"--model_type ligand_mpnn --ligand_mpnn_use_atom_context 1 "
                                     f"--ligand_mpnn_use_side_chain_context 1 "
                                     f"--fixed_residues_multi {output_dir}/masked_pos.jsonl --out_folder {output_dir} "
                                     f"--number_of_batches {conf.ligand_mpnn.mpnn_outputs_per_temperature} --temperature {T} "
                                     f"--file_ending _{T} "
                                     f"--omit_AA {conf.ligand_mpnn.mpnn_omit_AAs} --pdb_path {f} "
                                     f"--checkpoint_ligand_mpnn {conf.ligand_mpnn.model_params_file}")

    cmds_filename_mpnn = os.path.join(conf.ligand_mpnn.output_dir,"commands_mpnn.sh")
    with open(cmds_filename_mpnn, "w") as file:
        file.write("\n".join(commands_mpnn))

    print("Example MPNN command:")
    print(commands_mpnn[-1])


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf: HydraConfig) -> None:
    run_ligand(conf)


if __name__ == '__main__':
    main()
