from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, TextIO, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
import yaml
from jinja2 import Template, Environment

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

ContigMappings = List[Tuple[Tuple[int, int], Tuple[int, int]]]
ContigDict = Dict[str, ContigMappings]


def parse_contig(contig: str) -> ContigDict:
    """
    Parse contig to create, for each chain, a list of index ranges
    between the base chain and the partial chain for mutable residues.

    Example:
    Input:  "A1-2,3-4,B1-5,B286-292,8-8,B301-308"
    Output: {
    {
        "A": [(3, 5), (3, 6)],
        "B": [(293, 300), (13, 20)]
    }
    """

    cur_chain = 'A'
    intervals = contig.split(",")
    reduced_index = 0
    prev_end_index = 0
    res = {}
    current_mappings = []
    for interval in intervals:
        start_str, end_str = interval.split("-")
        end = int(end_str)
        if start_str[0].isalpha():  # residues included, just update indices
            chain = start_str[0]
            start = int(start_str[1:])
            n_residues_to_keep = end - start + 1
            # If switching chains, flush current
            if chain != cur_chain:
                if cur_chain:
                    res[cur_chain] = current_mappings
                current_mappings = []
                cur_chain = chain
                reduced_index = 0

            reduced_index += n_residues_to_keep
            prev_end_index = end
        else:  # residues modified, create intervals
            n_modify_origin = int(start_str)
            n_in_new = int(end_str)
            current_mappings.append(((prev_end_index + 1, prev_end_index + n_modify_origin),
                                     (reduced_index + 1, reduced_index + n_in_new)))
            prev_end_index += n_modify_origin
            reduced_index += n_in_new
    if cur_chain:
        res[cur_chain] = current_mappings

    logging.info("Modified chains from contig:\n" + "\n".join(f"  {k}: {len(v)} interval(s)" for k, v in res.items()))
    return res


def read_fasta_chains(source) -> List[Dict[str, Any]]:
    """
    Reads FASTA chains from a file path or file-like object.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a FASTA entry with:
            - 'name': the first part of the header (before any commas)
            - other key-value pairs parsed from header parts with 'key=value' format
            - 'chain': the corresponding chain string
    """
    entries = []

    if isinstance(source, str):
        f = open(source, "r")
        close_when_done = True
    else:
        f = source
        close_when_done = False

    try:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = lines[i][1:].strip()
                chain = lines[i + 1].strip()
            else:
                continue

            header_parts = [part.strip() for part in header.split(",")]
            entry = {}

            if header_parts:
                entry["name"] = header_parts[0]

            for part in header_parts[1:]:
                if "=" in part:
                    key, val = part.split("=", 1)
                    entry[key.strip()] = val.strip()

            entry["merged_partial_chain"] = chain
            entries.append(entry)

    finally:
        if close_when_done:
            f.close()

    return entries[1:]


def chain_to_list(chain: str) -> List[Tuple[int, str]]:
    return [(i + 1, char) for i, char in enumerate(chain)]


def list_to_chain(sl: List[Tuple[int, str]]) -> str:
    chars = [t[1] for t in sl]
    return "".join(chars)


def find_index(lst: List[Tuple[int, str]], index: int) -> int:
    for i, x in enumerate(lst):
        if x[0] == index:
            return i
    return -1


def update_base_chain_from_partial(base_chain_list: List[Tuple[int, str]], partial_chain: str,
                                   mappings: ContigMappings) -> None:
    """Replacing indices in base chain with corresponding (from contig) indices in the partial chain."""
    for mapping in mappings:
        base_ind_start = mapping[0][0]
        base_ind_end = mapping[0][1]
        partial_ind_start = mapping[1][0]
        partial_ind_end = mapping[1][1]
        # prepare list of modified amino acids, accounting for indices to start from 0 in Python lists
        replace = [(-1, l) for l in list(partial_chain)[partial_ind_start - 1:partial_ind_end]]
        # Find where to apply modifications in the base chain by using the original indices,
        # stored as the first element of each tuple. This enables multiple sequential modifications,
        # and allows insertions or deletions (i.e., the modified segment may be longer or shorter
        # than the original).
        i1 = find_index(base_chain_list, base_ind_start)
        i2 = find_index(base_chain_list, base_ind_end)
        base_chain_list[i1:i2 + 1] = replace


def calculate_modified_chain(base_chain: str, merged_partial_chain: str, partial_chain_start: Optional[str],
                             contig_dict: ContigDict) -> str:
    """
    Calculate a single modified chain by replacing residues at indices
    based on a mapping parsed from a contig string.
    """
    if partial_chain_start:
        partial_chains = merged_partial_chain.split(partial_chain_start)[1:]
        partial_chains = [partial_chain_start + partial_chain for partial_chain in partial_chains]
    else:
        partial_chains=[merged_partial_chain]
    # convert string representation to list, to simplify modification
    chain_list = chain_to_list(base_chain)
    for i, partial_chain in enumerate(partial_chains):
        update_base_chain_from_partial(chain_list, partial_chain, list(contig_dict.values())[i])
    return list_to_chain(chain_list)


def split_chain(chain: str, alpha_beta_split_string: Optional[str]) -> Tuple[str, str]:
    if alpha_beta_split_string:
        alpha, beta = chain.split(alpha_beta_split_string, 1)
        return alpha, alpha_beta_split_string + beta
    else:
        return chain, None


def get_modified_chains_from_fasta_file(source: str | TextIO, base_chain: str, partial_chain_start: Optional[str],
                                        alpha_beta_split_string: Optional[str], contig_dict: ContigDict) -> List[
    Dict[str, Any]]:
    chains = read_fasta_chains(source)
    for chain_dict in chains:
        res = calculate_modified_chain(base_chain, chain_dict["merged_partial_chain"], partial_chain_start, contig_dict)
        alpha, beta = split_chain(res, alpha_beta_split_string)
        chain_dict["modified_chain"] = res
        chain_dict["modified_alpha"] = alpha
        chain_dict["modified_beta"] = beta
    return chains

def get_modified_chains_from_dir(dir_name: str, pattern: str, base_chain: str, partial_chain_start: Optional[str],
                                 alpha_beta_split_string: Optional[str], contig_dict: ContigDict) -> List[
    Dict[str, Any]]:
    folder = Path(dir_name)
    fa_files = list(folder.glob(f"{pattern}"))
    res = []
    logging.info(f"Processing {len(fa_files)} fasta files from folder {dir_name}")

    for file in fa_files:
        res += get_modified_chains_from_fasta_file(file.resolve().as_posix(), base_chain, partial_chain_start,
                                                   alpha_beta_split_string,
                                                   contig_dict)

    return res


def render_boltz_input(chain: Dict[str, Any], template_path: str, cif_file: str, molecule_smiles: str,
                       output_dir: str, conf, msa_file) -> None:
    context = {
        "alpha_seq": chain["modified_alpha"],
        "beta_seq": chain["modified_beta"],
        "smiles": molecule_smiles,
        "cif_file": cif_file,
        "constraints": OmegaConf.to_container(conf.boltz.constraints, resolve=True),
        "properties": OmegaConf.to_container(conf.boltz.properties, resolve=True),
    }
    if not conf.boltz.boltz_params.use_msa_server:
        context["msa_file"] = os.path.join(output_dir, conf.boltz.colabfold.output_folder, msa_file)
    template_path = Path(template_path)
    template_str = template_path.read_text()

    env = Environment()
    env.filters['to_yaml'] = lambda value: yaml.safe_dump(value, default_flow_style=False, sort_keys=False)

    template = env.from_string(template_str)
    return template.render(context)


def save_chain(chain: Dict[str, Any], output_dir: str, template_path: str, cif_file: str, molecule_smiles: str,
               conf) -> None:
    chain_name = f"{chain['name']}_{chain['T']}_{chain['id']}"

    msa_file = f"{chain_name}"
    rendered_input = render_boltz_input(chain, template_path, cif_file, molecule_smiles, output_dir, conf,
                                        msa_file)

    for model in range(conf.boltz.models_per_sequence):
        fname = f"{chain_name}_model_{model}.yaml"
        dir_path = Path(output_dir) / conf.boltz.yaml_files_dir
        output_path = dir_path / fname
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered_input)


def copy_sequence(input: str, n_copies: int) -> str:
    inputs = [input] * n_copies
    return ":".join(inputs)


def save_fasta(merge_fasta: bool, chains: List[Dict[str, Any]], n_copies: int,output_dir: str):
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    if merge_fasta:
        output_path = dir_path / "fasta.fa"
        output_path.write_text("")
    for chain in chains:
        if chain['modified_beta']:
            chain_str = f"{copy_sequence(chain['modified_alpha'],n_copies)}:{copy_sequence(chain['modified_beta'],n_copies)}"
        else:
            chain_str = f"{copy_sequence(chain['modified_alpha'],n_copies)}"
        fasta_input = (
                f">{chain['name']}_{chain['T']}_{chain['id']}\n" + chain_str
        )
        if not merge_fasta:
            fname_fasta = f"{chain['name']}_{chain['T']}_{chain['id']}.fa"
            output_path = dir_path / fname_fasta
            output_path.write_text(fasta_input)
        else:
            with output_path.open("a") as f:
                f.write(fasta_input + "\n")  # add newline if needed


def save_chains(chains: List[Dict[str, Any]], output_dir: str, template_path: str, cif_file: str,
                molecule_smiles: str, conf) -> None:
    logging.info(
        f"Writing {len(chains)} modified chains/{conf.boltz.models_per_sequence} model(s) each to folder {output_dir}")

    for chain in chains:
        save_chain(chain, output_dir, template_path, cif_file, molecule_smiles, conf)


def prepare_colabfold_search_command(conf):
    input_files_dir = Path(conf.boltz.input_files_dir)
    folders = [d for d in input_files_dir.iterdir() if d.is_dir()]

    commands_colabfold = []
    for folder in folders:
        fasta_folder = folder / conf.boltz.colabfold.fasta_output_folder
        colabfold_files = list(fasta_folder.glob("*.fa"))

        for file in colabfold_files:
            commands_colabfold.append(f"{conf.boltz.colabfold.search_command} {file} "
                                      f"{conf.boltz.colabfold.database} {folder / conf.boltz.colabfold.output_folder}"
                                      )

    print("Example colabfold_search command:")
    print(commands_colabfold[-1])

    cmds_filename_colabfold = os.path.join(conf.boltz.output_dir, "commands_colabfold_search.sh")
    with open(cmds_filename_colabfold, "w") as file:
        file.write("\n".join(commands_colabfold))


def prepare_msas_convert(conf):
    input_files_dir = Path(conf.boltz.input_files_dir)
    folders = [d for d in input_files_dir.iterdir() if d.is_dir()]

    commands_msas_convert = []
    for folder in folders:
        colabfold_search_output_folder = folder / conf.boltz.colabfold.output_folder
        yaml_output_folder = folder / conf.boltz.yaml_files_dir
        yaml_files = list(yaml_output_folder.glob("*_model_0.yaml"))
        for file in yaml_files:
            fname = str(colabfold_search_output_folder / file.name.replace("_model_0.yaml", ""))
            csv_file_alpha = fname + "_alpha.csv"
            csv_file_beta = fname + "_beta.csv"
            msas_file = fname + ".a3m"
            commands_msas_convert.append(f"{conf.boltz.colabfold.convert_command}  --msas_file {msas_file} "
                                         f" --csv_alpha {csv_file_alpha}  --csv_beta {csv_file_beta}"
                                         )

    print("Example msas_convert command:")
    print(commands_msas_convert[-1])

    cmds_filename_colabfold = os.path.join(conf.boltz.output_dir, "commands_msas_convert.sh")
    with open(cmds_filename_colabfold, "w") as file:
        file.write("\n".join(commands_msas_convert))



def prepare_colabfold_command(conf):
    input_files_dir = Path(conf.colabfold.input_files_dir)
    folders = [d for d in input_files_dir.iterdir() if d.is_dir()]
    commands_colabfold = []
    for folder in folders:
        colabfold_files = list(folder.glob("*.fa"))
        for file in colabfold_files:
            commands_colabfold.append(f"{conf.colabfold.command} "
                                  f"{'--templates --custom-template-path' if conf.colabfold.use_templates else ''} "
                                  f"{conf.colabfold.custom_template_path  if conf.colabfold.use_templates else ''} "
                                  f"--data {conf.colabfold.af2_weights_folder} "
                                  f"--msa-mode {conf.colabfold.msa_mode} "
                                  f"{conf.colabfold.extra_params if conf.colabfold.extra_params else ''} "
                                  f"{file} "
                                  f"{conf.colabfold.output_dir}/colabfold_{file.stem}"
                                  )
    print("Example colabfold command:")
    print(commands_colabfold[-1])

    cmds_filename_colabfold = os.path.join(conf.colabfold.output_dir, "commands_colabfold.sh")
    with open(cmds_filename_colabfold, "w") as file:
        file.write("\n".join(commands_colabfold))


def prepare_boltz_command(conf):
    input_files_dir = Path(conf.boltz.input_files_dir)
    folders = [d for d in input_files_dir.iterdir() if d.is_dir()]

    commands_boltz = []
    for folder in folders:
        boltz_yaml_folder = folder / conf.boltz.yaml_files_dir

        if conf.boltz.batch_processing:
            boltz_files = [boltz_yaml_folder.resolve()]
        else:
            boltz_files = list(boltz_yaml_folder.glob("*.yaml"))

        for file in boltz_files:
            commands_boltz.append(f"{conf.boltz.command} {file} "
                                  f"--model {conf.boltz.boltz_params.model} "
                                  f"--output_format {conf.boltz.boltz_params.output_format} "
                                  f"{'--use_msa_server' if conf.boltz.boltz_params.use_msa_server else ''} "
                                  f"{'--use_potentials' if conf.boltz.boltz_params.use_potentials else ''} "
                                  f"{'--affinity_mw_correction' if conf.boltz.boltz_params.affinity_mw_correction else ''} "
                                  f"{'--no_kernels' if conf.boltz.boltz_params.no_kernels else ''} "
                                  f"--cache {conf.boltz.boltz_params.cache} "
                                  f"--recycling_steps {conf.boltz.boltz_params.recycling_steps} "
                                  f"--sampling_steps {conf.boltz.boltz_params.sampling_steps} "
                                  f"--diffusion_samples {conf.boltz.boltz_params.diffusion_samples} "
                                  f"{conf.boltz.boltz_params.extra_params if conf.boltz.boltz_params.extra_params else ''} "
                                  f"--out_dir {conf.boltz.boltz_params.output_dir}/{folder.name}"
                                  )
    print("Example Boltz command:")
    print(commands_boltz[-1])

    cmds_filename_boltz = os.path.join(conf.boltz.output_dir, "commands_boltz2.sh")
    with open(cmds_filename_boltz, "w") as file:
        file.write("\n".join(commands_boltz))


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf: HydraConfig) -> None:
    conf.base_dir = os.path.abspath(conf.base_dir)

    logging.info(
        "Preparing boltz inputs:\n" +
        f"contig_map: {conf.contig_map}\n" +
        OmegaConf.to_yaml(conf.boltz, resolve=True)
    )

    contig_dict = parse_contig(conf.contig_map)

    matches = glob.glob(conf.boltz.input_dir_pattern)
    dirs = [d for d in matches if os.path.isdir(d)]
    all_modified_chains = []
    for input_dir in dirs:
        input_file = os.path.basename(os.path.dirname(input_dir))

        modified_chains = get_modified_chains_from_dir(input_dir, conf.boltz.file_pattern, conf.boltz.base_chain,
                                                       conf.boltz.partial_chain_start, conf.boltz.betachain_start,
                                                       contig_dict)
        all_modified_chains += modified_chains
        save_chains(modified_chains, os.path.join(conf.boltz.input_files_dir, input_file),
                    conf.boltz.boltz_input_template, conf.boltz.cif_file,
                    conf.boltz.molecule_smiles,
                    conf)

        if not conf.boltz.boltz_params.use_msa_server:
            save_fasta(True, modified_chains, 1, os.path.join(conf.boltz.input_files_dir, input_file,conf.boltz.colabfold.fasta_output_folder))
        if conf.colabfold.enable:
            save_fasta(False, modified_chains, 4, os.path.join(conf.colabfold.input_files_dir, input_file))

    prepare_boltz_command(conf)
    if conf.colabfold.enable:
        prepare_colabfold_command(conf)

    if not conf.boltz.boltz_params.use_msa_server:
        prepare_colabfold_search_command(conf)
        prepare_msas_convert(conf)


if __name__ == "__main__":
    main()
