from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, TextIO, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from jinja2 import Template
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

    cur_chain = None
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


def calculate_modified_chain(base_chain: str, merged_partial_chain: str, partial_chain_start: str,
                             contig_dict: ContigDict) -> str:
    """
    Calculate a single modified chain by replacing residues at indices
    based on a mapping parsed from a contig string.
    """
    partial_chains = merged_partial_chain.split(partial_chain_start)[1:]
    partial_chains = [partial_chain_start + partial_chain for partial_chain in partial_chains]
    # convert string representation to list, to simplify modification
    chain_list = chain_to_list(base_chain)
    for i, partial_chain in enumerate(partial_chains):
        update_base_chain_from_partial(chain_list, partial_chain, list(contig_dict.values())[i])
    return list_to_chain(chain_list)


def split_chain(chain: str, alpha_beta_split_string: str) -> Tuple[str, str]:
    alpha, beta = chain.split(alpha_beta_split_string, 1)
    return alpha, alpha_beta_split_string + beta


def get_modified_chains_from_fasta_file(source: str | TextIO, base_chain: str, partial_chain_start: str,
                                        alpha_beta_split_string: str, contig_dict: ContigDict) -> List[Dict[str, Any]]:
    chains = read_fasta_chains(source)
    for chain_dict in chains:
        res = calculate_modified_chain(base_chain, chain_dict["merged_partial_chain"], partial_chain_start, contig_dict)
        alpha, beta = split_chain(res, alpha_beta_split_string)
        chain_dict["modified_chain"] = res
        chain_dict["modified_alpha"] = alpha
        chain_dict["modified_beta"] = beta
    return chains


def get_modified_chains_from_dir(dir_name: str, pattern: str, base_chain: str, partial_chain_start: str,
                                 alpha_beta_split_string: str, contig_dict: ContigDict) -> List[Dict[str, Any]]:
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
                       msa_file: Optional[str] = None) -> None:
    context = {
        "alpha_seq": chain["modified_alpha"],
        "beta_seq": chain["modified_beta"],
        "smiles": molecule_smiles,
        "cif_file": cif_file,
    }
    if msa_file:
        context["msa_file"] = msa_file
    template_path = Path(template_path)
    template_str = template_path.read_text()
    template = Template(template_str)
    return template.render(context)


def save_chain(chain: Dict[str, Any], output_dir: str, template_path: str, cif_file: str, molecule_smiles: str,
               msa_file: Optional[str] = None) -> None:
    rendered_input = render_boltz_input(chain, template_path, cif_file, molecule_smiles, msa_file)

    fname = f"{chain['name']}_{chain['T']}_{chain['id']}.yaml"
    dir_path = Path(output_dir)
    output_path = dir_path / fname
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_input)


def save_chains(chains: List[Dict[str, Any]], output_dir: str, template_path: str, cif_file: str,
                molecule_smiles: str, msa_file: Optional[str] = None) -> None:
    logging.info(f"Writing {len(chains)} modified chains to folder {output_dir}")

    for chain in chains:
        save_chain(chain, output_dir, template_path, cif_file, molecule_smiles, msa_file)


def prepare_boltz_command(conf):
    yaml_dir = Path(conf.boltz.yaml_files_dir)
    folders = [d for d in yaml_dir.iterdir() if d.is_dir()]

    commands_boltz = []
    for folder in folders:
        if conf.boltz.batch_processing:
            boltz_files = [folder.resolve()]
        else:
            boltz_files = list(folder.glob("*.yaml"))

        for file in boltz_files:
            commands_boltz.append(f"{conf.boltz.command} {file} "
                                  f"--model {conf.boltz.boltz_params.model} "
                                  f"--output_format {conf.boltz.boltz_params.output_format} "
                                  f"{'--use_msa_server' if conf.boltz.boltz_params.use_msa_server else ''} "
                                  f"{'--use_potentials' if conf.boltz.boltz_params.use_potentials else ''} "
                                  f"--cache {conf.boltz.boltz_params.cache} "
                                  f"--out_dir {conf.boltz.boltz_params.output_dir}"
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

    for input_dir in dirs:
        input_file = os.path.basename(os.path.dirname(input_dir))

        modified_chains = get_modified_chains_from_dir(input_dir, conf.boltz.file_pattern, conf.boltz.base_chain,
                                                       conf.boltz.partial_chain_start, conf.boltz.betachain_start,
                                                       contig_dict)
        save_chains(modified_chains, os.path.join(conf.boltz.yaml_files_dir, input_file),
                    conf.boltz.boltz_input_template, conf.boltz.cif_file,
                    conf.boltz.molecule_smiles,
                    conf.boltz.boltz_params.msa if not conf.boltz.boltz_params.use_msa_server else None)

    prepare_boltz_command(conf)


if __name__ == "__main__":
    main()
