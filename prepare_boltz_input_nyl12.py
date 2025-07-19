import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, TextIO

from jinja2 import Template

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


def read_fasta_chains(source: str | TextIO) -> List[Dict[str, Any]]:
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


def render_boltz_input(chain: Dict[str, Any], template_path: str, cif_file: str, molecule_smiles: str):
    context = {
        "alpha_seq": chain["modified_alpha"],
        "beta_seq": chain["modified_beta"],
        "smiles": molecule_smiles,
        "cif_file": cif_file,
    }
    template_path = Path(template_path)
    template_str = template_path.read_text()
    template = Template(template_str)
    return template.render(context)


def save_chain(chain: Dict[str, Any], output_dir: str, template_path: str, cif_file: str, molecule_smiles: str) -> None:
    rendered_input = render_boltz_input(chain, template_path, cif_file, molecule_smiles)

    fname = f"{chain['name']}_{chain['T']}_{chain['id']}.yaml"
    dir_path = Path(output_dir)
    output_path = dir_path / fname
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(rendered_input)


def save_chains(chains: List[Dict[str, Any]], output_dir: str, template_path: str, cif_file: str,
                molecule_smiles: str) -> None:
    logging.info(f"Writing {len(chains)} modified chains to folder {output_dir}")

    for chain in chains:
        save_chain(chain, output_dir, template_path, cif_file, molecule_smiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Input directory with partialmpnn files")
    parser.add_argument("--file-pattern", default="nyl12*.fa", help="Fasta file pattern in input directory")
    parser.add_argument("--base-chain",
                        default="MAASSTDNILHFDFPEVQIGTAINPEGPTGITLFYFPKGVQASVDIQGGSVGTFFTQEKMQQGEAYLDGVAFTGGGILGLEAVAGAVSSLFADQTKNEVQFRRMPLISGAVIFDYTPRQNMIYPDKALGQKAFAALSAGQFVQGRHGAGVSASVGKLLRDGFQLAGQGGAFAQIGKTKIAVFTVVNAVGVILDEKGEVIYGLPKGATKQTLNQQVTELLQQPKKPFWPEPKNTTLTIVITNEKLAPRHLKQLGRQVHHALSQVIHPYATILDGDVLYTVSTRSIESDLYAPGADIESDLNAKFIYLGMVAGELAKQAVWSAVGYSHRP",
                        help="Base chain to be modified")
    parser.add_argument("--partial-chain-start", default="MAASS", help="chain defining start of subchain")
    parser.add_argument("--betachain-start", default="TTLTIVIT", help="chain defining start of beta chain")
    parser.add_argument("--cif-file", help="Path to the template cif file")
    parser.add_argument("--boltz-input-template", default="templates/boltz.yaml.j2",
                        help="Path to the template file for boltz input")

    parser.add_argument("--molecule-smiles", default="[NH3+]CCCCCC(NCCCCCC(NCCCCCC(NCCCCCC([O-])=O)=O)=O)=O",
                        help="Molecule in smiles format")
    parser.add_argument("--contig-string",
                        default="A1-5,A18-114,7-7,A122-150,14-14,A165-221,A233-328,B1-5,B286-292,8-8,B301-308,C1-5,D1-5,D29-35,5-5,D41-95,9-9,D105-110",
                        help="Contig string defining conditional information and variable regions (RFdiffusion format)")
    parser.add_argument("--output-dir", help="Output directory with inputs for boltz")
    args = parser.parse_args()

    logging.info("Preparing boltz input:\n" + "\n".join(f"  {k}: {v}" for k, v in vars(args).items()))

    contig_dict = parse_contig(args.contig_string)
    modified_chains = get_modified_chains_from_dir(args.input_dir, args.file_pattern, args.base_chain,
                                                   args.partial_chain_start, args.betachain_start, contig_dict)
    save_chains(modified_chains, args.output_dir, args.boltz_input_template, args.cif_file, args.molecule_smiles)


if __name__ == "__main__":
    main()
