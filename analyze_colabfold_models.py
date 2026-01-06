import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from tabulate import tabulate
from pymol import cmd

import hydra
from hydra.core.hydra_config import HydraConfig


def parse_contig(contig: str):
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

    return res


def find_prediction_files(base_path: Path):
    result = []
    for folder in base_path.glob("colabfold*"):
        if folder.is_dir():
            json_file = list(folder.glob("*scores_rank_001_*model*.json"))[0]
            match = re.search(r"model_(\d+)", json_file.name)
            if match:
                model_number = int(match.group(1))
                pdb_file = list(folder.glob(f"*model_{model_number}*.pdb"))[0]
                sequence_name = folder.name.removeprefix("colabfold_")
                result.append({"model": pdb_file, "sequence_name": sequence_name,
                               "scores": json_file})
    return result


def get_model_loop_scores(conf, model_file):
    confidences = []
    contig_dict = parse_contig(conf.contig_map)
    with open(model_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                b_factor_str = line[60:66].strip()
                if b_factor_str:
                    confidences.append(float(b_factor_str))
    base_chain = conf.boltz.base_chain
    base_chain_start = conf.boltz.betachain_start
    start_idx = base_chain.find(base_chain_start)

    len_alpha = start_idx
    len_beta = len(base_chain) - len_alpha
    chains = [None] * 4
    for i in range(4):
        chains[i] = confidences[len_alpha * i:len_alpha * (i + 1)] + confidences[
            len_alpha * 4 + len_beta * i:len_alpha * 4 + len_beta * (i + 1)]
    loop_scores = {}
    n_loops = 0
    for loops in contig_dict.values():
        for loop in loops:
            n_loops += 1
            loop_score = 0
            for i in range(4):
                loop_score += np.array(chains[i][loop[1][0] - 1:loop[1][1] - 1]).mean()
            loop_score /= 4
            loop_scores[f"loop_{n_loops}"] = loop_score
    return loop_scores


def process_model(conf, model):
    with open(model["scores"], "r") as f:
        data = json.load(f)
        ptm = data["ptm"]
        iptm = data["iptm"]
        actifptm = data["actifptm"]
        plddt_array = np.array(data["plddt"])
        plddt = plddt_array.mean()

    loop_scores = get_model_loop_scores(conf, model["model"])
    res = loop_scores
    res["ptm"] = ptm
    res["iptm"] = iptm
    res["plddt"] = plddt
    res["actifptm"] = actifptm

    return res


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf: HydraConfig) -> None:
    if not conf.filtering.enable:
        return

    conf.base_dir = os.path.abspath(conf.base_dir)

    base_path = Path(conf.colabfold.output_dir)
    Path(conf.filtering.output_dir).mkdir(parents=True, exist_ok=True)


    models = find_prediction_files(base_path)

    rows = []
    for model in models:
        scores = {"sequence_name": model["sequence_name"]}
        scores.update(process_model(conf, model))
        rows += [scores]

    df = pd.DataFrame(rows)
    output_file = Path(conf.filtering.output_dir) / "colabfold_results.csv"
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
