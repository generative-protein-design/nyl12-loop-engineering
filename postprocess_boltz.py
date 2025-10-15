import json
import os
import shutil
from pathlib import Path

import pandas as pd
from natsort import natsorted
from tabulate import tabulate
from pymol import cmd

import hydra
from hydra.core.hydra_config import HydraConfig


def load_atoms_from_csv(csv_file: str):
    """Read atom selections from CSV and return dict mapping atom_id → [chain, resi, atom_name]."""
    df = pd.read_csv(csv_file)
    atoms = {}
    for _, row in df.iterrows():
        atoms[row["atom_id"]] = [str(row["chain"]), str(row["resi"]), str(row["atom_name"])]
    return atoms


def distance(mol, atom1, atom2):
    a1 = "mol1//" + "/".join(atom1)
    a2 = "mol1//" + "/".join(atom2)
    d = cmd.distance("d1", a1, a2)
    return d


def angle(mol, atom1, atom2, atom3):
    a1 = "mol1//" + "/".join(atom1)
    a2 = "mol1//" + "/".join(atom2)
    a3 = "mol1//" + "/".join(atom3)
    a = cmd.angle("a1", a1, a2, a3)
    return a


def dihedral(mol, atom1, atom2, atom3, atom4):
    a1 = "mol1//" + "/".join(atom1)
    a2 = "mol1//" + "/".join(atom2)
    a3 = "mol1//" + "/".join(atom3)
    a4 = "mol1//" + "/".join(atom4)
    t = cmd.dihedral("t1", a1, a2, a3, a4)
    return t


def find_prediction_files(base_path: Path):
    triples = []
    for pdb_file in base_path.rglob("boltz*/predictions/nyl*/*.pdb"):
        parent = pdb_file.parent
        confidence_files = list(parent.glob("confidence*.json"))
        affinity_files = list(parent.glob("affinity*.json"))

        confidence = confidence_files[0] if confidence_files else None
        affinity = affinity_files[0] if affinity_files else None

        if confidence is not None:
            triples.append((confidence, affinity, pdb_file))

    return natsorted(triples, key=lambda x: str(x[2]))


def copy_filtered_pdbs(base_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdb_file in base_path.rglob("predictions/nyl*/*.pdb"):
        new_name = f"{pdb_file.name.replace('_model_0.pdb','.pdb')}"

        dest = output_dir / new_name
        shutil.copy2(pdb_file, dest)
        print(f"Copied {pdb_file} → {dest}")


def compute_metrics(mol, atoms):
    """Wrapper to compute geometry metrics using PyMOL API."""
    d1 = distance(mol, atoms["thr_og1"], atoms["carbonyl_c"])
    d2 = distance(mol, atoms["carbonyl_o"], atoms["backbone_n"])
    d3 = distance(mol, atoms["carbonyl_o"], atoms["sidechain_nd2"])
    a1 = angle(mol, atoms["thr_og1"], atoms["carbonyl_c"], atoms["carbonyl_o"])
    t1 = dihedral(mol, atoms["thr_og1"], atoms["carbonyl_c"], atoms["carbonyl_o"], atoms["substrate_c"])
    t2 = dihedral(mol, atoms["substrate_n"], atoms["substrate_c"], atoms["sidechain_nd2"], atoms["backbone_n"])
    return d1, d2, d3, a1, t1, t2


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf: HydraConfig) -> None:
    conf.base_dir = os.path.abspath(conf.base_dir)

    base_path = Path(conf.boltz.output_dir)

    # Load atoms of interest from CSV
    atoms = load_atoms_from_csv(conf.postprocessing.atom_selections_file)

    # Step 1: Find input files
    files = find_prediction_files(base_path)

    rows = []
    for confidence_json, affinity_json, mol in files:
        model_number = mol.parent.parent.parent.name.split("_")[-1]
        cmd.delete("all")
        cmd.load(str(mol), "mol1")

        # Compute geometry
        d1, d2, d3, a1, t1, t2 = compute_metrics(mol, atoms)

        # Read confidence.json
        with open(confidence_json, "r") as f:
            conf_data = json.load(f)

        affinity_pred_value = None
        affinity_probability_binary = None
        if affinity_json is not None:
            with open(affinity_json, "r") as f:
                aff_data = json.load(f)
            affinity_pred_value = aff_data.get("affinity_pred_value")
            affinity_probability_binary = aff_data.get("affinity_probability_binary")

        new_filename = f"{mol.name.replace('_model_0.pdb','.pdb')}"

        rows.append(
            {
                "model_num": model_number,
                "model": str(base_path / "pdb" / new_filename),
                "d1": round(d1, 2),
                "d2": round(d2, 2),
                "d3": round(d3, 2),
                "a1": round(a1, 1),
                "t1": round(t1, 1),
                "t2": round(t2, 1),
                "confidence": round(conf_data["confidence_score"], 3),
                "ligand_iptm": round(conf_data["ligand_iptm"], 3),
                "affinity_pred_value": (
                    round(affinity_pred_value, 3) if affinity_pred_value is not None else None
                ),
                "affinity_probability_binary": (
                    round(affinity_probability_binary, 3)
                    if affinity_probability_binary is not None
                    else None
                ),
            }
        )

    df = pd.DataFrame(rows)

    df = df.rename(
        columns={
            "confidence": "conf",
            "ligand_iptm": "lig_iptm",
            "affinity_pred_value": "aff_pred",
            "affinity_probability_binary": "aff_prob",
        }
    )

    df = df.sort_values(by="lig_iptm", ascending=False)
    df.to_csv( Path(conf.postprocessing.output_dir)/"full_metrics.csv", index=False)

    filtered = df[
        (df["d1"] < conf.postprocessing.d1_max)
        & (df["d2"] < conf.postprocessing.d2_max)
        & (df["d3"] < conf.postprocessing.d3_max)
        & (df["a1"].between(conf.postprocessing.a1_min, conf.postprocessing.a1_max))
        & (df["t1"].abs().between(conf.postprocessing.t1_min, conf.postprocessing.t1_max))
    ]

    copy_filtered_pdbs(base_path, Path(conf.postprocessing.output_dir)/"pdb")

    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
    print("\nModels that pass all filters:")
    if filtered.empty:
        print("None found.")
    else:
        print(tabulate(filtered, headers="keys", tablefmt="psql", showindex=False))
        output_file = Path(conf.postprocessing.output_dir)/"filtered_metrics.csv"
        filtered.to_csv(output_file, index=False)
        print(f"\nFiltered results saved to {output_file}")


if __name__ == "__main__":
    main()

