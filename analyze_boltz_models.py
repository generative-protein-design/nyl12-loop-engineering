import json
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
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
    result = []
    for pdb_file in base_path.rglob("boltz*/predictions/*/*.pdb"):
        parent = pdb_file.parent
        confidence_files = list(parent.glob("confidence*.json"))
        affinity_files = list(parent.glob("affinity*.json"))

        confidence = confidence_files[0] if confidence_files else None
        affinity = affinity_files[0] if affinity_files else None

        if confidence is not None:
            model_name = f"{pdb_file.stem.replace('_model_0', '', 1)}"
            sequence_name = model_name.rsplit('_model_', 1)[0]
            input_name = sequence_name.rsplit('_', 3)[0]
            result.append({"confidence": confidence, "affinity": affinity, "model": pdb_file, "model_name": model_name,
                           "sequence_name": sequence_name, "input_name": input_name})

    return result


def get_sequence_from_boltz_input(file):
    with open(file, 'r') as f:
        data = yaml.safe_load(f)

    protein_sequences = []

    for entry in data.get('sequences', []):
        if 'protein' in entry:
            seq = entry['protein'].get('sequence')
            if seq:
                protein_sequences.append(seq)

    return "".join(protein_sequences)


def copy_top_sequences(filtered_data, conf, output_dir: Path):
    seq_dir = output_dir / "top_sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    input_files_dir = Path(conf.boltz.input_files_dir)
    folders = [d for d in input_files_dir.iterdir() if d.is_dir()]
    yaml_files = []
    for folder in folders:
        yaml_output_folder = folder / conf.boltz.yaml_files_dir
        yaml_files += (list(yaml_output_folder.glob("*_model_*.yaml")))

    top_models = \
    filtered_data.drop_duplicates(subset='sequence_name', keep='first').head(conf.filtering.number_of_top_sequences)[
        'model'].tolist()
    yaml_files = [file for file in yaml_files if file.stem in top_models]
    for file in yaml_files:
        sequence_name = file.stem.rsplit('_model_', 1)[0]
        output_name = seq_dir / f"{sequence_name}.fa"
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(f">{sequence_name}\n")
            seq = get_sequence_from_boltz_input(file)
            f.write(seq)
    print(f"Wrote {len(yaml_files)} sequence(s) to {seq_dir}")


def copy_pdb_files(files, copy_relaxed: bool, output_dir: Path):
    original_dir = output_dir / "original_pdb"
    original_dir.mkdir(parents=True, exist_ok=True)
    if copy_relaxed:
        relaxed_dir = output_dir / "relaxed_pdb"
        relaxed_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        original_file = file['model']
        if copy_relaxed:
            relaxed_file = file['relaxed_model']
        else:
            relaxed_file = None
        new_name = f"{file['model_name']}.pdb"

        dest = original_dir / new_name
        shutil.copy2(original_file, dest)
        print(f"Copied {original_file} → {dest}")
        if relaxed_file:
            dest = relaxed_dir / new_name
            shutil.copy2(relaxed_file, dest)
            print(f"Copied {relaxed_file} → {dest}")


def compute_metrics(mol, atoms):
    """Wrapper to compute geometry metrics using PyMOL API."""
    d1 = distance(mol, atoms["thr_og1"], atoms["carbonyl_c"])
    d2 = distance(mol, atoms["carbonyl_o"], atoms["backbone_n"])
    d3 = distance(mol, atoms["carbonyl_o"], atoms["sidechain_nd2"])
    d4 = distance(mol, atoms["substrate_c73"], atoms["resi161_ca"])
    a1 = angle(mol, atoms["thr_og1"], atoms["carbonyl_c"], atoms["carbonyl_o"])
    t1 = dihedral(mol, atoms["thr_og1"], atoms["carbonyl_c"], atoms["carbonyl_o"], atoms["substrate_c"])
    t2 = dihedral(mol, atoms["substrate_n"], atoms["substrate_c"], atoms["sidechain_nd2"], atoms["backbone_n"])
    return d1, d2, d3, d4, a1, t1, t2


def add_relaxed_files(conf, files):
    relaxation_dir = Path(conf.relaxation.output_dir)
    for file in files:
        relaxed_model = relaxation_dir / file["model"].stem / "complex_min.pdb"
        file["relaxed_model"] = relaxed_model if relaxed_model.exists() else None
        energy_file = relaxation_dir / file["model"].stem / "energy.json"
        file["energy"] = energy_file if energy_file.exists() else None
    pass


def filter_by_affinity(files, conf: HydraConfig) -> None:
    atoms_unrelaxed = load_atoms_from_csv(conf.filtering.affinity.atom_selections_file)
    atoms_relaxed = load_atoms_from_csv(conf.filtering.affinity.atom_selections_file_relaxed)

    add_relaxed_files(conf, files)

    rows = []
    for file in files:
        confidence_json = file["confidence"]
        affinity_json = file["affinity"]
        energy_json = file["energy"]

        model_name = file["model_name"]

        if conf.filtering.affinity.remove_non_relaxable_models and not file["relaxed_model"]:
            print(f"relaxed model for {model_name} missing. Skipping")
            continue

        atoms = atoms_unrelaxed
        if conf.filtering.affinity.filter_relaxed_results:
            mol = file["relaxed_model"]
            if mol:
                mol = file["relaxed_model"]
                atoms = atoms_relaxed
            else:
                mol = file["model"]
                print(f"relaxed model for {model_name} missing. Using the unrelaxed one")
        else:
            mol = file["model"]

        cmd.delete("all")
        cmd.load(str(mol), "mol1")

        # Compute geometry
        d1, d2, d3, d4, a1, t1, t2 = compute_metrics(mol, atoms)

        if energy_json:
            with open(energy_json, "r") as f:
                energy_data = json.load(f)
        else:
            energy_data = {}

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

        interface_delta = energy_data.get("interface_delta", None)

        rows.append(
            {
                "sequence_name": file["sequence_name"],
                "model": model_name,
                "d1": round(d1, 2),
                "d2": round(d2, 2),
                "d3": round(d3, 2),
                "d4": round(d3, 2),
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
                "interface_delta":
                    float(interface_delta)
                    if interface_delta is not None
                    else None
                ,
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
    df_colabfold = pd.read_csv(Path(conf.filtering.output_dir) / "colabfold_results.csv")
    df = df.merge(df_colabfold, on="sequence_name", how="left")

    filtered = df[
        (df["d1"] < conf.filtering.affinity.d1_max)
        & (df["d2"] < conf.filtering.affinity.d2_max)
        & (df["d3"] < conf.filtering.affinity.d3_max)
        & (df["d4"] < conf.filtering.affinity.d4_max)
        & (df["interface_delta"] < conf.filtering.affinity.interface_delta_max)
        & (df["a1"].between(conf.filtering.affinity.a1_min, conf.filtering.affinity.a1_max))
        & (df["t1"].abs().between(conf.filtering.affinity.t1_min, conf.filtering.affinity.t1_max))
        ]

    return df, filtered


def filter_by_backbone(files, conf: HydraConfig) -> None:
    rows = []
    for file in files:
        confidence_json = file["confidence"]
        model_name = file["model_name"]
        # JMP: cmd.load... # load backbone model

        seqs = file["sequence_name"].split("_")
        seqs[-1], seqs[-2] = seqs[-2], seqs[-1]
        backbone_filename = "_".join(seqs) + ".pdb"
        backbone_file = Path(conf.ligand_mpnn.output_dir) / file["input_name"] / "backbones" / backbone_filename
        cmd.load(str(file["model"]), "mol1")
        cmd.load(str(backbone_file), "bb")
        # JMP: Compute RMSD between Boltz and corresponding RFDAA backbone model
        res = cmd.align("mol1", "bb")
        cmd.delete("mol1")
        cmd.delete("bb")

        # Read confidence.json
        with open(confidence_json, "r") as f:
            conf_data = json.load(f)

        rows.append(
            {
                "sequence_name": file["sequence_name"],
                "model": model_name,
                "confidence": round(conf_data["confidence_score"], 3),
                "ptm": round(conf_data["ptm"], 3),
                "complex_plddt": round(conf_data["complex_plddt"], 3),
                "ligand_iptm": round(conf_data["ligand_iptm"], 3),
                "rmsd": res[0],
            }
        )

    df = pd.DataFrame(rows)

    df = df.rename(
        columns={
            "confidence": "conf",
            "ligand_iptm": "lig_iptm",
        }
    )

    df = df.sort_values(by="conf", ascending=False)
    filtered = df[
        (df["rmsd"] < conf.filtering.backbone.max_rmsd)
    ]
    return df, filtered


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf: HydraConfig) -> None:
    conf.base_dir = os.path.abspath(conf.base_dir)
    base_path = Path(conf.boltz.output_dir)
    Path(conf.filtering.output_dir).mkdir(parents=True, exist_ok=True)
    files = find_prediction_files(base_path)

    if conf.filtering.affinity.enable:
        df, filtered = filter_by_affinity(files, conf)
    elif conf.filtering.backbone.enable:
        df, filtered = filter_by_backbone(files, conf)
    else:
        return

    df.to_csv(Path(conf.filtering.output_dir) / "full_metrics.csv", index=False)

    copy_pdb_files(files, conf.filtering.affinity.enable, Path(conf.filtering.output_dir))
    total_models = len(df)
    passed_models = len(filtered)

    copy_top_sequences(filtered, conf, Path(conf.filtering.output_dir))

    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
    if filtered.empty:
        print("None found.")
    else:
        print(tabulate(filtered, headers="keys", tablefmt="psql", showindex=False))
        output_file = Path(conf.filtering.output_dir) / "filtered_metrics.csv"
        filtered.to_csv(output_file, index=False)
        print(f"\nFiltered results saved to {output_file}")

    print(f"\n{passed_models} of {total_models} models passed all filters:")


if __name__ == "__main__":
    main()
