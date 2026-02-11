#!/usr/bin/env bash

set -e

export NTASKS_CPU=16
export NTASKS_GPU=2

export BASE_DIR=`pwd`

export CONFIG_NAME=${1:-config}

export SOURCE_PATH=$(dirname "$(realpath "$0")")

export OUTPUT_FOLDER=$(pixi run --manifest-path $SOURCE_PATH/pixi.toml --as-is -e rf python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.work_dir).name)")

export HYDRA_PARAMS="hydra.searchpath=[$SOURCE_PATH/config] +site=aster source_dir=$SOURCE_PATH --config-path=`pwd`/config --config-name=${CONFIG_NAME}"

enabled() {
  pixi run --manifest-path "$SOURCE_PATH/pixi.toml" --as-is -e rf python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml')
res = OmegaConf.select(cfg, '${1}.enable', default=False)
print(res)
"
}


run_task() {
    local DONE="$1"
    if [[ -f ${OUTPUT_FOLDER}/${DONE}.done ]]; then
        echo "[SKIP] $DONE"
        return
    fi
    echo "[BEGIN] $DONE"
    local start_time=$SECONDS
    cat | bash
    local duration=$(( SECONDS - start_time ))
    echo "[END] $DONE (took ${duration}s)"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Completed on: $timestamp in ${duration}s" > "${OUTPUT_FOLDER}/${DONE}.done"
}

if [ "$(enabled "diffusion")" == "True" ]; then
#rf-diffusion
run_task rf_diffusion <<'CMD'
pixi run --manifest-path $SOURCE_PATH/pixi.toml --as-is -e rf python $SOURCE_PATH/0_run_diffusion_nyl12.py $HYDRA_PARAMS &&
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/0_diffusion/commands_diffusion.sh
CMD
else
  echo "rf_diffusion is disabled."
fi

#ligand-mpnn
if [ "$(enabled "ligand_mpnn")" == "True" ]; then
run_task ligand_mpnn <<'CMD'
pixi run --as-is --manifest-path $SOURCE_PATH/pixi.toml -e ligand python $SOURCE_PATH/1_ligandmpnn_nyl12.py $HYDRA_PARAMS &&
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/1_ligandmpnn/commands_mpnn.sh
CMD
else
  echo "ligand_mpnn is disabled."
fi

#boltz2
if [[ "$(enabled "colabfold")" == "True" ]]; then
    pixi run --as-is --manifest-path "$SOURCE_PATH/pixi.toml" -e boltz \
        python "$SOURCE_PATH/prepare_boltz_input_nyl12.py" \
        colabfold.custom_template_path="\$CIF_FOLDER" $HYDRA_PARAMS
        echo local colabfold
elif [[ "$(enabled "boltz")" == "True" ]]; then
    pixi run --as-is --manifest-path "$SOURCE_PATH/pixi.toml" -e boltz \
        python "$SOURCE_PATH/prepare_boltz_input_nyl12.py" $HYDRA_PARAMS
fi

if [ "$(enabled "boltz.local_colabfold_search")" == "True" ]; then
run_task colabfold_search <<'CMD'
bash ${OUTPUT_FOLDER}/2_boltz/commands_colabfold_search.sh
CMD

run_task msas_convert <<'CMD'
parallel --halt soon,fail=1 -j $NTASKS_CPU --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/2_boltz/commands_msas_convert.sh
CMD

else
  echo "colabfold_search is disabled."
fi


if [ "$(enabled "boltz")" == "True" ]; then

run_task boltz <<'CMD'
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup CUDA_VISIBLE_DEVICES='$(({%} - 1))' bash -c "{}" :::: ${OUTPUT_FOLDER}/2_boltz/commands_boltz2.sh
CMD
else
  echo "boltz is disabled."
fi

if [ "$(enabled "colabfold")" == "True" ]; then
export COLABFOLD_TEMPLATE_FOLDER=$(pixi run --manifest-path $SOURCE_PATH/pixi.toml --as-is -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(cfg.colabfold.custom_template_path)")

#colabfold_local
run_task colabfold_local <<'CMD'
parallel -j "$NTASKS_GPU" 'mkdir -p /tmp/cif1_{%} && cp "$COLABFOLD_TEMPLATE_FOLDER"/*  /tmp/cif1_{%}/' ::: $(seq "$NTASKS_GPU") &&
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup CIF_FOLDER=/tmp/cif1_{%} CUDA_VISIBLE_DEVICES='$(({%} - 1))' bash -c "{}" :::: ${OUTPUT_FOLDER}/2b_colabfold/commands_colabfold.sh
chmod og+rwX -R /tmp/cif1_*
CMD
else
  echo "colabfold is disabled."
fi

if [ "$(enabled "relaxation")" == "True" ]; then

export RELAXATION_OUTPUT_FOLDER=$(pixi run --as-is --manifest-path $SOURCE_PATH/pixi.toml -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.relaxation.output_dir).name)")
export BOLTZ_OUTPUT_FOLDER=$(pixi run --as-is --manifest-path $SOURCE_PATH/pixi.toml -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.boltz.output_dir).name)")
export INPUT_MODEL=$(find $BASE_DIR/$OUTPUT_FOLDER/$BOLTZ_OUTPUT_FOLDER -type f -path "*/boltz*/predictions/*/*.pdb" | head -n 1)

run_task amber_params <<'CMD'
pixi run --as-is --manifest-path $SOURCE_PATH/pixi.toml -e analysis bash $SOURCE_PATH/src/compute_amber_params.sh --input_model=$INPUT_MODEL --output_folder=$BASE_DIR/$OUTPUT_FOLDER/$RELAXATION_OUTPUT_FOLDER --source_dir=$SOURCE_PATH
CMD

run_task relaxation <<'CMD'
bash $SOURCE_PATH/src/prepare_relaxation_commands.sh --command=$SOURCE_PATH/src/run_relaxation.sh --input_folder=$BASE_DIR/$OUTPUT_FOLDER/$BOLTZ_OUTPUT_FOLDER --output_folder=$BASE_DIR/$OUTPUT_FOLDER/$RELAXATION_OUTPUT_FOLDER --source_dir=$SOURCE_PATH &&
parallel -j $NTASKS_CPU --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/${RELAXATION_OUTPUT_FOLDER}/commands_relaxation.sh || true
CMD

else
  echo "relaxation is disabled."
fi

if [[ "$(enabled "filtering.backbone")" == "True" || "$(enabled "filtering.affinity")" == "True" ]]; then

run_task filtering <<'CMD'
pixi run --as-is --manifest-path $SOURCE_PATH/pixi.toml -e analysis python $SOURCE_PATH/analyze_colabfold_models.py $HYDRA_PARAMS &&
pixi run --as-is --manifest-path $SOURCE_PATH/pixi.toml -e analysis python $SOURCE_PATH/analyze_boltz_models.py $HYDRA_PARAMS
CMD

else
  echo "filtering is disabled."
fi



chmod og+rwX -R .
