#!/usr/bin/env bash

set -e

export NTASKS_CPU=16
export NTASKS_GPU=2

export BASE_DIR=`pwd`

export CONFIG_NAME=${1:-config}

export OUTPUT_FOLDER=$(pixi run -e rf python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.work_dir).name)")

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


#rf-diffusion
run_task rf_diffusion <<'CMD'
pixi run -e rf python 0_run_diffusion_nyl12.py +site=aster --config-name=$CONFIG_NAME
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/0_diffusion/commands_diffusion.sh
CMD

#ligand-mpnn
run_task ligand_mpnn <<'CMD'
pixi run -e ligand python 1_ligandmpnn_nyl12.py +site=aster --config-name=$CONFIG_NAME
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/1_ligandmpnn/commands_mpnn.sh
CMD

#boltz2
pixi run -e boltz python prepare_boltz_input_nyl12.py +site=aster colabfold.custom_template_path=\$CIF_FOLDER --config-name=$CONFIG_NAME

run_task colabfold_search <<'CMD'
bash ${OUTPUT_FOLDER}/2_boltz/commands_colabfold_search.sh
CMD

run_task msas_convert <<'CMD'
parallel --halt soon,fail=1 -j $NTASKS_CPU --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/2_boltz/commands_msas_convert.sh
CMD

run_task boltz <<'CMD'
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup CUDA_VISIBLE_DEVICES='$(({%} - 1))' bash -c "{}" :::: ${OUTPUT_FOLDER}/2_boltz/commands_boltz2.sh
CMD

export COLABFOLD_TEMPLATE_FOLDER=$(pixi run -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(cfg.colabfold.custom_template_path)")

#colabfold_local
run_task colabfold_local <<'CMD'
parallel -j "$NTASKS_GPU" 'mkdir -p /tmp/cif_{%} && cp "$COLABFOLD_TEMPLATE_FOLDER"/*  /tmp/cif_{%}/' ::: $(seq "$NTASKS_GPU")
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup CIF_FOLDER=/tmp/cif_{%} CUDA_VISIBLE_DEVICES='$(({%} - 1))' bash -c "{}" :::: ${OUTPUT_FOLDER}/2b_colabfold/commands_colabfold.sh
CMD


export RELAXATION_OUTPUT_FOLDER=$(pixi run -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.relaxation.output_dir).name)")
export BOLTZ_OUTPUT_FOLDER=$(pixi run -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.boltz.output_dir).name)")
export INPUT_MODEL=$(find $BASE_DIR/$OUTPUT_FOLDER/$BOLTZ_OUTPUT_FOLDER -type f -path "*/boltz*/predictions/*/*.pdb" | head -n 1)

#export INPUT_MODEL=/data/35y/nyl12-loop-engineering/output_xama/2_boltz/nyl12_xa.ma_jmp/boltz_results_nyl12_xa.ma_jmp_4_0.1_5_model_1/predictions/nyl12_xa.ma_jmp_4_0.1_5_model_1/nyl12_xa.ma_jmp_4_0.1_5_model_1_model_0.pdb

run_task amber_params <<'CMD'
pixi run -e analysis bash src/compute_amber_params.sh --input_model=$INPUT_MODEL --output_folder=$BASE_DIR/$OUTPUT_FOLDER/$RELAXATION_OUTPUT_FOLDER
CMD

run_task relaxation <<'CMD'
bash src/prepare_relaxation_commands.sh --command=$BASE_DIR/src/run_relaxation.sh --input_folder=$BASE_DIR/$OUTPUT_FOLDER/$BOLTZ_OUTPUT_FOLDER --output_folder=$BASE_DIR/$OUTPUT_FOLDER/$RELAXATION_OUTPUT_FOLDER
parallel -j $NTASKS --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/${RELAXATION_OUTPUT_FOLDER}/commands_relaxation.sh || true
CMD

run_task filtering <<'CMD'
pixi run -e analysis python analyze_colabfold_models.py +site=aster --config-name=$CONFIG_NAME
pixi run -e analysis python analyze_boltz_models.py +site=aster --config-name=$CONFIG_NAME
CMD


chmod og+rwX -R .
