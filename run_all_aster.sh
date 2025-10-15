#!/usr/bin/env bash

set -e

export NTASKS=2

export OUTPUT_FOLDER=`pixi run -e rf python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/config.yaml'); print(Path(cfg.work_dir).name)"`

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
pixi run -e rf python 0_run_diffusion_nyl12.py +site=aster
parallel -j $NTASKS --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/0_diffusion/commands_diffusion.sh
CMD

#ligand-mpnn
run_task ligand_mpnn <<'CMD'
pixi run -e ligand python 1_ligandmpnn_nyl12.py +site=aster
parallel -j $NTASKS --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/1_ligandmpnn/commands_mpnn.sh
CMD

#boltz2

pixi run -e boltz python prepare_boltz_input_nyl12.py +site=aster

run_task colabfold_search <<'CMD'
bash ${OUTPUT_FOLDER}/2_boltz/commands_colabfold_search.sh
CMD

run_task msas_convert <<'CMD'
parallel -j $NTASKS --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/2_boltz/commands_msas_convert.sh
CMD

run_task boltz <<'CMD'
parallel -j $NTASKS --ungroup CUDA_VISIBLE_DEVICES='$(({%} - 1))' bash -c "{}" :::: ${OUTPUT_FOLDER}/2_boltz/commands_boltz2.sh
CMD

run_task postprocess <<'CMD'
pixi run -e pymol python postprocess_boltz.py +site=aster
CMD
