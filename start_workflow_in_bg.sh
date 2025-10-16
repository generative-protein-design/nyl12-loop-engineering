OUTPUT_FOLDER=`pixi run -e rf python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/config.yaml'); print(Path(cfg.work_dir).name)"`
LOGFILE="${OUTPUT_FOLDER}/output_$(date '+%Y-%m-%d_%H-%M-%S').log"

echo starting protein engineering
echo output folder: $OUTPUT_FOLDER
echo logs:  $LOGFILE

setsid ./run_all_aster.sh &> "$LOGFILE" &



