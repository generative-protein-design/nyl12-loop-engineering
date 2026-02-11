#!/bin/bash

set -e

for ARG in "$@"; do
    case $ARG in
        --input_model=*)
            INPUT_MODEL="${ARG#*=}"
            ;;
        --output_folder=*)
            OUTPUT_FOLDER="${ARG#*=}"
            ;;
        --source_dir=*)
            SOURCE_DIR="${ARG#*=}"
            ;;
        *)
            echo "Unknown argument: $ARG"
            exit 1
            ;;
    esac
done

BASE_DIR=`pwd`
MODEL=$(basename "$INPUT_MODEL" .pdb)
mkdir -p $OUTPUT_FOLDER/$MODEL

pixi run --as-is --manifest-path $SOURCE_DIR/pixi.toml -e analysis python $SOURCE_DIR/src/add_hydrogens_pymol.py $INPUT_MODEL $OUTPUT_FOLDER/$MODEL

# Generate Amber prmtop and inpcrd files with tleap. Run this step for every unique sequence.
#https://ambermd.org/tutorials/basic/tutorial4b/

cp $BASE_DIR/config/leap2.in $OUTPUT_FOLDER/$MODEL
cd $OUTPUT_FOLDER/$MODEL
grep -v -e LIG -e CONECT -e '^[[:space:]]*$' ${INPUT_MODEL} > protein.pdb
ln -sf $OUTPUT_FOLDER/amber_params/lig.frcmod .
ln -sf $OUTPUT_FOLDER/amber_params/lig.lib .

pixi run --as-is --manifest-path $SOURCE_DIR/pixi.toml -e analysis tleap -f leap2.in
pixi run --as-is --manifest-path $SOURCE_DIR/pixi.toml -e analysis python $SOURCE_DIR/src/run_openmm.py

pixi run --as-is --manifest-path $SOURCE_DIR/pixi.toml -e analysis python $SOURCE_DIR/src/molfile_to_params.py \
	-p LIG \
	-n LIG \
	--long-names \
	--clobber \
	--keep-names \
	lig_h.mol2

pixi run --as-is --manifest-path $SOURCE_DIR/pixi.toml -e analysis python $SOURCE_DIR/src/pyrosetta_interface_delta.py

cd - > /dev/null


