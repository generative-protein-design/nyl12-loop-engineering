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
        *)
            echo "Unknown argument: $ARG"
            exit 1
            ;;
    esac
done


BASE_DIR=`pwd`

MODEL=$(basename "$INPUT_MODEL" .pdb)


echo computing Amber parameters using $INPUT_MODEL ...

# Generate Amber parameters with antechamber.
mkdir -p $OUTPUT_FOLDER/amber_params
cp $BASE_DIR/config/leap.in $OUTPUT_FOLDER/amber_params
cd $OUTPUT_FOLDER/amber_params
pixi run -e analysis python $BASE_DIR/src/add_hydrogens_pymol.py $INPUT_MODEL .
pixi run -e analysis antechamber -i lig_h.pdb -fi pdb -o lig.mol2 -fo mol2 -c bcc -s 2
pixi run -e analysis parmchk2 -i lig.mol2 -f mol2 -o lig.frcmod
pixi run -e analysis tleap -f leap.in
cd - > /dev/null

echo Amber parameters are saved in $OUTPUT_FOLDER/amber_params




