#!/usr/bin/env bash

set -e

export BASE_DIR=$(realpath "`pwd`")
export SOURCE_PATH=$(dirname "$(realpath "$0")")

if [ "$BASE_DIR" == "$SOURCE_PATH" ]; then
  echo Cannot create simulation in source folder
  exit
fi

PROJECT_PATH=$BASE_DIR/$1


mkdir -p $PROJECT_PATH

mkdir $PROJECT_PATH/config $PROJECT_PATH/templates $PROJECT_PATH/cif

cp $SOURCE_PATH/config/config.yaml $PROJECT_PATH/config
cp $SOURCE_PATH/templates/boltz.yaml.j2 $PROJECT_PATH/templates/boltz.yaml.j2
cp -r $SOURCE_PATH/cif/all_loops $PROJECT_PATH/cif
cp -r $SOURCE_PATH/cif/Nyl12_refine13_no_tail.cif $PROJECT_PATH/cif

cat << EOF > $PROJECT_PATH/README
============================================================
SIMULATION WORKFLOW
============================================================

1. ENVIRONMENT SETUP
   Ensure $BASE_DIR is included in your PATH variable.

2. PROJECT DIRECTORY
   cd $PROJECT_PATH

3. CONFIGURATION
   Modify config files and templates as required. See $SOURCE_PATH/config for examples

4. EXECUTION
   Run the simulation using one of the following:

   - Interactive: run_all_aster.sh
   - Background:  start_workflow_in_bg.sh

============================================================
EOF



cat $PROJECT_PATH/README
