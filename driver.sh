#!/bin/bash

if [[ -z "${SLURM_NODEID}" ]]; then
    echo "need \$SLURM_NODEID set"
    exit
fi
if [[ -z "${SLURM_NNODES}" ]]; then
    echo "need \$SLURM_NNODES set"
    exit
fi
cat $1 |                                               \
awk -v NNODE="$SLURM_NNODES" -v NODEID="$SLURM_NODEID" \
'NR % NNODE == NODEID' |                               \


#export PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/m906/protein-design/shared_images

parallel --halt soon,fail=1 --ungroup -j 4 CUDA_VISIBLE_DEVICES='$(({%} - 1))' bash -c "{}"
