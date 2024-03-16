#!/bin/bash
#SBATCH --output=slurm/output.%A.%a.%j.out
#SBATCH --signal=B:SIGTERM@1200
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2


mkdir -p $SCRATCH/logs

ulimit -n `ulimit -H -n`
. ~/mambaforge/etc/profile.d/conda.sh
. ~/mambaforge/etc/profile.d/mamba.sh
mamba activate jax311

which python
date
echo $*

mkdir -p $SCRATCH/logs
exit_script() {
    date
    trap - SIGTERM EXIT
    pkill -P $$
    sleep 3
    cp -rv $SLURM_TMPDIR/logs/* $SCRATCH/logs
    kill -- -$$
}

trap exit_script SIGTERM EXIT
python $1 run $2 ${SLURM_ARRAY_TASK_ID}
