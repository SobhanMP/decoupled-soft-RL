#!/bin/bash
#SBATCH --output=slurm/output.%A.%a.%j.out
#SBATCH --signal=B:SIGTERM@3600
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1


mkdir -p $SCRATCH/logs

ulimit -n `ulimit -H -n`
ulimit -c 0
. ~/mambaforge/etc/profile.d/conda.sh
. ~/mambaforge/etc/profile.d/mamba.sh
mamba activate gfn

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
srun python ${@:1} ${SLURM_ARRAY_TASK_ID}
