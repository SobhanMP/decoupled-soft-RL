You can install the project with `pip install -e .` and then install `jaxlib`. Otherwise install the project as `pip install -e.[cuda11_pip]` or `pip install -e.[cuda12_pip]`.

You can run the experiments with `python rexp.py exec <exp> : <ids>` where `<exp>` is `auto` or `static` and`<ids>` is a list of valid cuda device id or the string `"cpu"`. Use multiple devices for parallelism. Otherwise use the command `python rexp.py sbatch <exp>` for the slurm command. See `slurm.sh` for a possible valid slurm conf.

The ndim grid experiment is in the `ndim_grid.py` file.