#!/bin/env python
import datetime
import itertools
import multiprocessing as mp
import os
from pathlib import Path
import socket
import subprocess as sp

import click

TF = [True, False]


def name(i, group, exp):
    hostname = socket.gethostname()
    fn = f"{i}-{hostname}-{datetime.datetime.now():%Y-%m-%d_%H_%M_%S}"

    jobid = os.environ.get("SLURM_JOB_ID")
    if jobid is not None:
        fn = f"{fn}-{jobid}"
    arrayid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if arrayid is not None:
        fn = f"{fn}-{arrayid}"

    tmpdir = os.environ.get("SLURM_TMPDIR")
    if tmpdir is not None:
        d = Path(tmpdir) / "logs"
    else:
        d = Path("logs")

    return d / group / exp / fn


def seeds(x):
    return [i * 107 for i in range(1, x + 1)]


def fmt(name, runs):
    for i, j in enumerate(runs):
        yield (name, i, j)


def parser_id(exps, name, x, inds=False):
    runs = exps[name]
    x = x.strip()
    xs = x.split(":")
    assert len(xs) <= 2

    if x[0] == ":":
        i = 0
    else:
        i = int(xs[0])

    if x[-1] == ":":
        j = len(runs)
    else:
        j = int(xs[1])
    if inds:
        return i, j
    else:
        return itertools.islice(
            fmt(name, runs),
            i,
            j,
        )


def worker(gpu, q: mp.Queue):
    env = os.environ.copy()
    if gpu != "any":
        CVD = "CUDA_VISIBLE_DEVICES"
        if gpu in ["none", "cpu"]:
            env[CVD] = ""
        else:
            env[CVD] = gpu

    while True:
        x = q.get()
        if x is None:
            break
        print(" ".join(x))
        sp.run(x, env=env, check=False)


def make_cli(com, exps):
    @click.group()
    def cli():
        pass

    @cli.command()
    @click.argument("name")
    def size(name):
        print(len(exps[name]))

    def parsegpu(i):
        x = i.split("*")
        if len(x) <= 1:
            return [i]
        elif len(x) == 2:
            return x[0] * int(x[1])
        else:
            raise NotImplementedError()

    @cli.command()
    @click.argument("name")
    @click.argument("id")
    @click.argument("gpu", nargs=-1)
    def exec(name, id, gpu):  # pylint: disable=redefined-builtin
        if len(gpu) == 0:
            gpu = ["any"]
        gpu = [j for i in gpu for j in parsegpu(i)]
        ctx = mp.get_context("forkserver")
        q = ctx.Queue()
        process = [ctx.Process(target=worker, args=(i, q)) for i in gpu]
        for p in process:
            p.start()
        for l, i, v in parser_id(exps, name, id):
            q.put(com(l, i, v))
        for _ in process:
            q.put(None)
        for p in process:
            p.join()

    @cli.command()
    @click.argument("name")
    @click.argument("index", default=":")
    @click.option("--template", default="slurm.sh")
    def sbatch(name, index, template):
        import __main__  # pylint: disable=import-outside-toplevel

        i, j = parser_id(exps, name, index, inds=True)
        print(
            " ".join(
                [
                    "sbatch",
                    f"--comment={name}",
                    f"--array={i}-{j-1}",
                    template,
                    __main__.__file__,
                    name,
                ]
            )
        )

    @cli.command()
    @click.argument("name")
    @click.argument("i", type=int)
    def run(name, i):
        e = next(iter(itertools.islice(exps[name], i, i + 1)))
        c = com(name, i, e)
        print(" ".join(c))
        sp.run(c, check=False)

    return cli
