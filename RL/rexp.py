from copy import deepcopy
import datetime
import itertools
import os
from pathlib import Path
import socket

import numpy as np
from exps import make_cli, name, seeds


gym = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "HumanoidStandup-v4",
    "Humanoid-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    "Pusher-v4",
    "Reacher-v4",
    "Swimmer-v4",
    "Walker2d-v4",
]

dmc = [
    "AcrobotSwingup-v1",
    "BallInCupCatch-v1",
    "CartpoleBalance-v1",
    "CartpoleSwingup-v1",
    "CheetahRun-v1",
    "FingerSpin-v1",
    "FingerTurnEasy-v1",
    "FingerTurnHard-v1",
    "FishUpright-v1",
    "FishSwim-v1",
    "HopperStand-v1",
    "HopperHop-v1",
    "HumanoidStand-v1",
    "HumanoidWalk-v1",
    "HumanoidRun-v1",
    "ManipulatorBringBall-v1",
    "PendulumSwingup-v1",
    "PointMassEasy-v1",
    "ReacherEasy-v1",
    "ReacherHard-v1",
    "WalkerWalk-v1",
    "WalkerRun-v1",
    "SwimmerSwimmer6-v1",
    "SwimmerSwimmer15-v1",
]



N_runs = 10


exps_cc_ = {
    "auto": [
        {
            "log.name": "ours" if heur == "UNIFORM" else "default",
            "algo.auto_temp": True,
            "algo.heuristic_mult": mult,
            "algo.temp_heuristic": heur,
            "env.scale": scale,
            "algo.loss_fn": "mse",
        }
        for scale in [0.1, 0.25, 1, 4]
        for (mult, heur) in [(0.1, "UNIFORM"), (1.0, "SAC")]
    ],
    "static": [
        {
            "log.name": "ours" if decoupled else "default",
            "algo.decoupled": decoupled,
            "algo.temp": 0.25,
            "env.scale": scale,
            "algo.loss_fn": "mse",
        }
        for decoupled in [True, False]
        for scale in [
            1.0,
            2.0,
            4.0,
            0.5,
            0.25,
        ]
    ],
}



class multiply:  # for __len__
    def __init__(self, xs, env) -> None:
        self.xs = xs
        self.env_list = env

    def __len__(self):
        return N_runs * len(self.xs) * len(self.env_list)

    def __iter__(self):
        for env in self.env_list:
            for index, seed in enumerate(seeds(N_runs)):
                for x in self.xs:
                    if "seed" in x:
                        xseed = seed + x["seed"]
                    else:
                        xseed = seed
                    yield {
                        **x,
                        "seed": xseed,
                        "env.id": env,
                        "log.name": f"{x['log.name']}-{index}",
                    }


exps_cc = {
    f"{k}-{env_name}": multiply(v, env_list)
    for k, v in exps_cc_.items()
    for env_name, env_list in [
        ("dmc", dmc),
    ]
}


def command(l, i, x):
    d = name(i, "rl", l)

    return [
        "python",
        "main.py",
        f"log.dir={d}",
        f"log.run={l}",
        *[f"{k}={str(v)}" for k, v in x.items()],
    ]

print(exps_cc.keys())
if __name__ == "__main__":
    make_cli(command, exps_cc)()
