from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from queue import Queue
import sys
import threading
import time
from typing import Optional
import numpy as np
from tensorboardX import SummaryWriter
from omegaconf import MISSING, OmegaConf


from jac.consts import jnp, dc


# stolen from recursion
def create_logger(name="main_log", logfile=None, stdout=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - {} - %(message)s".format(name),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a"))
    if stdout:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@dataclass(eq=True, kw_only=True, unsafe_hash=True)
class LogConf:
    dump_every: int = MISSING
    dir: Optional[str] = MISSING
    wandb: bool = MISSING
    tensorboard: bool = MISSING
    info: bool = MISSING
    thread: bool = MISSING
    name: Optional[str] = MISSING
    proj_name: str = MISSING
    proj_entity: str = MISSING
    run: Optional[str] = MISSING
    stdout: bool = MISSING

    def __call__(self):
        return Logger(self)


def get_item(x):
    if hasattr(x, "item"):
        return x.item()
    else:
        return x


class Logger:
    def __init__(self, cfg: LogConf, wcfg=None, loading=False):
        if not loading:
            print(OmegaConf.to_yaml(wcfg or cfg))
        if cfg.dir is None and cfg.tensorboard:
            raise ValueError("dir must be specified if using tensorboard")

        if cfg.dir is not None and not loading:
            d = Path(cfg.dir)
            d.mkdir(parents=True, exist_ok=True)
            self.dir = d
            self.dump_dir = d / "dumps"
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            with open(d / "conf.yaml", "w", encoding="utf8") as fd:
                fd.write(OmegaConf.to_yaml(wcfg or cfg))

        if cfg.info and not loading:
            if cfg.dir is not None:
                ld = d / "info.log"
            else:
                ld = None
            self.logger = create_logger(logfile=ld, stdout=cfg.stdout)
            self.logger.info(OmegaConf.to_yaml(wcfg or cfg))

        if cfg.tensorboard and not loading:
            self.writer = SummaryWriter(
                logdir=cfg.dir, max_queue=1000, flush_secs=10 * 60
            )
        if cfg.wandb and not loading:
            import wandb  # slows things down on CC

            wandb.init(
                project=cfg.proj_name,
                entity=cfg.proj_entity,
                name=cfg.name,
                config=OmegaConf.to_container(OmegaConf.structured(wcfg or cfg)),
            )

        self.cfg = cfg
        self.wcfg = wcfg

        self.t = time.time()
        self.logs = []
        self.reset()
        if cfg.thread and not loading:
            self.q = Queue()
            self.thread = threading.Thread(target=self.worker, daemon=True)
            self.thread.start()

    def pretty(self, x):
        return {k: v for k, v in x.items() if ":" not in k}

    @classmethod
    def restore(cls, x):
        logs, cfg, wcfg = x
        cfg = dc.replace(cfg, wandb=False, tensorboard=False, info=False, dir=None)
        self = cls(cfg, wcfg, loading=True)
        self.logs = logs
        return self

    def dump(self):
        return (self.logs, self.cfg, self.wcfg)

    def worker(self):
        while True:
            ld = self.q.get()
            if ld is None:
                break
            self.store(*ld)
            time.sleep(1e-3)  # yield

    def store(self, it, ld):
        ld = {k: np.mean(list(map(get_item, v))) for k, v in ld.items()}

        self.logs.append((it, ld))
        if self.cfg.tensorboard:
            for k, v in ld.items():
                self.writer.add_scalar(k, v, it)
        pld = self.pretty(ld)
        if self.cfg.wandb:
            import wandb

            wandb.log(pld, step=it)
        if self.cfg.info:
            self.logger.info(pld)

    def reset(self):
        self.auxes = defaultdict(list)

    def append(self, aux):
        for k, v in aux.items():
            self.auxes[k].append(v)

    def extend(self, aux):
        for k, v in aux.items():
            if len(v) > 0:
                self.auxes[k].extend(v)

    def __call__(self, it):
        ld = self.auxes
        ld["time"] = [time.time() - self.t]
        if self.cfg.thread:
            assert self.thread.is_alive()
            self.q.put((it, ld))
        else:
            self.store(it, ld)

        self.reset()
        return ld

    def __getitem__(self, key):
        x = [x for (x, _) in self.logs]
        y = [y[key] for (_, y) in self.logs]
        return x, y

    def killthread(self):
        if hasattr(self, "q"):
            self.q.put(None)
        if hasattr(self, "thread"):
            self.thread.join()

    def close(self):
        self.killthread()
        if hasattr(self, "writer"):
            self.writer.close()

    def pickle(self, epoch, **kwargs):
        log_dir = self.cfg.dir
        if log_dir is None:
            return
        if self.cfg.thread:
            while not self.q.empty():
                time.sleep(1e-3)
        for k, w in kwargs.items():
            with open(self.dump_dir / f"{k}-{epoch}.pkl", "wb") as file:
                pickle.dump(w, file)
        with open(self.dir / "logs.pkl", "wb") as file:
            pickle.dump(self.dump(), file)
