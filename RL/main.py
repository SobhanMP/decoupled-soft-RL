import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from jac.env_wrapper import EnvPoolConf, MultiGymConf
from jac.trans import TransRBConf
from jac.training import Conf, train
from jac.sac import SACConf

# from jax import config
# config.update("jax_disable_jit", True)
# config.update("jax_debug_nans", True)

cs = ConfigStore.instance()
cs.store(name="base_config", node=Conf)
cs.store(group="replay", name="base_trans", node=TransRBConf)

cs.store(group="algo", name="base_sac", node=SACConf)

cs.store(group="env", name="base_envpool", node=EnvPoolConf)
cs.store(group="env", name="base_multigym", node=MultiGymConf)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(cfg: Conf):
    return train(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    main()
