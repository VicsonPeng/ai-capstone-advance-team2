import gymnasium as gym

gym.register(
    id="HCIS-FridgeOrganization-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fridge_organization_env_cfg:FridgeOrganizationEnvCfg",
    },
)