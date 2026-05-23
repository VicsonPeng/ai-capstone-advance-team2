import gymnasium as gym

gym.register(
    id="HCIS-FridgeOrganization-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
        "simulator.tasks.fridge_organization.fridge_organization_env_cfg:FridgeOrganizationEnvCfg",
    },
)