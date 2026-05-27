import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from simulator import ASSETS_ROOT
from simulator.assets.scenes.kitchen import KITCHEN_CFG

from simulator.tasks.template import mdp
from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

# =========================================================
# Asset Paths
# =========================================================

FRIDGE_ORG_ROOT = ASSETS_ROOT / "scenes" / "fridge_organization"
FRIDGE_OBJECTS_ROOT = FRIDGE_ORG_ROOT / "objects"

# =========================================================
# Scene
# =========================================================


@configclass
class FridgeOrganizationSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration for bookcase organization."""

    scene: AssetBaseCfg = KITCHEN_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Scene"
    )

    # cameras inherited from parent — wrist + front both active

    # -----------------------------------------------------
    # Bookcase (USD model, static)
    # -----------------------------------------------------

    # Bookcase (was fridge)
    fridge: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FRIDGE_OBJECTS_ROOT / "fridge" / "model_fridge_1.usd"),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.37, -0.2, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Apple
    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FRIDGE_OBJECTS_ROOT / "apple" / "model_FakeFruit_5404E_RomeRedApple_69323.usd"),
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.5, 0.30),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Drink
    drink: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/drink",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FRIDGE_OBJECTS_ROOT / "drink" / "model_drink002.usd"),
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, -0.5, 0.30),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Snack
    snack: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/snack",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FRIDGE_OBJECTS_ROOT / "snack" / "model_snack012.usd"),
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.15, -0.5, 0.30),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# =========================================================
# Observations — inherit from parent (includes wrist + front cameras)
# =========================================================


@configclass
class FridgeOrganizationObservationsCfg(SingleArmFrankaObservationsCfg):
    pass


# =========================================================
# Success Condition
# =========================================================


def apple_on_bookcase(
    env,
    apple_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Success: apple placed on bookcase shelf."""

    apple: RigidObject = env.scene[apple_cfg.name]
    apple_pos = apple.data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # bookcase region — adjust after seeing it in simulator
    done = torch.logical_and(done, apple_pos[:, 0] > 0.25)
    done = torch.logical_and(done, apple_pos[:, 0] < 0.65)
    done = torch.logical_and(done, apple_pos[:, 1] > 0.20)
    done = torch.logical_and(done, apple_pos[:, 1] < 0.60)
    done = torch.logical_and(done, apple_pos[:, 2] > 0.10)

    return done


# =========================================================
# Terminations
# =========================================================


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=apple_on_bookcase,
        params={"apple_cfg": SceneEntityCfg("apple")},
    )


# =========================================================
# Main Environment
# =========================================================


@configclass
class FridgeOrganizationEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Bookcase organization task."""

    scene: FridgeOrganizationSceneCfg = FridgeOrganizationSceneCfg(env_spacing=8.0)
    observations: FridgeOrganizationObservationsCfg = FridgeOrganizationObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "pick up the object and place it on the bookcase shelf."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.5, 0.0, 0.5)

        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)

        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -math.pi / 4.0,
            "panda_joint3": 0.0,
            "panda_joint4": -3.0 * math.pi / 4.0,
            "panda_joint5": 0.0,
            "panda_joint6": math.pi / 2.0,
            "panda_joint7": math.pi / 4.0,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }
