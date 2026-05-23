import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets

from simulator import ASSETS_ROOT
from simulator.assets.scenes.kitchen import (
    KITCHEN_CFG,
    KITCHEN_USD_PATH,
)

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
    """Scene configuration for fridge organization."""

    # use kitchen background scene
    scene: AssetBaseCfg = KITCHEN_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Scene"
    )

    # -----------------------------------------------------
    # Fridge (static large object)
    # -----------------------------------------------------

    fridge: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(
                FRIDGE_OBJECTS_ROOT
                / "fridge"
                / "model_fridge_1.usd"
            ),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # -----------------------------------------------------
    # Apple
    # -----------------------------------------------------

    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(
                FRIDGE_OBJECTS_ROOT
                / "apple"
                / "model_FakeFruit_5404E_RomeRedApple_69323.usd"
            ),
            mass_props=MassPropertiesCfg(mass=0.05),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.20, 0.85),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # -----------------------------------------------------
    # Drink
    # -----------------------------------------------------

    drink: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/drink",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(
                FRIDGE_OBJECTS_ROOT
                / "drink"
                / "model_drink002.usd"
            ),
            mass_props=MassPropertiesCfg(mass=0.15),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.00, 0.85),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # -----------------------------------------------------
    # Snack
    # -----------------------------------------------------

    snack: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/snack",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(
                FRIDGE_OBJECTS_ROOT
                / "snack"
                / "model_snack012.usd"
            ),
            mass_props=MassPropertiesCfg(mass=0.05),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.20, 0.85),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# =========================================================
# Success Condition
# =========================================================


def apple_inside_fridge(
    env,
    apple_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Success condition:
    apple reaches fridge area.
    """

    apple: RigidObject = env.scene[apple_cfg.name]

    apple_pos = apple.data.root_pos_w - env.scene.env_origins

    done = torch.ones(
        env.num_envs,
        dtype=torch.bool,
        device=env.device,
    )

    # fridge region bounds
    done = torch.logical_and(done, apple_pos[:, 0] > 0.60)
    done = torch.logical_and(done, apple_pos[:, 0] < 0.90)

    done = torch.logical_and(done, apple_pos[:, 1] > -0.25)
    done = torch.logical_and(done, apple_pos[:, 1] < 0.25)

    done = torch.logical_and(done, apple_pos[:, 2] > 0.50)

    return done


# =========================================================
# Terminations
# =========================================================


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    """Termination configuration."""

    success = DoneTerm(
        func=apple_inside_fridge,
        params={
            "apple_cfg": SceneEntityCfg("apple"),
        },
    )


# =========================================================
# Main Environment
# =========================================================


@configclass
class FridgeOrganizationEnvCfg(
    SingleArmFrankaTaskEnvCfg
):
    """Fridge organization task."""

    scene: FridgeOrganizationSceneCfg = (
        FridgeOrganizationSceneCfg(env_spacing=8.0)
    )

    observations: SingleArmFrankaObservationsCfg = (
        SingleArmFrankaObservationsCfg()
    )

    terminations: TerminationsCfg = (
        TerminationsCfg()
    )

    task_description: str = (
        "pick up the apple and place it into the fridge."
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # -------------------------------------------------
        # Viewer
        # -------------------------------------------------

        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.5, 0.0, 0.5)

        self.dynamic_reset_gripper_effort_limit = False

        # -------------------------------------------------
        # Robot Initial Pose
        # -------------------------------------------------

        self.scene.robot.init_state.pos = (
            0.35,
            -0.74,
            0.01,
        )

        self.scene.robot.init_state.rot = (
            0.707,
            0.0,
            0.0,
            0.707,
        )

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

        # -------------------------------------------------
        # Parse Kitchen Scene
        # -------------------------------------------------

        parse_usd_and_create_subassets(
            KITCHEN_USD_PATH,
            self,
        )