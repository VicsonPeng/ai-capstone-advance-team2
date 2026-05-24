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

    # cameras disabled for headless operation; re-enable on GlowsAI with --enable_cameras
    wrist = None
    front = None

    # -----------------------------------------------------
    # Fridge (static large object)
    # -----------------------------------------------------

    # fridge is static — use AssetBaseCfg, no RigidBodyAPI needed
    # 把原本的 fridge AssetBaseCfg 整段換掉
    fridge: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/fridge",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.30, 0.60),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.9)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.00, 3.00, 0.30),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # -----------------------------------------------------
    # Apple
    # -----------------------------------------------------

    # placeholder shapes — replace with complete USD packages once available
    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/apple",
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1)
            ),
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
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.3, 0.8)
            ),
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
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.05, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.7, 0.2)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.20, 0.85),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# =========================================================
# Observations (no cameras — headless compatible)
# =========================================================


@configclass
class FridgeOrganizationObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


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

    observations: FridgeOrganizationObservationsCfg = (
        FridgeOrganizationObservationsCfg()
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
