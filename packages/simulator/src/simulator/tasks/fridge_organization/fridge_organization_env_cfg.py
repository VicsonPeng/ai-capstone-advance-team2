import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from simulator.assets.scenes.kitchen import KITCHEN_CFG

from simulator.tasks.template import mdp
from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

# =========================================================
# Container geometry constants
# =========================================================

W = 0.15   # inner width (x)
D = 0.15   # inner depth (y)
H = 0.15   # inner height (z)
T = 0.02   # wall thickness

# Red container center (bottom center)
RX, RY, RZ = 0.27, -0.10, 0.05

# Blue container center (bottom center)
BX, BY, BZ = 0.47, -0.10, 0.05

# Red container center (bottom center)
ORX, ORY, ORZ = 0.0, -0.12, 0.10

# Blue container center (bottom center)
OBX, OBY, OBZ = 0.65, -0.45, 0.10


def _kinematic_box(color, size, pos):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Scene/{pos[0]}_{pos[1]}_{pos[2]}".replace(".", ""),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# =========================================================
# Scene
# =========================================================


@configclass
class FridgeOrganizationSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration — two open-top containers + two boxes."""

    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # cameras inherited from parent

    # --------------------------------------------------
    # Red container  (center: RX, RY, RZ=bottom)
    # --------------------------------------------------

    red_bottom: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_bottom",
        spawn=sim_utils.CuboidCfg(
            size=(W + 2*T, D + 2*T, T),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(RX, RY, RZ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    red_front: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_front",
        spawn=sim_utils.CuboidCfg(
            size=(W + 2*T, T, H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(RX, RY - D/2 - T/2, RZ + T/2 + H/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    red_back: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_back",
        spawn=sim_utils.CuboidCfg(
            size=(W + 2*T, T, H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(RX, RY + D/2 + T/2, RZ + T/2 + H/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    red_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_left",
        spawn=sim_utils.CuboidCfg(
            size=(T, D, H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(RX - W/2 - T/2, RY, RZ + T/2 + H/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    red_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_right",
        spawn=sim_utils.CuboidCfg(
            size=(T, D, H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(RX + W/2 + T/2, RY, RZ + T/2 + H/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # --------------------------------------------------
    # Blue container  (center: BX, BY, BZ=bottom)
    # --------------------------------------------------

    blue_bottom: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_bottom",
        spawn=sim_utils.CuboidCfg(
            size=(W + 2*T, D + 2*T, T),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BX, BY, BZ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    blue_front: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_front",
        spawn=sim_utils.CuboidCfg(
            size=(W + 2*T, T, H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BX, BY - D/2 - T/2, BZ + T/2 + H/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    blue_back: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_back",
        spawn=sim_utils.CuboidCfg(
            size=(W + 2*T, T, H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BX, BY + D/2 + T/2, BZ + T/2 + H/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    blue_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_left",
        spawn=sim_utils.CuboidCfg(
            size=(T, D, H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BX - W/2 - T/2, BY, BZ + T/2 + H/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    blue_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_right",
        spawn=sim_utils.CuboidCfg(
            size=(T, D, H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BX + W/2 + T/2, BY, BZ + T/2 + H/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # --------------------------------------------------
    # Objects to pick — red cube and blue cube
    # --------------------------------------------------

    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/apple",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(ORX, ORY, ORZ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    drink: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/drink",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBX, OBY, OBZ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# =========================================================
# Observations
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
    apple: RigidObject = env.scene[apple_cfg.name]
    apple_pos = apple.data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # inside red container
    done = torch.logical_and(done, apple_pos[:, 0] > RX - W/2)
    done = torch.logical_and(done, apple_pos[:, 0] < RX + W/2)
    done = torch.logical_and(done, apple_pos[:, 1] > RY - D/2)
    done = torch.logical_and(done, apple_pos[:, 1] < RY + D/2)
    done = torch.logical_and(done, apple_pos[:, 2] > RZ)

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

    scene: FridgeOrganizationSceneCfg = FridgeOrganizationSceneCfg(env_spacing=8.0)
    observations: FridgeOrganizationObservationsCfg = FridgeOrganizationObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "pick up the red cube and place it into the red container."

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