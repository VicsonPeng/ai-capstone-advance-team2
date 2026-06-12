================================================================================
Team2 — AI Capstone Advanced Submission
README.txt
================================================================================

TASK
----
Fridge Organization: pick up the red cube and place it into the red container;
pick up the blue cube and place it into the blue container.

This is a custom environment implemented as an Advanced-level task on top of the
provided Isaac Lab / LeIsaac simulator framework.

No additional files or custom configurations are needed beyond what is in this
repository and the trained policy checkpoint.

================================================================================
PREREQUISITES
================================================================================

- Linux machine with an Nvidia GPU (verify: nvidia-smi)
- Docker installed
- uv installed (https://docs.astral.sh/uv/)
- Repository cloned:
    git clone https://github.com/VicsonPeng/ai-capstone-advance-team2.git
    cd ai-capstone-advance-team2

================================================================================
ENVIRONMENT SETUP (Host Machine)
================================================================================

Install all Python dependencies:

    uv sync

Activate the virtual environment:

    source .venv/bin/activate

Set your Hugging Face username (required for dataset/model download):

    export HF_USER=<your-huggingface-username>

================================================================================
ADVANCED TASK — FRIDGE ORGANIZATION
================================================================================

The custom task is defined at:

    packages/simulator/src/simulator/tasks/fridge_organization/
        __init__.py                    <- gym registration
        fridge_organization_env_cfg.py <- full environment configuration
        mdp/
            __init__.py

Gym ID:  HCIS-FridgeOrganization-SingleArm-v0

Scene:
  - Two open-top containers (red and blue) placed on a kitchen countertop
  - Red container center:  x=0.27, y=-0.10, z=0.05
  - Blue container center: x=0.47, y=-0.10, z=0.05
  - Container size: 15 cm x 15 cm x 15 cm (inner), wall thickness 2 cm
  - Two 5 cm cubes (red = "apple", blue = "drink") spawned at randomized
    positions within x=[0.20, 0.50], y=[-0.45, -0.25] on each episode reset,
    with a minimum separation of 8 cm to avoid collision

Success condition:
  - The red cube (apple) is inside the red container bounds

Robot:
  - Franka Panda single-arm
  - Base position: (0.35, -0.74, 0.01), rotated 90 deg around z-axis
  - Initial joint configuration: standard ready pose

================================================================================
DATA GENERATION (inside Isaac Lab container)
================================================================================

Step 1 — Launch the Isaac Lab container:

    make launch-isaaclab-glowsai-4090   # RTX 4090
    # or
    make launch-isaaclab-glowsai-l40s   # L40S

Step 2 — Inside the container, run data generation:

    python scripts/datagen/generate.py \
        --task HCIS-FridgeOrganization-SingleArm-v0 \
        --num_envs 1 \
        --device cuda \
        --enable_cameras \
        --record \
        --use_lerobot_recorder \
        --lerobot_dataset_repo_id ${HF_USER}/<repo_id> \
        --object_poses data/<demo_directory>/object_poses.json

================================================================================
TRAINING (Host Machine)
================================================================================

    lerobot-train \
      --dataset.repo_id=${HF_USER}/<repo_id> \
      --policy.type=diffusion \
      --output_dir=<your-output-dir> \
      --job_name=fridge_organization \
      --policy.device=cuda \
      --policy.repo_id=${HF_USER}/my_policy

After training, upload checkpoint to Hugging Face:

    hf upload ${HF_USER}/my_policy <your-output-dir>/pretrained_model --revision v1

================================================================================
ROLLOUT / EVALUATION (inside Isaac Lab container)
================================================================================

Step 1 — Launch the Isaac Lab container (same as data generation step 1).

Step 2 — Place the submitted Checkpoint Folder into the repository:

    Copy the submitted Checkpoint Folder (from the Google Drive submission)
    into the repository root as:

        checkpoints/...... 
        
    The directory should contain pretrained_model/ and train_config.json.

Step 3 — Run rollout:

    python scripts/rollout.py \
        --task HCIS-FridgeOrganization-SingleArm-v0 \
        --policy_type=lerobot-diffusion \
        --policy_checkpoint_path=checkpoints/...... \
        --policy_action_horizon=1 \
        --device=cuda \
        --enable_cameras \
        --eval_rounds=50 \
        --episode_length_s=20

================================================================================
CONFIGURATIONS FOLDER CONTENTS
================================================================================

The submitted Configurations Folder contains the standalone environment
configuration for the custom Fridge Organization task:

    fridge_organization/
        __init__.py                    <- gym.register call
        fridge_organization_env_cfg.py <- FridgeOrganizationEnvCfg class
        mdp/
            __init__.py

These files are importable as:
    simulator.tasks.fridge_organization.fridge_organization_env_cfg:FridgeOrganizationEnvCfg

No external assets beyond the shared kitchen USD scene are required.
The kitchen USD is included in the repository under:
    packages/simulator/assets/scenes/

================================================================================
NOTES
================================================================================

- All simulation runs require a Linux machine with an Nvidia GPU.
- The data generation and rollout steps must be run inside the Docker container.
- Training runs on the host machine (not inside Docker) for performance reasons.
- Full pipeline documentation is available in the docs/ directory.

================================================================================
