"""Constants for the Dog quadruped robot."""

from etils import epath


ROOT_PATH = epath.Path(__file__).parent
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_flat_terrain.xml"


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "flat_terrain": FLAT_TERRAIN_XML,
    }[task_name]


# 4 feet sites
FEET_SITES = [
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
]

# Foot collision geoms (spheres at foot tips)
FR_FEET_GEOMS = [
    "FR_foot_collision",
]

FL_FEET_GEOMS = [
    "FL_foot_collision",
]

RR_FEET_GEOMS = [
    "RR_foot_collision",
]

RL_FEET_GEOMS = [
    "RL_foot_collision",
]

FEET_GEOMS = FR_FEET_GEOMS + FL_FEET_GEOMS + RR_FEET_GEOMS + RL_FEET_GEOMS

HIP_JOINT_NAMES = [
    "FR_hip_joint",
    "FL_hip_joint",
    "RR_hip_joint",
    "RL_hip_joint",
]

THIGH_JOINT_NAMES = [
    "FR_thigh_joint",
    "FL_thigh_joint",
    "RR_thigh_joint",
    "RL_thigh_joint",
]

CALF_JOINT_NAMES = [
    "FR_calf_joint",
    "FL_calf_joint",
    "RR_calf_joint",
    "RL_calf_joint",
]

# All 12 actuated joints in order
JOINTS_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "trunk"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
