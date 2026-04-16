import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ROBOT_ASSETS_DIR = "C:/IsaacLab/source/robot_assets"

WANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS_DIR}/wanda.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2875),
        joint_pos={
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_tendonDriver_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg( #ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_tendonDriver_joint"],
            effort_limit_sim=37.0,
            saturation_effort=37.0,
            velocity_limit=21.0,
            friction=0.0,
            # stiffness=25.0,
            # damping=0.5,
            stiffness={
                "f[l,r]_shoulder_pitch_joint": 91.5,
                "f[l,r]_shoulder_roll_joint": 309.46545,
                "f[l,r]_tendonDriver_joint": 5.52904,
                
                "b[l,r]_shoulder_pitch_joint": 114.07796,
                "b[l,r]_shoulder_roll_joint": 366.98553,
                "b[l,r]_tendonDriver_joint": 5.49093,
            },
            damping={
                "f[l,r]_shoulder_pitch_joint": 0.03664,
                "f[l,r]_shoulder_roll_joint": 0.12379,
                "f[l,r]_tendonDriver_joint": 0.00221,
                
                "b[l,r]_shoulder_pitch_joint": 0.04563,
                "b[l,r]_shoulder_roll_joint": 0.14679,
                "b[l,r]_tendonDriver_joint": 0.0022,
            },
        ),
    },
)