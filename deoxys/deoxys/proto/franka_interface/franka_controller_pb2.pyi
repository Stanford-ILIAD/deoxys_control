from google.protobuf import any_pb2 as _any_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JointGoal(_message.Message):
    __slots__ = ("is_delta", "q1", "q2", "q3", "q4", "q5", "q6", "q7")
    IS_DELTA_FIELD_NUMBER: _ClassVar[int]
    Q1_FIELD_NUMBER: _ClassVar[int]
    Q2_FIELD_NUMBER: _ClassVar[int]
    Q3_FIELD_NUMBER: _ClassVar[int]
    Q4_FIELD_NUMBER: _ClassVar[int]
    Q5_FIELD_NUMBER: _ClassVar[int]
    Q6_FIELD_NUMBER: _ClassVar[int]
    Q7_FIELD_NUMBER: _ClassVar[int]
    is_delta: bool
    q1: float
    q2: float
    q3: float
    q4: float
    q5: float
    q6: float
    q7: float
    def __init__(self, is_delta: bool = ..., q1: _Optional[float] = ..., q2: _Optional[float] = ..., q3: _Optional[float] = ..., q4: _Optional[float] = ..., q5: _Optional[float] = ..., q6: _Optional[float] = ..., q7: _Optional[float] = ...) -> None: ...

class Goal(_message.Message):
    __slots__ = ("is_delta", "x", "y", "z", "ax", "ay", "az")
    IS_DELTA_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    AX_FIELD_NUMBER: _ClassVar[int]
    AY_FIELD_NUMBER: _ClassVar[int]
    AZ_FIELD_NUMBER: _ClassVar[int]
    is_delta: bool
    x: float
    y: float
    z: float
    ax: float
    ay: float
    az: float
    def __init__(self, is_delta: bool = ..., x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., ax: _Optional[float] = ..., ay: _Optional[float] = ..., az: _Optional[float] = ...) -> None: ...

class ExponentialSmoothingConfig(_message.Message):
    __slots__ = ("alpha_q", "alpha_dq", "alpha_eef", "alpha_eef_vel")
    ALPHA_Q_FIELD_NUMBER: _ClassVar[int]
    ALPHA_DQ_FIELD_NUMBER: _ClassVar[int]
    ALPHA_EEF_FIELD_NUMBER: _ClassVar[int]
    ALPHA_EEF_VEL_FIELD_NUMBER: _ClassVar[int]
    alpha_q: float
    alpha_dq: float
    alpha_eef: float
    alpha_eef_vel: float
    def __init__(self, alpha_q: _Optional[float] = ..., alpha_dq: _Optional[float] = ..., alpha_eef: _Optional[float] = ..., alpha_eef_vel: _Optional[float] = ...) -> None: ...

class FrankaStateEstimatorMessage(_message.Message):
    __slots__ = ("is_estimation", "estimator_type", "config")
    class EstimatorType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        NO_ESTIMATOR: _ClassVar[FrankaStateEstimatorMessage.EstimatorType]
        EXPONENTIAL_SMOOTHING_ESTIMATOR: _ClassVar[FrankaStateEstimatorMessage.EstimatorType]
    NO_ESTIMATOR: FrankaStateEstimatorMessage.EstimatorType
    EXPONENTIAL_SMOOTHING_ESTIMATOR: FrankaStateEstimatorMessage.EstimatorType
    IS_ESTIMATION_FIELD_NUMBER: _ClassVar[int]
    ESTIMATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    is_estimation: bool
    estimator_type: FrankaStateEstimatorMessage.EstimatorType
    config: _any_pb2.Any
    def __init__(self, is_estimation: bool = ..., estimator_type: _Optional[_Union[FrankaStateEstimatorMessage.EstimatorType, str]] = ..., config: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class FrankaOSCControllerConfig(_message.Message):
    __slots__ = ("residual_mass_vec",)
    RESIDUAL_MASS_VEC_FIELD_NUMBER: _ClassVar[int]
    residual_mass_vec: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, residual_mass_vec: _Optional[_Iterable[float]] = ...) -> None: ...

class FrankaGripperHomingMessage(_message.Message):
    __slots__ = ("homing",)
    HOMING_FIELD_NUMBER: _ClassVar[int]
    homing: bool
    def __init__(self, homing: bool = ...) -> None: ...

class FrankaGripperMoveMessage(_message.Message):
    __slots__ = ("width", "speed")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    width: float
    speed: float
    def __init__(self, width: _Optional[float] = ..., speed: _Optional[float] = ...) -> None: ...

class FrankaGripperStopMessage(_message.Message):
    __slots__ = ("stop",)
    STOP_FIELD_NUMBER: _ClassVar[int]
    stop: bool
    def __init__(self, stop: bool = ...) -> None: ...

class FrankaGripperGraspMessage(_message.Message):
    __slots__ = ("width", "speed", "force", "epsilon_inner", "epsilon_outer")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    EPSILON_INNER_FIELD_NUMBER: _ClassVar[int]
    EPSILON_OUTER_FIELD_NUMBER: _ClassVar[int]
    width: float
    speed: float
    force: float
    epsilon_inner: float
    epsilon_outer: float
    def __init__(self, width: _Optional[float] = ..., speed: _Optional[float] = ..., force: _Optional[float] = ..., epsilon_inner: _Optional[float] = ..., epsilon_outer: _Optional[float] = ...) -> None: ...

class FrankaGripperControlMessage(_message.Message):
    __slots__ = ("termination", "control_msg")
    TERMINATION_FIELD_NUMBER: _ClassVar[int]
    CONTROL_MSG_FIELD_NUMBER: _ClassVar[int]
    termination: bool
    control_msg: _any_pb2.Any
    def __init__(self, termination: bool = ..., control_msg: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class FrankaDummyControllerMessage(_message.Message):
    __slots__ = ("goal", "termination")
    GOAL_FIELD_NUMBER: _ClassVar[int]
    TERMINATION_FIELD_NUMBER: _ClassVar[int]
    goal: Goal
    termination: bool
    def __init__(self, goal: _Optional[_Union[Goal, _Mapping]] = ..., termination: bool = ...) -> None: ...

class FrankaOSCPoseControllerMessage(_message.Message):
    __slots__ = ("goal", "translational_stiffness", "rotational_stiffness", "termination", "config")
    GOAL_FIELD_NUMBER: _ClassVar[int]
    TRANSLATIONAL_STIFFNESS_FIELD_NUMBER: _ClassVar[int]
    ROTATIONAL_STIFFNESS_FIELD_NUMBER: _ClassVar[int]
    TERMINATION_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    goal: Goal
    translational_stiffness: _containers.RepeatedScalarFieldContainer[float]
    rotational_stiffness: _containers.RepeatedScalarFieldContainer[float]
    termination: bool
    config: FrankaOSCControllerConfig
    def __init__(self, goal: _Optional[_Union[Goal, _Mapping]] = ..., translational_stiffness: _Optional[_Iterable[float]] = ..., rotational_stiffness: _Optional[_Iterable[float]] = ..., termination: bool = ..., config: _Optional[_Union[FrankaOSCControllerConfig, _Mapping]] = ...) -> None: ...

class FrankaJointPositionControllerMessage(_message.Message):
    __slots__ = ("goal", "kp_gains", "kd_gains", "speed_factor")
    GOAL_FIELD_NUMBER: _ClassVar[int]
    KP_GAINS_FIELD_NUMBER: _ClassVar[int]
    KD_GAINS_FIELD_NUMBER: _ClassVar[int]
    SPEED_FACTOR_FIELD_NUMBER: _ClassVar[int]
    goal: JointGoal
    kp_gains: float
    kd_gains: float
    speed_factor: float
    def __init__(self, goal: _Optional[_Union[JointGoal, _Mapping]] = ..., kp_gains: _Optional[float] = ..., kd_gains: _Optional[float] = ..., speed_factor: _Optional[float] = ...) -> None: ...

class FrankaJointImpedanceControllerMessage(_message.Message):
    __slots__ = ("goal", "kp", "kd")
    GOAL_FIELD_NUMBER: _ClassVar[int]
    KP_FIELD_NUMBER: _ClassVar[int]
    KD_FIELD_NUMBER: _ClassVar[int]
    goal: JointGoal
    kp: _containers.RepeatedScalarFieldContainer[float]
    kd: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, goal: _Optional[_Union[JointGoal, _Mapping]] = ..., kp: _Optional[_Iterable[float]] = ..., kd: _Optional[_Iterable[float]] = ...) -> None: ...

class FrankaCartesianVelocityControllerMessage(_message.Message):
    __slots__ = ("goal", "kp_gains", "kd_gains", "speed_factor")
    GOAL_FIELD_NUMBER: _ClassVar[int]
    KP_GAINS_FIELD_NUMBER: _ClassVar[int]
    KD_GAINS_FIELD_NUMBER: _ClassVar[int]
    SPEED_FACTOR_FIELD_NUMBER: _ClassVar[int]
    goal: Goal
    kp_gains: float
    kd_gains: float
    speed_factor: float
    def __init__(self, goal: _Optional[_Union[Goal, _Mapping]] = ..., kp_gains: _Optional[float] = ..., kd_gains: _Optional[float] = ..., speed_factor: _Optional[float] = ...) -> None: ...

class FrankaJointVelocityControllerMessage(_message.Message):
    __slots__ = ("goal", "kp_gains", "kd_gains")
    GOAL_FIELD_NUMBER: _ClassVar[int]
    KP_GAINS_FIELD_NUMBER: _ClassVar[int]
    KD_GAINS_FIELD_NUMBER: _ClassVar[int]
    goal: JointGoal
    kp_gains: float
    kd_gains: float
    def __init__(self, goal: _Optional[_Union[JointGoal, _Mapping]] = ..., kp_gains: _Optional[float] = ..., kd_gains: _Optional[float] = ...) -> None: ...

class FrankaJointTorqueControllerMessage(_message.Message):
    __slots__ = ("goal", "kp_gains", "kd_gains")
    GOAL_FIELD_NUMBER: _ClassVar[int]
    KP_GAINS_FIELD_NUMBER: _ClassVar[int]
    KD_GAINS_FIELD_NUMBER: _ClassVar[int]
    goal: JointGoal
    kp_gains: float
    kd_gains: float
    def __init__(self, goal: _Optional[_Union[JointGoal, _Mapping]] = ..., kp_gains: _Optional[float] = ..., kd_gains: _Optional[float] = ...) -> None: ...

class FrankaControlMessage(_message.Message):
    __slots__ = ("termination", "controller_type", "traj_interpolator_type", "traj_interpolator_time_fraction", "control_msg", "timeout", "state_estimator_msg")
    class ControllerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        NO_CONTROL: _ClassVar[FrankaControlMessage.ControllerType]
        OSC_POSE: _ClassVar[FrankaControlMessage.ControllerType]
        OSC_POSITION: _ClassVar[FrankaControlMessage.ControllerType]
        JOINT_POSITION: _ClassVar[FrankaControlMessage.ControllerType]
        JOINT_IMPEDANCE: _ClassVar[FrankaControlMessage.ControllerType]
        JOINT_VELOCITY: _ClassVar[FrankaControlMessage.ControllerType]
        TORQUE: _ClassVar[FrankaControlMessage.ControllerType]
        OSC_YAW: _ClassVar[FrankaControlMessage.ControllerType]
        CARTESIAN_VELOCITY: _ClassVar[FrankaControlMessage.ControllerType]
    NO_CONTROL: FrankaControlMessage.ControllerType
    OSC_POSE: FrankaControlMessage.ControllerType
    OSC_POSITION: FrankaControlMessage.ControllerType
    JOINT_POSITION: FrankaControlMessage.ControllerType
    JOINT_IMPEDANCE: FrankaControlMessage.ControllerType
    JOINT_VELOCITY: FrankaControlMessage.ControllerType
    TORQUE: FrankaControlMessage.ControllerType
    OSC_YAW: FrankaControlMessage.ControllerType
    CARTESIAN_VELOCITY: FrankaControlMessage.ControllerType
    class TrajInterpolatorType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        NO_OP: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
        LINEAR_POSITION: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
        LINEAR_POSE: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
        MIN_JERK_POSE: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
        SMOOTH_JOINT_POSITION: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
        MIN_JERK_JOINT_POSITION: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
        LINEAR_JOINT_POSITION: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
        COSINE_CARTESIAN_VELOCITY: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
        LINEAR_CARTESIAN_VELOCITY: _ClassVar[FrankaControlMessage.TrajInterpolatorType]
    NO_OP: FrankaControlMessage.TrajInterpolatorType
    LINEAR_POSITION: FrankaControlMessage.TrajInterpolatorType
    LINEAR_POSE: FrankaControlMessage.TrajInterpolatorType
    MIN_JERK_POSE: FrankaControlMessage.TrajInterpolatorType
    SMOOTH_JOINT_POSITION: FrankaControlMessage.TrajInterpolatorType
    MIN_JERK_JOINT_POSITION: FrankaControlMessage.TrajInterpolatorType
    LINEAR_JOINT_POSITION: FrankaControlMessage.TrajInterpolatorType
    COSINE_CARTESIAN_VELOCITY: FrankaControlMessage.TrajInterpolatorType
    LINEAR_CARTESIAN_VELOCITY: FrankaControlMessage.TrajInterpolatorType
    TERMINATION_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_TYPE_FIELD_NUMBER: _ClassVar[int]
    TRAJ_INTERPOLATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    TRAJ_INTERPOLATOR_TIME_FRACTION_FIELD_NUMBER: _ClassVar[int]
    CONTROL_MSG_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    STATE_ESTIMATOR_MSG_FIELD_NUMBER: _ClassVar[int]
    termination: bool
    controller_type: FrankaControlMessage.ControllerType
    traj_interpolator_type: FrankaControlMessage.TrajInterpolatorType
    traj_interpolator_time_fraction: float
    control_msg: _any_pb2.Any
    timeout: float
    state_estimator_msg: FrankaStateEstimatorMessage
    def __init__(self, termination: bool = ..., controller_type: _Optional[_Union[FrankaControlMessage.ControllerType, str]] = ..., traj_interpolator_type: _Optional[_Union[FrankaControlMessage.TrajInterpolatorType, str]] = ..., traj_interpolator_time_fraction: _Optional[float] = ..., control_msg: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., timeout: _Optional[float] = ..., state_estimator_msg: _Optional[_Union[FrankaStateEstimatorMessage, _Mapping]] = ...) -> None: ...
