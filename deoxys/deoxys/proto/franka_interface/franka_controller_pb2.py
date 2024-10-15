# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: franka_controller.proto
# Protobuf Python Version: 5.28.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    2,
    '',
    'franka_controller.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17\x66ranka_controller.proto\x1a\x19google/protobuf/any.proto\"q\n\tJointGoal\x12\x10\n\x08is_delta\x18\x01 \x01(\x08\x12\n\n\x02q1\x18\x02 \x01(\x01\x12\n\n\x02q2\x18\x03 \x01(\x01\x12\n\n\x02q3\x18\x04 \x01(\x01\x12\n\n\x02q4\x18\x05 \x01(\x01\x12\n\n\x02q5\x18\x06 \x01(\x01\x12\n\n\x02q6\x18\x07 \x01(\x01\x12\n\n\x02q7\x18\x08 \x01(\x01\"]\n\x04Goal\x12\x10\n\x08is_delta\x18\x01 \x01(\x08\x12\t\n\x01x\x18\x02 \x01(\x01\x12\t\n\x01y\x18\x03 \x01(\x01\x12\t\n\x01z\x18\x04 \x01(\x01\x12\n\n\x02\x61x\x18\x05 \x01(\x01\x12\n\n\x02\x61y\x18\x06 \x01(\x01\x12\n\n\x02\x61z\x18\x07 \x01(\x01\"i\n\x1a\x45xponentialSmoothingConfig\x12\x0f\n\x07\x61lpha_q\x18\x01 \x01(\x01\x12\x10\n\x08\x61lpha_dq\x18\x02 \x01(\x01\x12\x11\n\talpha_eef\x18\x03 \x01(\x01\x12\x15\n\ralpha_eef_vel\x18\x04 \x01(\x01\"\xe6\x01\n\x1b\x46rankaStateEstimatorMessage\x12\x15\n\ris_estimation\x18\x01 \x01(\x08\x12\x42\n\x0e\x65stimator_type\x18\x02 \x01(\x0e\x32*.FrankaStateEstimatorMessage.EstimatorType\x12$\n\x06\x63onfig\x18\x05 \x01(\x0b\x32\x14.google.protobuf.Any\"F\n\rEstimatorType\x12\x10\n\x0cNO_ESTIMATOR\x10\x00\x12#\n\x1f\x45XPONENTIAL_SMOOTHING_ESTIMATOR\x10\x01\"6\n\x19\x46rankaOSCControllerConfig\x12\x19\n\x11residual_mass_vec\x18\x01 \x03(\x01\",\n\x1a\x46rankaGripperHomingMessage\x12\x0e\n\x06homing\x18\x01 \x01(\x08\"8\n\x18\x46rankaGripperMoveMessage\x12\r\n\x05width\x18\x01 \x01(\x01\x12\r\n\x05speed\x18\x02 \x01(\x01\"(\n\x18\x46rankaGripperStopMessage\x12\x0c\n\x04stop\x18\x01 \x01(\x08\"v\n\x19\x46rankaGripperGraspMessage\x12\r\n\x05width\x18\x01 \x01(\x01\x12\r\n\x05speed\x18\x02 \x01(\x01\x12\r\n\x05\x66orce\x18\x03 \x01(\x01\x12\x15\n\repsilon_inner\x18\x04 \x01(\x01\x12\x15\n\repsilon_outer\x18\x05 \x01(\x01\"]\n\x1b\x46rankaGripperControlMessage\x12\x13\n\x0btermination\x18\x01 \x01(\x08\x12)\n\x0b\x63ontrol_msg\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any\"H\n\x1c\x46rankaDummyControllerMessage\x12\x13\n\x04goal\x18\x01 \x01(\x0b\x32\x05.Goal\x12\x13\n\x0btermination\x18\x02 \x01(\x08\"\xb5\x01\n\x1e\x46rankaOSCPoseControllerMessage\x12\x13\n\x04goal\x18\x01 \x01(\x0b\x32\x05.Goal\x12\x1f\n\x17translational_stiffness\x18\x02 \x03(\x01\x12\x1c\n\x14rotational_stiffness\x18\x03 \x03(\x01\x12\x13\n\x0btermination\x18\x04 \x01(\x08\x12*\n\x06\x63onfig\x18\x05 \x01(\x0b\x32\x1a.FrankaOSCControllerConfig\"z\n$FrankaJointPositionControllerMessage\x12\x18\n\x04goal\x18\x01 \x01(\x0b\x32\n.JointGoal\x12\x10\n\x08kp_gains\x18\x02 \x01(\x01\x12\x10\n\x08kd_gains\x18\x03 \x01(\x01\x12\x14\n\x0cspeed_factor\x18\x04 \x01(\x01\"Y\n%FrankaJointImpedanceControllerMessage\x12\x18\n\x04goal\x18\x01 \x01(\x0b\x32\n.JointGoal\x12\n\n\x02kp\x18\x02 \x03(\x01\x12\n\n\x02kd\x18\x03 \x03(\x01\"y\n(FrankaCartesianVelocityControllerMessage\x12\x13\n\x04goal\x18\x01 \x01(\x0b\x32\x05.Goal\x12\x10\n\x08kp_gains\x18\x02 \x01(\x01\x12\x10\n\x08kd_gains\x18\x03 \x01(\x01\x12\x14\n\x0cspeed_factor\x18\x04 \x01(\x01\"d\n$FrankaJointVelocityControllerMessage\x12\x18\n\x04goal\x18\x01 \x01(\x0b\x32\n.JointGoal\x12\x10\n\x08kp_gains\x18\x02 \x01(\x01\x12\x10\n\x08kd_gains\x18\x03 \x01(\x01\"b\n\"FrankaJointTorqueControllerMessage\x12\x18\n\x04goal\x18\x01 \x01(\x0b\x32\n.JointGoal\x12\x10\n\x08kp_gains\x18\x02 \x01(\x01\x12\x10\n\x08kd_gains\x18\x03 \x01(\x01\"\xf5\x05\n\x14\x46rankaControlMessage\x12\x13\n\x0btermination\x18\x01 \x01(\x08\x12=\n\x0f\x63ontroller_type\x18\x02 \x01(\x0e\x32$.FrankaControlMessage.ControllerType\x12J\n\x16traj_interpolator_type\x18\x03 \x01(\x0e\x32*.FrankaControlMessage.TrajInterpolatorType\x12\'\n\x1ftraj_interpolator_time_fraction\x18\x04 \x01(\x01\x12)\n\x0b\x63ontrol_msg\x18\x05 \x01(\x0b\x32\x14.google.protobuf.Any\x12\x0f\n\x07timeout\x18\x06 \x01(\x01\x12\x39\n\x13state_estimator_msg\x18\x07 \x01(\x0b\x32\x1c.FrankaStateEstimatorMessage\"\xae\x01\n\x0e\x43ontrollerType\x12\x0e\n\nNO_CONTROL\x10\x00\x12\x0c\n\x08OSC_POSE\x10\x01\x12\x10\n\x0cOSC_POSITION\x10\x02\x12\x12\n\x0eJOINT_POSITION\x10\x03\x12\x13\n\x0fJOINT_IMPEDANCE\x10\x04\x12\x12\n\x0eJOINT_VELOCITY\x10\x05\x12\n\n\x06TORQUE\x10\x06\x12\x0b\n\x07OSC_YAW\x10\x07\x12\x16\n\x12\x43\x41RTESIAN_VELOCITY\x10\x08\"\xeb\x01\n\x14TrajInterpolatorType\x12\t\n\x05NO_OP\x10\x00\x12\x13\n\x0fLINEAR_POSITION\x10\x01\x12\x0f\n\x0bLINEAR_POSE\x10\x02\x12\x11\n\rMIN_JERK_POSE\x10\x03\x12\x19\n\x15SMOOTH_JOINT_POSITION\x10\x04\x12\x1b\n\x17MIN_JERK_JOINT_POSITION\x10\x05\x12\x19\n\x15LINEAR_JOINT_POSITION\x10\x06\x12\x1d\n\x19\x43OSINE_CARTESIAN_VELOCITY\x10\x07\x12\x1d\n\x19LINEAR_CARTESIAN_VELOCITY\x10\x08\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'franka_controller_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_JOINTGOAL']._serialized_start=54
  _globals['_JOINTGOAL']._serialized_end=167
  _globals['_GOAL']._serialized_start=169
  _globals['_GOAL']._serialized_end=262
  _globals['_EXPONENTIALSMOOTHINGCONFIG']._serialized_start=264
  _globals['_EXPONENTIALSMOOTHINGCONFIG']._serialized_end=369
  _globals['_FRANKASTATEESTIMATORMESSAGE']._serialized_start=372
  _globals['_FRANKASTATEESTIMATORMESSAGE']._serialized_end=602
  _globals['_FRANKASTATEESTIMATORMESSAGE_ESTIMATORTYPE']._serialized_start=532
  _globals['_FRANKASTATEESTIMATORMESSAGE_ESTIMATORTYPE']._serialized_end=602
  _globals['_FRANKAOSCCONTROLLERCONFIG']._serialized_start=604
  _globals['_FRANKAOSCCONTROLLERCONFIG']._serialized_end=658
  _globals['_FRANKAGRIPPERHOMINGMESSAGE']._serialized_start=660
  _globals['_FRANKAGRIPPERHOMINGMESSAGE']._serialized_end=704
  _globals['_FRANKAGRIPPERMOVEMESSAGE']._serialized_start=706
  _globals['_FRANKAGRIPPERMOVEMESSAGE']._serialized_end=762
  _globals['_FRANKAGRIPPERSTOPMESSAGE']._serialized_start=764
  _globals['_FRANKAGRIPPERSTOPMESSAGE']._serialized_end=804
  _globals['_FRANKAGRIPPERGRASPMESSAGE']._serialized_start=806
  _globals['_FRANKAGRIPPERGRASPMESSAGE']._serialized_end=924
  _globals['_FRANKAGRIPPERCONTROLMESSAGE']._serialized_start=926
  _globals['_FRANKAGRIPPERCONTROLMESSAGE']._serialized_end=1019
  _globals['_FRANKADUMMYCONTROLLERMESSAGE']._serialized_start=1021
  _globals['_FRANKADUMMYCONTROLLERMESSAGE']._serialized_end=1093
  _globals['_FRANKAOSCPOSECONTROLLERMESSAGE']._serialized_start=1096
  _globals['_FRANKAOSCPOSECONTROLLERMESSAGE']._serialized_end=1277
  _globals['_FRANKAJOINTPOSITIONCONTROLLERMESSAGE']._serialized_start=1279
  _globals['_FRANKAJOINTPOSITIONCONTROLLERMESSAGE']._serialized_end=1401
  _globals['_FRANKAJOINTIMPEDANCECONTROLLERMESSAGE']._serialized_start=1403
  _globals['_FRANKAJOINTIMPEDANCECONTROLLERMESSAGE']._serialized_end=1492
  _globals['_FRANKACARTESIANVELOCITYCONTROLLERMESSAGE']._serialized_start=1494
  _globals['_FRANKACARTESIANVELOCITYCONTROLLERMESSAGE']._serialized_end=1615
  _globals['_FRANKAJOINTVELOCITYCONTROLLERMESSAGE']._serialized_start=1617
  _globals['_FRANKAJOINTVELOCITYCONTROLLERMESSAGE']._serialized_end=1717
  _globals['_FRANKAJOINTTORQUECONTROLLERMESSAGE']._serialized_start=1719
  _globals['_FRANKAJOINTTORQUECONTROLLERMESSAGE']._serialized_end=1817
  _globals['_FRANKACONTROLMESSAGE']._serialized_start=1820
  _globals['_FRANKACONTROLMESSAGE']._serialized_end=2577
  _globals['_FRANKACONTROLMESSAGE_CONTROLLERTYPE']._serialized_start=2165
  _globals['_FRANKACONTROLMESSAGE_CONTROLLERTYPE']._serialized_end=2339
  _globals['_FRANKACONTROLMESSAGE_TRAJINTERPOLATORTYPE']._serialized_start=2342
  _globals['_FRANKACONTROLMESSAGE_TRAJINTERPOLATORTYPE']._serialized_end=2577
# @@protoc_insertion_point(module_scope)
