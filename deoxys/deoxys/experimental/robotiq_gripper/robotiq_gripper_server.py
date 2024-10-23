"""Robotiq Gripper code adapted from polymetis, server side (NUC)"""
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import logging
import threading

import zmq
from google.protobuf import any_pb2 as _any_pb2

import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
import deoxys.proto.franka_interface.franka_robot_state_pb2 as franka_robot_state_pb2
from deoxys.utils.yaml_config import YamlConfig

from .third_party.robotiq_2finger_grippers.robotiq_2f_gripper import (
    Robotiq2FingerGripper,
)

log = logging.getLogger(__name__)

DEFAULT_GRASP_FORCE = 2.0

class RobotiqGripperServer:
    """ZMQ server that exposes controls of a Robotiq gripper
    Communicates with the gripper through modbus
    """

    def __init__(self, general_cfg_file: str = "config/local-host.yml", comport="/dev/ttyUSB0"):

        # Connect to gripper
        general_cfg = YamlConfig(general_cfg_file).as_easydict()
        self._gripper_pub_port = general_cfg.NUC.GRIPPER_PUB_PORT
        self._gripper_sub_port = general_cfg.NUC.GRIPPER_SUB_PORT
        self._gripper_sub_ip = general_cfg.PC.IP
        self._gripper_pub_rate = 40.
        if "GRIPPER" in general_cfg and "PUB_RATE" in general_cfg.GRIPPER:
            sel._gripper_pub_rate = general_cfg.GRIPPER.PUB_RATE

        self.comport = comport
        self.init_gripper()

        # Connect to client

        self._context = zmq.Context()
        self._gripper_publisher = self._context.socket(zmq.PUB)
        self._gripper_subscriber = self._context.socket(zmq.SUB)

        # publisher (sends gripper states to client)
        self._gripper_publisher.bind(f"tcp://*:{self._gripper_pub_port}")

        # subscriber (receives gripper commands from client)
        self._gripper_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._gripper_subscriber.connect(f"tcp://{self._gripper_sub_ip}:{self._gripper_sub_port}")

        self._latest_gripper_msg = None

    def init_gripper(self):
        self.gripper = Robotiq2FingerGripper(comport=self.comport)

        if not self.gripper.init_success:
            raise Exception(f"Unable to open comport to {comport}")

        if not self.gripper.getStatus():
            raise Exception(f"Failed to contact gripper on port {comport}... ABORTING")

        print("Activating gripper...")
        self.gripper.home()
        if (
            self.gripper.is_ready()
            and self.gripper.sendCommand()
            and self.gripper.getStatus()
        ):
            print("Activated.")
        else:
            raise Exception(f"Unable to activate!")


    def get_gripper_state(self):
        gripper_state = (
            franka_robot_state_pb2.FrankaGripperStateMessage()
        )

        if not self.gripper.getStatus():
            # NOTE: getStatus returns False and does not update state if modbus read fails
            # gripper_state.error_code = 1
            log.warning(
                "Failed to read gripper state. Returning last observed state instead."
            )

        gripper_state.width = self.gripper.get_pos()
        gripper_state.max_width = self.gripper.stroke
        gripper_state.is_grasped = self.gripper.object_detected()
        gripper_state.temperature = 0

        curr_time = time.time()
        # sec, msec
        gripper_state.time.toSec = curr_time
        gripper_state.time.toMSec = int(curr_time * 1000)
        #    franka_robot_state_pb2.FrankaGripperStateMessage.Duration(toSec=curr_time, toMSec=int(curr_time * 1000))
        #)
        # gripper_state.is_moving = self.gripper.is_moving()

        return gripper_state

    def apply_gripper_command(self, cmd: _any_pb2.Any):
        send_after = True

        h = franka_controller_pb2.FrankaGripperHomingMessage()
        s = franka_controller_pb2.FrankaGripperStopMessage()
        m = franka_controller_pb2.FrankaGripperMoveMessage()
        g = franka_controller_pb2.FrankaGripperGraspMessage()


        # STOP
        if cmd.Unpack(s): 
            self.gripper.stop()
            #print("recieved stop")

        # MOVE
        elif cmd.Unpack(m): 
            self.gripper.goto(pos=m.width, vel=m.speed, force=DEFAULT_GRASP_FORCE)
            #print("recieved move")

        # GRASP
        elif cmd.Unpack(g):
            # use default force for grasp if not specified
            if g.force == 0.:
                g.force = DEFAULT_GRASP_FORCE
            self.gripper.goto(pos=g.width, vel=g.speed, force=g.force)
            #print("recieved grasp")

        # HOMING
        elif cmd.Unpack(h):
            log.info("Homing Disabled (bug: unable to reactivate gripper after).")
            #self.init_gripper()

            send_after = False  # homing does its own sending.

        # error otherwise
        else:
            raise NotImplementedError(f"{type(cmd)} is not implemented for robotiq gripper!")

        # finalize the command by sending it to the gripper over modbus
        if send_after:
            self.gripper.sendCommand()

    def get_state_loop(self):
        # thread for publishing the state at a fixed frequency
        while True:
            start_time = time.time()
            #try:
            self._latest_gripper_msg = self.get_gripper_state()
            self._gripper_publisher.send(self._latest_gripper_msg.SerializeToString())
            #except Exception as e:
            #    log.warning(f"GRIPPER PUB ERROR: {str(e)}")
            remaining_time = (1. / self._gripper_pub_rate) - (time.time() - start_time)
            if remaining_time > 1e-5:
                time.sleep(remaining_time)

    def run(self):

        # start threads for publishing state
        self._state_pub_thread = threading.Thread(target=self.get_state_loop)
        self._state_pub_thread.daemon = True
        self._state_pub_thread.start()

        log.info(f"Publishing gripper state to localhost:{self._gripper_pub_port}, subscribing to {self._gripper_sub_ip}:{self._gripper_sub_port}.")

        # main thread for reading commands and applying it on gripper.
        running = True
        while running:
            # read the state
            #try:
            control_msg = franka_controller_pb2.FrankaGripperControlMessage()
                
            # blocking, wait for new command.
            message = self._gripper_subscriber.recv()
            control_msg.ParseFromString(message)

            # apply various types of gripper control 
            # (BLOCKING, commands will not interrupt each other)
            self.apply_gripper_command(control_msg.control_msg)

            if control_msg.termination:
                log.warning("Commanded Gripper Termination!")
                running = False

            #except Exception as e:
            #    log.warning(e)
