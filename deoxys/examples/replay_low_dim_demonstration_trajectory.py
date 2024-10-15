"""Replay demonstration trajectories."""

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict

from deoxys import config_root
from deoxys.experimental.motion_utils import follow_joint_traj, reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="recorded_trajecotry.hdf5",
    )
    robot_config_parse_args(parser)
    return parser.parse_args()


def main():

    args = parse_args()

    # Load recorded demonstration trajectories
    with open(args.dataset, "r") as f:
        demo_file = h5py.File(args.dataset)

        config = json.loads(demo_file["data"].attrs["config"])

        joint_sequence = demo_file["data/joint_states"]
        action_sequence = demo_file["data/actions"]

    # # Initialize franka interface
    # device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    # device.start_control()

    # Franka Interface
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg))

    # Reset to the same initial joint configuration
    logger.info("Resetting to the initial configuration")
    reset_joints_to(robot_interface, joint_sequence[0])

    # Command the same sequence of actions
    visited_joints = []

    if "OSC" in config["controller_type"]:
        logger.info("Start replay recorded actions using a OSC-family controller")
        for action in action_sequence:
            visited_joints.append(robot_interface.last_q)
            robot_interface.control(
                controller_type=config["controller_type"],
                action=action,
                controller_cfg=EasyDict(config["controller_cfg"]),
            )
    elif config["controller_type"] == "JOINT_IMPEDANCE":
        follow_joint_traj(robot_interface, joint_sequence)
    logger.info("Finish replaying.")
    robot_interface.close()

    visited_joints = np.array(visited_joints)
    print(visited_joints.shape, joint_sequence.shape)

    err = np.linalg.norm(visited_joints - joint_sequence, axis=-1)
    logger.info(f"Final L2 joints seq error: {err.mean()} \\pm {err.std()}")

if __name__ == "__main__":
    main()
