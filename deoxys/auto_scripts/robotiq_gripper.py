
import argparse
import os
from deoxys import config_root
from deoxys.deoxys.experimental.robotiq_gripper.robotiq_gripper_server import RobotiqGripperServer
from deoxys.deoxys.utils.yaml_config import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="local-host.yml")
    parser.add_argument("--comport", type=str, default="/dev/ttyUSB0")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Franka Interface
    if args.cfg[0] != "/":
        config_path = os.path.join(config_root, args.cfg)
    else:
        config_path = args.cfg

    cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()

    gripper_server = RobotiqGripperServer(cfg, comport=args.comport)

    gripper_server.run()