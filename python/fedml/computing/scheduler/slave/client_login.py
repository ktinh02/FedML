import argparse
import os
import fedml
from fedml.computing.scheduler.scheduler_core.general_constants import MarketplaceType
from fedml.computing.scheduler.slave.slave_agent import FedMLLaunchSlaveAgent


def logout():
    FedMLLaunchSlaveAgent.logout()


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="client")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--api_key", "-k", type=str, default="")
    parser.add_argument("--no_gpu_check", "-ngc", type=int, default=1)
    parser.add_argument("--local_on_premise_platform_host", "-lp", type=str, default="127.0.0.1")
    parser.add_argument("--local_on_premise_platform_port", "-lpp", type=int, default=80)
    parser.add_argument("--market_place_type", "-mpt", type=str, default=MarketplaceType.SECURE.name)
    parser.add_argument("--price_per_hour", "-pph", type=str, default="0.0")

    args = parser.parse_args()
    args.user = args.user
    if args.api_key == "":
        args.api_key = args.user

    if args.local_on_premise_platform_host != "127.0.0.1":
        fedml.set_local_on_premise_platform_host(args.local_on_premise_platform_host)
    if args.local_on_premise_platform_port != 80:
        fedml.set_local_on_premise_platform_port(args.local_on_premise_platform_port)

    fedml.set_env_version(args.version)
    slave_agent = FedMLLaunchSlaveAgent()
    if args.type == 'login':
        slave_agent.login(userid=args.api_key, api_key=args.api_key, device_id=args.device_id,
                          os_name=args.os_name, role=args.role, marketplace_type=args.market_place_type,
                          price_per_hour=args.price_per_hour)
    else:
        FedMLLaunchSlaveAgent.logout()
