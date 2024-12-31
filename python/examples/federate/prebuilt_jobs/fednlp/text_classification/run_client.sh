#!/usr/bin/env bash
RANK=$1
/usr/bin/python3 torch_main.py --cf config/fedml_config.yaml --rank $RANK --role client --client_id 2 --run_id fednlp_tc
