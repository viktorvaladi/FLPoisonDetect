#!/usr/bin/env python

import argparse
from client import FLClient
from server import FLServer
from cinic10_ds import get_train_ds, get_test_val_ds
import os
import yaml

def run():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--model-config", type=str, required=False)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--total-clients", type=int, default=10)
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--server-address", type=str, default='0.0.0.0')
    parser.add_argument("--is-poisoned", action="store_true")
    parser.add_argument("--nr_of_split_per_round", type=int, default=4)
    args = parser.parse_args()

    model_config_file = args.model_config or 'model_config.yml'
    model_config = {}

    if os.path.isfile(model_config_file):
        with open(model_config_file) as f:
            model_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    rounds = model_config.get('rounds') or args.rounds
    epochs = model_config.get('epochs') or args.epochs
    nr_of_split_per_round = model_config.get('nr_of_split_per_round') or args.nr_of_split_per_round

    if args.server:
        FLServer(rounds, epochs, nr_of_split_per_round).start()
    else:
        FLClient(
            *get_train_ds(args.total_clients, args.client_index),
            *get_test_val_ds(),
            is_poisoned=args.is_poisoned,
        ).start(args.server_address)
