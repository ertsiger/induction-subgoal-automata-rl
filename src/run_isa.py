#!/usr/bin/env python3

import argparse
import gym
import pickle
import os
import sys

from gym_subgoal_automata.envs.base.base_env import BaseEnv
from reinforcement_learning.isa_base_algorithm import ISAAlgorithmBase
from reinforcement_learning.isa_qrm_algorithm import ISAAlgorithmQRM
from reinforcement_learning.isa_hrl_algorithm import ISAAlgorithmHRL
from reinforcement_learning.tabular_qlearning import TabularQLearning
from utils import utils

ENV_SUBGOAL_AUTOMATA_PREFIX = "gym_subgoal_automata:"


def get_environment_classes(environment_names):
    environment_classes = []

    # office world
    if "coffee" in environment_names:
        environment_classes.append(("OfficeWorldDeliverCoffee-v0", {}))
    if "coffee-drop" in environment_names:
        environment_classes.append(("OfficeWorldDeliverCoffee-v0", {"drop_coffee_enable": True}))
    if "mail" in environment_names:
        environment_classes.append(("OfficeWorldDeliverMail-v0", {}))
    if "coffee-mail" in environment_names:
        environment_classes.append(("OfficeWorldDeliverCoffeeAndMail-v0", {}))
    if "coffee-mail-drop" in environment_names:
        environment_classes.append(("OfficeWorldDeliverCoffeeAndMail-v0", {"drop_coffee_enable": True}))
    if "coffee-or-mail" in environment_names:
        environment_classes.append(("OfficeWorldDeliverCoffeeOrMail-v0", {}))
    if "visit-ab" in environment_names:
        environment_classes.append(("OfficeWorldPatrolAB-v0", {}))
    if "visit-ab-strict" in environment_names:
        environment_classes.append(("OfficeWorldPatrolABStrict-v0", {}))
    if "visit-abc" in environment_names:
        environment_classes.append(("OfficeWorldPatrolABC-v0", {}))
    if "visit-abc-strict" in environment_names:
        environment_classes.append(("OfficeWorldPatrolABCStrict-v0", {}))
    if "visit-abcd" in environment_names:
        environment_classes.append(("OfficeWorldPatrolABCD-v0", {}))
    if "visit-abcd-strict" in environment_names:
        environment_classes.append(("OfficeWorldPatrolABCDStrict-v0", {}))

    # water world
    if "water-rg" in environment_names:
        environment_classes.append(("WaterWorldRedGreen-v0", {}))
    if "water-bc" in environment_names:
        environment_classes.append(("WaterWorldBlueCyan-v0", {}))
    if "water-my" in environment_names:
        environment_classes.append(("WaterWorldMagentaYellow-v0", {}))
    if "water-rg-bc" in environment_names:
        environment_classes.append(("WaterWorldRedGreenAndBlueCyan-v0", {}))
    if "water-bc-my" in environment_names:
        environment_classes.append(("WaterWorldBlueCyanAndMagentaYellow-v0", {}))
    if "water-rg-my" in environment_names:
        environment_classes.append(("WaterWorldRedGreenAndMagentaYellow-v0", {}))
    if "water-rg-bc-my" in environment_names:
        environment_classes.append(("WaterWorldRedGreenAndBlueCyanAndMagentaYellow-v0", {}))
    if "water-rgb-cmy" in environment_names:
        environment_classes.append(("WaterWorldRedGreenBlueAndCyanMagentaYellow-v0", {}))
    if "water-rg-strict" in environment_names:
        environment_classes.append(("WaterWorldRedGreenStrict-v0", {}))
    if "water-rgb-strict" in environment_names:
        environment_classes.append(("WaterWorldRedGreenBlueStrict-v0", {}))
    if "water-cmy-strict" in environment_names:
        environment_classes.append(("WaterWorldCyanMagentaYellowStrict-v0", {}))
    if "water-r-bc" in environment_names:
        environment_classes.append(("WaterWorldRedAndBlueCyan-v0", {}))
    if "water-rg-b" in environment_names:
        environment_classes.append(("WaterWorldRedGreenAndBlue-v0", {}))
    if "water-rgb" in environment_names:
        environment_classes.append(("WaterWorldRedGreenBlue-v0", {}))
    if "water-rgbc" in environment_names:
        environment_classes.append(("WaterWorldRedGreenBlueCyan-v0", {}))
    if "water-rgbcy" in environment_names:
        environment_classes.append(("WaterWorldRedGreenBlueCyanYellow-v0", {}))
    if "water-r-g-b" in environment_names:
        environment_classes.append(("WaterWorldRedAndGreenAndBlue-v0", {}))
    if "water-rg-avoid-m" in environment_names:
        environment_classes.append(("WaterWorldRedGreenAvoidMagenta-v0", {}))
    if "water-rg-avoid-my" in environment_names:
        environment_classes.append(("WaterWorldRedGreenAvoidMagentaYellow-v0", {}))
    if "water-r-avoid-m" in environment_names:
        environment_classes.append(("WaterWorldRedAvoidMagenta-v0", {}))

    # craft world
    if "make-plank" in environment_names:
        environment_classes.append(("CraftWorldMakePlank-v0", {}))
    if "make-stick" in environment_names:
        environment_classes.append(("CraftWorldMakeStick-v0", {}))
    if "make-cloth" in environment_names:
        environment_classes.append(("CraftWorldMakeCloth-v0", {}))
    if "make-rope" in environment_names:
        environment_classes.append(("CraftWorldMakeRope-v0", {}))
    if "make-bridge" in environment_names:
        environment_classes.append(("CraftWorldMakeBridge-v0", {}))
    if "make-bed" in environment_names:
        environment_classes.append(("CraftWorldMakeBed-v0", {}))
    if "make-axe" in environment_names:
        environment_classes.append(("CraftWorldMakeAxe-v0", {}))
    if "make-shears" in environment_names:
        environment_classes.append(("CraftWorldMakeShears-v0", {}))
    if "get-gold" in environment_names:
        environment_classes.append(("CraftWorldGetGold-v0", {}))
    if "get-gem" in environment_names:
        environment_classes.append(("CraftWorldGetGem-v0", {}))

    # three coloured rooms
    if "cookie" in environment_names:
        environment_classes.append(("Cookie-v0", {}))
    if "symbol" in environment_names:
        environment_classes.append(("Symbol-v0", {}))
    if "two-keys" in environment_names:
        environment_classes.append(("TwoKeys-v0", {}))

    return environment_classes


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", help="which algorithm to use with interleaved automata learning (qrm, hrl)")
    parser.add_argument("config_file", help="path to the input configuration")
    return parser


def get_target_automata(environment_classes):
    return [gym.make(ENV_SUBGOAL_AUTOMATA_PREFIX + env_class, params={**env_params, "generation": "random"}).get_automaton()
            for env_class, env_params in environment_classes]


def get_random_tasks(environment_classes, config):
    tasks = []

    use_seed = get_param(config, "use_environment_seed")
    num_tasks = get_param(config, "num_tasks")

    for env_class, env_params in environment_classes:
        domain_tasks = []
        for task_id in range(num_tasks):
            seed = task_id + get_param(config, "starting_environment_seed") if use_seed else None
            task_params = {**env_params, "generation": "random", BaseEnv.RANDOM_SEED_FIELD: seed, **config}
            domain_tasks.append(gym.make(ENV_SUBGOAL_AUTOMATA_PREFIX + env_class, params=task_params))
        tasks.append(domain_tasks)
    return tasks


def get_predefined_tasks(environment_classes, config):
    if len(environment_classes) > 1:
        raise RuntimeError("Error: Only one environment is supported when tasks predefined.")

    environment = get_param(config, "environments")[0]
    if environment == "coffee":
        maps = [{"A": (1, 1), "f": [(1, 8)], "g": (1, 3), "n": [(0, 4)], "m": (11, 0), "a": (4, 4), "b": (7, 4),
                 "c": (8, 8), "d": (9, 0)},
                {"A": (1, 1), "f": [(1, 3)], "g": (1, 3), "n": [(0, 4)], "m": (11, 0), "a": (4, 4), "b": (7, 4),
                 "c": (8, 8), "d": (9, 0)}]
    elif environment == "coffee-mail":
        maps = [{"A": (1, 1), "f": [(0, 5)], "g": (1, 3), "n": [(1, 4)], "m": (2, 3), "a": (4, 4), "b": (7, 4),
                 "c": (8, 8), "d": (9, 0)},
                {"A": (1, 1), "f": [(0, 3)], "g": (1, 3), "n": [(1, 4)], "m": (1, 7), "a": (4, 4), "b": (7, 4),
                 "c": (8, 8), "d": (9, 0)},
                {"A": (1, 1), "f": [(1, 5)], "g": (1, 3), "n": [(0, 4)], "m": (1, 5), "a": (4, 4), "b": (7, 4),
                 "c": (8, 8), "d": (9, 0)},
                {"A": (1, 1), "f": [(1, 3)], "g": (1, 3), "n": [(0, 4)], "m": (1, 3), "a": (4, 4), "b": (7, 4),
                 "c": (8, 8), "d": (9, 0)},
                {"A": (1, 1), "f": [(0, 5)], "g": (1, 3), "n": [(0, 4)], "m": (1, 3), "a": (4, 4), "b": (7, 4),
                 "c": (8, 8), "d": (9, 0)},
                {"A": (1, 1), "f": [(1, 3)], "g": (1, 3), "n": [(0, 4)], "m": (0, 5), "a": (4, 4), "b": (7, 4),
                 "c": (8, 8), "d": (9, 0)}]
    elif environment == "visit-abcd":
        maps = [{"A": (1, 1), "a": (2, 2), "b": (0, 2), "c": (0, 0), "d": (2, 0), "n": [(1, 2)], "f": [(0, 3)],
                 "g": (1, 3), "m": (1, 3)}]
    else:
        raise RuntimeError("Error: There are not predefined tasks for '{}'.".format(environment))

    tasks = []
    for m in maps:
        tasks.append(gym.make(environment_classes[0], params={"generation": "params", "map": m, **config}))

    return [tasks], len(tasks)


def get_algorithm(algorithm_name, config):
    environment_classes = get_environment_classes(get_param(config, "environments"))

    task_generation_method = get_param(config, "task_generation_method")
    if task_generation_method == "random":
        args = [get_random_tasks(environment_classes, config), get_param(config, "num_tasks")]
    elif task_generation_method == "predefined":
        predefined_tasks, num_tasks = get_predefined_tasks(environment_classes, config)
        args = [predefined_tasks, num_tasks]
    else:
        raise RuntimeError("Error: Unknown task generation method {}.".format(task_generation_method))

    args.extend([get_param(config, "folder_names"), config])

    if algorithm_name == "qrm" or algorithm_name == "hrl":
        args.extend([get_target_automata(environment_classes),
                     os.path.join(os.path.dirname(sys.argv[0]), "bin")])
        algorithm_class = ISAAlgorithmQRM if algorithm_name == "qrm" else ISAAlgorithmHRL
    elif algorithm_name == "qlearning":
        algorithm_class = TabularQLearning
    else:
        raise RuntimeError("Error: Unknown algorithm %s." % algorithm_name)
    return algorithm_class(*args)


def get_param(config, param_name):
    param_value = utils.get_param(config, param_name)
    if param_value is None:
        raise RuntimeError("Error: The configuration parameters \'%s\' cannot be undefined." % param_name)
    return param_value


def get_checkpoint_filenames(checkpoint_folder):
    if os.path.exists(checkpoint_folder):
        return [x for x in os.listdir(checkpoint_folder) if x.startswith("checkpoint")]
    return []


def checkpoints_exist(checkpoint_folder):
    return len(get_checkpoint_filenames(checkpoint_folder)) > 0


def get_last_checkpoint_filename(checkpoint_folder):
    checkpoint_filenames = get_checkpoint_filenames(checkpoint_folder)
    checkpoint_filenames.sort(key=lambda x: int(x[len("checkpoint-"):-len(".pickle")]))
    return os.path.join(checkpoint_folder, checkpoint_filenames[-1])


def load_last_checkpoint(checkpoint_folder):
    with open(get_last_checkpoint_filename(checkpoint_folder), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    args = get_argparser().parse_args()
    config = utils.read_json_file(args.config_file)

    loaded_checkpoint = False

    if get_param(config, ISAAlgorithmBase.CHECKPOINT_ENABLE) \
            and checkpoints_exist(get_param(config, ISAAlgorithmBase.CHECKPOINT_FOLDER)):
        isa_algorithm = load_last_checkpoint(get_param(config, ISAAlgorithmBase.CHECKPOINT_FOLDER))
        loaded_checkpoint = True
    else:
        isa_algorithm = get_algorithm(args.algorithm, config)
    isa_algorithm.run(loaded_checkpoint)

