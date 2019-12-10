import argparse
import gym
from reinforcement_learning.isa_algorithm import ISAAlgorithm
from utils import utils
import os
import sys


def get_environment_classes(environment_names):
    environment_classes = []

    # office world
    if "coffee" in environment_names:
        environment_classes.append("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0")
    if "coffee-mail" in environment_names:
        environment_classes.append("gym_subgoal_automata:OfficeWorldDeliverCoffeeAndMail-v0")
    if "visit-abc" in environment_names:
        environment_classes.append("gym_subgoal_automata:OfficeWorldPatrolABC-v0")
    if "visit-abc-strict" in environment_names:
        environment_classes.append("gym_subgoal_automata:OfficeWorldPatrolABCStrict-v0")
    if "visit-abcd" in environment_names:
        environment_classes.append("gym_subgoal_automata:OfficeWorldPatrolABCD-v0")
    if "visit-abcd-strict" in environment_names:
        environment_classes.append("gym_subgoal_automata:OfficeWorldPatrolABCDStrict-v0")

    # water world
    if "water-rg" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreen-v0")
    if "water-bc" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldBlueCyan-v0")
    if "water-my" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldMagentaYellow-v0")
    if "water-rg-bc" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreenAndBlueCyan-v0")
    if "water-bc-my" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldBlueCyanAndMagentaYellow-v0")
    if "water-rg-my" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreenAndMagentaYellow-v0")
    if "water-rg-bc-my" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreenAndBlueCyanAndMagentaYellow-v0")
    if "water-rgb-cmy" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreenBlueAndCyanMagentaYellow-v0")
    if "water-rgb-strict" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreenBlueStrict-v0")
    if "water-cmy-strict" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldCyanMagentaYellowStrict-v0")
    if "water-r-bc" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedAndBlueCyan-v0")
    if "water-rg-b" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreenAndBlue-v0")
    if "water-rgb" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreenBlue-v0")
    if "water-rgbc" in environment_names:
        environment_classes.append("gym_subgoal_automata:WaterWorldRedGreenBlueCyan-v0")

    return environment_classes


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to the input configuration")
    return parser


def get_initial_automata(environment_classes, config):
    initial_automata = []
    if not utils.get_param(config, ISAAlgorithm.INTERLEAVED_FIELD, False):
        initial_automata = [gym.make(env_class).get_automaton() for env_class in environment_classes]
    return initial_automata


def get_tasks(environment_classes, config):
    tasks = []

    use_seed = utils.get_param(config, "use_seed")
    num_tasks = utils.get_param(config, "num_tasks")

    if use_seed is None or num_tasks is None:
        raise RuntimeError("Error: The configuration parameters \'use_seed\' and \'num_tasks\' cannot be undefined.")

    for env_class in environment_classes:
        domain_tasks = []
        for task_id in range(num_tasks):
            seed = task_id if use_seed else None
            domain_tasks.append(gym.make(env_class, params={"generation": "random", "seed": seed, **config}))
        tasks.append(domain_tasks)
    return tasks


if __name__ == "__main__":
    args = get_argparser().parse_args()
    config = utils.read_json_file(args.config_file)

    environment_names = utils.get_param(config, "environments")
    experiment_names = utils.get_param(config, "folder_names")

    if environment_names is None or experiment_names is None:
        raise RuntimeError("Error: The configuration parameters \'environments\' and \'folder_names\' cannot be undefined.")

    environment_classes = get_environment_classes(environment_names)
    initial_automata = get_initial_automata(environment_classes, config)
    tasks = get_tasks(environment_classes, config)

    binary_folder_name = os.path.join(os.path.dirname(sys.argv[0]), "bin")
    isa_algorithm = ISAAlgorithm(len(tasks), initial_automata, experiment_names, config, binary_folder_name)
    isa_algorithm.run(tasks)

