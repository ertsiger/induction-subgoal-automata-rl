import argparse
import os

from utils import utils
from reinforcement_learning.learning_algorithm import LearningAlgorithm
from reinforcement_learning.isa_qrm_algorithm import ISAAlgorithmBase, ISAAlgorithmQRM
from reinforcement_learning.isa_hrl_algorithm import ISAAlgorithmHRL

BASE_ISA_SEEDS = [25101993, 25041996, 31121960, 20091953, 30422020, 31071993, 11091714, 25071992, 1123581321, 31415926]


def _set_environment_config(config, args, environments, experiment_path):
    config["environments"] = environments
    config["task_generation_method"] = "random"
    config["num_tasks"] = args.num_tasks
    config["use_environment_seed"] = True
    config["starting_environment_seed"] = args.seed
    config["folder_names"] = [os.path.join(experiment_path, env) for env in environments]


def _set_gpu_config(config, use_gpu):
    config[ISAAlgorithmBase.USE_GPU] = use_gpu


def _set_seed_config(config, isa_seed_value):
    config[ISAAlgorithmBase.USE_SEED] = isa_seed_value is not None
    if isa_seed_value is not None:
        config[ISAAlgorithmBase.SEED_VALUE] = isa_seed_value


def _set_checkpoint_config(config, environments, multitask, experiment_path):
    config[ISAAlgorithmBase.CHECKPOINT_ENABLE] = True
    config[ISAAlgorithmBase.CHECKPOINT_FREQUENCY] = 1000  # int(config[LearningAlgorithm.NUM_EPISODES] / 10)
    if multitask:
        config[ISAAlgorithmBase.CHECKPOINT_FOLDER] = experiment_path
    else:
        config[ISAAlgorithmBase.CHECKPOINT_FOLDER] = os.path.join(experiment_path, environments[0])


def _set_interleaved_config(config, args):
    config[ISAAlgorithmBase.INTERLEAVED_FIELD] = args.interleaved_learning
    config[LearningAlgorithm.USE_COMPRESSED_TRACES] = args.use_compressed_traces
    config[LearningAlgorithm.IGNORE_EMPTY_OBSERVATIONS] = args.ignore_empty_observations
    if args.interleaved_learning:
        config[ISAAlgorithmBase.ILASP_TIMEOUT_FIELD] = 7200
        config[ISAAlgorithmBase.ILASP_VERSION_FIELD] = "2"
        config[ISAAlgorithmBase.AVOID_LEARNING_ONLY_NEGATIVE] = args.avoid_learning_negative_only_formulas
        config[ISAAlgorithmBase.USE_RESTRICTED_OBSERVABLES] = args.use_restricted_observables
        config[ISAAlgorithmBase.MAX_DISJUNCTION_SIZE] = args.max_disj_size
        config[ISAAlgorithmBase.LEARN_ACYCLIC_GRAPH] = args.learn_acyclic
        if args.symmetry_breaking_method is not None:
            config[ISAAlgorithmBase.SYMMETRY_BREAKING_METHOD] = args.symmetry_breaking_method
        config[ISAAlgorithmBase.PRIORITIZE_OPTIMAL_SOLUTIONS] = args.prioritize_optimal_solutions
    else:
        config[ISAAlgorithmBase.INITIAL_AUTOMATON_MODE] = "target"


def _set_gridworld_rl_config(config, args):
    config["hide_state_variables"] = True
    if args.domain == "craftworld":
        config["enforce_single_observable_per_location"] = True
        config["height"] = 39
        config["width"] = 39
    config[LearningAlgorithm.DEBUG] = False
    config[LearningAlgorithm.TRAIN_MODEL] = True
    config[LearningAlgorithm.NUM_EPISODES] = 10000
    config[LearningAlgorithm.MAX_EPISODE_LENGTH] = args.maximum_episode_length  # 250 is good for officeworld
    config[LearningAlgorithm.LEARNING_RATE] = 0.1
    config[LearningAlgorithm.EXPLORATION_RATE] = 0.1
    config[LearningAlgorithm.DISCOUNT_RATE] = 0.99
    config[LearningAlgorithm.IS_TABULAR_CASE] = True


def _set_waterworld_rl_config(config, args):
    config["random_restart"] = True
    config[LearningAlgorithm.DEBUG] = False
    config[LearningAlgorithm.TRAIN_MODEL] = True
    config[LearningAlgorithm.NUM_EPISODES] = 50000
    config[LearningAlgorithm.MAX_EPISODE_LENGTH] = args.maximum_episode_length  # 100 can be good for waterworld (higher causes not focus on goal)
    config[LearningAlgorithm.LEARNING_RATE] = 1e-5
    config[LearningAlgorithm.EXPLORATION_RATE] = 0.1
    config[LearningAlgorithm.DISCOUNT_RATE] = 0.9
    config[LearningAlgorithm.IS_TABULAR_CASE] = False
    config[LearningAlgorithm.GREEDY_EVALUATION_FREQUENCY] = 500
    config[LearningAlgorithm.GREEDY_EVALUATION_EPISODES] = 10
    config[ISAAlgorithmBase.USE_DOUBLE_DQN] = True
    config[ISAAlgorithmBase.TARGET_NET_UPDATE_FREQUENCY] = 100
    config[ISAAlgorithmBase.NUM_HIDDEN_LAYERS] = 4
    config[ISAAlgorithmBase.NUM_NEURONS_PER_LAYER] = 64
    config[ISAAlgorithmBase.USE_EXPERIENCE_REPLAY] = True
    config[ISAAlgorithmBase.EXPERIENCE_REPLAY_BUFFER_SIZE] = 50000
    config[ISAAlgorithmBase.EXPERIENCE_REPLAY_BATCH_SIZE] = 32
    config[ISAAlgorithmBase.EXPERIENCE_REPLAY_START_SIZE] = 1000


def _set_algorithm_config(config, domain, algorithm, rl_guidance_method):
    if algorithm == "qrm":
        config[ISAAlgorithmQRM.USE_REWARD_SHAPING] = rl_guidance_method is not None
        if rl_guidance_method is not None:
            config[ISAAlgorithmQRM.REWARD_SHAPING_METHOD] = rl_guidance_method
    elif algorithm == "hrl":
        config[ISAAlgorithmHRL.USE_NUM_POSITIVE_MATCHINGS] = True
        config[ISAAlgorithmHRL.ALWAYS_REUSE_QFUNCTION] = False
        config[ISAAlgorithmHRL.UPDATE_ALL_POLICY_BANK] = False if domain == "waterworld" else True  # very costly for DQN approach to update everything...
        if rl_guidance_method is None:
            config[ISAAlgorithmHRL.ENABLE_PSEUDOREWARD_ON_DEADEND] = False
            config[ISAAlgorithmHRL.PSEUDOREWARD_AFTER_STEP] = 0.0
        else:
            config[ISAAlgorithmHRL.ENABLE_PSEUDOREWARD_ON_DEADEND] = True
            config[ISAAlgorithmHRL.PSEUDOREWARD_AFTER_STEP] = -0.01


def _get_experiment_config(args, experiment_path, isa_seed_value, environments):
    config = {}
    _set_environment_config(config, args, environments, experiment_path)

    if args.domain == "officeworld":
        _set_gridworld_rl_config(config, args)
    elif args.domain == "craftworld":
        _set_gridworld_rl_config(config, args)
    elif args.domain == "waterworld":
        _set_waterworld_rl_config(config, args)

    _set_gpu_config(config, args.use_gpu)
    _set_seed_config(config, isa_seed_value)
    _set_checkpoint_config(config, environments, args.multitask, experiment_path)
    _set_interleaved_config(config, args)
    _set_algorithm_config(config, args.domain, args.algorithm, args.rl_guidance_method)
    return config


def _create_experiments(args, experiment_path, isa_seed_value, experiment_directories):
    utils.mkdir(experiment_path)
    environments = args.environments if args.environments is not None else _get_tasks(args.domain)

    if args.multitask:
        config = _get_experiment_config(args, experiment_path, isa_seed_value, environments)
        utils.write_json_obj(config, os.path.join(experiment_path, "config.json"))
        experiment_directories.append(experiment_path)
    else:
        for env in environments:
            env_path = os.path.join(experiment_path, env)
            utils.mkdir(env_path)
            config = _get_experiment_config(args, experiment_path, isa_seed_value, [env])
            utils.write_json_obj(config, os.path.join(env_path, "config.json"))
            experiment_directories.append(env_path)


def _get_officeworld_tasks():
    return ["coffee", "coffee-mail", "visit-abcd"]


def _get_craftworld_tasks():
    return ["make-plank", "make-stick", "make-cloth", "make-rope", "make-bridge", "make-bed", "make-axe", "make-shears",
            "get-gold", "get-gem"]


def _get_waterworld_tasks():
    return ["water-rgb", "water-rg-b", "water-rg"]


def _get_tasks(domain):
    if domain == "officeworld":
        return _get_officeworld_tasks()
    elif domain == "craftworld":
        return _get_craftworld_tasks()
    elif domain == "waterworld":
        return _get_waterworld_tasks()
    else:
        raise RuntimeError("Error: Unknown domain '%s'." % domain)


def _generate_isa_random_seeds(num_runs):
    isa_random_seeds = []
    seed_sum = 0
    for i in range(num_runs):
        seed = BASE_ISA_SEEDS[i % len(BASE_ISA_SEEDS)] + seed_sum
        isa_random_seeds.append(seed)
        if (i + 1) % len(BASE_ISA_SEEDS) == 0:
            seed_sum += 1
    return isa_random_seeds


def _get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", help="domain whose tasks will be used (officeworld, craftworld, waterworld)")
    parser.add_argument("algorithm", help="which algorithm to use with interleaved automata learning (qrm, hrl)")
    parser.add_argument("num_runs", type=int, help="how many runs to create")
    parser.add_argument("root_experiments_path", help="folder where the experiment folders are created")
    parser.add_argument("experiment_folder_name", help="name of the experiment folder")

    parser.add_argument("--maximum_episode_length", "-m", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num_tasks", "-t", type=int, default=1, help="number of MDPs used")
    parser.add_argument("--seed", type=int, default=0, help="starting environment seed (the 't' tasks have starting from this value)")
    parser.add_argument("--interleaved_learning", "-i", action="store_true", help="whether to learn automata while learning")
    parser.add_argument("--use_restricted_observables", "-r", action="store_true", help="whether to only use the observables required by the task")
    parser.add_argument("--max_disj_size", "-d", type=int, default=1, help="the maximum number of edges between two states")
    parser.add_argument("--learn_acyclic", "-a", action="store_true", help="whether the learned automaton is enforced to be acyclic")
    parser.add_argument("--symmetry_breaking_method", "-s", default=None, help="symmetry breaking method to use (bfs, bfs-alternative, increasing-path)")
    parser.add_argument("--use_compressed_traces", "-c", action="store_true", help="whether to use compressed traces")
    parser.add_argument("--ignore_empty_observations", "-e", action="store_true", help="whether to ignore empty observations")
    parser.add_argument("--prioritize_optimal_solutions", "-p", action="store_true", help="whether to use additional criteria for ranking optimal solutions")
    parser.add_argument("--rl_guidance_method", "-g", default=None, help="use method to guide the RL agent (qrm: max_distance, min_distance, hrl: pseudorewards)")
    parser.add_argument("--avoid_learning_negative_only_formulas", "-n", action="store_true", help="whether to allow learning formulas formed only by negative literals")
    parser.add_argument("--environments", nargs='+', default=None, help="list of environments of the specified domain")

    parser.add_argument("--use_gpu", action="store_true", help="whether to use the gpu")
    parser.add_argument("--timed", action="store_true", help="whether it is an experiment whose running time should be compared with others")
    parser.add_argument("--multitask", action="store_true", help="whether the experiments are multitask")
    return parser


if __name__ == "__main__":
    args = _get_argparser().parse_args()

    seeds = _generate_isa_random_seeds(args.num_runs)

    root_experiments_path = os.path.abspath(args.root_experiments_path)
    folder_name = os.path.join(root_experiments_path, args.experiment_folder_name)

    for i in range(1, args.num_runs + 1):
        experiment_directories = []
        experiment_path = os.path.join(folder_name, "batch_%d" % i)

        _create_experiments(args, experiment_path, seeds[i - 1], experiment_directories)

