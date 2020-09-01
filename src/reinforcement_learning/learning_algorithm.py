from abc import ABC, abstractmethod
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
from timeit import default_timer as timer
import torch
from utils import utils


class LearningAlgorithm(ABC):
    """
    Generic class for the different implemented learning algorithms.
    """
    DEBUG = "debug"                            # whether to print messages for debugging
    TRAIN_MODEL = "train_model"                # whether we are training or testing
    NUM_EPISODES = "num_episodes"              # number of episodes to execute the agent
    MAX_EPISODE_LENGTH = "max_episode_length"  # maximum number of steps per episode
    LEARNING_RATE = "learning_rate"
    DISCOUNT_RATE = "discount_rate"
    IS_TABULAR_CASE = "is_tabular_case"        # whether we use tabular q-learning or function approximation
    USE_GPU = "use_gpu"                        # whether to use the gpu (e.g., for deep rl)

    EXPLORATION_RATE = "exploration_rate"                              # exploration rate value (use only if annealing is not enabled)
    USE_EXPLORATION_RATE_ANNEALING = "use_exploration_rate_annealing"  # whether to use linear annealing over exploration rate
    INITIAL_EXPLORATION_RATE = "initial_exploration_rate"              # initial exploration rate for annealing
    FINAL_EXPLORATION_RATE = "final_exploration_rate"                  # final exploration rate for annealing

    GREEDY_EVALUATION_ENABLE = "greedy_evaluation_enable"        # whether to periodically evaluate the greedy policy
    GREEDY_EVALUATION_FREQUENCY = "greedy_evaluation_frequency"  # how many episodes are executed between evaluations of the greedy policy
    GREEDY_EVALUATION_EPISODES = "greedy_evaluation_episodes"    # how many episodes are used to evaluate the greedy policy

    USE_COMPRESSED_TRACES = "use_compressed_traces"          # whether to used compressed traces for learning
    IGNORE_EMPTY_OBSERVATIONS = "ignore_empty_observations"  # whether to ignore empty observations

    USE_SEED = "use_seed"  # whether to use a seed for Python's random, numpy and torch
    SEED_VALUE = "seed"    # value of the seed

    CHECKPOINT_ENABLE = "checkpoint_enable"        # whether to save progress checkpoints
    CHECKPOINT_FOLDER = "checkpoint_folder"        # where are checkpoints saved
    CHECKPOINT_FILENAME = "checkpoint_%d.pickle"   # checkpoint name pattern
    CHECKPOINT_FREQUENCY = "checkpoint_frequency"  # every how many episodes a checkpoint is produced

    REWARD_STEPS_FOLDER = "reward_steps_logs"                # folder where the reward-steps are saved
    REWARD_STEPS_GREEDY_FOLDER = "reward_steps_greedy_logs"  # folder where the reward-steps for the greedy evaluation are saved
    REWARD_STEPS_FILENAME = "reward_steps-%d.txt"            # reward-steps log file pattern
    REWARD_STEPS_TEST_FILENAME = "reward_steps_test-%d.txt"  # reward-steps log file pattern when evaluating a given model

    ABSOLUTE_RUNNING_TIME_FILENAME = "running_time.txt"  # name of the file registering the total running time of the algorithm
    MODELS_FOLDER = "models"                             # where are the final models saved at the end of the learning

    def __init__(self, tasks, num_tasks, export_folder_names, params):
        self.num_domains = len(tasks)
        self.num_tasks = num_tasks
        self.tasks = tasks
        self.export_folder_names = export_folder_names

        use_gpu = utils.get_param(params, LearningAlgorithm.USE_GPU, False)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.debug = utils.get_param(params, LearningAlgorithm.DEBUG, False)
        self.train_model = utils.get_param(params, LearningAlgorithm.TRAIN_MODEL, True)
        self.num_episodes = utils.get_param(params, LearningAlgorithm.NUM_EPISODES, 20000)
        self.max_episode_length = utils.get_param(params, LearningAlgorithm.MAX_EPISODE_LENGTH, 100)
        self.learning_rate = utils.get_param(params, LearningAlgorithm.LEARNING_RATE, 0.1)
        self.discount_rate = utils.get_param(params, LearningAlgorithm.DISCOUNT_RATE, 0.99)
        self.is_tabular_case = utils.get_param(params, LearningAlgorithm.IS_TABULAR_CASE, True)

        self.exploration_rate = utils.get_param(params, LearningAlgorithm.EXPLORATION_RATE, 0.1)
        self.use_exploration_rate_annealing = utils.get_param(params, LearningAlgorithm.USE_EXPLORATION_RATE_ANNEALING, False)
        self.final_exploration_rate = utils.get_param(params, LearningAlgorithm.FINAL_EXPLORATION_RATE, 0.01)
        if self.use_exploration_rate_annealing:
            self.exploration_rate = utils.get_param(params, LearningAlgorithm.INITIAL_EXPLORATION_RATE, 1.0)
            self.exploration_decay_rate = (self.exploration_rate - self.final_exploration_rate) / self.num_episodes

        self.greedy_evaluation_enable = utils.get_param(params, LearningAlgorithm.GREEDY_EVALUATION_ENABLE, True)
        self.greedy_evaluation_frequency = utils.get_param(params, LearningAlgorithm.GREEDY_EVALUATION_FREQUENCY, 1)
        self.greedy_evaluation_episodes = utils.get_param(params, LearningAlgorithm.GREEDY_EVALUATION_EPISODES, 1)

        self.use_compressed_traces = utils.get_param(params, LearningAlgorithm.USE_COMPRESSED_TRACES, True)
        self.ignore_empty_observations = utils.get_param(params, LearningAlgorithm.IGNORE_EMPTY_OBSERVATIONS, False)

        # learning progress attributes
        self.current_episode = 1
        self.current_domain_id = 0
        self.current_task_id = 0

        self.running_time = 0.0
        self.last_timestamp = None

        # seed attributes
        self.use_seed = utils.get_param(params, LearningAlgorithm.USE_SEED, False)
        self.seed_value = utils.get_param(params, LearningAlgorithm.SEED_VALUE, None)
        self.python_seed_state = None
        self.numpy_seed_state = None
        self.torch_seed_state = None

        if self.use_seed:
            self._set_random_seed()  # need to set these here, especially before creating the model in the subclasses

        # checkpoint attributes
        self.checkpoint_enable = utils.get_param(params, LearningAlgorithm.CHECKPOINT_ENABLE, False)
        self.checkpoint_folder = utils.get_param(params, LearningAlgorithm.CHECKPOINT_FOLDER, ".")
        self.checkpoint_frequency = utils.get_param(params, LearningAlgorithm.CHECKPOINT_FREQUENCY, 5)

        # logs for the different tasks
        self.reward_steps_loggers = []
        self.reward_steps_greedy_loggers = []

        if self.train_model:
            utils.rm_dirs(self.get_reward_episodes_folders())
            utils.rm_dirs(self.get_reward_episodes_greedy_folders())
            utils.rm_dirs(self.get_models_folders())
            utils.rm_files(self.get_running_time_files())

    def __getstate__(self):
        # the loggers must be removed to produce a checkpoint
        state = self.__dict__.copy()
        del state['reward_steps_loggers']
        if self.greedy_evaluation_enable:
            del state['reward_steps_greedy_loggers']
        return state

    '''
    Learning Loop (main loop, what happens when an episode ends, changes or was not completed)
    '''
    def run(self, loaded_checkpoint=False):
        if self.checkpoint_enable and loaded_checkpoint:
            self._restore_uncheckpointed_files()
            if self.use_seed:
                self._load_seed_states()

        if not self.train_model:
            self._import_models()

        self._init_reward_steps_loggers()

        self.last_timestamp = timer()
        self._run_tasks()
        self._write_running_time_files()

        self._export_models()

    def _run_tasks(self):
        while self.current_episode <= self.num_episodes:
            completed_episode, total_reward, episode_length, ended_terminal, observation_history, compressed_history = \
                self._run_episode(self.current_domain_id, self.current_task_id)

            history = compressed_history if self.use_compressed_traces else observation_history
            previous_episode = self.current_episode
            self._on_episode_end(completed_episode, ended_terminal, total_reward, episode_length, history)

            if previous_episode != self.current_episode:
                self._on_episode_change(previous_episode)

            # make a checkpoint
            if self.checkpoint_enable and (not completed_episode or (previous_episode % self.checkpoint_frequency == 0)):
                self._make_checkpoint(previous_episode)

    @abstractmethod
    def _run_episode(self, domain_id, task_id):
        pass

    def _on_episode_end(self, completed_episode, ended_terminal, total_reward, episode_length, history):
        # logging
        self._show_learning_msg(self.current_domain_id, self.current_task_id, self.current_episode, ended_terminal,
                                total_reward, episode_length, history)
        self._log_reward_and_steps(self.reward_steps_loggers, self.current_domain_id, self.current_task_id, total_reward,
                                   episode_length)

        # needed to log when an automaton is learned (see subclasses)
        if not completed_episode:
            self._on_incomplete_episode(self.current_domain_id)

        # update next domain, task and episode to work with
        if self.num_domains > 1:
            self.current_domain_id += 1
            if self.current_domain_id == self.num_domains:
                self.current_domain_id = 0
                self._update_task_and_episode_counters()
        else:
            self._update_task_and_episode_counters()

    @abstractmethod
    def _on_incomplete_episode(self, current_domain_id):
        pass

    def _on_episode_change(self, previous_episode):
        # update exploration rate according to annealing
        if self.use_exploration_rate_annealing:
            self.exploration_rate = max(self.exploration_rate - self.exploration_decay_rate, self.final_exploration_rate)

        # perform evaluation of the greedy policies
        if self.train_model and self.greedy_evaluation_enable and previous_episode % self.greedy_evaluation_frequency == 0:
            self._evaluate_greedy_policies()

    '''
    Task Management Methods (tasks from ids, next task to interact with)
    '''
    def _get_task(self, domain_id, task_id):
        return self.tasks[domain_id][task_id]

    def _update_task_and_episode_counters(self):
        self.current_task_id = (self.current_task_id + 1) % self.num_tasks
        if self.current_task_id == 0:
            self.current_episode += 1

    '''
    Action Selection (epsilon-greedy)
    '''
    def _choose_egreedy_action(self, task, state, q_table):
        if self.train_model:
            prob = random.uniform(0, 1)
            if prob <= self.exploration_rate:
                return self._get_random_action(task)
        return self._get_greedy_action(task, state, q_table)

    def _get_greedy_action(self, task, state, q_table):
        if self.is_tabular_case:
            q_values = [q_table[(state, action)] for action in range(task.action_space.n)]
            return utils.randargmax(q_values)
        else:
            state_v = torch.tensor(state).to(self.device)
            q_values = q_table(state_v)
            return utils.randargmax(q_values.detach().cpu().numpy())

    def _get_random_action(self, task):
        return random.choice(range(0, task.action_space.n))

    '''
    History and Observation Management
    '''
    def _get_observations_as_ordered_tuple(self, observation_set):
        observations_list = list(observation_set)
        utils.sort_by_ord(observations_list)
        return tuple(observations_list)

    def _update_histories(self, observation_history, compressed_observation_history, observations):
        # update histories only if the observation is non-empty
        if self.ignore_empty_observations and len(observations) == 0:
            return False

        observations_tuple = self._get_observations_as_ordered_tuple(observations)
        observation_history.append(observations_tuple)

        observations_changed = len(compressed_observation_history) == 0 or observations_tuple != compressed_observation_history[-1]
        if observations_changed:
            compressed_observation_history.append(observations_tuple)

        if self.use_compressed_traces:
            return observations_changed
        return True  # all observations are relevant if the traces are uncompressed

    '''
    Greedy Policy Evaluation
    '''
    def _evaluate_greedy_policies(self):
        self.train_model = False
        for domain_id in range(self.num_domains):
            for task_id in range(self.num_tasks):
                self._evaluate_greedy_policies_helper(domain_id, task_id)
        self.train_model = True

    def _evaluate_greedy_policies_helper(self, domain_id, task_id):
        sum_total_reward, sum_episode_length = 0, 0
        for evaluation_episode in range(self.greedy_evaluation_episodes):
            _, total_reward, episode_length, _, _, _ = self._run_episode(domain_id, task_id)
            sum_total_reward += total_reward
            sum_episode_length += episode_length
        avg_total_reward = sum_total_reward / self.greedy_evaluation_episodes
        avg_episode_length = sum_episode_length / self.greedy_evaluation_episodes
        self._log_reward_and_steps(self.reward_steps_greedy_loggers, domain_id, task_id, avg_total_reward, avg_episode_length)

    '''
    Logging
    '''
    def _show_learning_msg(self, domain_id, task_id, episode, ended_terminal, total_reward, episode_length, history):
        if self.debug:
            print("Domain: " + str(domain_id) +
                  " - Task: " + str(task_id) +
                  " - Episode: " + str(episode) +
                  " - Terminal: " + str(ended_terminal) +
                  " - Reward: " + str(total_reward) +
                  " - Steps: " + str(episode_length) +
                  " - Observations: " + str(history))

    def _init_reward_steps_loggers(self):
        if self.train_model:
            self.reward_steps_loggers = self._init_reward_steps_loggers_helper(LearningAlgorithm.REWARD_STEPS_FOLDER,
                                                                               LearningAlgorithm.REWARD_STEPS_FILENAME)
            if self.greedy_evaluation_enable:
                self.reward_steps_greedy_loggers = self._init_reward_steps_loggers_helper(LearningAlgorithm.REWARD_STEPS_GREEDY_FOLDER,
                                                                                          LearningAlgorithm.REWARD_STEPS_FILENAME)
        else:
            self.reward_steps_loggers = self._init_reward_steps_loggers_helper(LearningAlgorithm.REWARD_STEPS_FOLDER,
                                                                               LearningAlgorithm.REWARD_STEPS_TEST_FILENAME)

    def _init_reward_steps_loggers_helper(self, folder_name, filename_pattern):
        reward_steps_loggers = []
        for domain_id in range(self.num_domains):
            folder_name = os.path.join(self.export_folder_names[domain_id], folder_name)
            utils.mkdir(folder_name)

            task_loggers = []
            for task_id in range(self.num_tasks):
                filename = filename_pattern % task_id

                name = os.path.join(folder_name, filename)
                handler = logging.FileHandler(name)

                logger = logging.getLogger(name)
                logger.setLevel(logging.INFO)
                logger.addHandler(handler)

                task_loggers.append(logger)

            reward_steps_loggers.append(task_loggers)
        return reward_steps_loggers

    def _log_reward_and_steps(self, reward_steps_loggers, domain_id, task_id, episode_reward, episode_length):
        reward_steps_loggers[domain_id][task_id].info(";".join([str(episode_reward), str(episode_length)]))

    '''
    Checkpoint Management
    '''
    def _make_checkpoint(self, episode):
        self._update_running_time()

        if self.use_seed:
            self._save_seed_states()

        filename = LearningAlgorithm.CHECKPOINT_FILENAME % episode
        file_path = os.path.join(self.checkpoint_folder, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def _restore_uncheckpointed_files(self):  # inherited by subclasses
        self._unlog_uncheckpointed_episodes()

    def _unlog_uncheckpointed_episodes(self):
        """Removes the lines for uncheckpointed episodes."""
        for domain_id in range(self.num_domains):
            self._unlog_uncheckpointed_episodes_helper(self.get_reward_episodes_folder(domain_id),
                                                       LearningAlgorithm.REWARD_STEPS_FILENAME,
                                                       self.current_episode - 1)
            if self.greedy_evaluation_enable:
                self._unlog_uncheckpointed_episodes_helper(self.get_reward_episodes_greedy_folder(domain_id),
                                                           LearningAlgorithm.REWARD_STEPS_FILENAME,
                                                           int((self.current_episode - 1) / self.greedy_evaluation_frequency))

    def _unlog_uncheckpointed_episodes_helper(self, folder_name, filename_pattern, num_logged_episodes):
        if utils.path_exists(folder_name):
            for task_id in range(self.num_tasks):
                reward_episodes_file = filename_pattern % task_id
                reward_episodes_file_path = os.path.join(folder_name, reward_episodes_file)
                if utils.path_exists(reward_episodes_file_path):
                    try:
                        df = pd.read_csv(reward_episodes_file_path, nrows=num_logged_episodes, sep=';', header=None)
                        df.to_csv(reward_episodes_file_path, sep=';', index=False, header=None)
                    except pd.errors.EmptyDataError:
                        pass

    def get_reward_episodes_folders(self):
        return [self.get_reward_episodes_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_reward_episodes_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], LearningAlgorithm.REWARD_STEPS_FOLDER)

    def get_reward_episodes_greedy_folders(self):
        return [self.get_reward_episodes_greedy_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_reward_episodes_greedy_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], LearningAlgorithm.REWARD_STEPS_GREEDY_FOLDER)

    '''
    Management of the file keeping track of the total running time
    '''
    def _update_running_time(self):
        current_timestamp = timer()
        self.running_time += current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp

    def _write_running_time_files(self):
        self._update_running_time()
        for filename in self.get_running_time_files():
            with open(filename, 'w') as f:
                f.write(str(self.running_time))

    def get_running_time_files(self):
        return [self.get_running_time_file(domain_id) for domain_id in range(self.num_domains)]

    def get_running_time_file(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], LearningAlgorithm.ABSOLUTE_RUNNING_TIME_FILENAME)

    '''
    Random Seed Management
    '''
    def _set_random_seed(self):
        if not isinstance(self.seed_value, int):
            raise RuntimeError("Error: the seed must be an integer value.")

        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        self._set_torch_cudnn()

    def _set_torch_cudnn(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

    def _load_seed_states(self):
        assert self.python_seed_state is not None
        assert self.numpy_seed_state is not None
        assert self.torch_seed_state is not None

        random.setstate(self.python_seed_state)
        np.random.set_state(self.numpy_seed_state)
        torch.set_rng_state(self.torch_seed_state)

        self._set_torch_cudnn()

    def _save_seed_states(self):
        self.python_seed_state = random.getstate()
        self.numpy_seed_state = np.random.get_state()
        self.torch_seed_state = torch.get_rng_state()

    '''
    Model Management
    '''
    @abstractmethod
    def _export_models(self):
        pass

    @abstractmethod
    def _import_models(self):
        pass

    def get_models_folders(self):
        return [self.get_models_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_models_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], LearningAlgorithm.MODELS_FOLDER)
