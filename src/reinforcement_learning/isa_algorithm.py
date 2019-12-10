import os
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from utils import utils
from utils.ilasp.generator import ilasp_task_generator
from utils.ilasp.solver import ilasp_solution_parser
from utils.ilasp.solver.ilasp_solver import solve_ilasp_task
from reinforcement_learning.dqn_model import DQN
from reinforcement_learning.experience_replay import ExperienceBuffer, Experience
from reinforcement_learning.learning_algorithm import LearningAlgorithm
from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton
from timeit import default_timer as timer


class MultipleConditionsHoldException(Exception):
    def __init__(self):
        super(MultipleConditionsHoldException, self).__init__("Error: Multiple conditions cannot hold at the same time.")


class ISAAlgorithm(LearningAlgorithm):
    ACCEPTING_STATE_NAME = "s_acc"
    REJECTING_STATE_NAME = "s_rej"

    LOAD_LAST_AUTOMATON_SOLUTION = "load_last_automaton_solution"  # whether to load last automaton solution from experiment folder

    INTERLEAVED_FIELD = "interleaved_automaton_learning"  # whether RL is interleaved with ILASP automaton learner
    ILASP_TIMEOUT_FIELD = "ilasp_timeout"  # time that ILASP has for finding a single automaton solution
    ILASP_VERSION_FIELD = "ilasp_version"  # ILASP version to run
    USE_COMPRESSED_TRACES_FIELD = "use_compressed_traces"  # whether to used compressed traces for learning
    STARTING_NUM_STATES_FIELD = "starting_num_states"  # number of states that the starting automaton has
    USE_RESTRICTED_OBSERVABLES = "use_restricted_observables"  # use the restricted set of observables (the ones that define the goal for the task)
    MAX_DISJUNCTION_SIZE = "max_disjunction_size"  # related to the previous flag: maximum number of literals inside a disjunction
    LEARN_ACYCLIC_GRAPH = "learn_acyclic_graph"  # whether the target automata has cycles or not
    SYMMETRY_BREAKING_METHOD = "symmetry_breaking_method"  # which symmetry breaking method is used to break symmetries in the graph
    REWARD_ON_REJECTING_STATE = "reward_rejecting_state"  # whether to give a reward of -1 in the rejecting state

    USE_REWARD_SHAPING = "use_reward_shaping"  # whether reward shaping is used

    USE_EXPERIENCE_REPLAY = "use_experience_replay"  # whether to use the experience replay buffer for learning (automatically active for deep learning approach)
    EXPERIENCE_REPLAY_BUFFER_SIZE = "experience_replay_buffer_size"  # size of the ER buffer
    EXPERIENCE_REPLAY_BATCH_SIZE = "experience_replay_batch_size"  # size of the batches sampled from the ER buffer
    EXPERIENCE_REPLAY_START_SIZE = "experience_replay_start_size"  # size of the ER after which learning starts

    USE_DOUBLE_DQN = "use_double_dqn"  # whether double DQN is used instead of simple DQN
    TARGET_NET_UPDATE_FREQUENCY = "target_net_update_frequency"  # how many steps happen between target DQN updates
    NUM_HIDDEN_LAYERS = "num_hidden_layers"  # number of hidden layers that the network has
    NUM_NEURONS_PER_LAYER = "num_neurons_per_layer"  # number of neurons per hidden layer

    AUTOMATON_TASK_FOLDER = "automaton_tasks"
    AUTOMATON_TASK_FILENAME = "task-%d.las"

    AUTOMATON_SOLUTION_FOLDER = "automaton_solutions"
    AUTOMATON_SOLUTION_FILENAME = "solution-%d.txt"

    AUTOMATON_PLOT_FOLDER = "automaton_plots"
    AUTOMATON_PLOT_FILENAME = "plot-%d.png"

    LEARNT_POLICIES_FOLDER = "learnt_policies"
    LEARNT_POLICY_FILENAME = "policy-%d.txt"

    REWARD_STEPS_FOLDER = "reward_steps_episodes"
    REWARD_STEPS_FILENAME = "reward_steps-%d.txt"
    REWARD_STEPS_TEST_FILENAME = "reward_steps_test-%d.txt"

    ABSOLUTE_RUNNING_TIME_FILENAME = "running_time.txt"
    AUTOMATON_LEARNING_EPISODES_FILENAME = "automaton_learning_episodes.txt"

    MODELS_FOLDER = "models"
    MODEL_FILENAME = "model-%d-%s.pt"

    def __init__(self, num_domains, automata, export_folder_names, params, binary_folder_name=None):
        super().__init__(params)

        self.num_domains = num_domains
        self.export_folder_names = export_folder_names
        self.binary_folder_name = binary_folder_name

        self.load_last_automaton_solution = utils.get_param(params, ISAAlgorithm.LOAD_LAST_AUTOMATON_SOLUTION, False)

        # interleaved automaton learning params
        self.interleaved_automaton_learning = utils.get_param(params, ISAAlgorithm.INTERLEAVED_FIELD, False)
        self.ilasp_timeout = utils.get_param(params, ISAAlgorithm.ILASP_TIMEOUT_FIELD, 120)
        self.ilasp_version = utils.get_param(params, ISAAlgorithm.ILASP_VERSION_FIELD, "2")
        self.use_compressed_traces = utils.get_param(params, ISAAlgorithm.USE_COMPRESSED_TRACES_FIELD, True)
        self.num_starting_states = utils.get_param(params, ISAAlgorithm.STARTING_NUM_STATES_FIELD, 3)
        self.num_automaton_states = None
        self.use_restricted_observables = utils.get_param(params, ISAAlgorithm.USE_RESTRICTED_OBSERVABLES, False)
        self.max_disjunction_size = utils.get_param(params, ISAAlgorithm.MAX_DISJUNCTION_SIZE, 1)
        self.learn_acyclic_graph = utils.get_param(params, ISAAlgorithm.LEARN_ACYCLIC_GRAPH, True)
        self.symmetry_breaking_method = utils.get_param(params, ISAAlgorithm.SYMMETRY_BREAKING_METHOD, None)
        self.has_observed_pos_example = None
        self.reward_on_rejecting_state = utils.get_param(params, ISAAlgorithm.REWARD_ON_REJECTING_STATE, False)

        self.use_reward_shaping = utils.get_param(params, ISAAlgorithm.USE_REWARD_SHAPING, False)

        # experience replay
        self.use_experience_replay = utils.get_param(params, ISAAlgorithm.USE_EXPERIENCE_REPLAY, False) or not self.is_tabular_case
        self.experience_replay_buffer_size = utils.get_param(params, ISAAlgorithm.EXPERIENCE_REPLAY_BUFFER_SIZE, 50000)
        self.experience_replay_batch_size = utils.get_param(params, ISAAlgorithm.EXPERIENCE_REPLAY_BATCH_SIZE, 32)
        self.experience_replay_start_size = utils.get_param(params, ISAAlgorithm.EXPERIENCE_REPLAY_START_SIZE, 1000)
        self.experience_replay_buffers = None

        # q-tables in the case of tabular QRM and dqn in the case of DQRM
        self.q_functions = []

        # deep q-learning
        self.double_dqn = utils.get_param(params, ISAAlgorithm.USE_DOUBLE_DQN, True)
        self.num_layers = utils.get_param(params, ISAAlgorithm.NUM_HIDDEN_LAYERS, 6)
        self.num_neurons_per_layer = utils.get_param(params, ISAAlgorithm.NUM_NEURONS_PER_LAYER, 64)
        self.target_net_update_frequency = utils.get_param(params, ISAAlgorithm.TARGET_NET_UPDATE_FREQUENCY, 100)
        self.target_net_update_counter = 0
        self.target_q_functions = []
        self.optimizers = []

        # set of automata per domain
        self.automata = None
        self._set_automata(automata)

        # sets of examples
        self.pos_examples = None
        self.neg_examples = None
        self.inc_examples = None

        # keep track of the number of learnt automata per domain
        self.automaton_counters = None

        # set of tasks to be learnt
        self.tasks = None

        # logs for the different tasks
        self.reward_steps_loggers = []
        self.automaton_learning_episodes_loggers = []

        if self.train_model:  # if the tasks are learnt, remove previous folders if they exist
            utils.rm_dirs(self.get_automaton_task_folders())
            utils.rm_dirs(self.get_automaton_solution_folders())
            utils.rm_dirs(self.get_automaton_plot_folders())
            utils.rm_dirs(self.get_learnt_policies_folders())
            utils.rm_dirs(self.get_reward_episodes_folders())
            utils.rm_dirs(self.get_models_folders())
            utils.rm_files(self.get_automaton_learning_episodes_files())
            utils.rm_files(self.get_running_time_files())

    def get_automaton_task_folders(self):
        return [self.get_automaton_task_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_task_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.AUTOMATON_TASK_FOLDER)

    def get_automaton_solution_folders(self):
        return [self.get_automaton_solution_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_solution_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.AUTOMATON_SOLUTION_FOLDER)

    def get_automaton_plot_folders(self):
        return [self.get_automaton_plot_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_plot_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.AUTOMATON_PLOT_FOLDER)

    def get_learnt_policies_folders(self):
        return [self.get_learnt_policies_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_learnt_policies_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.LEARNT_POLICIES_FOLDER)

    def get_reward_episodes_folders(self):
        return [self.get_reward_episodes_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_reward_episodes_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.REWARD_STEPS_FOLDER)

    def get_models_folders(self):
        return [self.get_models_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_models_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.MODELS_FOLDER)

    def get_automaton_learning_episodes_files(self):
        return [self.get_automaton_learning_episodes_file(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_learning_episodes_file(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.AUTOMATON_LEARNING_EPISODES_FILENAME)

    def get_running_time_files(self):
        return [self.get_running_time_file(domain_id) for domain_id in range(self.num_domains)]

    def get_running_time_file(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.ABSOLUTE_RUNNING_TIME_FILENAME)

    def get_automata(self):
        return self.automata

    def _set_automata(self, automata):
        if len(automata) == self.num_domains:
            self.automata = automata
        else:
            if self.load_last_automaton_solution:
                self._load_last_automata_solutions()
            else:
                self._set_initial_automata()

    def _set_initial_automata(self):
        self.automata = []

        for _ in range(self.num_domains):
            # the initial automaton is an automaton that doesn't accept nor reject anything
            automaton = SubgoalAutomaton()
            automaton.add_state("s0")
            automaton.set_initial_state("s0")
            self.automata.append(automaton)

    def _load_last_automata_solutions(self):
        self.automata = []

        for i in range(self.num_domains):
            automaton_solution_folder = self.get_automaton_solution_folder(i)
            last_automaton_filename = self._get_last_solution_filename(automaton_solution_folder)
            automaton = ilasp_solution_parser.parse_ilasp_solutions(last_automaton_filename)
            automaton.set_initial_state("s0")
            automaton.set_accept_state(ISAAlgorithm.ACCEPTING_STATE_NAME)
            automaton.set_reject_state(ISAAlgorithm.REJECTING_STATE_NAME)
            self.automata.append(automaton)

    def _get_last_solution_filename(self, automaton_solution_folder):
        automaton_solutions = os.listdir(automaton_solution_folder)
        automaton_solutions.sort(key=lambda k: int(k[:-len(".txt")].split("-")[1]))
        automaton_solutions_path = [os.path.join(automaton_solution_folder, f) for f in automaton_solutions]

        if len(automaton_solutions_path) > 1 and utils.is_file_empty(automaton_solutions_path[-1]):
            return automaton_solutions_path[-2]

        return automaton_solutions_path[-1]

    def run(self, tasks):
        self.tasks = tasks
        self.automaton_counters = [0] * self.num_domains

        self._build_q_functions()
        self._reset_examples()

        self.num_automaton_states = self.num_domains * [self.num_starting_states]

        if self.use_experience_replay:
            self._build_experience_replay_buffers()

        if not self.train_model:
            self._import_models()

        start = timer()
        self._run_tasks()
        self._write_running_time_files(timer() - start)

        if self.is_tabular_case:
            self._export_policies()
        else:
            self._export_models()

    def _init_loggers(self):
        self._init_reward_steps_loggers()

        if self.interleaved_automaton_learning:
            self._init_automaton_learning_episodes_loggers()

    def _init_reward_steps_loggers(self):
        self.reward_steps_loggers = []

        for domain_id in range(self.num_domains):
            folder_name = os.path.join(self.export_folder_names[domain_id], ISAAlgorithm.REWARD_STEPS_FOLDER)
            utils.mkdir(folder_name)

            task_loggers = []
            for task_id in range(len(self.tasks[domain_id])):
                if self.train_model:
                    filename = ISAAlgorithm.REWARD_STEPS_FILENAME
                else:
                    filename = ISAAlgorithm.REWARD_STEPS_TEST_FILENAME

                filename = filename % task_id

                name = os.path.join(folder_name, filename)
                handler = logging.FileHandler(name)

                logger = logging.getLogger(name)
                logger.setLevel(logging.INFO)
                logger.addHandler(handler)

                task_loggers.append(logger)

            self.reward_steps_loggers.append(task_loggers)

    def _init_automaton_learning_episodes_loggers(self):
        self.automaton_learning_episodes_loggers = []

        for domain_id in range(self.num_domains):
            utils.mkdir(self.export_folder_names[domain_id])

            name = self.get_automaton_learning_episodes_file(domain_id)
            handler = logging.FileHandler(name)

            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            self.automaton_learning_episodes_loggers.append(logger)

    def _log_reward_and_steps(self, domain_id, task_id, episode_reward, episode_length):
        self.reward_steps_loggers[domain_id][task_id].info(str(episode_reward) + ";" + str(episode_length))

    def _register_automaton_learning_log(self, domain_id, episode):
        self.automaton_learning_episodes_loggers[domain_id].info(str(episode))

    def _write_running_time_files(self, running_time):
        for filename in self.get_running_time_files():
            with open(filename, 'w') as f:
                f.write(str(running_time))

    def _run_tasks(self):
        self._init_loggers()  # init reward + steps and learning episodes loggers

        self.target_net_update_counter = 0
        self.has_observed_pos_example = [False] * self.num_domains

        # initialize domain, task and episode counters (updated after every episode execution)
        current_domain_id = 0
        current_task_id = 0
        current_episode = 1

        while current_episode <= self.num_episodes:
            # run the episode
            completed_episode, total_reward, episode_length, ended_terminal, observation_history, compressed_history = \
                self._run_episode(current_domain_id, current_task_id)

            if completed_episode:  # change domain, task and episode counters
                history = compressed_history if self.use_compressed_traces else observation_history
                current_domain_id, current_task_id, current_episode = self._on_episode_done(current_domain_id,
                                                                                            current_task_id,
                                                                                            current_episode,
                                                                                            ended_terminal,
                                                                                            total_reward,
                                                                                            episode_length, history)
            else:
                # if the episode was interrupted, log the learning episode
                self._register_automaton_learning_log(current_domain_id, current_episode)

    def _on_episode_done(self, current_domain_id, current_task_id, current_episode, ended_terminal, total_reward,
                         episode_length, history):
        self._show_learning_msg(current_domain_id, current_task_id, current_episode, ended_terminal, total_reward,
                                episode_length, history)

        self._log_reward_and_steps(current_domain_id, current_task_id, total_reward, episode_length)

        next_domain_id, next_task_id, next_episode = current_domain_id, current_task_id, current_episode

        if self.num_domains > 1:
            next_domain_id = current_domain_id + 1
            if next_domain_id == self.num_domains:
                next_domain_id = 0
                next_task_id, next_episode = self._get_next_task_and_episode(current_domain_id, current_task_id,
                                                                             current_episode)
        else:
            next_domain_id = current_domain_id
            next_task_id, next_episode = self._get_next_task_and_episode(current_domain_id, current_task_id,
                                                                         current_episode)
        return next_domain_id, next_task_id, next_episode

    def _get_next_task_and_episode(self, current_domain_id, current_task_id, current_episode):
        next_task_id = (current_task_id + 1) % self._get_num_tasks(current_domain_id)
        next_episode = current_episode
        if next_task_id == 0:
            next_episode += 1
        return next_task_id, next_episode

    def _show_learning_msg(self, domain_id, task_id, episode, ended_terminal, total_reward, episode_length, history):
        if self.debug:
            print("Domain: " + str(domain_id) +
                  " - Task: " + str(task_id) +
                  " - Episode: " + str(episode) +
                  " - Terminal: " + str(ended_terminal) +
                  " - Reward: " + str(total_reward) +
                  " - Steps: " + str(episode_length) +
                  " - Observations: " + str(history))

    def _get_task(self, domain_id, task_id):
        return self.tasks[domain_id][task_id]

    def _get_num_tasks(self, domain_id):
        return len(self.tasks[domain_id])

    def _get_automaton(self, domain_id):
        return self.automata[domain_id]

    def _set_automaton(self, domain_id, automaton):
        self.automata[domain_id] = automaton

    def _get_q_function(self, domain_id, task_id):
        return self.q_functions[domain_id][task_id]

    def _get_target_q_function(self, domain_id, task_id):
        return self.target_q_functions[domain_id][task_id]

    def _get_optimizer(self, domain_id, task_id):
        return self.optimizers[domain_id][task_id]

    def _get_experience_replay_buffer(self, task_id):
        return self.experience_replay_buffers[task_id]

    def _get_actual_initial_automaton_state(self, task, domain_id, observation_history, compressed_observation_history,
                                            initial_observations):
        current_automaton_states = self._get_initial_automaton_state_successors(domain_id, initial_observations)
        if len(current_automaton_states) > 1:
            if self.interleaved_automaton_learning:
                self._perform_interleaved_automaton_learning(task, domain_id, None, observation_history,
                                                             compressed_observation_history)
                current_automaton_states = self._get_initial_automaton_state_successors(domain_id, initial_observations)
            else:
                # if we are not learning the automaton, notify that the provided automaton is non-deterministic
                raise MultipleConditionsHoldException
        return next(iter(current_automaton_states))

    def _get_initial_automaton_state_successors(self, domain_id, observations):
        automaton = self._get_automaton(domain_id)
        initial_state = automaton.get_initial_state()
        return automaton.get_next_states(initial_state, observations)

    def _get_task_observations(self, task):
        observations = task.get_observations()

        if self.use_restricted_observables:
            restricted_observables = task.get_restricted_observables()
            return observations.intersection(restricted_observables)

        return observations

    def _set_has_observed_pos_example(self, domain_id, task):
        if task.is_goal_achieved() and not self.has_observed_pos_example[domain_id]:
            self.has_observed_pos_example[domain_id] = True

    def _can_learn_new_automaton(self, domain_id, task):
        self._set_has_observed_pos_example(domain_id, task)
        return self.has_observed_pos_example[domain_id]

    def _run_episode(self, domain_id, task_id):
        task = self._get_task(domain_id, task_id)  # get the task to learn

        # initialize reward and steps counters and reset the task to its initial state
        total_reward, episode_length = 0, 0
        current_state = task.reset()

        # get initial observations and initialise histories
        initial_observations = self._get_task_observations(task)
        initial_observations_tuple = self._get_observations_as_ordered_tuple(initial_observations)
        observation_history, compressed_observation_history = [initial_observations_tuple], [initial_observations_tuple]

        # get actual initial automaton state (performs verification that there is only one possible initial state!)
        current_automaton_state = self._get_actual_initial_automaton_state(task_id, domain_id, observation_history,
                                                                           compressed_observation_history,
                                                                           initial_observations)

        # update the automaton if the goal is initial state achieves the goal and the example is not covered
        if self.interleaved_automaton_learning and self._can_learn_new_automaton(domain_id, task):
            updated_automaton = self._perform_interleaved_automaton_learning(task, domain_id,
                                                                             current_automaton_state,
                                                                             observation_history,
                                                                             compressed_observation_history)
            if updated_automaton:
                # get the actual initial state as done before (there shouldn't be problems with multiple initial
                # states this time)
                current_automaton_state = self._get_actual_initial_automaton_state(task_id, domain_id,
                                                                                   observation_history,
                                                                                   compressed_observation_history,
                                                                                   initial_observations)

        # get reward for current automaton state
        automaton = self._get_automaton(domain_id)
        total_reward = self._get_automaton_state_reward(automaton, current_automaton_state)

        # whether the episode execution must be stopped (an automaton is learnt in the middle)
        interrupt_episode = False

        while not task.is_terminal() and episode_length < self.max_episode_length and not interrupt_episode:
            # get q_function for the <domain, task> pair, choose an action and perform it
            q_function = self._get_q_function(domain_id, task_id)
            action = self._choose_egreedy_action(task, current_state, q_function[current_automaton_state])
            next_state, _, is_terminal, _ = task.step(action)
            observations = self._get_task_observations(task)

            observations_changed = self._update_histories(observation_history, compressed_observation_history,
                                                          observations)

            if self.train_model:
                self._update_q_functions(task_id, current_state, action, next_state, is_terminal, observations,
                                         observations_changed)

            next_automaton_states = self._get_next_automaton_states(automaton, current_automaton_state, observations,
                                                                    observations_changed)

            # check if there can be more than one next automaton state (i.e., non-deterministic)
            if len(next_automaton_states) > 1:
                if self.interleaved_automaton_learning:  # learn a new automaton which is deterministic w.r.t. new trace
                    interrupt_episode = self._perform_interleaved_automaton_learning(task, domain_id, None,
                                                                                     observation_history,
                                                                                     compressed_observation_history)
                else:
                    raise MultipleConditionsHoldException

            next_automaton_state = next(iter(next_automaton_states))

            # episode has to be interrupted if an automaton is learnt
            if not interrupt_episode and self.interleaved_automaton_learning and self._can_learn_new_automaton(domain_id, task):
                interrupt_episode = self._perform_interleaved_automaton_learning(task, domain_id, next_automaton_state,
                                                                                 observation_history,
                                                                                 compressed_observation_history)

            if not interrupt_episode:
                total_reward = self._get_automaton_transition_reward(automaton, current_automaton_state, next_automaton_state)

                if not self.is_tabular_case:  # update target DQN weights if the episode is not interrupted
                    self._update_target_deep_q_functions()

            # update current environment and automaton states and increase episode length
            current_state = next_state
            current_automaton_state = next_automaton_state
            episode_length += 1

        completed_episode = not interrupt_episode

        return completed_episode, total_reward, episode_length, task.is_terminal(), observation_history, compressed_observation_history

    def _get_automaton_state_reward(self, automaton, automaton_state):
        if automaton.is_accept_state(automaton_state):
            return 1
        elif self.reward_on_rejecting_state and automaton.is_reject_state(automaton_state):
            return -1
        return 0

    def _get_automaton_transition_reward(self, automaton, current_automaton_state, next_automaton_state):
        if automaton.is_accepting_transition(current_automaton_state, next_automaton_state):
            return 1
        elif self.reward_on_rejecting_state and automaton.is_rejecting_transition(current_automaton_state,
                                                                                  next_automaton_state):
            return -1
        return 0

    def _update_histories(self, observation_history, compressed_observation_history, observations):
        observations_tuple = self._get_observations_as_ordered_tuple(observations)
        observation_history.append(observations_tuple)

        observations_changed = False
        if observations_tuple != compressed_observation_history[-1]:
            observations_changed = True
            compressed_observation_history.append(observations_tuple)

        return observations_changed

    def _update_q_functions(self, task_id, current_state, action, next_state, is_terminal, observations, observations_changed):
        if self.use_experience_replay:
            experience = Experience(current_state, action, next_state, is_terminal, observations, observations_changed)
            experience_replay_buffer = self._get_experience_replay_buffer(task_id)
            experience_replay_buffer.append(experience)
            if len(experience_replay_buffer) >= self.experience_replay_start_size:
                experience_batch = experience_replay_buffer.sample(self.experience_replay_batch_size)

                if self.is_tabular_case:
                    for exp in experience_batch:
                        self._update_tabular_q_functions(task_id, (exp.state, exp.action), exp.next_state,
                                                         exp.observations, exp.observations_changed)
                else:
                    self._update_deep_q_functions(task_id, experience_batch)
        else:
            # update all q-tables of the current task (not other tasks because state spaces might be different!)
            current_pair = (current_state, action)
            self._update_tabular_q_functions(task_id, current_pair, next_state, observations, observations_changed)

    def _get_next_automaton_states(self, automaton, current_automaton_state, observations, observations_changed):
        # automaton has to be navigated with compressed traces if specified (just when a change occurs)
        if not self.use_compressed_traces or observations_changed:
            return automaton.get_next_states(current_automaton_state, observations)
        return {current_automaton_state}

    def _build_q_functions(self):
        self.q_functions = [{} for _ in range(self.num_domains)]
        if self.is_tabular_case:
            for domain_id in range(0, len(self.tasks)):
                self._build_domain_tabular_q_functions(domain_id)
        else:
            self.target_q_functions = [{} for _ in range(self.num_domains)]
            self.optimizers = [{} for _ in range(self.num_domains)]

            for domain_id in range(0, len(self.tasks)):
                self._build_domain_deep_q_functions(domain_id)

    def _build_domain_tabular_q_functions(self, domain_id):
        q_tables = self.q_functions[domain_id]
        q_tables.clear()

        current_tasks = self.tasks[domain_id]
        current_automaton = self._get_automaton(domain_id)

        for task_id in range(len(current_tasks)):
            q_tables[task_id] = {}
            task = current_tasks[task_id]

            num_states = task.observation_space.n
            num_actions = task.action_space.n
            state_action_list = [(s, a) for s in range(num_states) for a in range(num_actions)]

            for state in current_automaton.get_states():
                q_tables[task_id][state] = {}
                for sa_pair in state_action_list:
                    q_tables[task_id][state][sa_pair] = 0.0

    def _build_domain_deep_q_functions(self, domain_id):
        q_functions = self.q_functions[domain_id]
        target_q_functions = self.target_q_functions[domain_id]
        optimizers = self.optimizers[domain_id]

        q_functions.clear()
        target_q_functions.clear()
        optimizers.clear()

        current_tasks = self.tasks[domain_id]
        current_automaton = self._get_automaton(domain_id)

        for task_id in range(len(current_tasks)):
            q_functions[task_id] = {}
            target_q_functions[task_id] = {}
            optimizers[task_id] = {}

            task = current_tasks[task_id]

            num_states = task.observation_space.n
            num_actions = task.action_space.n

            for state in current_automaton.get_states():
                q_functions[task_id][state] = DQN(num_states, num_actions, self.num_layers, self.num_neurons_per_layer)
                q_functions[task_id][state].to(self.device)
                target_q_functions[task_id][state] = DQN(num_states, num_actions, self.num_layers, self.num_neurons_per_layer)
                target_q_functions[task_id][state].to(self.device)
                target_q_functions[task_id][state].load_state_dict(q_functions[task_id][state].state_dict())
                optimizers[task_id][state] = optim.Adam(q_functions[task_id][state].parameters(), lr=self.learning_rate)

    def _reset_q_functions(self, domain_id):
        if self.is_tabular_case:
            self._build_domain_tabular_q_functions(domain_id)
        else:
            self.target_net_update_counter = 0
            self._build_domain_deep_q_functions(domain_id)

    def _reset_all_q_functions(self):
        for domain_id in range(self.num_domains):
            self._reset_q_functions(domain_id)

    def _build_experience_replay_buffers(self):
        num_tasks = len(self.tasks[0])  # all domains should be assigned the same number of tasks
        self.experience_replay_buffers = [ExperienceBuffer(self.experience_replay_buffer_size) for _ in range(num_tasks)]

    def _get_pseudoreward(self, automaton, from_automaton_state, to_automaton_state):
        """Returns a pseudoreward for doing reward shaping based on the transition between two states in a given
        automaton. The provided pseudoreward is given according to the formulation from the paper 'Policy invariance
        under reward transformations: Theory and application to reward shaping'."""
        if not automaton.has_accept_state():
            return 0.0

        phi_from = self._get_phi_pseudoreward(automaton, from_automaton_state)
        phi_to = self._get_phi_pseudoreward(automaton, to_automaton_state)

        return self.discount_rate * phi_to - phi_from

    def _get_phi_pseudoreward(self, automaton, automaton_state):
        if automaton.has_accept_state():
            phi = automaton.get_num_states() - automaton.get_distance_to_accept_state(automaton_state)
        else:
            phi = float("-inf")
        if phi == float("-inf"):
            phi = -100.0
        return phi

    def _update_tabular_q_functions(self, task_id, current_pair, next_state, observations, observations_changed):
        for domain_id in range(self.num_domains):
            task = self._get_task(domain_id, task_id)
            automaton = self._get_automaton(domain_id)
            q_table = self._get_q_function(domain_id, task_id)

            for automaton_state in automaton.get_states():
                # although the automaton is non-deterministic we can still perform the update of its states
                for next_automaton_state in self._get_next_automaton_states(automaton, automaton_state, observations, observations_changed):
                    next_action = self._get_greedy_action(task, next_state, q_table[next_automaton_state])
                    next_pair = (next_state, next_action)

                    reward = self._get_automaton_transition_reward(automaton, automaton_state, next_automaton_state)
                    if self.use_reward_shaping:
                        reward += self._get_pseudoreward(automaton, automaton_state, next_automaton_state)

                    if automaton.is_terminal_state(next_automaton_state):
                        q_table[automaton_state][current_pair] += self.learning_rate * (reward - q_table[automaton_state][current_pair])
                    else:
                        q_table[automaton_state][current_pair] += self.learning_rate * (
                                    reward + self.discount_rate * q_table[next_automaton_state][next_pair] -
                                    q_table[automaton_state][current_pair])

    def _update_deep_q_functions(self, task_id, experience_batch):
        states, actions, next_states, is_terminal, observations, observations_changed = zip(*experience_batch)

        states_v = torch.tensor(states).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.long).to(self.device)  # torch.tensor() causes memory leak for integer/long
        next_states_v = torch.tensor(next_states).to(self.device)

        for domain_id in range(self.num_domains):
            automaton = self._get_automaton(domain_id)
            q_functions = self._get_q_function(domain_id, task_id)
            target_q_functions = self._get_target_q_function(domain_id, task_id)
            optimizers = self._get_optimizer(domain_id, task_id)

            for automaton_state in automaton.get_states():
                if not automaton.is_terminal_state(automaton_state):
                    self._update_deep_q_function(q_functions, target_q_functions, optimizers, automaton,
                                                 automaton_state, states_v, actions_v, next_states_v, observations,
                                                 observations_changed)

    def _update_deep_q_function(self, q_functions, target_q_functions, optimizers, automaton, automaton_state, states_v,
                                actions_v, next_states_v, observations, observations_changed):
        loss = self._compute_deep_q_function_loss(q_functions, target_q_functions, automaton, automaton_state, states_v,
                                                  actions_v, next_states_v, observations, observations_changed)
        optimizer = optimizers[automaton_state]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _compute_deep_q_function_loss(self, q_functions, target_q_functions, automaton, automaton_state, states_v,
                                      actions_v, next_states_v, observations, observations_changed):
        # get next automaton states for each item in the batch based on current aut. state and the observations
        next_automaton_states = np.array(self._get_next_automaton_states_batch(automaton, automaton_state, observations,
                                                                               observations_changed))

        # get the rewards based on the transitions taken in the automaton for each item in the batch
        rewards = self._get_transition_rewards_batch(automaton, automaton_state, next_automaton_states)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # perform forward pass for the current states in the current automaton state DQN
        net = q_functions[automaton_state]
        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        # get next state action values
        next_state_action_values = self._get_next_state_action_values(q_functions, target_q_functions, automaton,
                                                                      next_states_v, observations,
                                                                      next_automaton_states)

        expected_state_action_values = rewards_v + self.discount_rate * next_state_action_values
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def _get_next_automaton_states_batch(self, automaton, automaton_state, observations, observations_changed):
        return [next(iter(self._get_next_automaton_states(automaton, automaton_state, obs, observations_changed)))
                for obs in observations]

    def _get_transition_rewards_batch(self, automaton, automaton_state, next_automaton_states):
        transition_rewards_batch = []
        for next_automaton_state in next_automaton_states:
            reward = self._get_automaton_transition_reward(automaton, automaton_state, next_automaton_state)
            if self.use_reward_shaping:
                reward += self._get_pseudoreward(automaton, automaton_state, next_automaton_state)
            transition_rewards_batch.append(reward)
        return transition_rewards_batch

    def _get_next_state_action_values(self, q_functions, target_q_functions, automaton, next_states_v, observations,
                                      next_automaton_states):
        # initialize the next state-action values to zero with the size of the batch
        next_state_action_values = torch.zeros(len(observations), dtype=torch.float32, device=self.device)

        # get the set of possible next automaton states (to remove duplicates)
        next_automaton_states_set = set(next_automaton_states)

        # for each possible next automaton state, perform a forward pass in the corresponding q-function for the whole
        # batch of next states (just if the next state is non-terminal)
        for next_automaton_state in next_automaton_states_set:
            # if the automaton state is terminal we keep the value of 0.0 (no forward pass applied)
            if not automaton.is_terminal_state(next_automaton_state):
                # get the indices of the batch corresponding to the next automaton state
                indices = np.where(next_automaton_states == next_automaton_state)[0]

                # get the q-function of the next automaton state, apply the forward pass to the entire batch, select the
                # value of the maximising action and select only those indices we took before (the ones corresponding to
                # the next automaton state)
                target_net = target_q_functions[next_automaton_state]

                if self.double_dqn:
                    net = q_functions[next_automaton_state]
                    next_state_actions = net(next_states_v).max(1)[1]
                    next_state_action_values[indices] = target_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)[indices]
                else:
                    next_state_action_values[indices] = target_net(next_states_v).max(1)[0][indices]

        next_state_action_values = next_state_action_values.detach()
        return next_state_action_values

    def _update_target_deep_q_functions(self):
        self.target_net_update_counter += 1
        if self.target_net_update_counter % self.target_net_update_frequency == 0:
            for domain_id in range(self.num_domains):
                automaton = self._get_automaton(domain_id)
                for task_id in range(len(self.tasks[domain_id])):
                    for automaton_state in automaton.get_states():
                        if not automaton.is_terminal_state(automaton_state):  # no need to update terminal states policy
                            net = self.q_functions[domain_id][task_id][automaton_state]
                            target_net = self.target_q_functions[domain_id][task_id][automaton_state]
                            target_net.load_state_dict(net.state_dict())
            self.target_net_update_counter = 0

    def _export_policies(self):
        for domain_id in range(self.num_domains):
            for task_id in range(len(self.tasks[domain_id])):
                self._export_policy(domain_id, task_id)

    def _export_policy(self, domain_id, task_id):
        domain_folder = self.get_learnt_policies_folder(domain_id)
        utils.mkdir(domain_folder)

        with open(os.path.join(domain_folder, ISAAlgorithm.LEARNT_POLICY_FILENAME % task_id), 'w') as f:
            task = self.tasks[domain_id][task_id]
            automaton = self._get_automaton(domain_id)
            q_table = self._get_q_function(domain_id, task_id)

            current_state = task.reset()
            current_automaton_state = automaton.get_initial_state()

            is_terminal = False
            episode_length = 0

            while not is_terminal and episode_length < self.max_episode_length:
                action = self._get_greedy_action(task, current_state, q_table[current_automaton_state])
                current_state, _, is_terminal, observations = task.step(action)
                next_states = automaton.get_next_states(current_automaton_state, observations)
                if len(next_states) > 1:
                    f.write("STOP: MULTIPLE AUTOMATA STATES ARE AVAILABLE NEXT - THE FINAL AUTOMATON SHOULD BE DETERMINISTIC!\n")
                current_automaton_state = next(iter(next_states))
                f.write("State: " + str(current_state) + " -  Action: " + str(action) + "\n")
                episode_length += 1

            if episode_length >= self.max_episode_length:
                f.write("STOP: MAXIMUM EPISODE LENGTH SURPASSED!\n")

    def _get_model_path(self, domain_folder, task_id, automaton_state):
        model_name = ISAAlgorithm.MODEL_FILENAME % (task_id, automaton_state)
        return os.path.join(domain_folder, model_name)

    def _export_models(self):
        for domain_id in range(self.num_domains):
            automaton = self._get_automaton(domain_id)
            for task_id in range(len(self.tasks[domain_id])):
                for automaton_state in automaton.get_states():
                    self._export_model(domain_id, task_id, automaton_state)

    def _export_model(self, domain_id, task_id, automaton_state):
        domain_folder = self.get_models_folder(domain_id)
        utils.mkdir(domain_folder)

        model_path = self._get_model_path(domain_folder, task_id, automaton_state)

        q_functions = self._get_q_function(domain_id, task_id)
        model = q_functions[automaton_state]
        torch.save(model.state_dict(), model_path)

    def _import_models(self):
        for domain_id in range(self.num_domains):
            automaton = self._get_automaton(domain_id)
            for task_id in range(len(self.tasks[domain_id])):
                for automaton_state in automaton.get_states():
                    self._import_model(domain_id, task_id, automaton_state)

    def _import_model(self, domain_id, task_id, automaton_state):
        domain_folder = self.get_models_folder(domain_id)

        model_path = self._get_model_path(domain_folder, task_id, automaton_state)

        q_functions = self._get_q_function(domain_id, task_id)
        model = q_functions[automaton_state]
        model.load_state_dict(torch.load(model_path))
        model.eval()

    def _perform_interleaved_automaton_learning(self, task, domain_id, current_automaton_state, observation_history,
                                                compressed_observation_history):
        """Updates the set of examples based on the current observed trace. In case the set of example is updated, it
        makes a call to the automata learner. Returns True if a new automaton has been learnt, False otherwise."""
        updated_examples = self._update_examples(task, domain_id, current_automaton_state, observation_history,
                                                 compressed_observation_history)
        if updated_examples:
            if self.debug:
                if self.use_compressed_traces:
                    counterexample = str(compressed_observation_history)
                else:
                    counterexample = str(observation_history)
                print("Updating automaton " + str(domain_id) + "... The counterexample is: " + counterexample)
            self._update_automaton(task, domain_id)
            return True  # whether a new automaton has been learnt

        return False

    def _update_examples(self, task, domain_id, current_automaton_state, observation_history, compressed_observation_history):
        """Updates the set of examples. Returns True if the set of examples has been updated and False otherwise. Note
        that an update of the set of examples can be forced by setting 'current_automaton_state' to None."""
        automaton = self._get_automaton(domain_id)

        if task.is_terminal():
            if task.is_goal_achieved():
                if current_automaton_state is None or not automaton.is_accept_state(current_automaton_state):
                    self._update_example_set(self.pos_examples[domain_id], observation_history, compressed_observation_history)
                    return True
            elif current_automaton_state is None or not automaton.is_reject_state(current_automaton_state):
                self._update_example_set(self.neg_examples[domain_id], observation_history, compressed_observation_history)
                return True
        elif current_automaton_state is None or automaton.is_terminal_state(current_automaton_state):
            self._update_example_set(self.inc_examples[domain_id], observation_history, compressed_observation_history)
            return True
        return False  # whether example sets have been updated

    def _update_example_set(self, example_set, observation_history, compressed_observation_history):
        """Updates the a given example set with the corresponding history of observations depending on whether
        compressed traces are used or not to learn the automata. An exception is thrown if a trace is readded."""
        if self.use_compressed_traces:
            history_tuple = tuple(compressed_observation_history)
        else:
            history_tuple = tuple(observation_history)

        if history_tuple not in example_set:
            example_set.add(history_tuple)
        else:
            raise RuntimeError("An example that an automaton is currently covered cannot be uncovered afterwards!")

    def _reset_examples(self):
        # there is a set of examples for each domain
        self.pos_examples = [set() for _ in range(self.num_domains)]
        self.neg_examples = [set() for _ in range(self.num_domains)]
        self.inc_examples = [set() for _ in range(self.num_domains)]

    def _generate_ilasp_task(self, task, domain_id):
        utils.mkdir(self.get_automaton_task_folder(domain_id))

        ilasp_task_filename = os.path.join(self.get_automaton_task_folder(domain_id),
                                           ISAAlgorithm.AUTOMATON_TASK_FILENAME % self.automaton_counters[domain_id])

        observables = task.get_observables()
        if self.use_restricted_observables:
            observables = task.get_restricted_observables()

        ilasp_task_generator.generate_ilasp_task(self.num_automaton_states[domain_id],
                                                 ISAAlgorithm.ACCEPTING_STATE_NAME, ISAAlgorithm.REJECTING_STATE_NAME,
                                                 observables, self.pos_examples[domain_id],
                                                 self.neg_examples[domain_id], self.inc_examples[domain_id],
                                                 ilasp_task_filename, self.symmetry_breaking_method,
                                                 self.max_disjunction_size, self.learn_acyclic_graph)

    def _solve_ilasp_task(self, domain_id):
        utils.mkdir(self.get_automaton_solution_folder(domain_id))

        ilasp_task_filename = os.path.join(self.get_automaton_task_folder(domain_id),
                                           ISAAlgorithm.AUTOMATON_TASK_FILENAME % self.automaton_counters[domain_id])

        ilasp_solution_filename = os.path.join(self.get_automaton_solution_folder(domain_id),
                                               ISAAlgorithm.AUTOMATON_SOLUTION_FILENAME % self.automaton_counters[domain_id])

        return solve_ilasp_task(ilasp_task_filename, ilasp_solution_filename, timeout=self.ilasp_timeout,
                                version=self.ilasp_version, binary_folder_name=self.binary_folder_name)

    def _update_automaton(self, task, domain_id):
        self.automaton_counters[domain_id] += 1  # increment the counter of the number of aut. learnt for a domain

        self._generate_ilasp_task(task, domain_id)  # generate the automata learning task

        solver_success = self._solve_ilasp_task(domain_id)  # run the task solver
        if solver_success:
            ilasp_solution_filename = os.path.join(self.get_automaton_solution_folder(domain_id),
                                                   ISAAlgorithm.AUTOMATON_SOLUTION_FILENAME % self.automaton_counters[domain_id])
            candidate_automaton = ilasp_solution_parser.parse_ilasp_solutions(ilasp_solution_filename)

            if candidate_automaton.get_num_states() > 0:
                # set initial, accepting and rejecting states in the automaton
                candidate_automaton.set_initial_state("s0")
                candidate_automaton.set_accept_state(ISAAlgorithm.ACCEPTING_STATE_NAME)
                candidate_automaton.set_reject_state(ISAAlgorithm.REJECTING_STATE_NAME)
                self._set_automaton(domain_id, candidate_automaton)

                # the previous q-functions are not valid anymore since the automata structure has changed, so we reset
                # them
                self._reset_q_functions(domain_id)

                # plot the new automaton
                candidate_automaton.plot(self.get_automaton_plot_folder(domain_id),
                                         ISAAlgorithm.AUTOMATON_PLOT_FILENAME % self.automaton_counters[domain_id])
            else:
                # if the task is UNSATISFIABLE, it means the number of states is not enough to cover the examples, so
                # the number of states is incremented by 1 and try again
                self.num_automaton_states[domain_id] += 1

                if self.debug:
                    print("The number of states in the automaton has been increased to " + str(self.num_automaton_states[domain_id]))
                    print("Updating automaton...")

                self._update_automaton(task, domain_id)
        else:
            raise RuntimeError("Error: Couldn't find an automaton under the specified timeout!")
