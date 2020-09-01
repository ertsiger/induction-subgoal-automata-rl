import numpy as np
import os
from reinforcement_learning.learning_algorithm import LearningAlgorithm
from utils import utils


class TabularQLearning(LearningAlgorithm):
    TABULAR_MODEL_FILENAME = "model-%d.npy"  # %d = task id

    def __init__(self, tasks, num_tasks, export_folder_names, params):
        super().__init__(tasks, num_tasks, export_folder_names, params)

        self.q_functions = []
        self._build_q_functions()

    def _run_episode(self, domain_id, task_id):
        task = self._get_task(domain_id, task_id)

        # initialize reward and steps counters, histories and reset the task to its initial state
        total_reward, episode_length = 0, 0
        observation_history, compressed_observation_history = [], []
        current_state = task.reset()

        # get initial observations and initialise histories
        initial_observations = task.get_observations()
        self._update_histories(observation_history, compressed_observation_history, initial_observations)

        while not task.is_terminal() and episode_length < self.max_episode_length:
            action = self._choose_egreedy_action(task, current_state, self.q_functions[domain_id][task_id])
            next_state, reward, is_terminal, observations = task.step(action)

            self._update_histories(observation_history, compressed_observation_history, observations)

            if self.train_model:
                self._update_q_function(domain_id, task_id, current_state, action, next_state, reward, is_terminal)

            current_state = next_state
            total_reward += reward
            episode_length += 1

        return True, total_reward, episode_length, task.is_terminal(), observation_history, compressed_observation_history

    def _on_incomplete_episode(self, current_domain_id):
        pass

    def _build_q_functions(self):
        self.q_functions = [{} for _ in range(self.num_domains)]
        for domain_id in range(self.num_domains):
            for task_id in range(self.num_tasks):
                task = self._get_task(domain_id, task_id)
                self.q_functions[domain_id][task_id] = np.zeros((task.observation_space.n, task.action_space.n),
                                                                dtype=np.float32)

    def _update_q_function(self, domain_id, task_id, current_state, action, next_state, reward, is_terminal):
        q_function = self.q_functions[domain_id][task_id]

        next_state_value = 0.0
        if not is_terminal:
            task = self._get_task(domain_id, task_id)
            next_action = self._get_greedy_action(task, next_state, q_function)
            next_pair = (next_state, next_action)
            next_state_value = self.discount_rate * q_function[next_pair]

        current_pair = (current_state, action)
        q_function[current_pair] += self.learning_rate * (reward + next_state_value - q_function[current_pair])

    def _get_tabular_model_path(self, domain_folder, task_id):
        model_name = TabularQLearning.TABULAR_MODEL_FILENAME % task_id
        return os.path.join(domain_folder, model_name)

    def _export_models(self):
        for domain_id in range(self.num_domains):
            for task_id in range(self.num_tasks):
                self._export_model(domain_id, task_id)

    def _export_model(self, domain_id, task_id):
        domain_folder = self.get_models_folder(domain_id)
        utils.mkdir(domain_folder)
        model_path = self._get_tabular_model_path(domain_folder, task_id)
        np.save(model_path, self.q_functions[domain_id][task_id])

    def _import_models(self):
        for domain_id in range(self.num_domains):
            for task_id in range(self.num_tasks):
                self._import_model(domain_id, task_id)

    def _import_model(self, domain_id, task_id):
        domain_folder = self.get_models_folder(domain_id)
        model_path = self._get_tabular_model_path(domain_folder, task_id)
        self.q_functions[domain_id][task_id] = np.load(model_path)
