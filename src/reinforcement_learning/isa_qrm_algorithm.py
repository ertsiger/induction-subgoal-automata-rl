import collections
import os
import numpy as np
import torch
from torch import nn
from torch import optim

from reinforcement_learning.isa_base_algorithm import ISAAlgorithmBase
from utils import utils

from reinforcement_learning.dqn_model import DQN
from reinforcement_learning.experience_replay import ExperienceBuffer

# Experience class used to instantiate experiences added to the experience replay buffer when DQNs are used to
# approximate the q-functions.
Experience = collections.namedtuple("Experience", field_names=["state", "action", "next_state", "is_terminal",
                                                               "observations", "observations_changed"])


class ISAAlgorithmQRM(ISAAlgorithmBase):
    """
    Implementation of the method that interleaves RL and automata learning using QRM (Q-learning for Reward Machines).
    The implementation assumes that a reward of 1 is given when the agent transitions to the accepting state and 0
    otherwise. There is a Q-function for each automaton state: its associated policy attempts to satisfy any of the (best)
    outgoing conditions from that state.
    """
    USE_REWARD_SHAPING = "use_reward_shaping"                             # whether reward shaping is used
    REWARD_SHAPING_ONLY_WHEN_ACCEPTING = "reward_shaping_only_accepting"  # apply reward shaping only when there is an accepting state
    REWARD_SHAPING_METHOD = "reward_shaping_method"                       # min_distance or max_distance to accepting state
    REWARD_SHAPING_UNREACHABLE_ACCEPTING_DISTANCE = 1000000.0

    REWARD_ON_REJECTING_STATE = "reward_rejecting_state"  # whether to give a reward of -1 in the rejecting state

    TABULAR_MODEL_FILENAME = "model-%d.npy"  # %d = task id
    DQN_MODEL_FILENAME = "model-%d-%s.pt"    # %d = task id, %s = automaton state

    def __init__(self, tasks, num_tasks, export_folder_names, params, target_automata, binary_folder_name):
        super().__init__(tasks, num_tasks, export_folder_names, params, target_automata, binary_folder_name)

        # reward shaping
        self.use_reward_shaping = utils.get_param(params, ISAAlgorithmQRM.USE_REWARD_SHAPING, False)
        self.use_reward_shaping_only_when_accepting = utils.get_param(params, ISAAlgorithmQRM.REWARD_SHAPING_ONLY_WHEN_ACCEPTING, True)
        self.reward_shaping_method = utils.get_param(params, ISAAlgorithmQRM.REWARD_SHAPING_METHOD, "min_distance")

        # attributes of reward function encoded by the automaton
        self.reward_on_rejecting_state = utils.get_param(params, ISAAlgorithmQRM.REWARD_ON_REJECTING_STATE, False)

        # experience replay
        self.experience_replay_buffers = None
        if self.use_experience_replay:
            self._build_experience_replay_buffers()

        # q-tables in the case of tabular QRM and dqn in the case of DQRM
        self.q_functions = []
        self.target_q_functions = []
        self.optimizers = []
        self.target_net_update_counter = {}
        self._build_q_functions()

    '''
    Building of Q-functions <automaton state, MDP state, action> (one for each <domain, task>)
    '''
    def _build_q_functions(self):
        self.q_functions = [{} for _ in range(self.num_domains)]
        if self.is_tabular_case:
            for domain_id in range(self.num_domains):
                self._build_domain_tabular_q_functions(domain_id)
        else:
            self._init_target_net_update_counter()
            self.target_q_functions = [{} for _ in range(self.num_domains)]
            self.optimizers = [{} for _ in range(self.num_domains)]

            for domain_id in range(self.num_domains):
                self._build_domain_deep_q_functions(domain_id)

    def _build_domain_tabular_q_functions(self, domain_id):
        num_automaton_states = self._get_automaton(domain_id).get_num_states()

        for task_id in range(self.num_tasks):
            task = self._get_task(domain_id, task_id)
            self.q_functions[domain_id][task_id] = np.zeros((num_automaton_states, task.observation_space.n,
                                                             task.action_space.n), dtype=np.float32)

    def _build_domain_deep_q_functions(self, domain_id):
        q_functions = self.q_functions[domain_id]
        target_q_functions = self.target_q_functions[domain_id]
        optimizers = self.optimizers[domain_id]
        target_net_update_counters = self.target_net_update_counter[domain_id]

        q_functions.clear()
        target_q_functions.clear()
        optimizers.clear()
        target_net_update_counters.clear()

        current_tasks = self.tasks[domain_id]
        current_automaton = self._get_automaton(domain_id)

        for task_id in range(len(current_tasks)):
            q_functions[task_id] = {}
            target_q_functions[task_id] = {}
            optimizers[task_id] = {}
            target_net_update_counters[task_id] = {}

            task = current_tasks[task_id]

            num_states = task.observation_space.n
            num_actions = task.action_space.n

            for state_id in range(current_automaton.get_num_states()):
                q_functions[task_id][state_id] = DQN(num_states, num_actions, self.num_layers, self.num_neurons_per_layer)
                q_functions[task_id][state_id].to(self.device)
                target_q_functions[task_id][state_id] = DQN(num_states, num_actions, self.num_layers, self.num_neurons_per_layer)
                target_q_functions[task_id][state_id].to(self.device)
                target_q_functions[task_id][state_id].load_state_dict(q_functions[task_id][state_id].state_dict())
                optimizers[task_id][state_id] = optim.Adam(q_functions[task_id][state_id].parameters(), lr=self.learning_rate)
                target_net_update_counters[task_id][state_id] = 0

    def _build_experience_replay_buffers(self):
        self.experience_replay_buffers = [ExperienceBuffer(self.experience_replay_buffer_size) for _ in range(self.num_tasks)]

    def _init_target_net_update_counter(self):
        self.target_net_update_counter = [{} for _ in range(self.num_domains)]

    '''
    Action choice during the episode
    '''
    def _choose_action(self, domain_id, task_id, current_state, automaton, current_automaton_state):
        q_function = self._get_q_function(domain_id, task_id)
        task = self._get_task(domain_id, task_id)
        return self._choose_egreedy_action(task, current_state, q_function[current_automaton_state])

    '''
    Update of Q functions
    '''
    def _get_automaton_transition_reward(self, automaton, current_automaton_state, next_automaton_state):
        if automaton.is_accepting_transition(current_automaton_state, next_automaton_state):
            return 1
        elif self.reward_on_rejecting_state and automaton.is_rejecting_transition(current_automaton_state, next_automaton_state):
            return -1
        return 0

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
                                                         exp.is_terminal, exp.observations, exp.observations_changed)
                else:
                    self._update_deep_q_functions(task_id, experience_batch)
        else:
            # update all q-tables of the current task (not other tasks because state spaces might be different!)
            current_pair = (current_state, action)
            self._update_tabular_q_functions(task_id, current_pair, next_state, is_terminal, observations, observations_changed)

    def _update_tabular_q_functions(self, task_id, current_pair, next_state, is_terminal, observations, observations_changed):
        for domain_id in range(self.num_domains):
            task = self._get_task(domain_id, task_id)
            automaton = self._get_automaton(domain_id)
            q_table = self._get_q_function(domain_id, task_id)

            for automaton_state in automaton.get_states():
                automaton_state_id = automaton.get_state_id(automaton_state)

                next_automaton_state = self._get_next_automaton_state(automaton, automaton_state, observations, observations_changed)
                next_automaton_state_id = automaton.get_state_id(next_automaton_state)

                next_action = self._get_greedy_action(task, next_state, q_table[next_automaton_state_id])
                next_pair = (next_state, next_action)

                reward = self._get_automaton_transition_reward(automaton, automaton_state, next_automaton_state)
                if self.use_reward_shaping:
                    reward += self._get_pseudoreward(automaton, automaton_state, next_automaton_state)

                if automaton.is_terminal_state(next_automaton_state):
                    q_table[automaton_state_id][current_pair] += self.learning_rate * (reward - q_table[automaton_state_id][current_pair])
                else:
                    q_table[automaton_state_id][current_pair] += self.learning_rate * (
                                reward + self.discount_rate * q_table[next_automaton_state_id][next_pair] -
                                q_table[automaton_state_id][current_pair])

    def _update_deep_q_functions(self, task_id, experience_batch):
        states, actions, next_states, is_terminal, observations, observations_changed = zip(*experience_batch)

        states_v = torch.tensor(states).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.long).to(self.device)
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
        optimizer = optimizers[automaton.get_state_id(automaton_state)]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _compute_deep_q_function_loss(self, q_functions, target_q_functions, automaton, automaton_state, states_v,
                                      actions_v, next_states_v, observations, observations_changed):
        # get next automaton states for each item in the batch based on current aut. state and the observations
        next_automaton_states = np.array(self._get_next_automaton_state_batch(automaton, automaton_state, observations,
                                                                              observations_changed))

        # get the rewards based on the transitions taken in the automaton for each item in the batch
        rewards = self._get_transition_rewards_batch(automaton, automaton_state, next_automaton_states)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # perform forward pass for the current states in the current automaton state DQN
        net = q_functions[automaton.get_state_id(automaton_state)]
        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        # get next state action values
        next_state_action_values = self._get_next_state_action_values(q_functions, target_q_functions, automaton,
                                                                      next_states_v, observations,
                                                                      next_automaton_states)

        expected_state_action_values = rewards_v + self.discount_rate * next_state_action_values
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def _get_next_automaton_state_batch(self, automaton, automaton_state, observations, observations_changed):
        return [self._get_next_automaton_state(automaton, automaton_state, obs, observations_changed)
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
                next_automaton_state_id = automaton.get_state_id(next_automaton_state)
                target_net = target_q_functions[next_automaton_state_id]

                if self.use_double_dqn:
                    net = q_functions[next_automaton_state_id]
                    next_state_actions = net(next_states_v).max(1)[1]
                    next_state_action_values[indices] = target_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)[indices]
                else:
                    next_state_action_values[indices] = target_net(next_states_v).max(1)[0][indices]

        next_state_action_values = next_state_action_values.detach()
        return next_state_action_values

    '''
    What to do when a step in the environment has been completed
    '''
    def _on_performed_step(self, domain_id, task_id, next_state, reward, is_terminal, observations, automaton, current_automaton_state, next_automaton_state, episode_length):
        if not self.is_tabular_case:  # update target DQN weights if the episode is not interrupted
            current_automaton_state_id = automaton.get_state_id(current_automaton_state)
            self._update_target_deep_q_functions(domain_id, task_id, current_automaton_state_id)

    def _update_target_deep_q_functions(self, domain_id, task_id, automaton_state_id):
        self.target_net_update_counter[domain_id][task_id][automaton_state_id] += 1
        if self.target_net_update_counter[domain_id][task_id][automaton_state_id] % self.target_net_update_frequency == 0:
            net = self.q_functions[domain_id][task_id][automaton_state_id]
            target_net = self.target_q_functions[domain_id][task_id][automaton_state_id]
            target_net.load_state_dict(net.state_dict())
            self.target_net_update_counter[domain_id][task_id][automaton_state_id] = 0

    '''
    Reward Shaping Methods
    '''
    def _get_pseudoreward(self, automaton, from_automaton_state, to_automaton_state):
        """Returns a pseudoreward for doing reward shaping based on the transition between two states in a given
        automaton. The provided pseudoreward is given according to the formulation from the paper 'Policy invariance
        under reward transformations: Theory and application to reward shaping'."""
        if (self.use_reward_shaping_only_when_accepting and not automaton.has_accept_state()) or \
            automaton.is_accept_state(from_automaton_state) or \
            automaton.is_reject_state(from_automaton_state):
            return 0.0

        phi_from = self._get_phi_pseudoreward(automaton, from_automaton_state)
        phi_to = self._get_phi_pseudoreward(automaton, to_automaton_state)

        return self.discount_rate * phi_to - phi_from

    def _get_phi_pseudoreward(self, automaton, automaton_state):
        if automaton.has_accept_state():
            phi = automaton.get_num_states() - automaton.get_distance_to_accept_state(automaton_state,
                                                                                      method=self.reward_shaping_method)
            # case of -inf (unreachable accepting) is passed to a large number
            phi = max(phi, -ISAAlgorithmQRM.REWARD_SHAPING_UNREACHABLE_ACCEPTING_DISTANCE)
        else:
            phi = -ISAAlgorithmQRM.REWARD_SHAPING_UNREACHABLE_ACCEPTING_DISTANCE  # unreachable, so phi would return this
        return phi

    '''
    What to do when a new automaton is learned
    '''
    def _on_automaton_learned(self, domain_id):
        # the previous q-functions are not valid anymore since the automata structure has changed, so we reset them
        self._reset_q_functions(domain_id)

    def _reset_q_functions(self, domain_id):
        if self.is_tabular_case:
            self._build_domain_tabular_q_functions(domain_id)
        else:
            self._build_domain_deep_q_functions(domain_id)

    def _reset_all_q_functions(self):
        for domain_id in range(self.num_domains):
            self._reset_q_functions(domain_id)

    '''
    Model exporting and importing
    '''
    def _get_tabular_model_path(self, domain_folder, task_id):
        model_name = ISAAlgorithmQRM.TABULAR_MODEL_FILENAME % task_id
        return os.path.join(domain_folder, model_name)

    def _get_dqn_model_path(self, domain_folder, task_id, automaton_state):
        model_name = ISAAlgorithmQRM.DQN_MODEL_FILENAME % (task_id, automaton_state)
        return os.path.join(domain_folder, model_name)

    def _export_models(self):
        for domain_id in range(self.num_domains):
            automaton = self._get_automaton(domain_id)
            for task_id in range(self.num_tasks):
                if self.is_tabular_case:
                    self._export_tabular_model(domain_id, task_id)
                else:
                    for automaton_state in automaton.get_states():
                        self._export_dqn_model(domain_id, task_id, automaton, automaton_state)

    def _export_tabular_model(self, domain_id, task_id):
        domain_folder = self.get_models_folder(domain_id)
        utils.mkdir(domain_folder)

        model_path = self._get_tabular_model_path(domain_folder, task_id)
        q_table = self._get_q_function(domain_id, task_id)
        np.save(model_path, q_table)

    def _export_dqn_model(self, domain_id, task_id, automaton, automaton_state):
        domain_folder = self.get_models_folder(domain_id)
        utils.mkdir(domain_folder)

        model_path = self._get_dqn_model_path(domain_folder, task_id, automaton_state)
        model = self._get_q_function(domain_id, task_id)[automaton.get_state_id(automaton_state)]
        torch.save(model.state_dict(), model_path)

    def _import_models(self):
        for domain_id in range(self.num_domains):
            automaton = self._get_automaton(domain_id)
            for task_id in range(self.num_tasks):
                if self.is_tabular_case:
                    self._import_tabular_model(domain_id, task_id)
                else:
                    for automaton_state in automaton.get_states():
                        self._import_dqn_model(domain_id, task_id, automaton, automaton_state)

    def _import_tabular_model(self, domain_id, task_id):
        domain_folder = self.get_models_folder(domain_id)
        model_path = self._get_tabular_model_path(domain_folder, task_id)
        self._set_q_function(domain_id, task_id, np.load(model_path))

    def _import_dqn_model(self, domain_id, task_id, automaton, automaton_state):
        domain_folder = self.get_models_folder(domain_id)
        model_path = self._get_dqn_model_path(domain_folder, task_id, automaton_state)

        q_functions = self._get_q_function(domain_id, task_id)
        model = q_functions[automaton.get_state_id(automaton_state)]
        model.load_state_dict(torch.load(model_path))
        model.eval()

    '''
    Getters and setters
    '''
    def _set_q_function(self, domain_id, task_id, q_function):
        self.q_functions[domain_id][task_id] = q_function

    def _get_q_function(self, domain_id, task_id):
        return self.q_functions[domain_id][task_id]

    def _get_target_q_function(self, domain_id, task_id):
        return self.target_q_functions[domain_id][task_id]

    def _get_optimizer(self, domain_id, task_id):
        return self.optimizers[domain_id][task_id]

    def _get_experience_replay_buffer(self, task_id):
        return self.experience_replay_buffers[task_id]
