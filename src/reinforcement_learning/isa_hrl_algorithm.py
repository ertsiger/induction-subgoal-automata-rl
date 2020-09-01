import collections
import numpy as np
import os
import random
import torch
from torch import optim, nn

from gym_subgoal_automata.utils.condition import EdgeCondition
from reinforcement_learning.dqn_model import DQN
from reinforcement_learning.experience_replay import ExperienceBuffer
from reinforcement_learning.isa_base_algorithm import ISAAlgorithmBase
from utils import utils

# Experience classes used when the q-functions are approximated by DQNs. The class 'Experience' corresponds to the
# low-level experiences (actions) and is used to update the option policies. The class 'OptionExperience' corresponds to
# the high-level experiences (options) and is used to update the policies over options (i.e. the metacontrollers).
Experience = collections.namedtuple("Experience", field_names=["state", "action", "next_state", "is_terminal",
                                                               "is_goal_achieved", "observations"])
OptionExperience = collections.namedtuple("OptionExperience", field_names=["state", "option", "next_state", "is_terminal",
                                                                           "reward", "num_steps", "next_options_mask"])


class ISAAlgorithmHRL(ISAAlgorithmBase):
    """
    Implementation of the method that interleaves RL and automata learning using Hierarchical RL. Specifically, there is
    a policy for each formula in the automaton, which attempts to find an observation that satisfies the formula. Then,
    there is an option for each outgoing transition from an automaton state. A metacontroller in each automaton state
    must select an option when the agent reaches that state.
    """
    ENABLE_PSEUDOREWARD_ON_DEADEND = "enable_pseudoreward_on_deadend"      # whether to give a pseudoreward on deadend MDP states
    PSEUDOREWARD_CONDITION_SATISFIED = "pseudoreward_condition_satisfied"  # amount of reward when the condition of an option is satisfied
    PSEUDOREWARD_AFTER_STEP = "pseudoreward_after_step"                    # amount of pseudoreward given to an option after a step is performed (can be negative to encourage reaching the goal faster)

    USE_NUM_POSITIVE_MATCHINGS = "use_num_positive_matchings"  # use the number of positive matchings only to determine which Q-function to reuse
    ALWAYS_REUSE_QFUNCTION = "always_reuse_qfunction"          # always reuse a Q-function regardless of whether the condition is in the policy bank
    UPDATE_ALL_POLICY_BANK = "update_all_policy_bank"          # whether to update all the option q-functions in the policy bank or only those currently appearing in the automaton

    TABULAR_MODEL_FILENAME = "model-%d-%d.npy"  # filename pattern for tabular models of options q-functions
    DQN_MODEL_FILENAME = "model-%d-%d.pt"       # filename pattern for deep models of options q-functions

    TABULAR_META_MODEL_FILENAME = "meta-%d-%s.npy"  # filename patther for tabular models of q-functions over options
    DQN_META_MODEL_FILENAME = "meta-%d.pt"          # filename patther for deep models of q-functions over options

    def __init__(self, tasks, num_tasks, export_folder_names, params, target_automata, binary_folder_name):
        super().__init__(tasks, num_tasks, export_folder_names, params, target_automata, binary_folder_name)

        # pseudorewards to update the q-functions of the conditions
        self.enable_pseudoreward_on_deadend_state = utils.get_param(params, ISAAlgorithmHRL.ENABLE_PSEUDOREWARD_ON_DEADEND, False)
        self.pseudoreward_condition_satisfied = utils.get_param(params, ISAAlgorithmHRL.PSEUDOREWARD_CONDITION_SATISFIED, 1.0)
        self.pseudoreward_after_step = utils.get_param(params, ISAAlgorithmHRL.PSEUDOREWARD_AFTER_STEP, 0.0)

        # policy bank q-function update flags
        self.use_num_positive_matchings = utils.get_param(params, ISAAlgorithmHRL.USE_NUM_POSITIVE_MATCHINGS, True)
        self.always_reuse_qfunction = utils.get_param(params, ISAAlgorithmHRL.ALWAYS_REUSE_QFUNCTION, False)
        self.update_all_policy_bank = utils.get_param(params, ISAAlgorithmHRL.UPDATE_ALL_POLICY_BANK, False)

        # option related structures
        self.has_terminated = {}        # whether the option has terminated
        self.selected_option = {}       # option currently being executed
        self.last_state = {}            # state where the option started being executed
        self.num_option_steps = {}      # number of steps between the last option initiation and termination
        self.option_reward = {}

        # q-functions for selecting among options (policies over options)
        self.meta_q_functions = []
        self.target_meta_q_functions = []
        self.target_meta_counters = {}
        self.meta_optimizers = {}
        self.meta_experience_replay_buffers = None

        # q-functions for satisfying a particular condition
        self.policy_bank = {}
        self.policy_bank_update_counter = {}
        self.target_policy_bank = {}
        self.target_policy_bank_counter = {}
        self.policy_bank_optimizers = {}
        self.experience_replay_buffers = None

        self._build_q_functions()
        self._build_experience_replay_buffers()

    '''
    Building of Q-functions <automaton state, MDP state, action> (one for each <domain, task>)
    '''
    def _build_q_functions(self):
        self._build_option_tracking_structs()  # initialize structures for properly updating the q-tables
        self._build_policy_bank()              # initialize the q-tables for each of the conditions in the automata
        self._build_meta_q_functions()         # initialize the q-tables for choosing between options (edges)

    def _build_option_tracking_structs(self):
        self.has_terminated.clear()
        self.selected_option.clear()
        self.last_state.clear()
        self.num_option_steps.clear()
        self.option_reward.clear()

        for domain_id in range(self.num_domains):
            self.has_terminated[domain_id] = {}
            self.selected_option[domain_id] = {}
            self.last_state[domain_id] = {}
            self.num_option_steps[domain_id] = {}
            self.option_reward[domain_id] = {}

            current_tasks = self.tasks[domain_id]
            for task_id in range(len(current_tasks)):
                self.has_terminated[domain_id][task_id] = True
                self.selected_option[domain_id][task_id] = None
                self.last_state[domain_id][task_id] = None
                self.num_option_steps[domain_id][task_id] = None
                self.option_reward[domain_id][task_id] = None

    def _build_policy_bank(self):
        self._init_target_net_update_counter()
        for domain_id in range(self.num_domains):
            self._build_domain_policy_bank(domain_id, False)

    def _init_target_net_update_counter(self):
        self.target_policy_bank_counter = {}

    def _build_domain_policy_bank(self, domain_id, copy_similar):
        automaton = self.automata[domain_id]

        for task_id in range(self.num_tasks):
            # initialize container for the given task id
            if task_id not in self.policy_bank:
                self.policy_bank[task_id] = {}
                self.policy_bank_update_counter[task_id] = {}

                if not self.is_tabular_case:
                    self.target_policy_bank[task_id] = {}
                    self.policy_bank_optimizers[task_id] = {}
                    self.target_policy_bank_counter[task_id] = {}

            task = self.tasks[domain_id][task_id]

            # for each possible condition/option, initialize a table or a DQN. If an option with a similar termination
            # condition is found, then copy it (if the flag allows it).
            for condition in automaton.get_all_conditions():
                if condition not in self.policy_bank[task_id] or self.always_reuse_qfunction:
                    if copy_similar:
                        self._build_function_from_existing_condition(task_id, task, condition)
                    else:
                        self._initialize_function_for_condition(task_id, task, condition)

                    if not self.is_tabular_case:
                        self.target_policy_bank[task_id][condition] = DQN(task.observation_space.n, task.action_space.n, self.num_layers, self.num_neurons_per_layer)
                        self.target_policy_bank[task_id][condition].to(self.device)
                        self.target_policy_bank[task_id][condition].load_state_dict(self.policy_bank[task_id][condition].state_dict())
                        self.policy_bank_optimizers[task_id][condition] = optim.Adam(self.policy_bank[task_id][condition].parameters(), lr=self.learning_rate)
                        self.target_policy_bank_counter[task_id][condition] = 0

    def _initialize_function_for_condition(self, task_id, task, condition):
        if self.is_tabular_case:
            self.policy_bank[task_id][condition] = np.zeros((task.observation_space.n, task.action_space.n),
                                                            dtype=np.float32)
        else:
            self.policy_bank[task_id][condition] = DQN(task.observation_space.n, task.action_space.n, self.num_layers,
                                                       self.num_neurons_per_layer)
            self.policy_bank[task_id][condition].to(self.device)
        self.policy_bank_update_counter[task_id][condition] = 0

    def _build_function_from_existing_condition(self, task_id, task, condition):
        max_num_matchings = 1
        max_conditions, max_conditions_updates = [], []

        # take the existing conditions in the policy bank with highest number of symbol matchings
        for existing_condition in self.policy_bank[task_id]:
            if self.use_num_positive_matchings:
                num_matchings = condition.get_num_positive_matching_symbols(existing_condition)
            else:
                num_matchings = condition.get_num_matching_symbols(existing_condition)
            if num_matchings > max_num_matchings:
                max_num_matchings = num_matchings
                max_conditions = [existing_condition]
                max_conditions_updates = [self.policy_bank_update_counter[task_id][existing_condition]]
            elif num_matchings == max_num_matchings:
                max_conditions.append(existing_condition)
                max_conditions_updates.append(self.policy_bank_update_counter[task_id][existing_condition])

        self._initialize_function_for_condition(task_id, task, condition)
        if len(max_conditions) > 0:
            # take the condition with more matchings that has been updated the most
            max_condition = max_conditions[utils.randargmax(max_conditions_updates)]
            if self.is_tabular_case:
                self.policy_bank[task_id][condition] = np.copy(self.policy_bank[task_id][max_condition])
            else:
                self.policy_bank[task_id][condition].load_state_dict(self.policy_bank[task_id][max_condition].state_dict())

    def _build_meta_q_functions(self):
        self.meta_q_functions = [{} for _ in range(self.num_domains)]
        self.target_meta_q_functions = [{} for _ in range(self.num_domains)]
        self.target_meta_counters = [{} for _ in range(self.num_domains)]
        self.meta_optimizers = [{} for _ in range(self.num_domains)]

        for domain_id in range(self.num_domains):
            if self.is_tabular_case:
                self._build_tabular_meta_q_functions(domain_id)
            else:
                self._build_dqn_meta_q_functions(domain_id)

    def _build_tabular_meta_q_functions(self, domain_id):
        q_tables = self.meta_q_functions[domain_id]
        q_tables.clear()

        current_tasks = self.tasks[domain_id]
        current_automaton = self._get_automaton(domain_id)

        for task_id in range(len(current_tasks)):
            q_tables[task_id] = {}
            task = current_tasks[task_id]

            for state in current_automaton.get_states():
                if current_automaton.get_num_outgoing_edges(state) > 0:
                    q_tables[task_id][state] = np.zeros((task.observation_space.n, current_automaton.get_num_outgoing_edges(state)),
                                                        dtype=np.float32)
                else:
                    q_tables[task_id][state] = np.zeros((task.observation_space.n, task.action_space.n),
                                                        dtype=np.float32)

    def _build_dqn_meta_q_functions(self, domain_id):
        q_functions = self.meta_q_functions[domain_id]
        target_q_functions = self.target_meta_q_functions[domain_id]
        target_meta_counters = self.target_meta_counters[domain_id]
        optimizers = self.meta_optimizers[domain_id]

        q_functions.clear()
        target_q_functions.clear()
        optimizers.clear()

        current_tasks = self.tasks[domain_id]
        current_automaton = self._get_automaton(domain_id)

        for task_id in range(len(current_tasks)):
            task = current_tasks[task_id]
            num_states, num_actions = task.observation_space.n, task.action_space.n
            num_automaton_states = current_automaton.get_num_states()

            q_functions[task_id] = DQN(num_states + num_automaton_states,
                                       current_automaton.get_num_unique_conditions() + num_actions,
                                       self.num_layers, self.num_neurons_per_layer)
            q_functions[task_id].to(self.device)

            target_q_functions[task_id] = DQN(num_states + num_automaton_states,
                                              current_automaton.get_num_unique_conditions() + num_actions,
                                              self.num_layers, self.num_neurons_per_layer)
            target_q_functions[task_id].to(self.device)
            target_q_functions[task_id].load_state_dict(q_functions[task_id].state_dict())

            target_meta_counters[task_id] = 0

            optimizers[task_id] = optim.Adam(q_functions[task_id].parameters(), lr=self.learning_rate)

    def _build_experience_replay_buffers(self):
        self.experience_replay_buffers = [ExperienceBuffer(self.experience_replay_buffer_size)
                                          for _ in range(self.num_tasks)]
        self._build_meta_experience_replay_buffers()

    def _build_meta_experience_replay_buffers(self):
        self.meta_experience_replay_buffers = {}
        for domain_id in range(self.num_domains):
            self.meta_experience_replay_buffers[domain_id] = [ExperienceBuffer(self.experience_replay_buffer_size)
                                                              for _ in range(self.num_tasks)]

    def _run_episode(self, domain_id, task_id):
        """Overrides parent method but adding a line for resetting the last automaton state to None in order to choose
        a new option when the episode starts."""
        self.has_terminated[domain_id][task_id] = True
        return super()._run_episode(domain_id, task_id)

    '''
    Action and option choice during the episode
    '''
    def _choose_action(self, domain_id, task_id, current_state, automaton, current_automaton_state_id):
        current_automaton_state = automaton.get_states()[current_automaton_state_id]

        task = self._get_task(domain_id, task_id)

        if self.interleaved_automaton_learning and not self.has_observed_goal_example[domain_id]:
            return self._get_random_action(task)

        # if the current automaton has terminated it means we have to select a new option to run
        if self.has_terminated[domain_id][task_id]:
            self.has_terminated[domain_id][task_id] = False
            self.selected_option[domain_id][task_id] = self._choose_egreedy_option(domain_id, task_id, current_state,
                                                                                   automaton, current_automaton_state)
            self.last_state[domain_id][task_id] = current_state
            self.num_option_steps[domain_id][task_id] = 0
            self.option_reward[domain_id][task_id] = 0

        option = self.selected_option[domain_id][task_id]
        if automaton.get_num_outgoing_edges(current_automaton_state) > 0:
            return self._choose_egreedy_action(task, current_state, self.policy_bank[task_id][option])
        return option  # it is a primitive action

    def _choose_egreedy_option(self, domain_id, task_id, current_state, automaton, automaton_state):
        """Epsilon greedy option selection at a given automaton state for a given state."""
        if self.train_model:
            prob = random.uniform(0, 1)
            if prob <= self.exploration_rate:
                return self._get_random_option(domain_id, task_id, automaton, automaton_state)
        return self._get_greedy_option(domain_id, task_id, current_state, automaton, automaton_state)

    def _get_random_option(self, domain_id, task_id, automaton, automaton_state):
        if automaton.get_num_outgoing_edges(automaton_state) > 0:
            return random.choice(automaton.get_outgoing_conditions(automaton_state))
        else:
            return self._get_random_action(self._get_task(domain_id, task_id))

    def _get_greedy_option(self, domain_id, task_id, current_state, automaton, current_automaton_state):
        """Returns the option with the highest Q-value at the given automaton state for a given state."""
        if self.is_tabular_case:
            return self._get_greedy_option_tabular(domain_id, task_id, current_state, automaton, current_automaton_state)
        else:
            return self._get_greedy_option_deep(domain_id, task_id, current_state, automaton, current_automaton_state)

    def _get_greedy_option_tabular(self, domain_id, task_id, current_state, automaton, current_automaton_state):
        meta_q_function = self.meta_q_functions[domain_id][task_id]

        if automaton.get_num_outgoing_edges(current_automaton_state) > 0:
            available_options = automaton.get_outgoing_conditions(current_automaton_state)
        else:
            available_options = [i for i in range(self._get_task(domain_id, task_id).action_space.n)]

        q_values = [meta_q_function[current_automaton_state][(current_state, option)] for option in range(len(available_options))]
        return available_options[utils.randargmax(q_values)]

    def _get_greedy_option_deep(self, domain_id, task_id, current_state, automaton, current_automaton_state):
        meta_q_function = self.meta_q_functions[domain_id][task_id]
        state_v = self._get_state_and_automaton_state_vector(current_state, automaton, current_automaton_state)
        state_v = torch.tensor(state_v).to(self.device)
        q_values = meta_q_function(state_v)
        task = self._get_task(domain_id, task_id)
        mask = torch.tensor(self._get_available_option_mask(task, automaton, current_automaton_state), dtype=torch.bool).logical_not()
        q_values[mask] = -100000000000.0
        option_index = utils.randargmax(q_values.detach().cpu().numpy())
        all_conditions = automaton.get_all_conditions()
        if option_index < len(all_conditions):  # an option
            return all_conditions[option_index]
        return option_index - len(all_conditions)  # a primitive action

    '''
    What to do when a step in the environment has been completed (includes update of policies over options - metapolicies)
    '''
    def _on_performed_step(self, domain_id, task_id, next_state, reward, is_terminal, observations, automaton,
                           current_automaton_state, next_automaton_state, episode_length):
        if self.interleaved_automaton_learning and not self.has_observed_goal_example[domain_id]:
            return

        # update option attributes
        discount = self.discount_rate ** self.num_option_steps[domain_id][task_id]
        self.option_reward[domain_id][task_id] += discount * reward
        self.num_option_steps[domain_id][task_id] += 1
        self.has_terminated[domain_id][task_id] = automaton.is_terminal_state(current_automaton_state) \
                                                  or current_automaton_state != next_automaton_state \
                                                  or is_terminal \
                                                  or episode_length >= self.max_episode_length - 1

        if self.train_model:
            if self.has_terminated[domain_id][task_id]:  # if the option terminates, the metacontroller is updated
                if self.is_tabular_case:
                    self._update_tabular_meta_q_functions(domain_id, task_id, next_state, is_terminal, automaton,
                                                          current_automaton_state, next_automaton_state)
                else:
                    self._update_deep_meta_q_functions(domain_id, task_id, next_state, is_terminal, automaton,
                                                       current_automaton_state, next_automaton_state)

            if not self.is_tabular_case:
                self._update_target_deep_q_functions(domain_id, task_id)

    def _update_tabular_meta_q_functions(self, domain_id, task_id, next_state, is_terminal, automaton, current_automaton_state, next_automaton_state):
        """Applies SMDP Q-Learning update to the function for the given (domain, task) pair."""
        meta_q_function = self.meta_q_functions[domain_id][task_id]
        delta = self.option_reward[domain_id][task_id]

        if not is_terminal:
            if automaton.get_num_outgoing_edges(next_automaton_state) > 0:
                next_greedy_option = self._get_greedy_option(domain_id, task_id, next_state, automaton, next_automaton_state)
                next_greedy_option_id = automaton.get_outgoing_condition_id(next_automaton_state, next_greedy_option)
            else:
                next_greedy_option_id = self._get_greedy_option(domain_id, task_id, next_state, automaton, next_automaton_state)
            discount = self.discount_rate ** self.num_option_steps[domain_id][task_id]
            delta += discount * meta_q_function[next_automaton_state][(next_state, next_greedy_option_id)]

        if automaton.get_num_outgoing_edges(current_automaton_state) > 0:
            selected_option_id = automaton.get_outgoing_condition_id(current_automaton_state, self.selected_option[domain_id][task_id])
        else:
            selected_option_id = self.selected_option[domain_id][task_id]

        updated_so_pair = (self.last_state[domain_id][task_id], selected_option_id)
        delta -= meta_q_function[current_automaton_state][updated_so_pair]
        meta_q_function[current_automaton_state][updated_so_pair] += self.learning_rate * delta

    def _update_deep_meta_q_functions(self, domain_id, task_id, next_state, is_terminal, automaton, current_automaton_state, next_automaton_state):
        self._add_to_meta_experience_replay_buffer(domain_id, task_id, next_state, is_terminal, automaton,
                                                   current_automaton_state, next_automaton_state)

        er_buffer = self.meta_experience_replay_buffers[domain_id][task_id]
        if len(er_buffer) >= self.experience_replay_start_size:
            experience_batch = er_buffer.sample(self.experience_replay_batch_size)
            self._update_deep_meta_q_functions_from_batch(domain_id, task_id, experience_batch)

    def _add_to_meta_experience_replay_buffer(self, domain_id, task_id, next_state, is_terminal, automaton, current_automaton_state, next_automaton_state):
        state_v = self._get_state_and_automaton_state_vector(self.last_state[domain_id][task_id], automaton, current_automaton_state)

        if isinstance(self.selected_option[domain_id][task_id], EdgeCondition):  # option
            option_id = automaton.get_all_conditions().index(self.selected_option[domain_id][task_id])
        else:  # primitive action
            option_id = self.selected_option[domain_id][task_id] + automaton.get_num_unique_conditions()
        next_state_v = self._get_state_and_automaton_state_vector(next_state, automaton, next_automaton_state)
        reward = self.option_reward[domain_id][task_id]
        num_steps = self.num_option_steps[domain_id][task_id]

        task = self._get_task(domain_id, task_id)
        mask = self._get_available_option_mask(task, automaton, next_automaton_state)  # mask for outgoing conditions of the next automaton state

        experience = OptionExperience(state_v, option_id, next_state_v, is_terminal, reward, num_steps, mask)
        self.meta_experience_replay_buffers[domain_id][task_id].append(experience)

    def _update_deep_meta_q_functions_from_batch(self, domain_id, task_id, experience_batch):
        states, conditions, next_states, is_terminal, rewards, num_steps, next_options_masks = zip(*experience_batch)

        # convert batch components to tensors
        states_v = torch.tensor(states).to(self.device)
        conditions_v = torch.tensor(conditions, dtype=torch.long).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        is_terminal_v = torch.tensor(is_terminal, dtype=torch.bool).to(self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        num_steps_v = torch.tensor(num_steps, dtype=torch.float32).to(self.device)
        next_options_masks_v = torch.tensor(next_options_masks, dtype=torch.float32).to(self.device)

        # get Q-values for all options at the current state
        net = self.meta_q_functions[domain_id][task_id]
        state_option_values = net(states_v).gather(1, conditions_v.unsqueeze(-1)).squeeze(-1)

        # get target Q-values masking those options not available at the given (state, automaton state) with very
        # negative values in order not to select them
        target_net = self.target_meta_q_functions[domain_id][task_id]

        if self.use_double_dqn:
            next_state_option_values = net(next_states_v)
            next_state_option_values += -100000000000.0 * (1.0 - next_options_masks_v)
            next_state_options = next_state_option_values.max(1)[1]
            next_state_option_values = target_net(next_states_v).gather(1, next_state_options.unsqueeze(-1)).squeeze(-1)
        else:
            next_state_option_values = target_net(next_states_v)
            next_state_option_values += -100000000000.0 * (1.0 - next_options_masks_v)
            next_state_option_values = next_state_option_values.max(1)[0]

        next_state_option_values[is_terminal_v] = 0.0
        next_state_option_values = next_state_option_values.detach()

        # SMDP Q-learning discount
        discount = self.discount_rate ** num_steps_v

        expected_state_action_values = rewards_v + discount * next_state_option_values
        loss = nn.MSELoss()(state_option_values, expected_state_action_values)

        self.meta_optimizers[domain_id][task_id].zero_grad()
        loss.backward()
        self.meta_optimizers[domain_id][task_id].step()

    def _get_one_hot_automaton_state(self, automaton, automaton_state):
        automaton_states = sorted(list(automaton.get_states()))
        automaton_state_v = np.zeros(len(automaton_states), dtype=np.float32)
        automaton_state_v[automaton_states.index(automaton_state)] = 1.0
        return automaton_state_v

    def _get_available_option_mask(self, task, automaton, automaton_state):
        all_conditions = automaton.get_all_conditions()
        if automaton.get_num_outgoing_edges(automaton_state) > 0:
            state_conditions = automaton.get_outgoing_conditions(automaton_state)
            mask = [1 if c in state_conditions else 0 for c in all_conditions] + [0] * task.action_space.n
        else:
            mask = [0] * len(all_conditions) + [1] * task.action_space.n
        return mask

    def _get_state_and_automaton_state_vector(self, state, automaton, automaton_state):
        return np.concatenate((state, self._get_one_hot_automaton_state(automaton, automaton_state)))

    def _update_q_functions(self, task_id, current_state, action, next_state, is_terminal, observations, _):
        task = self._get_task(0, task_id)

        if self.use_experience_replay:
            experience = Experience(current_state, action, next_state, is_terminal, task.is_goal_achieved(), observations)
            experience_replay_buffer = self._get_experience_replay_buffer(task_id)
            experience_replay_buffer.append(experience)
            if len(experience_replay_buffer) >= self.experience_replay_start_size:
                experience_batch = experience_replay_buffer.sample(self.experience_replay_batch_size)

                if self.is_tabular_case:
                    for exp in experience_batch:
                        self._update_tabular_q_functions(task_id, (exp.state, exp.action), exp.next_state,
                                                         exp.is_terminal, exp.is_goal_achieved, exp.observations)
                else:
                    self._update_deep_q_functions(task_id, experience_batch)
        else:
            # update all q-tables of the current task (not other tasks because state spaces might be different!)
            current_pair = (current_state, action)
            self._update_tabular_q_functions(task_id, current_pair, next_state, is_terminal, task.is_goal_achieved(), observations)

    def _get_experience_replay_buffer(self, task_id):
        return self.experience_replay_buffers[task_id]

    def _get_all_automata_conditions(self):
        all_conditions = set()
        for domain_id in range(self.num_domains):
            all_conditions.update(self.automata[domain_id].get_all_conditions())
        return sorted(all_conditions)

    def _update_tabular_q_functions(self, task_id, current_pair, next_state, is_terminal, is_goal_achieved, observations):
        # take any task, just needed for the number of actions
        task = self._get_task(0, task_id)

        if self.update_all_policy_bank:
            conditions = self.policy_bank[task_id]
        else:
            conditions = self._get_all_automata_conditions()

        for condition in conditions:
            reward, is_terminal_local = self._get_pseudoreward_for_condition(condition, observations, is_terminal, is_goal_achieved)
            if is_terminal_local:
                delta = reward
            else:
                next_action = self._get_greedy_action(task, next_state, self.policy_bank[task_id][condition])
                next_pair = (next_state, next_action)
                delta = reward + self.discount_rate * self.policy_bank[task_id][condition][next_pair]

            self.policy_bank[task_id][condition][current_pair] += self.learning_rate * (delta - self.policy_bank[task_id][condition][current_pair])
            self.policy_bank_update_counter[task_id][condition] += 1

    def _get_pseudoreward_for_condition(self, condition, observations, is_terminal, is_goal_achieved):
        """Returns the pseudoreward used to update the policy that aims to achieve the condition passes as a parameter."""
        valid_observation = not self.ignore_empty_observations or len(observations) > 0

        if condition.is_satisfied(observations) and valid_observation:
            return self.pseudoreward_condition_satisfied, True
        elif is_terminal:
            if not is_goal_achieved and self.enable_pseudoreward_on_deadend_state:
                # note that if the MDP state is terminal and the goal is not achieved, the state is a deadend state
                return float(-self.max_episode_length), True
            return 0.0, True
        return self.pseudoreward_after_step, False

    def _update_deep_q_functions(self, task_id, experience_batch):
        states, actions, next_states, is_terminal, is_goal_achieved, observations = zip(*experience_batch)

        states_v = torch.tensor(states).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.long).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)

        if self.update_all_policy_bank:
            conditions = self.policy_bank[task_id]
        else:
            conditions = self._get_all_automata_conditions()

        for condition in conditions:
            rewards, is_terminal_local = self._get_transition_rewards_terminal_batches(condition, observations,
                                                                                       is_terminal, is_goal_achieved)
            rewards_v = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            is_terminal_v = torch.tensor(is_terminal_local, dtype=torch.bool).to(self.device)

            net = self.policy_bank[task_id][condition]
            target_net = self.target_policy_bank[task_id][condition]

            state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

            if self.use_double_dqn:
                next_state_actions = net(next_states_v).max(1)[1]
                next_state_action_values = target_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
            else:
                next_state_action_values = target_net(next_states_v).max(1)[0]

            next_state_action_values[is_terminal_v] = 0.0
            next_state_action_values = next_state_action_values.detach()

            expected_state_action_values = rewards_v + self.discount_rate * next_state_action_values
            loss = nn.MSELoss()(state_action_values, expected_state_action_values)

            self.policy_bank_optimizers[task_id][condition].zero_grad()
            loss.backward()
            self.policy_bank_optimizers[task_id][condition].step()
            self.policy_bank_update_counter[task_id][condition] += 1

    def _get_transition_rewards_terminal_batches(self, condition, observations, is_terminal, is_goal_achieved):
        transition_rewards_batch = []
        transition_terminal_batch = []
        for i in range(len(observations)):
            reward, is_terminal_local = self._get_pseudoreward_for_condition(condition, observations[i], is_terminal[i],
                                                                             is_goal_achieved[i])
            transition_rewards_batch.append(reward)
            transition_terminal_batch.append(is_terminal_local)
        return transition_rewards_batch, transition_terminal_batch

    def _update_target_deep_q_functions(self, domain_id, task_id):
        self._update_target_deep_meta_q_functions(domain_id, task_id)
        self._update_target_deep_policy_bank(domain_id, task_id)

    def _update_target_deep_meta_q_functions(self, domain_id, task_id):
        self.target_meta_counters[domain_id][task_id] += 1
        if self.target_meta_counters[domain_id][task_id] % self.target_net_update_frequency == 0:
            self.target_meta_q_functions[domain_id][task_id].load_state_dict(self.meta_q_functions[domain_id][task_id].state_dict())
            self.target_meta_counters[domain_id][task_id] = 0

    def _update_target_deep_policy_bank(self, domain_id, task_id):
        option = self.selected_option[domain_id][task_id]
        if not isinstance(option, np.int64) and not isinstance(option, int):
            self.target_policy_bank_counter[task_id][option] += 1

            if self.target_policy_bank_counter[task_id][option] % self.target_net_update_frequency == 0:
                net = self.policy_bank[task_id][option]
                target_net = self.target_policy_bank[task_id][option]
                target_net.load_state_dict(net.state_dict())
                self.target_policy_bank_counter[task_id][option] = 0

    '''
    What to do when a new automaton is learned
    '''
    def _on_automaton_learned(self, domain_id):
        self._reset_q_functions(domain_id)
        self._reset_meta_experience_replay(domain_id)

    def _reset_q_functions(self, domain_id):
        """Rebuild Q-functions when an automaton is learned."""
        self._build_domain_policy_bank(domain_id, True)

        if self.is_tabular_case:
            self._build_tabular_meta_q_functions(domain_id)
        else:
            self._build_dqn_meta_q_functions(domain_id)

    def _reset_meta_experience_replay(self, domain_id):
        """When a new automaton is learned, the learned policies over options are no longer useful."""
        if self.use_experience_replay:
            for task_id in range(self.num_tasks):
                self.meta_experience_replay_buffers[domain_id][task_id].clear()

    '''
    Model exporting and importing
    '''
    def _export_models(self):
        for domain_id in range(self.num_domains):
            utils.mkdir(self.get_models_folder(domain_id))
            for task_id in range(self.num_tasks):
                self._export_policy_bank(domain_id, task_id)
                self._export_meta_functions(domain_id, task_id)

    def _export_policy_bank(self, domain_id, task_id):
        automaton = self._get_automaton(domain_id)
        conditions = automaton.get_all_conditions()

        for i in range(len(conditions)):
            condition = conditions[i]

            if self.is_tabular_case:
                model_path = os.path.join(self.get_models_folder(domain_id),
                                          ISAAlgorithmHRL.TABULAR_MODEL_FILENAME % (task_id, i))
                np.save(model_path, self.policy_bank[task_id][condition])
            else:
                model_path = os.path.join(self.get_models_folder(domain_id),
                                          ISAAlgorithmHRL.DQN_MODEL_FILENAME % (task_id, i))
                torch.save(self.policy_bank[task_id][condition].state_dict(), model_path)

    def _export_meta_functions(self, domain_id, task_id):
        if self.is_tabular_case:
            for automaton_state in self.meta_q_functions[domain_id][task_id]:
                model_path = os.path.join(self.get_models_folder(domain_id),
                                          ISAAlgorithmHRL.TABULAR_META_MODEL_FILENAME % (task_id, automaton_state))
                np.save(model_path, self.meta_q_functions[domain_id][task_id][automaton_state])
        else:
            model_path = os.path.join(self.get_models_folder(domain_id), ISAAlgorithmHRL.DQN_META_MODEL_FILENAME % task_id)
            torch.save(self.meta_q_functions[domain_id][task_id].state_dict(), model_path)

    def _import_models(self):
        for domain_id in range(self.num_domains):
            for task_id in range(self.num_tasks):
                self._import_policy_bank(domain_id, task_id)
                self._import_meta_functions(domain_id, task_id)

    def _import_policy_bank(self, domain_id, task_id):
        automaton = self._get_automaton(domain_id)
        conditions = automaton.get_all_conditions()

        for i in range(len(conditions)):
            condition = conditions[i]

            if self.is_tabular_case:
                model_path = os.path.join(self.get_models_folder(domain_id),
                                          ISAAlgorithmHRL.TABULAR_MODEL_FILENAME % (task_id, i))
                self.policy_bank[task_id][condition] = np.load(model_path)
            else:
                model_path = os.path.join(self.get_models_folder(domain_id),
                                          ISAAlgorithmHRL.DQN_MODEL_FILENAME % (task_id, i))
                model = self.policy_bank[task_id][condition]
                model.load_state_dict(torch.load(model_path))
                model.eval()

    def _import_meta_functions(self, domain_id, task_id):
        if self.is_tabular_case:
            for automaton_state in self.meta_q_functions[domain_id][task_id]:
                model_path = os.path.join(self.get_models_folder(domain_id),
                                          ISAAlgorithmHRL.TABULAR_META_MODEL_FILENAME % (task_id, automaton_state))
                self.meta_q_functions[domain_id][task_id][automaton_state] = np.load(model_path)
        else:
            model_path = os.path.join(self.get_models_folder(domain_id), ISAAlgorithmHRL.DQN_META_MODEL_FILENAME % task_id)
            model = self.meta_q_functions[domain_id][task_id]
            model.load_state_dict(torch.load(model_path))
            model.eval()
