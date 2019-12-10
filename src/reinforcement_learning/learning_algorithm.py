from utils import utils
import random
import torch


class LearningAlgorithm:
    DEBUG = "debug"  # whether to print messages for debugging
    TRAIN_MODEL = "train_model"  # whether we are training or testing
    NUM_EPISODES_FIELD = "num_episodes"  # number of episodes to execute the agent
    MAX_EPISODE_LENGTH_FIELD = "max_episode_length"  # maximum number of steps per episode
    LEARNING_RATE_FIELD = "learning_rate"
    EXPLORATION_RATE_FIELD = "exploration_rate"
    DISCOUNT_RATE_FIELD = "discount_rate"
    IS_TABULAR_CASE = "is_tabular_case"  # whether we use tabular q-learning or function approximation

    def __init__(self, params=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug = utils.get_param(params, LearningAlgorithm.DEBUG, False)
        self.train_model = utils.get_param(params, LearningAlgorithm.TRAIN_MODEL, True)
        self.num_episodes = utils.get_param(params, LearningAlgorithm.NUM_EPISODES_FIELD, 20000)
        self.max_episode_length = utils.get_param(params, LearningAlgorithm.MAX_EPISODE_LENGTH_FIELD, 100)
        self.learning_rate = utils.get_param(params, LearningAlgorithm.LEARNING_RATE_FIELD, 0.1)
        self.exploration_rate = utils.get_param(params, LearningAlgorithm.EXPLORATION_RATE_FIELD, 0.1)
        self.discount_rate = utils.get_param(params, LearningAlgorithm.DISCOUNT_RATE_FIELD, 0.99)
        self.is_tabular_case = utils.get_param(params, LearningAlgorithm.IS_TABULAR_CASE, True)

    def run(self, task):
        raise NotImplementedError

    def _choose_egreedy_action(self, task, state, q_table):
        if self.train_model:
            prob = random.uniform(0, 1)
            if prob <= self.exploration_rate:
                return random.choice(range(0, task.action_space.n))
        return self._get_greedy_action(task, state, q_table)

    def _get_greedy_action(self, task, state, q_table):
        if self.is_tabular_case:
            q_values = [q_table[(state, action)] for action in range(task.action_space.n)]
            return utils.randargmax(q_values)
        else:
            state_v = torch.tensor(state).to(self.device)
            q_values = q_table(state_v)
            return utils.randargmax(q_values.detach().cpu().numpy())

    def _get_observations_as_ordered_tuple(self, observation_set):
        observations_list = list(observation_set)
        utils.sort_by_ord(observations_list)
        return tuple(observations_list)
