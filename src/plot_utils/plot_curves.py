import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

from reinforcement_learning.isa_base_algorithm import ISAAlgorithmBase
from utils.utils import read_json_file

CONFIG_COLOUR_ATTR = "colour"
CONFIG_FOLDERS_ATTR = "folders"
CONFIG_LABEL_ATTR = "label"
REWARD_IF_GOAL = 1.0


def moving_average(interval, window_size):
    interval_df = pd.DataFrame(interval)
    roll = interval_df.rolling(window=window_size, min_periods=1)
    mean, std_dev = roll.mean(), roll.std(ddof=0)
    return mean.squeeze().to_numpy(), std_dev.squeeze().to_numpy()


def read_reward_steps_list(filename, max_episode_length, use_greedy_traces, greedy_evaluation_frequency):
    reward_steps = pd.read_csv(filename, sep=';', header=None)
    rewards, steps = reward_steps[0], reward_steps[1]
    rewards_np, steps_np = rewards.squeeze().to_numpy(), steps.squeeze().to_numpy()
    steps_np[rewards_np < REWARD_IF_GOAL] = max_episode_length  # unsolved tasks are given the maximum episode length as #steps
    if use_greedy_traces and greedy_evaluation_frequency > 1:
        rewards_np_rep = np.repeat(np.concatenate(([0.0], rewards_np[:-1])), greedy_evaluation_frequency)
        rewards_np = np.concatenate((rewards_np_rep[:-1], [rewards_np[-1]]))
        steps_np_rep = np.repeat(np.concatenate(([max_episode_length], steps_np[:-1])), greedy_evaluation_frequency)
        steps_np = np.concatenate((steps_np_rep[:-1], [steps_np[-1]]))
    return rewards_np, steps_np


def read_automaton_learning_episodes(filename):
    automaton_learning_episodes = pd.read_csv(filename, header=None)
    return automaton_learning_episodes.squeeze().to_numpy()


def get_compressed_learning_episodes(learning_episodes, max_diff):
    new_episodes = []
    for ep in learning_episodes:
        if len(new_episodes) == 0 or (ep - new_episodes[-1]) >= max_diff:
            new_episodes.append(ep)
    return new_episodes


def plot_curve(figure_id, num_episodes, moving_avg, learning_episodes, learning_ep_compression, colour, label):
    plt.figure(figure_id)
    x_axis = range(1, num_episodes + 1)
    plt.plot(x_axis, moving_avg, color=colour, label=label)

    for vl in get_compressed_learning_episodes(sorted(learning_episodes), learning_ep_compression):
        if vl <= num_episodes:
            plt.axvline(x=vl, alpha=0.35, color=colour, dashes=(1, 2), zorder=-1000)


def plot_single_task_curve(
    figure_id, num_episodes, items, learning_episodes, learning_ep_compression, colour, label, window_size
):
    moving_avg, _ = moving_average(items, window_size)
    plot_curve(figure_id, num_episodes, moving_avg, learning_episodes, learning_ep_compression, colour, label)


def save_reward_plot(figure_id, task_id, plot_title, output_filename_base, output_path):
    plt.figure(figure_id)
    if plot_title is not None:
        plt.title(plot_title, fontsize=26)
    plt.xlabel("Number of episodes", fontsize=26)
    plt.ylabel("Average return", fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylim((0, REWARD_IF_GOAL + 0.1))
    plt.locator_params(nbins=6)
    plt.legend(fontsize=16, ncol=2, loc="lower right")
    output_filename = os.path.join(output_path, "{}_reward_{}".format(output_filename_base, task_id))
    plt.savefig(output_filename + ".png", bbox_inches='tight')
    plt.savefig(output_filename + ".pdf", bbox_inches='tight')


def save_steps_plot(figure_id, task_id, plot_title, max_steps, output_filename_base, output_path):
    plt.figure(figure_id)
    if plot_title is not None:
        plt.title(plot_title, fontsize=26)
    plt.xlabel("Number of episodes", fontsize=26)
    plt.ylabel("Average steps", fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylim((0, max_steps + 5))
    plt.locator_params(nbins=6)
    plt.legend(fontsize=16, ncol=2, loc="upper right")
    output_filename = os.path.join(output_path, "{}_steps_{}".format(output_filename_base, task_id))
    plt.savefig(output_filename + ".png", bbox_inches='tight')
    plt.savefig(output_filename + ".pdf", bbox_inches='tight')


def init_total_rewards_steps(config_obj, num_episodes):
    total_rewards_sum, total_steps_sum, automaton_learning_episodes = {}, {}, {}
    for setting_label in config_obj:
        label = setting_label[CONFIG_LABEL_ATTR]
        total_rewards_sum[label] = np.zeros(num_episodes, dtype=np.float64)
        total_steps_sum[label] = np.zeros(num_episodes, dtype=np.float64)
        automaton_learning_episodes[label] = set()
    return total_rewards_sum, total_steps_sum, automaton_learning_episodes


def get_reward_steps_sums_for_setting(task_id, setting, max_episode_length, use_greedy_traces, greedy_evaluation_frequency):
    if CONFIG_FOLDERS_ATTR not in setting:
        raise AttributeError("Error: Missing field 'folder' in setting.")

    task_setting_folders = setting[CONFIG_FOLDERS_ATTR]

    if len(task_setting_folders) != num_runs:
        raise RuntimeError("Error: There must be {} folders in setting '{}'.".format(num_runs, setting[CONFIG_LABEL_ATTR]))

    task_rewards, task_steps = np.zeros(num_episodes, dtype=np.float64), np.zeros(num_episodes, dtype=np.float64)
    task_automaton_learning_episodes = []

    for folder in task_setting_folders:
        rewards_steps_folder_path = os.path.join(folder, ISAAlgorithmBase.REWARD_STEPS_FOLDER)
        rewards_steps_log_files = os.listdir(rewards_steps_folder_path)

        if len(rewards_steps_log_files) != num_tasks:
            raise Exception("Error: The expected number of tasks is {} and was {}.".format(num_tasks, len(rewards_steps_log_files)))

        rewards_steps_folder = ISAAlgorithmBase.REWARD_STEPS_GREEDY_FOLDER if use_greedy_traces else ISAAlgorithmBase.REWARD_STEPS_FOLDER
        rewards_steps_log_file = os.path.join(folder, rewards_steps_folder, ISAAlgorithmBase.REWARD_STEPS_FILENAME % task_id)
        rewards, steps = read_reward_steps_list(rewards_steps_log_file, max_episode_length, use_greedy_traces, greedy_evaluation_frequency)

        if len(rewards) >= num_episodes:
            task_rewards += rewards[:num_episodes]
            task_steps += steps[:num_episodes]
            try:
                automaton_learning_episodes_path = os.path.join(folder, ISAAlgorithmBase.AUTOMATON_LEARNING_EPISODES_FILENAME)
                task_automaton_learning_episodes.extend(read_automaton_learning_episodes(automaton_learning_episodes_path))
            except (IOError, EmptyDataError):
                pass  # the automaton learning episodes file does not exist
        else:
            print("Error: The number of episodes in {} is less than {}. Rewards are set to 0, and steps to the maximum "
                  "episode length.".format(rewards_steps_log_file, num_episodes))

    return task_rewards, task_steps, task_automaton_learning_episodes


def process_tasks(
    config_obj, num_tasks, num_runs, num_episodes, max_episode_length, use_greedy_traces, greedy_evaluation_frequency,
    plot_task_curves, window_size, plot_title, learning_ep_compression, output_filename_base, output_path
):
    reward_fig_id, steps_fig_id = 0, 1
    total_rewards_sum, total_steps_sum, total_automaton_learning_episodes = init_total_rewards_steps(config_obj, num_episodes)

    for task_id in tqdm(range(num_tasks)):
        if plot_task_curves:
            reward_fig, steps_fig = plt.figure(reward_fig_id), plt.figure(steps_fig_id)

        for setting in config_obj:
            setting_label = setting[CONFIG_LABEL_ATTR]
            task_rewards, task_steps, task_automaton_learning_episodes = get_reward_steps_sums_for_setting(
                task_id, setting, max_episode_length, use_greedy_traces, greedy_evaluation_frequency
            )

            total_rewards_sum[setting_label] += task_rewards
            total_steps_sum[setting_label] += task_steps
            total_automaton_learning_episodes[setting_label].update(task_automaton_learning_episodes)

            if plot_task_curves:
                plot_single_task_curve(
                    reward_fig_id, num_episodes, task_rewards / num_runs, task_automaton_learning_episodes,
                    learning_ep_compression, setting[CONFIG_COLOUR_ATTR], setting_label, window_size
                )
                plot_single_task_curve(
                    steps_fig_id, num_episodes, task_steps / num_runs, task_automaton_learning_episodes,
                    learning_ep_compression, setting[CONFIG_COLOUR_ATTR], setting_label, window_size
                )

        if plot_task_curves:
            save_reward_plot(reward_fig_id, task_id, plot_title, output_filename_base, output_path)
            save_steps_plot(steps_fig_id, task_id, plot_title, max_episode_length, output_filename_base, output_path)
            plt.close(reward_fig)
            plt.close(steps_fig)

    return total_rewards_sum, total_steps_sum, total_automaton_learning_episodes


def plot_average_task_curves(
    config_obj, num_tasks, num_runs, max_episode_length, window_size, plot_title, learning_ep_compression,
    total_rewards_sum, total_steps_sum, total_automaton_learning_episodes, output_filename_base, output_path
):
    reward_fig_id, steps_fig_id = 0, 1
    reward_fig, steps_fig = plt.figure(reward_fig_id), plt.figure(steps_fig_id)
    for setting in config_obj:
        setting_label = setting[CONFIG_LABEL_ATTR]

        reward_mean, steps_mean = total_rewards_sum[setting_label] / (num_tasks * num_runs), total_steps_sum[setting_label] / (num_tasks * num_runs)

        reward_mean, _ = moving_average(reward_mean, window_size)
        steps_mean, _ = moving_average(steps_mean, window_size)

        plot_curve(
            reward_fig_id, num_episodes, reward_mean, total_automaton_learning_episodes[setting_label],
            learning_ep_compression, setting[CONFIG_COLOUR_ATTR], setting_label
        )
        plot_curve(
            steps_fig_id, num_episodes, steps_mean, total_automaton_learning_episodes[setting_label],
            learning_ep_compression, setting[CONFIG_COLOUR_ATTR], setting_label
        )

    save_reward_plot(reward_fig_id, "avg", plot_title, output_filename_base, output_path)
    save_steps_plot(steps_fig_id, "avg", plot_title, max_episode_length, output_filename_base, output_path)
    plt.close(reward_fig)
    plt.close(steps_fig)


def create_argparser():
    parser = argparse.ArgumentParser(description='Plots learning curves. It assumes rewards are on [0, 1] scale. '
                                                 'A reward of 1 is received if the goal is achieved and 0 otherwise.')
    parser.add_argument("config", help="file containing the configuration")
    parser.add_argument("num_tasks", type=int, help="number of tasks")
    parser.add_argument("num_runs", type=int, help="number of runs")
    parser.add_argument("num_episodes", type=int, help="number of episodes")
    parser.add_argument("--max_episode_length", type=int, default=100, help="maximum length of an episode")
    parser.add_argument("--plot_task_curves", action="store_true", help="whether to also plot task curves")
    parser.add_argument("--use_greedy_traces", "-g", action="store_true", help="whether to use the traces that use the greedy policy")
    parser.add_argument("--greedy_evaluation_frequency", type=int, default=1, help="every how many episodes was the greedy policy evaluated")
    parser.add_argument("--use_tex", action="store_true", help="whether to plot the strings using TeX")
    parser.add_argument("--window_size", "-w", type=int, default=10, help="size of the averaging window")
    parser.add_argument("--plot_title", "-t", default=None, help="the title of the plot")
    parser.add_argument("--learning_ep_compression", type=int, default=0, help="number of automaton learning episodes to compress in one")
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()

    plt.rc('text', usetex=args.use_tex)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')

    num_tasks, num_runs, num_episodes = args.num_tasks, args.num_runs, args.num_episodes
    config_obj = read_json_file(args.config)
    output_filename_base, output_path = os.path.basename(args.config)[:-len(".json")], os.path.abspath(os.path.dirname(args.config))
    total_rewards_sum, total_steps_sum, total_automaton_learning_episodes = process_tasks(
        config_obj, num_tasks, num_runs, num_episodes, args.max_episode_length, args.use_greedy_traces,
        args.greedy_evaluation_frequency, args.plot_task_curves, args.window_size, args.plot_title,
        args.learning_ep_compression, output_filename_base, output_path
    )
    plot_average_task_curves(
        config_obj, num_tasks, num_runs, args.max_episode_length, args.window_size, args.plot_title,
        args.learning_ep_compression, total_rewards_sum, total_steps_sum, total_automaton_learning_episodes,
        output_filename_base, output_path
    )
