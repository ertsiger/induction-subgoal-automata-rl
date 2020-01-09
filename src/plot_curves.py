import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from reinforcement_learning.isa_algorithm import ISAAlgorithm


def moving_average(interval, window_size):
    window = []
    moving_average = []
    moving_std = []

    for i in range(0, len(interval)):
        window.append(interval[i])
        if len(window) > window_size:
            window.pop(0)
        window_avg = np.mean(window)
        window_std = np.std(window)
        moving_average.append(window_avg)
        moving_std.append(window_std)

    return np.array(moving_average, copy=False), np.array(moving_std, copy=False)


def read_reward_steps_list(filename):
    reward_list, steps_list = [], []
    with open(filename) as f:
        for line in f:
            line_strip = line.strip()
            if len(line_strip) > 0:
                reward, steps = line_strip.split(";")
                reward_list.append(float(reward))
                steps_list.append(float(steps))
    return reward_list, steps_list


def read_automaton_learning_episodes(filename):
    vertical_lines = []
    with open(filename) as f:
        for line in f:
            line_strip = line.strip()
            if len(line_strip) > 0:
                vertical_lines.append(int(line_strip))
    return vertical_lines


def plot_curve(figure_id, num_episodes, moving_avg, learning_episodes, colour, label):
    plt.figure(figure_id)
    x_axis = range(1, num_episodes + 1)
    plt.plot(x_axis, moving_avg, color=colour, label=label)

    for vl in learning_episodes:
        plt.axvline(x=vl, alpha=0.35, color=colour, dashes=(1, 2), zorder=-1000)


def plot_single_task_curve(figure_id, num_episodes, items, learning_episodes, colour, label):
    moving_avg, _ = moving_average(items, 1000)
    plot_curve(figure_id, num_episodes, moving_avg, learning_episodes, colour, label)


def save_reward_plot(figure_id, task_id):
    plt.figure(figure_id)
    plt.xlabel("Number of episodes", fontsize=20)
    plt.ylabel("Average reward", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim((0, 1.05))
    plt.locator_params(nbins=6)
    plt.legend(fontsize=12, ncol=2, loc="lower right")
    plt.savefig("reward-" + str(task_id) + ".pdf", bbox_inches='tight')


def save_steps_plot(figure_id, task_id, min_steps=0, max_steps=100):
    plt.figure(figure_id)
    plt.xlabel("Number of episodes", fontsize=20)
    plt.ylabel("Average steps", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim((min_steps, max_steps))
    plt.locator_params(nbins=6)
    plt.legend(fontsize=12, ncol=2, loc="lower right")
    plt.savefig("steps-" + str(task_id) + ".pdf", bbox_inches='tight')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="file containing the configuration")
    parser.add_argument("num_tasks", type=int, help="number of tasks")
    parser.add_argument("num_runs", type=int, help="number of tasks")
    parser.add_argument("--plot_individual_tasks", action="store_true")
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    num_tasks = args.num_tasks
    num_runs = args.num_runs
    num_episodes = None
    legend_labels = []

    total_rewards_sum = {}
    total_steps_sum = {}
    automaton_learning_episodes = {}

    reward_fig_id, steps_fig_id = 0, 1

    with open(args.config) as f:
        config_obj = json.load(f)

    for task_id in range(num_tasks):
        if args.plot_individual_tasks:
            reward_fig, steps_fig = plt.figure(reward_fig_id), plt.figure(steps_fig_id)

        for setting in config_obj:
            if "folders" not in setting:
                raise AttributeError("Error: Missing field 'folder' in setting.")

            task_setting_folders = setting["folders"]

            if len(task_setting_folders) != num_runs:
                raise RuntimeError("There must be %d folders per setting!" % num_runs)

            setting_label = setting["label"]

            task_rewards, task_steps = None, None
            task_automaton_learning_episodes = []

            for folder in task_setting_folders:
                rewards_steps_folder_path = os.path.join(folder, ISAAlgorithm.REWARD_STEPS_FOLDER)
                rewards_steps_log_files = os.listdir(rewards_steps_folder_path)
                
                if len(rewards_steps_log_files) != num_tasks:
                    raise Exception("Error: The expected number of tasks is %d and was %d." % (num_tasks, len(rewards_steps_log_files)))

                try:
                    automaton_learning_episodes_path = os.path.join(folder, ISAAlgorithm.AUTOMATON_LEARNING_EPISODES_FILENAME)
                    current_automaton_learning_episodes = read_automaton_learning_episodes(automaton_learning_episodes_path)
                except IOError:
                    current_automaton_learning_episodes = []

                task_automaton_learning_episodes.extend(current_automaton_learning_episodes)

                if setting_label in automaton_learning_episodes:
                    automaton_learning_episodes[setting_label].update(current_automaton_learning_episodes)
                else:
                    automaton_learning_episodes[setting_label] = set(current_automaton_learning_episodes)

                rewards_steps_log_file = os.path.join(folder, ISAAlgorithm.REWARD_STEPS_FOLDER, ISAAlgorithm.REWARD_STEPS_FILENAME % task_id)
                rewards, steps = read_reward_steps_list(rewards_steps_log_file)
                num_episodes = len(rewards)

                np_rewards = np.array(rewards)

                if task_rewards is None:
                    task_rewards = np_rewards.copy()
                else:
                    task_rewards += np_rewards

                if setting_label in total_rewards_sum:
                    total_rewards_sum[setting_label] += np_rewards
                else:
                    total_rewards_sum[setting_label] = np_rewards.copy()

                np_steps = np.array(steps)

                if task_steps is None:
                    task_steps = np_steps.copy()
                else:
                    task_steps += np_steps

                if setting_label in total_steps_sum:
                    total_steps_sum[setting_label] += np_steps
                else:
                    total_steps_sum[setting_label] = np_steps.copy()

            if args.plot_individual_tasks:
                plot_single_task_curve(reward_fig_id, num_episodes, task_rewards / num_runs, task_automaton_learning_episodes, setting["colour"], setting_label)
                plot_single_task_curve(steps_fig_id, num_episodes, task_steps / num_runs, task_automaton_learning_episodes, setting["colour"], setting_label)

        if args.plot_individual_tasks:
            save_reward_plot(reward_fig_id, task_id)
            save_steps_plot(steps_fig_id, task_id)
            plt.close(reward_fig)
            plt.close(steps_fig)

        print("Finished processing task " + str(task_id))

    # save plot averages
    reward_fig, steps_fig = plt.figure(reward_fig_id), plt.figure(steps_fig_id)
    for setting in config_obj:
        setting_label = setting["label"]

        reward_mean, steps_mean = total_rewards_sum[setting_label] / (num_tasks * num_runs), total_steps_sum[setting_label] / (num_tasks * num_runs)

        reward_mean, _ = moving_average(reward_mean, 10)
        steps_mean, _ = moving_average(steps_mean, 10)

        plot_curve(reward_fig_id, num_episodes, reward_mean, automaton_learning_episodes[setting_label], setting["colour"], setting_label)
        plot_curve(steps_fig_id, num_episodes, steps_mean, automaton_learning_episodes[setting_label], setting["colour"], setting_label)

    save_reward_plot(reward_fig_id, "avg")
    save_steps_plot(steps_fig_id, "avg")
    plt.close(reward_fig)
    plt.close(steps_fig)
