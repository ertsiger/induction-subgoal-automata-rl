import argparse
import os
import numpy as np
from scipy import stats
import json
from tqdm import tqdm

from reinforcement_learning.isa_qrm_algorithm import ISAAlgorithmBase
from utils.utils import path_exists


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="config file")
    parser.add_argument("output_file", help="output file")
    return parser


def get_task_solving_time(ilasp_solution_file_path):
    """Returns the time that ILASP needed to find a solution to a given task."""
    with open(ilasp_solution_file_path) as f:
        lines = f.readlines()
        last_line = lines[-1]
        _, time = last_line.split(":")
        time = time.strip()[:-1]
        return float(time)


def _get_ilasp_tasks_stats(results_path):
    """Returns a dictionary with information about the ILASP tasks in the automaton_tasks directory (number of
    examples, maximum length of an example and an array with all the example lengths."""
    ilasp_tasks_folder = os.path.join(results_path, ISAAlgorithmBase.AUTOMATON_TASK_FOLDER)
    task_files = [f for f in os.listdir(ilasp_tasks_folder) if os.path.isfile(os.path.join(ilasp_tasks_folder, f)) and f.startswith("task-")]
    task_files.sort(key=lambda x: int(x[:-len(".las")].split("-")[1]))

    ilasp_tasks_stats = []

    for task_file in task_files:
        num_goal, num_dend, num_inc = 0, 0, 0
        example_lengths, goal_example_lengths, dend_example_lengths, inc_example_lengths = [], [], [], []

        with open(os.path.join(ilasp_tasks_folder, task_file)) as f:
            last_example_type = None
            for line in f:
                line_s = line.strip()
                if line_s.startswith("#pos({accept}"):
                    last_example_type = "goal"
                elif line_s.startswith("#pos({reject}"):
                    last_example_type = "dend"
                elif line_s.startswith("#pos({}"):
                    last_example_type = "inc"
                elif line_s.startswith("last("):
                    example_length = float(line_s[len("last("):-len(").")]) + 1
                    if last_example_type == "goal":
                        num_goal += 1
                        goal_example_lengths.append(example_length)
                    elif last_example_type == "dend":
                        num_dend += 1
                        dend_example_lengths.append(example_length)
                    elif last_example_type == "inc":
                        num_inc += 1
                        inc_example_lengths.append(example_length)
                    example_lengths.append(example_length)
            ilasp_tasks_stats.append(dict(num_goal=num_goal, num_dend=num_dend, num_inc=num_inc,
                                          max_example_length=np.max(example_lengths), example_lengths=example_lengths,
                                          goal_example_lengths=goal_example_lengths, dend_example_lengths=dend_example_lengths,
                                          inc_example_lengths=inc_example_lengths))

    return ilasp_tasks_stats


def _get_ilasp_solution_times(results_path):
    """Returns an array with all the ILASP running times for each of the tasks (except for those UNSAT, where the time
    is negligible so we set it to -1 to ignore it later)."""
    ilasp_solutions_folder = os.path.join(results_path, ISAAlgorithmBase.AUTOMATON_SOLUTION_FOLDER)
    solution_files = [f for f in os.listdir(ilasp_solutions_folder) if
                      os.path.isfile(os.path.join(ilasp_solutions_folder, f))]
    solution_files.sort(key=lambda x: int(x[:-len(".txt")].split("-")[1]))

    solution_times = []

    for sol_file in solution_files:
        sol_file_path = os.path.join(ilasp_solutions_folder, sol_file)
        try:
            time = get_task_solving_time(sol_file_path)
            solution_times.append(time)
        except:
            # sol_file_id = int(sol_file[:-len(".txt")].split("-")[1])
            # print("Task %d was UNSAT." % sol_file_id)
            solution_times.append(-1)

    return np.array(solution_times)


def _get_absolute_running_time(results_path):
    """Returns the total running time during which ISA run (reinforcement learning + automata learning)."""
    abs_time_file = os.path.join(results_path, ISAAlgorithmBase.ABSOLUTE_RUNNING_TIME_FILENAME)

    with open(abs_time_file) as f:
        running_time = float(f.readline())
        return running_time


def _has_run_finished(results_path):
    abs_time_file = os.path.join(results_path, ISAAlgorithmBase.ABSOLUTE_RUNNING_TIME_FILENAME)
    return path_exists(abs_time_file)


def _has_learned_automata(results_path):
    ilasp_tasks_folder = os.path.join(results_path, ISAAlgorithmBase.AUTOMATON_TASK_FOLDER)
    ilasp_solutions_folder = os.path.join(results_path, ISAAlgorithmBase.AUTOMATON_SOLUTION_FOLDER)
    return path_exists(ilasp_solutions_folder) and path_exists(ilasp_tasks_folder)


def _mean_std(arr):
    return np.round(np.mean(arr), 1), np.round(np.std(arr), 1)


def _mean_sem(arr):
    return np.round(np.mean(arr), 1), np.round(stats.sem(arr), 1)


def analyse_solutions(setting_name, results_paths):
    num_examples, num_goal, num_dend, num_inc = [], [], [], []
    total_solution_times, ilasp_last_time, all_solution_times, absolute_times = [], [], [], []
    ilasp_percent_times = []
    max_example_lengths, all_example_lengths, all_goal_example_lengths, all_dend_example_lengths, all_inc_example_lengths = [], [], [], [], []
    num_completed_runs = 0
    num_completed_and_started_automata_learning_runs = 0  # number of runs that found the goal at least once

    for i in tqdm(range(len(results_paths)), desc=setting_name):  # iterate through different runs
        results_path = results_paths[i]

        if _has_run_finished(results_path):
            num_completed_runs += 1
            if _has_learned_automata(results_path):
                ilasp_tasks_stats = _get_ilasp_tasks_stats(results_path)
                last_stats = ilasp_tasks_stats[-1]
                num_goal.append(last_stats["num_goal"])
                num_dend.append(last_stats["num_dend"])
                num_inc.append(last_stats["num_inc"])
                num_examples.append(num_goal[-1] + num_dend[-1] + num_inc[-1])
                max_example_lengths.append(last_stats["max_example_length"])
                all_example_lengths.extend(last_stats["example_lengths"])
                all_goal_example_lengths.extend(last_stats["goal_example_lengths"])
                all_dend_example_lengths.extend(last_stats["dend_example_lengths"])
                all_inc_example_lengths.extend(last_stats["inc_example_lengths"])

                # absolute running time
                absolute_time = _get_absolute_running_time(results_path)
                absolute_times.append(absolute_time)

                solution_times = _get_ilasp_solution_times(results_path)
                solution_found_times = solution_times[solution_times >= 0]
                ilasp_total_time = np.sum(solution_found_times)
                total_solution_times.append(ilasp_total_time)
                ilasp_last_time.append(solution_found_times[-1])
                all_solution_times.extend(solution_found_times)
                ilasp_percent_times.append(100.0 * ilasp_total_time / absolute_time)

                num_completed_and_started_automata_learning_runs += 1

    if num_completed_and_started_automata_learning_runs > 0:
        return {
            "num_examples": _mean_sem(num_examples),
            "num_goal_examples": _mean_sem(num_goal),
            "num_dend_examples": _mean_sem(num_dend),
            "num_inc_examples": _mean_sem(num_inc),
            "absolute_time": _mean_sem(absolute_times),
            "num_completed_runs": num_completed_runs,
            "num_found_goal_runs": num_completed_and_started_automata_learning_runs,
            "ilasp_total_time": _mean_sem(total_solution_times),
            "ilasp_percent_time": _mean_sem(ilasp_percent_times),
            "ilasp_last_time": _mean_sem(ilasp_last_time),
            "avg_time_per_automaton": _mean_sem(all_solution_times),
            "max_example_length": np.max(max_example_lengths),
            "example_length": _mean_std(all_example_lengths),
            "example_length_goal": _mean_std(all_goal_example_lengths),
            "example_length_dend": _mean_std(all_dend_example_lengths),
            "example_length_inc": _mean_std(all_inc_example_lengths)
        }
    return {}


if __name__ == "__main__":
    args = get_parser().parse_args()

    with open(args.config_file) as f:
        config_dict = json.load(f)

    results = {}
    for item in config_dict:
        results[item] = analyse_solutions(item, config_dict[item])

    with open(args.output_file, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
