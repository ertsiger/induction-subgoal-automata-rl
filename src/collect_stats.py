import argparse
import os
import numpy as np
from scipy import stats
import json

from reinforcement_learning.isa_algorithm import ISAAlgorithm


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
    ilasp_tasks_folder = os.path.join(results_path, ISAAlgorithm.AUTOMATON_TASK_FOLDER)
    task_files = [f for f in os.listdir(ilasp_tasks_folder) if os.path.isfile(os.path.join(ilasp_tasks_folder, f))]
    task_files.sort(key=lambda x: int(x[:-len(".las")].split("-")[1]))

    ilasp_tasks_stats = []

    for task_file in task_files:
        num_pos, num_neg, num_inc = 0, 0, 0
        example_lengths = []

        with open(os.path.join(ilasp_tasks_folder, task_file)) as f:
            for line in f:
                line_s = line.strip()
                if line_s.startswith("#pos({accept}, {reject}"):
                    num_pos += 1
                elif line_s.startswith("#pos({reject}, {accept}"):
                    num_neg += 1
                elif line_s.startswith("#pos({}, {accept, reject}"):
                    num_inc += 1
                elif line_s.startswith("last("):
                    example_length = float(line_s[len("last("):-len(").")])
                    example_lengths.append(example_length)

            ilasp_tasks_stats.append(dict(num_pos=num_pos, num_neg=num_neg, num_inc=num_inc,
                                          max_example_length=np.max(example_lengths), example_lengths=example_lengths))

    return ilasp_tasks_stats


def _get_ilasp_solution_times(results_path):
    """Returns an array with all the ILASP running times for each of the tasks (except for those UNSAT, where the time
    is negligible so we set it to -1 to ignore it later)."""
    ilasp_solutions_folder = os.path.join(results_path, ISAAlgorithm.AUTOMATON_SOLUTION_FOLDER)
    solution_files = [f for f in os.listdir(ilasp_solutions_folder) if
                      os.path.isfile(os.path.join(ilasp_solutions_folder, f))]
    solution_files.sort(key=lambda x: int(x[:-len(".txt")].split("-")[1]))

    solution_times = []

    for sol_file in solution_files:
        sol_file_id = int(sol_file[:-len(".txt")].split("-")[1])
        sol_file_path = os.path.join(ilasp_solutions_folder, sol_file)
        try:
            time = get_task_solving_time(sol_file_path)
            solution_times.append(time)
        except ValueError:
            # print("Task %d was UNSAT." % sol_file_id)
            solution_times.append(-1)

    return np.array(solution_times)


def _get_absolute_running_time(results_path):
    """Returns the total running time during which ISA run (reinforcement learning + automata learning)."""
    abs_time_file = os.path.join(results_path, ISAAlgorithm.ABSOLUTE_RUNNING_TIME_FILENAME)

    with open(abs_time_file) as f:
        running_time = float(f.readline())
        return running_time


def analyse_solutions(results_paths):
    num_examples, num_pos, num_neg, num_inc = [], [], [], []
    total_solution_times, ilasp_last_time, all_solution_times, absolute_times = [], [], [], []
    ilasp_percent_times = []
    max_example_lengths, all_example_lengths = [], []

    for i in range(len(results_paths)):  # iterate through different runs
        results_path = results_paths[i]

        # number of examples
        ilasp_tasks_stats = _get_ilasp_tasks_stats(results_path)
        last_stats = ilasp_tasks_stats[-1]
        num_pos.append(last_stats["num_pos"])
        num_neg.append(last_stats["num_neg"])
        num_inc.append(last_stats["num_inc"])
        num_examples.append(num_pos[-1] + num_neg[-1] + num_inc[-1])
        max_example_lengths.append(last_stats["max_example_length"])
        all_example_lengths.extend(last_stats["example_lengths"])

        # time
        solution_times = _get_ilasp_solution_times(results_path)
        absolute_time = _get_absolute_running_time(results_path)
        absolute_times.append(absolute_time)
        solution_found_times = solution_times[solution_times >= 0]
        ilasp_total_time = np.sum(solution_found_times)
        total_solution_times.append(ilasp_total_time)
        ilasp_last_time.append(solution_found_times[-1])
        all_solution_times.extend(solution_found_times)
        ilasp_percent_times.append(100.0 * ilasp_total_time / absolute_time)

    return {
        "num_examples": (np.mean(num_examples), stats.sem(num_examples)),
        "num_pos_examples": (np.mean(num_pos), stats.sem(num_pos)),
        "num_neg_examples": (np.mean(num_neg), stats.sem(num_neg)),
        "num_inc_examples": (np.mean(num_inc), stats.sem(num_inc)),
        "absolute_time": (np.mean(absolute_times), stats.sem(absolute_times)),
        "ilasp_total_time": (np.mean(total_solution_times), stats.sem(total_solution_times)),
        "ilasp_percent_time": (np.mean(ilasp_percent_times), stats.sem(ilasp_percent_times)),
        "ilasp_last_time": (np.mean(ilasp_last_time), stats.sem(ilasp_last_time)),
        "avg_time_per_automaton": (np.mean(all_solution_times), stats.sem(all_solution_times)),
        "max_example_length": np.max(max_example_lengths),
        "example_length": (np.mean(all_example_lengths), np.std(all_example_lengths))}


if __name__ == "__main__":
    args = get_parser().parse_args()

    with open(args.config_file) as f:
        config_dict = json.load(f)

    results = {}
    for item in config_dict:
        results[item] = analyse_solutions(config_dict[item])

    with open(args.output_file, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


