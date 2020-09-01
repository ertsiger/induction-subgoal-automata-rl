import argparse
from utils import utils
from ilasp.generator.ilasp_task_generator import generate_ilasp_task
from ilasp.solver.ilasp_solver import solve_ilasp_task
from ilasp.parser.ilasp_solution_parser import parse_ilasp_solutions


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_config", help="json file containing number of states, observables and examples")
    parser.add_argument("task_filename", help="filename of the ILASP task")
    parser.add_argument("solution_filename", help="filename of the ILASP task solution")
    parser.add_argument("plot_filename", help="filename of the automaton plot")
    parser.add_argument("--symmetry_breaking_method", "-s", default=None, help="method for symmetry breaking (bfs, increasing_path)")
    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    config = utils.read_json_file(args.task_config)

    generate_ilasp_task(config["num_states"], "u_acc", "u_rej", config["observables"], config["goal_examples"],
                        config["deadend_examples"], config["inc_examples"], ".", args.task_filename,
                        args.symmetry_breaking_method, config["max_disjunction_size"], config["learn_acyclic"],
                        config["use_compressed_traces"], config["avoid_learning_only_negative"],
                        config["prioritize_optimal_solutions"], binary_folder_name="../bin")

    solve_ilasp_task(args.task_filename, args.solution_filename, binary_folder_name="../bin")
    automaton = parse_ilasp_solutions(args.solution_filename)
    automaton.plot(".", args.plot_filename)

'''
Configuration File Example:

{
    "num_states": 6,
    "max_disjunction_size": 1,
    "learn_acyclic": true,
    "use_compressed_traces": true,
    "avoid_learning_only_negative": false,
    "prioritize_optimal_solutions": false,
    "observables": ["a", "b", "c", "d", "f", "g", "m", "n"],
    "goal_examples": [
        [["f"], ["m"], ["g"]],
        [["m"], ["f"], ["g"]],
        [["m", "f"], ["g"]],
        [["m", "f", "g"]],
        [["m"], ["f", "g"]],
        [["f"], ["m", "g"]]
    ],
    "deadend_examples": [
        [["n"]],
        [["f"], ["n"]],
        [["m"], ["n"]],
        [["f"], ["m"], ["n"]],
        [["m"], ["f"], ["n"]]
    ],
    "inc_examples": [
        [["f"]],
        [["g"]],
        [["m"]],
        [[], []],
        [[], ["g"]],
        [[], ["m"]],
        [[], ["f"]],
        [["m"], []],
        [["f"], []],
        [["m"], ["g"]],
        [["m"], [], ["g"]],
        [["f"], ["g"]],
        [["f"], [], ["g"]],
        [[], ["f"], ["g"]],
        [[], ["m"], ["g"]],
        [[], [], []],
        [[], ["f"], []],
        [[], ["m"], []],
        [["f"], [], []],
        [["m"], [], []],
        [[], [], ["g"]],
        [["f"], ["m"]],
        [["f"], ["m"], []],
        [["m"], ["f"], []],
        [["m", "f"]],
        [["g", "f"]],
        [["g", "m"]]
    ]
}
'''