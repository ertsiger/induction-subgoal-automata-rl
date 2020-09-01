import os
from ilasp.generator.utils.ilasp_task_generator_example import generate_examples
from ilasp.generator.utils.ilasp_task_generator_hypothesis import get_hypothesis_space
from ilasp.generator.utils.ilasp_task_generator_state import generate_state_statements
from ilasp.generator.utils.ilasp_task_generator_symmetry_breaking import generate_symmetry_breaking_statements
from ilasp.generator.utils.ilasp_task_generator_transition import generate_timestep_statements, generate_state_at_timestep_statements, generate_transition_statements


def generate_ilasp_task(num_states, accepting_state, rejecting_state, observables, goal_examples, dend_examples,
                        inc_examples, output_folder, output_filename, symmetry_breaking_method, max_disj_size,
                        learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                        prioritize_optimal_solutions, binary_folder_name):
    # statements will not be generated for the rejecting state if there are not deadend examples
    if len(dend_examples) == 0:
        rejecting_state = None

    with open(os.path.join(output_folder, output_filename), 'w') as f:
        task = _generate_ilasp_task_str(num_states, accepting_state, rejecting_state, observables, goal_examples,
                                        dend_examples, inc_examples, output_folder, symmetry_breaking_method,
                                        max_disj_size, learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                                        prioritize_optimal_solutions, binary_folder_name)
        f.write(task)


def _generate_ilasp_task_str(num_states, accepting_state, rejecting_state, observables, goal_examples, dend_examples,
                             inc_examples, output_folder, symmetry_breaking_method, max_disj_size, learn_acyclic,
                             use_compressed_traces, avoid_learning_only_negative, prioritize_optimal_solutions, binary_folder_name):
    task = generate_state_statements(num_states, accepting_state, rejecting_state)
    task += generate_timestep_statements(goal_examples, dend_examples, inc_examples)
    task += _generate_edge_indices_facts(max_disj_size)
    task += generate_state_at_timestep_statements(num_states, accepting_state, rejecting_state)
    task += generate_transition_statements(learn_acyclic, use_compressed_traces, avoid_learning_only_negative, prioritize_optimal_solutions)
    task += get_hypothesis_space(num_states, accepting_state, rejecting_state, observables, output_folder,
                                 symmetry_breaking_method, max_disj_size, learn_acyclic, binary_folder_name)

    if symmetry_breaking_method is not None:
        task += generate_symmetry_breaking_statements(num_states, accepting_state, rejecting_state, observables,
                                                      symmetry_breaking_method, max_disj_size, learn_acyclic)

    task += generate_examples(goal_examples, dend_examples, inc_examples)
    return task


def _generate_edge_indices_facts(max_disj_size):
    return "edge_id(1..%d).\n\n" % max_disj_size
