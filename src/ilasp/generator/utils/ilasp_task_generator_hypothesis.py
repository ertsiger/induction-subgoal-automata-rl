import os
from ilasp.ilasp_common import N_TRANSITION_STR, OBS_STR, CONNECTED_STR, generate_injected_statement
from ilasp.generator.utils.ilasp_task_generator_state import get_state_names
from ilasp.parser.ilasp_parser_utils import parse_edge_rule, parse_negative_transition_rule
from ilasp.solver.ilasp_solver import solve_ilasp_task

MAX_PENALTY = 100
TMP_OUTPUT_FILENAME = "tmp_task.las"
TMP_SEARCH_SPACE_FILENAME = "search_space.txt"


def get_hypothesis_space(num_states, accepting_state, rejecting_state, observables, output_folder,
                         symmetry_breaking_method, max_disj_size, learn_acyclic, binary_folder_name):
    tmp_output_path = os.path.join(output_folder, TMP_OUTPUT_FILENAME)
    tmp_search_space_path = os.path.join(output_folder, TMP_SEARCH_SPACE_FILENAME)

    # generate an ILASP task from which we can derivate the entire search space, which is needed
    # to add the injections afterwards
    _generate_base_ilasp_task(num_states, accepting_state, rejecting_state, observables, tmp_output_path,
                              symmetry_breaking_method, max_disj_size, learn_acyclic)

    # get the search space for the given ILASP task
    solve_ilasp_task(tmp_output_path, tmp_search_space_path, operation="search_space",
                     binary_folder_name=binary_folder_name)

    # return the new hypothesis space with the injections to pos and neg atoms
    return _get_hypothesis_space_with_injections(tmp_search_space_path)


def _generate_base_ilasp_task(num_states, accepting_state, rejecting_state, observables, output_path,
                              symmetry_breaking_method, max_disj_size, learn_acyclic):
    with open(output_path, 'w') as f:
        task = _generate_state_constants(num_states, accepting_state, rejecting_state)
        task += _generate_observables_constants(observables)
        task += _generate_edge_indices_constants(max_disj_size)
        task += _generate_hypothesis_space(num_states, accepting_state, rejecting_state, symmetry_breaking_method,
                                               max_disj_size, learn_acyclic)
        f.write(task)


def _generate_state_constants(num_states, accepting_state, rejecting_state):
    state_constants = ["#constant(node, " + s + ").\n" for s in get_state_names(num_states, accepting_state, rejecting_state)]
    return "".join(state_constants) + '\n'


def _generate_observables_constants(observables):
    observables_constants = ["#constant(observable, \"%s\").\n" % o for o in observables]
    return "".join(observables_constants) + '\n'


def _generate_edge_indices_constants(max_disj_size):
    edge_indices_constants = ["#constant(edge_id, %d).\n" % i for i in range(1, max_disj_size + 1)]
    return "".join(edge_indices_constants) + '\n'


def _generate_hypothesis_space(num_states, accepting_state, rejecting_state, symmetry_breaking_method, max_disj_size,
                               learn_acyclic):
    stmt = _generate_mode_bias(accepting_state, rejecting_state)

    # add hypothesis rules that say two states are connected
    stmt += _generate_connected_hypothesis_rules(num_states, accepting_state, rejecting_state,
                                                 symmetry_breaking_method, max_disj_size, learn_acyclic)
    return stmt


def _generate_mode_bias(accepting_state, rejecting_state):
    # maximum number of variables in each rule (each transition rule can only refer to one particular index)
    stmt = "#maxv(1).\n\n"
    stmt += "#modeh(%s(const(node), const(node), const(edge_id), var(step))).\n" % N_TRANSITION_STR
    stmt += "#modeb(%s(const(observable), var(step))).\n\n" % OBS_STR

    # rule for forcing observables to be mentioned in every rule
    stmt += "#bias(\":- not body(%s(_, _)), not body(naf(%s(_, _))).\").\n\n" % (OBS_STR, OBS_STR)

    # avoid learning rules to transition to itself
    stmt += "#bias(\":- head(%s(X, X, _, _)).\").\n" % N_TRANSITION_STR

    # avoid learning transitions from accepting and rejecting states
    stmt += "#bias(\":- head(%s(%s, _, _, _)).\").\n" % (N_TRANSITION_STR, accepting_state)
    if rejecting_state is not None:
        stmt += "#bias(\":- head(%s(%s, _, _, _)).\").\n" % (N_TRANSITION_STR, rejecting_state)
    stmt += "\n"
    return stmt


def _generate_connected_hypothesis_rules(num_states, accepting_state, rejecting_state, symmetry_breaking_method,
                                         max_disj_size, learn_acyclic):
    # connected = auxiliary facts of cost 2 to minimize the number of transitions
    stmt = ""
    states = get_state_names(num_states, accepting_state, rejecting_state)
    for s1 in [s for s in states if s != accepting_state and s != rejecting_state]:
        if learn_acyclic and symmetry_breaking_method == "increasing_path":
            # if we learn acyclic graphs and use the symmetry breaking increasing path, then we can remove some of
            # the rules of the hypothesis space: those that connect a state with one with lower id
            target_states = [states[i] for i in range(0, len(states)) if i > states.index(s1)]
        else:
            target_states = [s for s in states if s != s1]

        for s2 in target_states:
            for i in range(1, max_disj_size + 1):
                stmt += "2 ~ %s(%s, %s, %d).\n" % (CONNECTED_STR, s1, s2, i)
    stmt += "\n"
    return stmt


def _get_hypothesis_space_with_injections(hypothesis_space_filename):
    hypothesis_space = ["#max_penalty(%d).\n" % MAX_PENALTY]
    with open(hypothesis_space_filename) as f:
        counter = 0
        for line in f:
            line = line.strip()
            hypothesis_space.append(line)

            line = line.strip("2 ~ ")
            if line.startswith(N_TRANSITION_STR):
                hypothesis_space.append(_get_negative_transition_injection(parse_negative_transition_rule(line), counter))
            elif line.startswith(CONNECTED_STR):
                hypothesis_space.append(_get_edge_injection(parse_edge_rule(line), counter))

            counter += 1
    return "\n".join(hypothesis_space) + "\n\n"


def _get_negative_transition_injection(parsed_transition, counter):
    injection_str = ""
    for p in parsed_transition.pos:
        injection_str += _get_negative_transition_injection_helper(parsed_transition, p, True, counter)
    for n in parsed_transition.neg:
        injection_str += _get_negative_transition_injection_helper(parsed_transition, n, False, counter)
    return injection_str


def _get_negative_transition_injection_helper(parsed_transition, symbol, is_pos_obs, counter):
    predicate_name = "injected_"
    if is_pos_obs:  # remember it is a negative transition and, thus, observables have to be inverted
        predicate_name += "neg"
    else:
        predicate_name += "pos"
    return generate_injected_statement("%s(%s, %s, %s, %s) :- active(%d)." % (predicate_name, parsed_transition.src,
                                                                              parsed_transition.dst,
                                                                              parsed_transition.edge, symbol, counter))


def _get_edge_injection(parsed_edge, counter):
    return generate_injected_statement("injected_%s(%s, %s, %s) :- active(%d)." % (CONNECTED_STR, parsed_edge.src,
                                                                                   parsed_edge.dst, parsed_edge.edge,
                                                                                   counter))
