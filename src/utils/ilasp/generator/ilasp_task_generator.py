from utils.ilasp.common import *


def generate_ilasp_task(num_states, accepting_state, rejecting_state, fluents, pos_examples, neg_examples, inc_examples,
                        output_filename, symmetry_breaking_method=None, max_disj_size=1, learn_acyclic=True):
    with open(output_filename, 'w') as f:
        task = _generate_state_statements(num_states, accepting_state, rejecting_state)
        task += _generate_fluent_constants(fluents)
        task += _generate_timestep_statements(pos_examples, neg_examples, inc_examples)
        task += _generate_state_at_timestep_statements(num_states, accepting_state, rejecting_state)
        task += _generate_transition_statements(num_states, accepting_state, rejecting_state, max_disj_size, learn_acyclic)

        if symmetry_breaking_method is not None:
            task += _generate_state_priority_statements(num_states, accepting_state, rejecting_state)
            task += _generate_symmetry_breaking_statements(symmetry_breaking_method, learn_acyclic)

        task += _generate_examples(pos_examples, neg_examples, inc_examples)
        f.write(task)


def _generate_state_statements(num_states, accepting_state, rejecting_state):
    states = _get_state_names(num_states, accepting_state, rejecting_state)
    states_str = _generate_state_predicates(states)
    states_str += _generate_state_constants(states)
    return states_str


def _get_state_names(num_states, accepting_state, rejecting_state):
    states = ["s" + str(i) for i in range(num_states - 2)]
    states.append(accepting_state)
    states.append(rejecting_state)
    return states


def _generate_state_predicates(states):
    states_str = ""
    for s in states:
        states_str += "state(" + s + ").\n"
    states_str += "\n"
    return states_str


def _generate_state_constants(states):
    states_str = ""
    for s in states:
        states_str += "#constant(node, " + s + ").\n"
    states_str += "\n"
    return states_str


def _generate_state_priority_statements(num_states, accepting_state, rejecting_state):
    states = _get_state_names(num_states, accepting_state, rejecting_state)
    states_str = ""

    for i in range(0, len(states)):
        states_str += "state_id(%s, %d).\n" % (states[i], i)

    states_str += "\n"
    return states_str


def _generate_fluent_constants(fluents):
    fluents_str = ""
    for f in fluents:
        fluents_str += "#constant(character, \"%s\").\n" % f
    fluents_str += "\n"
    return fluents_str


def _generate_timestep_statements(pos_examples, neg_examples, inc_examples):
    stmt = "all_steps(0..%d).\n" % _get_longest_example_length(pos_examples, neg_examples, inc_examples)
    stmt += "step(T) :- all_steps(T), last(U), T<U+2.\n\n"
    return stmt


def _get_longest_example_length(pos_examples, neg_examples, inc_examples):
    max_pos = len(max(pos_examples, key=len)) if len(pos_examples) > 0 else 0
    max_neg = len(max(neg_examples, key=len)) if len(neg_examples) > 0 else 0
    max_inc = len(max(inc_examples, key=len)) if len(inc_examples) > 0 else 0
    return max(max_pos, max_neg, max_inc)


def _generate_state_at_timestep_statements(num_states, accepting_state, rejecting_state):
    stmt = _generate_initial_state_at_timestep(num_states)
    stmt += "st(T+1, Y) :- st(T, X), delta(X, Y, _, T), X!=%s, X!=%s.\n\n" % (accepting_state, rejecting_state)
    stmt += "accept :- last(T), st(T+1, %s).\n" % accepting_state
    stmt += "reject :- last(T), st(T+1, %s).\n\n" % rejecting_state
    return stmt


def _generate_initial_state_at_timestep(num_states):
    if num_states >= 3:
        stmt = "st(0, s0).\n"
    else:
        raise ValueError("The number of states should be >= 3.")
    return stmt


def _generate_transition_statements(num_states, accepting_state, rejecting_state, max_disj_size, learn_acyclic):
    stmt = _generate_edge_indices_constants(max_disj_size)

    # transitions are defined as the negative of the n_transitions (negative transitions)
    stmt += "external_delta(X, T) :- delta(X, Y, _, T), X!=Y.\n"
    stmt += "delta(X, X, 1, T) :- not external_delta(X, T), state(X), step(T).\n"
    stmt += "delta(X, Y, E, T) :- %s(X, Y, E), not %s(X, Y, E, T), step(T).\n\n" % (CONNECTED_STR, N_TRANSITION_STR)

    # determinism
    stmt += ":- delta(X, Y, _, T), delta(X, Z, _, T), Y!=Z.\n\n"

    # all non-terminal states must be connected to some other one
    stmt += ":- not %s(X, _, _), state(X), X!=%s, X!=%s.\n\n" % (CONNECTED_STR, accepting_state, rejecting_state)

    if learn_acyclic:
        stmt += "path(X, Y) :- %s(X, Y, _).\n" % CONNECTED_STR
        stmt += "path(X, Y) :- %s(X, Z, _), path(Z, Y).\n" % CONNECTED_STR
        stmt += ":- path(X, Y), path(Y, X).\n\n"

    # maximum number of variables in each rule (each transition rule can only refer to one particular index)
    stmt += "#maxv(1).\n\n"
    stmt += "#max_penalty(100).\n\n"

    stmt += "#modeh(%s(const(node), const(node), const(edge_id), var(step))).\n" % N_TRANSITION_STR
    stmt += "#modeb(%s(const(character), var(step))).\n\n" % OBS_STR

    # rule for forcing fluents to be mentioned in every rule
    stmt += "#bias(\":- not body(%s(_, _)), not body(naf(%s(_, _))).\").\n\n" % (OBS_STR, OBS_STR)

    # avoid learning rules to transition to itself
    stmt += "#bias(\":- head(%s(X, X, _, _)).\").\n" % N_TRANSITION_STR

    # avoid learning transitions from accepting and rejecting states
    stmt += "#bias(\":- head(%s(%s, _, _, _)).\").\n" % (N_TRANSITION_STR, accepting_state)
    stmt += "#bias(\":- head(%s(%s, _, _, _)).\").\n" % (N_TRANSITION_STR, rejecting_state)
    stmt += "\n"

    # connected = auxiliary facts of cost 2 to minimize the number of transitions
    states = _get_state_names(num_states, accepting_state, rejecting_state)
    for s1 in [s for s in states if s != accepting_state and s != rejecting_state]:
        for s2 in [s for s in states if s != s1]:
            for i in range(1, max_disj_size + 1):
                stmt += "2 ~ %s(%s, %s, %d).\n" % (CONNECTED_STR, s1, s2, i)

    stmt += "\n"

    return stmt


def _generate_edge_indices_constants(max_disj_size):
    edge_indices_str = "edge_id(1..%d).\n\n" % max_disj_size

    for i in range(1, max_disj_size + 1):
        edge_indices_str += "#constant(edge_id, %d).\n" % i

    edge_indices_str += "\n"

    return edge_indices_str


def _generate_symmetry_breaking_statements(symmetry_breaking_method, learn_acyclic):
    if symmetry_breaking_method == "increasing_path":
        if learn_acyclic:
            return _generate_increasing_acylic_path_symmetry_breaking_constraints()
        return _generate_increasing_cyclic_path_symmetry_breaking_constraints()
    else:
        raise RuntimeError("Error: Unknown symmetry breaking method \"%s\"" % symmetry_breaking_method)


def _generate_increasing_cyclic_path_symmetry_breaking_constraints():
    # NOTE: this method is not sound for graphs with cycles (although it considers them, there are some cases where it
    #       prunes candidate graphs that should not be pruned!)
    stmt = "first(0, s0).\n"
    stmt += "has_been_visited(T, STATE) :- st(T, STATE).\n"
    stmt += "has_been_visited(T, STATE) :- has_been_visited(T2, STATE), step(T), step(T2), T=T2+1.\n"
    stmt += "first(T, STATE) :- st(T, STATE), not has_been_visited(T2, STATE), step(T), step(T2), T=T2+1.\n"
    stmt += ":- first(T, STATE), first(T2, STATE2), T2>T, STATE!=STATE2, state_id(STATE, ID), state_id(STATE2, ID2), ID>ID2.\n"
    stmt += "\n"

    return stmt


def _generate_increasing_acylic_path_symmetry_breaking_constraints():
    return ":- st(TX, X), st(TY, Y), state_id(X, IDX), state_id(Y, IDY), TX<TY, IDX>IDY.\n\n"


def _generate_examples(pos_examples, neg_examples, inc_examples):
    examples = _generate_positive_examples(pos_examples)
    examples += _generate_negative_examples(neg_examples)
    examples += _generate_incomplete_examples(inc_examples)
    return examples


def _generate_positive_examples(examples):
    example_str = ""
    for example in examples:
        example_str += "#pos({accept}, {reject}, {\n"
        example_str += _generate_example(example)
        example_str += "}).\n\n"
    return example_str


def _generate_negative_examples(examples):
    example_str = ""
    for example in examples:
        example_str += "#pos({reject}, {accept}, {\n"
        example_str += _generate_example(example)
        example_str += "}).\n\n"
    return example_str


def _generate_incomplete_examples(examples):
    example_str = ""
    for example in examples:
        example_str += "#pos({}, {accept, reject}, {\n"
        example_str += _generate_example(example)
        example_str += "}).\n\n"
    return example_str


def _generate_example(example):
    example_str = "    "
    first = True

    for i in range(0, len(example)):
        for symbol in example[i]:
            if not first:
                example_str += " "
            example_str += "%s(\"%s\", %d)." % (OBS_STR, symbol, i)
            first = False

    if len(example) > 0:
        example_str += "\n"

    example_str += "    last(%d).\n" % (len(example) - 1)

    return example_str
