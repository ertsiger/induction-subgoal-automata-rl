from ilasp.ilasp_common import CONNECTED_STR, N_TRANSITION_STR, generate_injected_block
from ilasp.generator.utils.ilasp_task_generator_example import get_longest_example_length


def generate_timestep_statements(goal_examples, dend_examples, inc_examples):
    stmt = "all_steps(0..%d).\n" % get_longest_example_length(goal_examples, dend_examples, inc_examples)
    stmt += "step(T) :- all_steps(T), last(U), T<U+1.\n\n"
    return stmt


def generate_state_at_timestep_statements(num_states, accepting_state, rejecting_state):
    stmt = _generate_initial_state_at_timestep(num_states)
    stmt += "st(T+1, Y) :- st(T, X), delta(X, Y, T).\n\n"
    stmt += _generate_acceptance_rejection_rules(accepting_state, rejecting_state)
    return stmt


def _generate_initial_state_at_timestep(num_states):
    if num_states >= 3:
        stmt = "st(0, u0).\n"
    else:
        raise ValueError("The number of states should be >= 3.")
    return stmt


def _generate_acceptance_rejection_rules(accepting_state, rejecting_state):
    # a trace is accepted if it is at the accepting state in the last timestep (or reject if it is at the rejecting
    # state in the last timestep)
    stmt = "accept :- last(T), st(T+1, %s).\n" % accepting_state
    if rejecting_state is not None:
        stmt += "reject :- last(T), st(T+1, %s).\n\n" % rejecting_state
    return stmt


def generate_transition_statements(learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                                   prioritize_optimal_solutions):
    # transitions are defined as the negative of the n_transitions (negative transitions)
    stmt = "phi(X, Y, E, T) :- not %s(X, Y, E, T), %s(X, Y, E), step(T).\n" % (N_TRANSITION_STR, CONNECTED_STR)
    stmt += "out_phi(X, T) :- phi(X, _, _, T).\n"
    stmt += "delta(X, Y, T) :- phi(X, Y, _, T).\n"
    stmt += "delta(X, X, T) :- not out_phi(X, T), state(X), step(T).\n\n"

    # all states must be reachable from the initial state
    stmt += "reachable(u0).\n"
    stmt += "reachable(Y) :- reachable(X), ed(X, Y, _).\n"
    stmt += ":- not reachable(X), state(X).\n\n"

    if learn_acyclic:
        stmt += "path(X, Y) :- %s(X, Y, _).\n" % CONNECTED_STR
        stmt += "path(X, Y) :- %s(X, Z, _), path(Z, Y).\n" % CONNECTED_STR
        stmt += ":- path(X, Y), path(Y, X).\n\n"

    # extra edge constraints (e.g., determinism)
    stmt += _generate_edge_constraints(learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                                       prioritize_optimal_solutions)
    return stmt


def _generate_edge_constraints(learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                               prioritize_optimal_solutions):
    # the same observation cannot appear positive and negative in the same edge
    # they are actually not needed since ILASP attempts to find a minimal hypothesis and this case clearly
    # will appear only in non-minimal hypothesis (since the transition is cannot be taken, it is equivalent
    # to the loop transition used when no outgoing formulas are satisfied)
    stmts = [":- injected_pos(X, Y, E, O), injected_neg(X, Y, E, O)."]

    # determinism: edges must be mutually exclusive
    stmts.append("injected_mutex(X, Y, EY, Z, EZ) :- injected_pos(X, Y, EY, O), injected_neg(X, Z, EZ, O), Y<Z.")
    stmts.append("injected_mutex(X, Y, EY, Z, EZ) :- injected_neg(X, Y, EY, O), injected_pos(X, Z, EZ, O), Y<Z.")
    stmts.append(":- not injected_mutex(X, Y, EY, Z, EZ), injected_ed(X, Y, EY), injected_ed(X, Z, EZ), Y<Z.")

    if use_compressed_traces:
        # avoid learning edges without conditions
        stmts.append(":- injected_ed(X, Y, E), not injected_pos(X, Y, E, _), not injected_neg(X, Y, E, _).")

    # avoid learning edges only labeled by negative conditions
    if avoid_learning_only_negative:
        stmts.append(":- injected_neg(X, Y, E, _), not injected_pos(X, Y, E, _).")

    # choose one of the optimal solutions based on minimizing the criteria below;
    # note that the main criteria for an optimal solution is a optimal hypothesis, the rules below are just used
    # to prioritize (or rank) these hypothesis
    if prioritize_optimal_solutions:
        # solutions with less negative conditions are better
        stmts.append(":~ injected_neg(X, Y, E, O).[1@-1, X, Y, E, O]")

        # solutions without deadend states are better
        stmts.append("injected_deadend_state(X) :- injected_state(X), not injected_ed(X, _, _).")
        stmts.append(":~ injected_deadend_state(X).[1@-2, X]")

        # solutions without cycles are better (count just 1 if state X appears in a cycle)
        if not learn_acyclic:
            stmts.append("injected_path(X, Y) :- injected_ed(X, Y, _).")
            stmts.append("injected_path(X, Z) :- injected_path(X, Y), injected_ed(Y, Z, _).")
            stmts.append(":~ injected_path(X, Y), injected_path(Y, X), X!=Y.[1@-3, X]")

    return generate_injected_block(stmts) + '\n'

