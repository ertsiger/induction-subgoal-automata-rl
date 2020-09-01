from ilasp.ilasp_common import generate_injected_block
from ilasp.generator.utils.ilasp_task_generator_state import get_state_names, get_state_priorities


def generate_symmetry_breaking_statements(num_states, accepting_state, rejecting_state, observables,
                                          symmetry_breaking_method, max_disj_size, learn_acyclic):
    if symmetry_breaking_method == "increasing-path":
        if learn_acyclic:
            return _generate_increasing_path_symmetry_breaking(num_states, accepting_state, rejecting_state)
        else:
            raise RuntimeError("Error: The 'increasing-path' symmetry breaking only works for acyclic graphs.")
    elif symmetry_breaking_method.startswith("bfs"):
        return _generate_bfs_symmetry_breaking(num_states, accepting_state, rejecting_state, observables,
                                               symmetry_breaking_method, max_disj_size)
    else:
        raise RuntimeError("Error: Unknown symmetry breaking method \"%s\"" % symmetry_breaking_method)


def _generate_increasing_path_symmetry_breaking(num_states, accepting_state, rejecting_state):
    stmt = _generate_state_priority_statements(num_states, accepting_state, rejecting_state)
    stmt += ":- st(TX, X), st(TY, Y), state_id(X, IDX), state_id(Y, IDY), TX<TY, IDX>IDY.\n"
    stmt += ":- not ed(X, Y, E), ed(X, Y, E+1), edge_id(E).\n\n"  # do not leave lower edge ids unused
    return stmt


def _generate_state_priority_statements(num_states, accepting_state, rejecting_state):
    states_str = "\n".join(["state_id(%s, %d)." % (s, p)
                            for s, p in get_state_priorities(num_states, accepting_state, rejecting_state, True)])
    return states_str + '\n'


def _generate_bfs_symmetry_breaking(num_states, accepting_state, rejecting_state, observables, symmetry_breaking_method,
                                    max_disj_size):
    stmt = _generate_injected_state_statements(num_states, accepting_state, rejecting_state)
    stmt += _generate_injected_state_priority_statements(num_states, accepting_state, rejecting_state)
    stmt += _generate_injected_observable_statements(observables)
    if symmetry_breaking_method == "bfs":
        stmt += _generate_bfs_symmetry_breaking_rules(num_states, rejecting_state, max_disj_size)
    elif symmetry_breaking_method == "bfs-alternative":
        stmt += _generate_bfs_symmetry_breaking_rules_alternative(max_disj_size)
    return stmt


def _generate_injected_state_statements(num_states, accepting_state, rejecting_state):
    states = get_state_names(num_states, accepting_state, rejecting_state)
    return generate_injected_block(["injected_state(" + s + ")." for s in states]) + '\n'


def _generate_injected_state_priority_statements(num_states, accepting_state, rejecting_state):
    state_priorities = get_state_priorities(num_states, accepting_state, rejecting_state, False)
    return generate_injected_block(["injected_state_id(%s, %d)." % (s, i) for s, i in state_priorities]) + '\n'


def _generate_injected_observable_statements(observables):
    obs_list = sorted(observables)
    stmts = ["injected_obs_id(%s, %d)." % (obs_list[i], i + 1) for i in range(len(obs_list))]
    stmts.append("injected_num_obs(%d)." % len(observables))
    stmts.append("injected_valid_label(1..%d)." % (2 * len(observables)))
    return generate_injected_block(stmts) + '\n'


def _generate_bfs_symmetry_breaking_rules(num_states, rejecting_state, max_disj_size):
    stmts = _generate_bfs_symmetry_breaking_rules_helper()

    # nodes have a parent
    stmts.append("1{injected_pa(X, Y): injected_state(X), injected_state_lt(X, Y)} :- injected_state(Y), injected_state_id(Y, YID), YID>0.")

    # nodes have a single parent
    stmts.append(":- injected_pa(X, Y), injected_pa(XP, Y), injected_state_lt(X, XP), injected_state_lt(XP, Y).")

    # if there is a parent, there is a smallest index
    stmts.append("1{injected_sm(X, Y, E) : injected_edge_id(E)} :- injected_pa(X, Y).")

    # bfs order
    stmts.append(":- injected_pa(X, Y), injected_ed_sb(XP, YP, _), injected_state_lt(XP, X), injected_state_leq(Y, YP).")

    # smallest index implies BFS edge
    stmts.append(":- injected_sm(X, Y, _), not injected_pa(X, Y).")

    # smallest index is unique
    stmts.append(":- injected_sm(X, Y, E), injected_sm(X, Y, EP), E<EP.")

    # smallest index implies edge
    stmts.append(":- injected_sm(X, Y, E), not injected_ed_sb(X, Y, E).")

    # break ties with smallest index
    stmts.append(":- injected_sm(X, Y, E), injected_ed_sb(X, YP, EP), injected_state_lt(X, Y), injected_state_leq(Y, YP), EP<E.")

    # consecutive edge ids
    stmts.append(":- injected_map_ed(X, Y, E), not injected_map_ed(X, _, E-1), E>1.")

    # not duplicated edge indices
    stmts.append(":- injected_map_ed(X, Y, E), injected_map_ed(X, Z, E), Y<Z.")

    stmts.append(":- injected_lt(X, E-1, E, L), injected_label(X, E-1, L), not injected_lt(X, E-1, E, L-1), E>1, L>1.")
    stmts.append(":- injected_lt(X, E-1, E, L), injected_label(X, E-1, L), E>1, L=1.")

    stmts.append(":- injected_lt(X, E-1, E, L), not injected_label(X, E, L), not injected_lt(X, E-1, E, L-1), E>1, L>1.")
    stmts.append(":- injected_lt(X, E-1, E, L), not injected_label(X, E, L), E>1, L=1.")

    stmts.append("1{injected_lt(X, E-1, E, L) : injected_valid_label(L)} :- injected_map_ed(X, Y, E), E>1.")

    stmts.append(":- not injected_lt(X, E-1, E, L), injected_label(X, E-1, L), not injected_label(X, E, L), injected_map_ed(X, _, E), E>1.")

    return _generate_injected_edge_mapping_statements(num_states, rejecting_state, max_disj_size) + \
           generate_injected_block(stmts) + '\n'


def _generate_injected_edge_mapping_statements(num_states, rejecting_state, max_disj_size):
    max_num_outgoing_edges = (num_states - 1) * max_disj_size if rejecting_state is not None else (num_states - 2) * max_disj_size

    # make a mapping - edge ids should range between 1 and (N - 1) * k
    stmts = ["injected_edge_id(1..%d)." % max_num_outgoing_edges]
    stmts.append("1{injected_mapping(X, Y, E, EE) : injected_edge_id(EE)}1 :- injected_ed(X, Y, E).")

    # edges between two different nodes cannot be mapped to the same edge id
    stmts.append(":- injected_mapping(X, Y, _, E), injected_mapping(X, Z, _, E), Y<Z.")

    # different edge ids between two states must be different
    stmts.append(":- injected_mapping(X, Y, E, EE), injected_mapping(X, Y, EP, EE), E<EP.")

    # if an edge id is higher than another between two states previously, then the
    # new edge id should still be higher
    stmts.append(":- injected_ed(X, Y, E), injected_ed(X, Y, EP), E<EP, injected_mapping(X, Y, E, EE), "
                 "injected_mapping(X, Y, EP, EEP), EE>EEP.")

    # new edges using the edge id mapping
    stmts.append("injected_map_ed(X, Y, EP) :- injected_ed(X, Y, E), injected_mapping(X, Y, E, EP).")
    stmts.append("injected_map_pos(X, Y, EP, O) :- injected_pos(X, Y, E, O), injected_mapping(X, Y, E, EP).")
    stmts.append("injected_map_neg(X, Y, EP, O) :- injected_neg(X, Y, E, O), injected_mapping(X, Y, E, EP).")

    return generate_injected_block(stmts) + '\n'


def _generate_bfs_symmetry_breaking_rules_helper():
    # some helper for ordering edges
    stmts = ["injected_state_lt(X, Y) :- injected_state_id(X, XID), injected_state_id(Y, YID), XID<YID."]
    stmts.append("injected_state_leq(X, Y) :- injected_state_id(X, XID), injected_state_id(Y, YID), XID<=YID.")

    # redefine edges for symmetry breaking excluding those without state id (accepting and rejecting states)
    stmts.append("injected_ed_sb(X, Y, E) :- injected_map_ed(X, Y, E), injected_state_id(Y, _).")

    # define the label predicate, which maps pos and neg to numbers
    stmts.append("injected_label(X, E, OID) :- injected_map_pos(X, Y, E, O), injected_obs_id(O, OID).")
    stmts.append("injected_label(X, E, OID+N) :- injected_map_neg(X, Y, E, O), injected_obs_id(O, OID), injected_num_obs(N).")

    return stmts


def _generate_bfs_symmetry_breaking_rules_alternative(max_disj_size):
    # helper methods for comparing state ids
    stmts = ["injected_state_leq(X, Y) :- injected_state_id(X, XID), injected_state_id(Y, YID), XID<=YID.",
             "injected_state_lt(X, Y) :- injected_state_id(X, XID), injected_state_id(Y, YID), XID<YID."]

    # edge ids
    stmts.append("injected_edge_id(1..%d)." % max_disj_size)

    # use lower edge ids first
    stmts.append(":- injected_ed(X, Y, E), not injected_ed(X, Y, E-1), injected_edge_id(E), E>1.")

    # 1. choose an ordering between edges from a given state
    # 2. impose transitive relationships
    # 3. impose that an edge with a lower edge id is lower than another edge to the same state with a higher edge id
    stmts.append("1 { injected_ed_lt(X, (Y, E), (YP, EP)) ; injected_ed_lt(X, (YP, EP), (Y, E)) } 1 :- injected_ed(X, Y, E), injected_ed(X, YP, EP), (Y, E) < (YP, EP).")
    stmts.append(":- injected_ed_lt(X, Edge1, Edge2), injected_ed_lt(X, Edge2, Edge3), not injected_ed_lt(X, Edge1, Edge3), Edge1 != Edge3.")
    stmts.append(":- injected_ed_lt(X, (Y, E), (Y, EP)), injected_ed(X, Y, E), injected_ed(X, Y, EP), E>EP.")

    # parenting relationship
    # 1. define edges only for those states with ids (accepting and rejecting states are excluded)
    # 2. the parent is the node with lowest id with an edge to Y
    # 3. all nodes have a parent
    # 4. BFS ordering
    stmts.append("injected_ed_sb(X, Y, E) :- injected_ed(X, Y, E), injected_state_id(Y, _).")
    stmts.append("injected_pa(X, Y) :- injected_ed_sb(X, Y, _), injected_state_lt(X, Y), "
                 "#false : injected_ed_sb(Z, Y, _), injected_state_lt(Z, X).")
    stmts.append(":- injected_state_id(Y, YID), YID > 0, not injected_pa(_, Y).")
    stmts.append(":- injected_pa(X, Y), injected_ed_sb(XP, YP, _), injected_state_lt(XP, X), injected_state_leq(Y, YP).")

    # if X is the parent of Y, there is a smallest edge from X to Y (there is no destination YP such that Y<YP, but
    # the edge (X, YP) is lower than Y), no edge from X violates the order w.r.t. Y
    stmts.append("injected_state_ord(Y) :- injected_ed_sb(X, Y, E), injected_pa(X, Y), "
                 "#false : injected_ed_sb(X, YP, EP), injected_state_lt(Y, YP), injected_ed_lt(X, (YP, EP), (Y, E)).")
    stmts.append(":- injected_pa(_, Y), not injected_state_ord(Y).")

    # map pos and neg into integer labels
    stmts.append("injected_label(X, (Y, E), OID) :- injected_pos(X, Y, E, O), injected_obs_id(O, OID).")
    stmts.append("injected_label(X, (Y, E), OID+N) :- injected_neg(X, Y, E, O), injected_obs_id(O, OID), injected_num_obs(N).")

    # observation ordering should follow edge ordering
    stmts.append("injected_label_lt(X, Edge1, Edge2, L) :- injected_ed_lt(X, Edge1, Edge2), not injected_label(X, Edge1, L), injected_label(X, Edge2, L).")
    stmts.append("injected_label_lt(X, Edge1, Edge2, L+1) :- injected_label_lt(X, Edge1, Edge2, L), injected_valid_label(L+1).")
    stmts.append(":- injected_ed_lt(X, Edge1, Edge2), injected_label(X, Edge1, L), not injected_label(X, Edge2, L), not injected_label_lt(X, Edge1, Edge2, L).")

    return generate_injected_block(stmts) + '\n'
