from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton
from utils.ilasp.common import *

N_TRANSITION_STR_PREFIX = N_TRANSITION_STR + "("
N_TRANSITION_STR_SUFFIX = ", V0)"

CONNECTED_STR_PREFIX = CONNECTED_STR + "("
CONNECTED_STR_SUFFIX = ")."

POS_FLUENT_STR_PREFIX = OBS_STR + "(\""
NEG_FLUENT_STR_PREFIX = "not " + OBS_STR + "(\""
FLUENT_STR_SUFFIX = "\", V0"


def parse_ilasp_solutions(ilasp_learnt_filename):
    with open(ilasp_learnt_filename) as f:
        automaton = SubgoalAutomaton()
        edges = {}
        for line in f:
            line = line.strip()
            if line.startswith(N_TRANSITION_STR_PREFIX):
                transition, conditions = line.split(":-")

                transition = transition.strip()
                conditions = conditions.strip()

                from_state, to_state, edge_id = _parse_negative_transition(transition)
                current_edge = ((from_state, to_state), edge_id)

                if current_edge not in edges:
                    edges[current_edge] = []

                fluents = conditions.split("),")

                for fluent in fluents:
                    fluent = fluent.strip()
                    if fluent.startswith(POS_FLUENT_STR_PREFIX):
                        character = _parse_fluent(fluent, True)
                        edges[current_edge].append("~" + character)
                    elif fluent.startswith(NEG_FLUENT_STR_PREFIX):
                        character = _parse_fluent(fluent, False)
                        edges[current_edge].append(character)
            elif line.startswith(CONNECTED_STR_PREFIX):
                from_state, to_state, edge_id = [x.strip() for x in line[len(CONNECTED_STR_PREFIX):-len(CONNECTED_STR_SUFFIX)].split(",")]
                current_edge = ((from_state, to_state), edge_id)

                if current_edge not in edges:
                    edges[current_edge] = []

        for edge in edges:
            from_state, to_state = edge[0]
            automaton.add_state(from_state)
            automaton.add_state(to_state)
            automaton.add_edge(from_state, to_state, edges[edge])

        return automaton


def _parse_fluent(fluent_str, is_positive):
    prefix = POS_FLUENT_STR_PREFIX
    if not is_positive:
        prefix = NEG_FLUENT_STR_PREFIX
    return fluent_str[len(prefix):-len(FLUENT_STR_SUFFIX)]


def _parse_negative_transition(transition_str):
    states_edgeid = [x.strip() for x in transition_str[len(N_TRANSITION_STR_PREFIX):-len(N_TRANSITION_STR_SUFFIX)].split(",")]
    return states_edgeid[0], states_edgeid[1], states_edgeid[2]


def _edge_contains_conflict(new_condition, is_condition_negative, edge_conditions):
    if is_condition_negative:
        return ("~" + new_condition) in edge_conditions
    else:
        return new_condition in edge_conditions
