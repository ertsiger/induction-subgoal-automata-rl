from collections import namedtuple
from ilasp.ilasp_common import N_TRANSITION_STR, CONNECTED_STR, OBS_STR

N_TRANSITION_STR_PREFIX = N_TRANSITION_STR + "("
CONNECTED_STR_PREFIX = CONNECTED_STR + "("
CONNECTED_STR_SUFFIX = ")."
POS_FLUENT_STR_PREFIX = OBS_STR + "("
NEG_FLUENT_STR_PREFIX = "not " + POS_FLUENT_STR_PREFIX

ParsedNegativeTransitionRule = namedtuple("ParsedNegativeTransitionRule",
                                          field_names=["src", "dst", "edge", "pos", "neg"])
ParsedEdgeRule = namedtuple("ParsedEdgeRule", field_names=["src", "dst", "edge"])


def parse_edge_rule(edge_str):
    from_state, to_state, edge_id = [x.strip() for x in edge_str[len(CONNECTED_STR_PREFIX):-len(CONNECTED_STR_SUFFIX)].split(",")]
    return ParsedEdgeRule(from_state, to_state, edge_id)


def parse_negative_transition_rule(transition_str):
    head, body = transition_str.split(":-")
    head, body = head.strip(), body.strip()

    from_state, to_state, edge_id = _parse_negative_transition_head(head)
    pos_fluents, neg_fluents = _parse_negative_transition_body(body)
    return ParsedNegativeTransitionRule(from_state, to_state, edge_id, pos_fluents, neg_fluents)


def _parse_negative_transition_head(transition_head_str):
    states_edgeid = [x.strip() for x in transition_head_str[len(N_TRANSITION_STR_PREFIX):].split(",")]
    return states_edgeid[0], states_edgeid[1], states_edgeid[2]


def _parse_negative_transition_body(transition_body_str):
    fluents = transition_body_str.split(");")
    pos_fluents, neg_fluents = [], []

    for fluent in fluents:
        fluent = fluent.strip()
        if fluent.startswith(POS_FLUENT_STR_PREFIX):
            character = _parse_fluent(fluent, True)
            pos_fluents.append(character)
        elif fluent.startswith(NEG_FLUENT_STR_PREFIX):
            character = _parse_fluent(fluent, False)
            neg_fluents.append(character)

    return pos_fluents, neg_fluents


def _parse_fluent(fluent_str, is_positive):
    prefix = POS_FLUENT_STR_PREFIX
    if not is_positive:
        prefix = NEG_FLUENT_STR_PREFIX
    return fluent_str[len(prefix):].split(",")[0].strip("\"")

