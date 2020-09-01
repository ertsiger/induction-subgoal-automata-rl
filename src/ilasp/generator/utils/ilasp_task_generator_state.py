

def get_state_names(num_states, accepting_state, rejecting_state):
    states = ["u" + str(i) for i in range(num_states - 2)]
    states.append(accepting_state)
    if rejecting_state is not None:
        states.append(rejecting_state)
    return states


def generate_state_statements(num_states, accepting_state, rejecting_state):
    states = get_state_names(num_states, accepting_state, rejecting_state)
    return "".join(["state(" + s + ").\n" for s in states]) + '\n'


def get_state_priorities(num_states, accepting_state, rejecting_state, use_terminal_priority):
    states = get_state_names(num_states, accepting_state, rejecting_state)
    states_with_priority = states if use_terminal_priority \
                                  else [s for s in states if s != accepting_state and s != rejecting_state]
    state_priorities = []
    for s, priority in zip(states_with_priority, range(len(states_with_priority))):
        state_priorities.append((s, priority))
    return state_priorities

