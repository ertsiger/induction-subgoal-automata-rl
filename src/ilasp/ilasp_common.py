
N_TRANSITION_STR = "n_phi"
CONNECTED_STR = "ed"
OBS_STR = "obs"


def generate_injected_statements(stmts):
    return "\n".join([generate_injected_statement(stmt) for stmt in stmts]) + '\n'


def generate_injected_block(stmts):
    return generate_injected_statement('\n\t' + "\n\t".join(stmts) + '\n') + '\n'


def generate_injected_statement( stmt):
    return "#inject(\"" + stmt + "\")."
