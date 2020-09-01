from ilasp.ilasp_common import OBS_STR, generate_injected_statement


def generate_examples(goal_examples, dend_examples, inc_examples):
    is_rejecting = len(dend_examples) > 0  # whether the rejecting state is used

    examples = _generate_goal_examples(goal_examples, is_rejecting)
    examples += _generate_deadend_examples(dend_examples)
    examples += _generate_incomplete_examples(inc_examples, is_rejecting) + '\n'
    examples += _generate_examples_injection(goal_examples, dend_examples, inc_examples) + '\n'
    return examples


def _generate_examples_injection(goal_examples, dend_examples, inc_examples):
    num_examples = len(goal_examples) + len(dend_examples) + len(inc_examples)
    return generate_injected_statement("example_active(0..%d)." % (num_examples - 1))


def get_longest_example_length(goal_examples, dend_examples, inc_examples):
    max_goal = len(max(goal_examples, key=len)) if len(goal_examples) > 0 else 0
    max_dend = len(max(dend_examples, key=len)) if len(dend_examples) > 0 else 0
    max_inc = len(max(inc_examples, key=len)) if len(inc_examples) > 0 else 0
    return max(max_goal, max_dend, max_inc)


def _generate_goal_examples(examples, is_rejecting):
    example_str = ""
    for example in examples:
        if is_rejecting:
            example_str += "#pos({accept}, {reject}, {\n"
        else:
            example_str += "#pos({accept}, {}, {\n"
        example_str += _generate_example(example)
        example_str += "}).\n\n"
    return example_str


def _generate_deadend_examples(examples):
    example_str = ""
    for example in examples:
        example_str += "#pos({reject}, {accept}, {\n"
        example_str += _generate_example(example)
        example_str += "}).\n\n"
    return example_str


def _generate_incomplete_examples(examples, is_rejecting):
    example_str = ""
    for example in examples:
        if is_rejecting:
            example_str += "#pos({}, {accept, reject}, {\n"
        else:
            example_str += "#pos({}, {accept}, {\n"
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
