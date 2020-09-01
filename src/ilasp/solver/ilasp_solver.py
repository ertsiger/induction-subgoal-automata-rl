import os
import subprocess

ILASP_BINARY_NAME = "ILASP"
CLINGO_BINARY_NAME = "clingo"  # use clingo.rb for minimal solutions
CLINGO_MINIMAL_BINARY_NAME = "clingo.rb"
TIMEOUT_ERROR_CODE = 124

ILASP_OPERATION_SOLVE = "solve"
ILASP_OPERATION_SEARCH_SPACE = "search_space"


def solve_ilasp_task(ilasp_problem_filename, output_filename, version="2", max_body_literals=1, use_simple=True,
                     timeout=60*35, binary_folder_name=None, compute_minimal=False, operation=ILASP_OPERATION_SOLVE):
    with open(output_filename, 'w') as f:
        arguments = []
        if timeout is not None:
            arguments = ["timeout", str(timeout)]

        if binary_folder_name is None:
            ilasp_binary_path = ILASP_BINARY_NAME
        else:
            ilasp_binary_path = os.path.join(binary_folder_name, ILASP_BINARY_NAME)

        # other flags: -d -ni -ng -np
        arguments.extend([ilasp_binary_path,
                          "--version=" + version,  # test 2 and 2i
                          "--strict-types",
                          "-nc",  # omit constraints from the search space
                          "-ml=" + str(max_body_literals),
                          ilasp_problem_filename
                          ])

        if use_simple:
            arguments.append("--simple")  # simplify the representations of contexts

        if binary_folder_name is not None:
            arguments.append("--clingo")
            clingo_binary = CLINGO_MINIMAL_BINARY_NAME if compute_minimal else CLINGO_BINARY_NAME
            arguments.append("\"" + os.path.join(binary_folder_name, clingo_binary) + "\"")

        if operation == ILASP_OPERATION_SEARCH_SPACE:
            arguments.append("-s")

        return_code = subprocess.call(arguments, stdout=f)
        return return_code != TIMEOUT_ERROR_CODE
