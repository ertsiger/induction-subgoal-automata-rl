import os
import subprocess

ILASP_BINARY_NAME = "ILASP"
CLINGO_BINARY_NAME = "clingo"
TIMEOUT_ERROR_CODE = 124


def solve_ilasp_task(ilasp_problem_filename, output_filename, version="2", max_body_literals=1, use_simple=True,
                     timeout=60*35, binary_folder_name=None):
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
                          "--clingo5",  # generate clingo 5 programs
                          "-ml=" + str(max_body_literals),
                          ilasp_problem_filename
                          ])

        if use_simple:
            arguments.append("--simple")  # simplify the representations of contexts

        if binary_folder_name is not None:
            arguments.append("--clingo")
            arguments.append("\"" + os.path.join(binary_folder_name, CLINGO_BINARY_NAME) + "\"")

        return_code = subprocess.call(arguments, stdout=f)
        return return_code != TIMEOUT_ERROR_CODE
