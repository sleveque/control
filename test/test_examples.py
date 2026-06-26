import json
import os
import pathlib
import runpy


def run_example_notebook(filename, tmp_path):
    tmp_filename = tmp_path / "tmp.py"

    with open(filename, encoding="utf-8") as nb_h, \
            open(tmp_filename, "w", encoding="utf-8") as py_h:
        nb = json.load(nb_h)
        if nb["metadata"]["language_info"]["name"] != "python":
            raise RuntimeError("Expected a Python notebook")

        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                for line in cell["source"]:
                    py_h.write(line)
                py_h.write("\n\n")

    os.chdir(tmp_path)
    runpy.run_path(str(tmp_filename))


def test_0_stationary_linear_control(tmp_path):
    run_example_notebook(pathlib.Path(__file__).parent.parent / "documentation" / "0_stationary_linear_control.ipynb",
                         tmp_path)


def test_1_stationary_incompressible_linear_control(tmp_path):
    run_example_notebook(pathlib.Path(__file__).parent.parent / "documentation" / "1_stationary_incompressible_linear_control.ipynb",
                         tmp_path)


def test_2_stationary_non_linear_control(tmp_path):
    run_example_notebook(pathlib.Path(__file__).parent.parent / "documentation" / "2_stationary_non_linear_control.ipynb",
                         tmp_path)


def test_3_instationary_control(tmp_path):
    run_example_notebook(pathlib.Path(__file__).parent.parent / "documentation" / "3_instationary_control.ipynb",
                         tmp_path)


def test_4_instationary_incompressible_control(tmp_path):
    run_example_notebook(pathlib.Path(__file__).parent.parent / "documentation" / "4_instationary_incompressible_control.ipynb",
                         tmp_path)


def test_5_preconditioning_stationary_control(tmp_path):
    run_example_notebook(pathlib.Path(__file__).parent.parent / "documentation" / "5_preconditioning_stationary_control.ipynb",
                         tmp_path)
