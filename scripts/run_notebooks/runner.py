"""CLI to notebook or directory of notebooks.

Arguments
---------
--notebooks: Specific notebook or directory of notebooks to run.
--mock: Run notebooks with mock code. Default is True. If --no-mock is provided,
    notebooks will run without mock code.

Examples
--------
Run all notebooks in a directory with mock code:

.. code-block:: bash

    python scripts/run_notebooks/runner.py --notebooks notebooks/ --mock

Run a single notebook without mocked code:

.. code-block:: bash

    python scripts/run_notebooks/runner.py --notebooks notebooks/notebook.ipynb --no-mock

Run all the notebook is two different directories with mocked code (default):

.. code-block:: bash

    python scripts/run_notebooks/runner.py --notebooks notebooks/ notebooks2/

"""

from argparse import ArgumentParser

from rich.console import Console
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict
from uuid import uuid4

import papermill
from joblib import Parallel, delayed
from nbformat.notebooknode import NotebookNode
from papermill.iorw import load_notebook_node, write_ipynb

KERNEL_NAME: str = "python3"

HERE = Path(__file__).parent
INJECTED_CODE_FILE = HERE / "injected.py"
INJECTED_CODE = INJECTED_CODE_FILE.read_text()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def generate_random_id() -> str:
    return str(uuid4())


def inject_pymc_sample_mock_code(cells: list) -> None:
    cells.insert(
        0,
        NotebookNode(
            id=f"code-injection-{generate_random_id()}",
            execution_count=sum(map(ord, "Mock pm.sample")),
            cell_type="code",
            metadata={"tags": []},
            outputs=[],
            source=INJECTED_CODE,
        ),
    )


def mock_run(notebook_path: Path, i: int, total: int) -> None:
    nb = load_notebook_node(str(notebook_path))
    inject_pymc_sample_mock_code(nb.cells)
    with NamedTemporaryFile(suffix=".ipynb") as f:
        write_ipynb(nb, f.name)
        desc = f"({i} / {total}) Mocked {notebook_path.name}"
        papermill.execute_notebook(
            input_path=f.name,
            output_path=None,
            progress_bar=dict(desc=desc),
            kernel_name=KERNEL_NAME,
            cwd=notebook_path.parent,
        )


def actual_run(notebook_path: Path, i: int, total: int) -> None:
    papermill.execute_notebook(
        input_path=notebook_path,
        output_path=None,
        kernel_name=KERNEL_NAME,
        progress_bar={"desc": f"({i} / {total}) Running {notebook_path.name}"},
        cwd=notebook_path.parent,
    )


class NotebookFailure(TypedDict):
    notebook_path: Path
    error: str


def run_notebook(
    notebook_path: Path,
    i: int,
    total: int,
    mock: bool = True,
) -> NotebookFailure | None:
    logging.info(f"Running notebook: {notebook_path.name}")
    run = mock_run if mock else actual_run

    try:
        run(notebook_path, i=i, total=total)
    except Exception as e:
        logging.error(
            f"{e.__class__.__name__} encountered running notebook: {str(notebook_path)}"
        )
        return NotebookFailure(notebook_path=notebook_path, error=str(e))
    else:
        return


class RunParams(TypedDict):
    notebook_path: Path
    mock: bool
    i: int
    total: int


def run_parameters(notebook_paths: list[Path], mock: bool = True) -> list[RunParams]:
    def to_mock(notebook_path: Path, i: int) -> RunParams:
        return RunParams(
            notebook_path=notebook_path, mock=mock, i=i, total=len(notebook_paths)
        )

    return [
        to_mock(notebook_path, i=i)
        for i, notebook_path in enumerate(notebook_paths, start=1)
    ]


def main(notebooks_to_run: list[Path], mock: bool = True) -> None:
    console = Console()
    errors: list[NotebookFailure]
    setup_logging()
    logging.info("Starting notebook runner")
    logging.info(f"Running {len(notebooks_to_run)} notebook(s).")
    results = Parallel(n_jobs=-1)(
        delayed(run_notebook)(**run_params)
        for run_params in run_parameters(notebooks_to_run, mock=mock)
    )
    errors = [result for result in results if result is not None]

    if not errors:
        logging.info("Notebooks run successfully!")
        return

    for error in errors:
        console.rule(f"[bold red]Error running {error['notebook_path']}[/bold red]")
        console.print(error["error"])

    logging.error(f"{len(errors)} / {len(notebooks_to_run)} notebooks failed")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--notebooks",
        nargs="+",
        help="List of notebooks to run. If not provided, all notebooks will be run.",
    )
    mock_group = parser.add_mutually_exclusive_group()
    mock_group.add_argument(
        "--mock",
        action="store_true",
        help="Run notebooks with mock code",
        dest="mock",
    )
    mock_group.add_argument(
        "--no-mock",
        action="store_false",
        help="Run notebooks without mock code",
        dest="mock",
    )
    parser.set_defaults(mock=True)
    args = parser.parse_args()

    notebooks_to_run = []
    notebooks = args.notebooks
    notebooks = [Path(notebook) for notebook in notebooks]
    for notebook in notebooks:
        if notebook.is_dir():
            notebooks_to_run.extend(notebook.glob("*.ipynb"))
            notebooks_to_run.extend(notebook.glob("*/*.ipynb"))
        else:
            notebooks_to_run.append(notebook)

    args.notebooks = notebooks_to_run

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.notebooks, mock=args.mock)
