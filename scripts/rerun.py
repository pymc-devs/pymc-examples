"""
This script/module may be used to re-run, commit & push notebooks
from the CLI or from another Python script (via `import rerun`).

Run `python rerun.py -h` to show the CLI help.

The example below does the following:
1. Re-runs the BEST notebook
2. Commits changes to a branch "rerun-best"
3. Push that branch to a remote named "mine"
   Assuming you did something like: `git add remote mine https://github.com/yourgithubusername/pymc-examples`

```
python scripts/rerun.py --fp_notebook=examples/case_studies/BEST.ipynb --commit_to=rerun-best --push_to=mine
```
"""

import argparse
import logging
import pathlib
import subprocess


logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__file__)
DP_REPO = pathlib.Path(__file__).absolute().parent.parent

REPLACEMENTS = {
    "az.from_pymc3": "pm.to_inference_data",
    "arviz.from_pymc3": "pm.to_inference_data",
    "pymc3": "pymc",
    "PyMC3": "PyMC",
    "pymc3": "pymc",
    "PyMC3": "PyMC",
    "from theano import tensor as tt": "import aesara.tensor as at",
    "import theano.tensor as tt": "import aesara.tensor as at",
    "tt.": "at.",
    "aet": "at",
    "studenat": "studentt",
    "theano": "aesara",
    "Theano": "Aesara",
    "pm.sample()": "pm.sample(return_inferencedata=False)",
    ", return_inferencedata=True": "",
    "return_inferencedata=True, ": "",
    "return_inferencedata=True,": "",
    "return_inferencedata=True": "",
}


def apply_replacements(fp: pathlib.Path) -> bool:
    try:
        _log.info("⏳ Running API migration")
        with open(fp, "r", encoding="utf-8") as file:
            lines = file.read()

        for pattern, substitute in REPLACEMENTS.items():
            lines = lines.replace(pattern, substitute)

        with open(fp, "w", encoding="utf-8") as file:
            file.write(lines)
        return True
    except Exception as ex:
        _log.exception("❌ Failed to apply replacements.", exc_info=ex)
        return False


def run_precommit(fp: pathlib.Path, attempts: int = 2):
    try:
        for a in range(attempts):
            _log.info("⏳ Running pre-commit attempt %i on %s", a, fp)
            try:
                subprocess.check_call(["pre-commit", "run", "--files", str(fp)])
                break
            except subprocess.CalledProcessError:
                if a == attempts - 1:
                    raise
        return True
    except Exception as ex:
        _log.exception("❌ Failed to run pre-commit.", exc_info=ex)
        return False


def execute_notebook(fp: pathlib.Path) -> bool:
    try:
        _log.info("⏳ Executing notebook %s", fp)
        subprocess.check_call(
            [
                "jupyter",
                "nbconvert",
                "--ExecutePreprocessor.kernel_name=python3",
                "--ExecutePreprocessor.timeout=14000",
                "--execute",
                "--inplace",
                str(fp),
            ]
        )
        _log.info("✔ Notebook executed successfully.")
        return True
    except subprocess.CalledProcessError as ex:
        _log.exception("❌ Failed to commit.", exc_info=ex)
        return False


def commit(fp: pathlib.Path, branch: str) -> bool:
    try:
        _log.info("Switching to branch %s", branch)
        if branch not in subprocess.check_output(["git", "branch"]).decode("ascii"):
            subprocess.check_call(["git", "checkout", "-b", branch])
        else:
            subprocess.check_call(["git", "checkout", branch])

        _log.info("⏳ Committing changes in %s to %s", fp, branch)
        subprocess.check_call(["git", "stage", str(fp)])
        subprocess.check_call(["git", "commit", "-m", f"Re-run {fp.name} notebook"])
        _log.info("✔ Changes in %s were committed to branch %s.", fp, branch)
        return True
    except subprocess.CalledProcessError as ex:
        _log.exception("❌ Failed to commit.", exc_info=ex)
        return False


def push(branch, remote: str) -> bool:
    try:
        _log.info("⏳ Pushing %s to %s", branch, remote)
        subprocess.check_call(["git", "push", "-u", remote, f"{branch}:{branch}"])
        _log.info("✔ Pushed %s to %s/%s.", branch, remote, branch)
        return True
    except subprocess.CalledProcessError as ex:
        _log.exception("❌ Failed push.", exc_info=ex)
        return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp_notebook",
        type=str,
        help=f"Absolute or relative path to a Jupyter notebook in {str(DP_REPO)}.",
        required=True,
    )
    parser.add_argument(
        "--commit_to",
        type=str,
        help="Name of a git branch to commit to on success.",
        required=False,
    )
    parser.add_argument(
        "--push_to", type=str, help="Name of a git remote to push to on success.", required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    fp = pathlib.Path(args.fp_notebook)
    if not fp.exists():
        raise FileNotFoundError(f"Notebook file {fp} does not exist.")

    success = True
    success = success and apply_replacements(fp)
    success = success and run_precommit(fp)
    success = success and execute_notebook(fp)
    if args.commit_to:
        success = success and run_precommit(fp)
        success = success and commit(fp, args.commit_to)
        if success and args.push_to:
            success = success and push(args.commit_to, args.push_to)

    if success:
        _log.info("✔ All steps succeeded.")
    else:
        _log.error("❌ Manual investigation needed.")
