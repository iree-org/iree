# IREE regression testing suite

This project defines tooling and tests comprising IREE's model regression
testing suite. It aims operate at the compiler-input level and choreographs
the normal tool flows of `iree-compile`, `iree-run-module`,
`iree-benchmark-module`, etc.

## Quick Start

If you have IREE tools on your path or have a virtual environment setup:

```bash
pip install -e experimental/regression_suite
PATH=../iree-build/tools:$PATH \
pytest experimental/regression_suite
```

Useful options:

* `-s`: Stream all test output.
* `-m MARKEXPR`: Select subsets of the test suite.

Common marker selections:

* `-m "plat_host_cpu and presubmit"`: Run the host-CPU tests configured for
  presubmit on the CI.
* `-m "plat_rdna3_vulkan and presubmit"`: Run the host-CPU tests configured for
  presubmit on the CI.

You can display all markers with `pytest experimental/regression_suite --markers`

## Setting up a venv

NOTE: For this to work, you must previously have installed GitHub command line
tools and authenticated (`gh auth`). See https://cli.github.com/.

The test suite doesn't care how you get tools on your path, but a common
case is to run the regression suite from built compilers and tools from a
GitHub presubmit or postsubmit run. This can be done in one step by setting
up a venv:

```bash
deactivate  # If have any venv active.
python ./build_tools/pkgci/setup_venv.py \
  /tmp/iree_gh_venv \
  --fetch-gh-workflow=${RUN_ID} \
  [--compiler-variant=asserts] [--runtime-variant=asserts]
source /tmp/iree_gh_venv/bin/activate

# Then install the regression suite and run pytest as usual:
pip install -e experimental/regression_suite
```

In the above, `<<RUN_ID>>` is the value in any GitHub action presubmit/postsubmit
workflow which has built package artifacts. As an example, if looking at a GitHub
Actions status page on:
`https://github.com/iree-org/iree/actions/runs/5957351746/job/16159877442`, then
the run id is the first number in the URL (5957351746).

Running the above will allow you to run `pytest` and you will have tools as built
at the commit from which the workflow run originated.
