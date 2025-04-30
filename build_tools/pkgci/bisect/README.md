# Package bisect scripting

This scripting connects the `git bisect` tool
(<https://git-scm.com/docs/git-bisect>) with IREE's package builds, allowing
developers to run tests through commit history efficiently. For example, this
can be used to spot at which commit an `iree-compile` command started failing.

At each stage of the bisect process, this bisect tool will download the IREE
packages (i.e. `iree-base-compiler` and `iree-base-runtime`) and prepend their
installed location to the `PATH` environment variable. If you want to run a
bisect that _does not_ need to run `iree-compile`, just use `git bisect`
directly. However, if you _do_ need to run `iree-compile`, this can save
substantial time by avoiding the need to build it from source at each test
commit.

## Prerequisites

### System requirements

Requirement | Details
----------- | -------
Linux | (at least until IREE builds packages for other systems at each commit)
`git` | <https://git-scm.com/>
`gh` CLI | <https://cli.github.com/>
iree-org/iree repository read access | Needed to [download workflow artifacts](https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-workflow-runs/downloading-workflow-artifacts). See also [obtaining commit access](https://iree.dev/developers/general/contributing/#obtaining-commit-access).
`python3.11` with `venv` support | (Version must match what PkgCI builds) `sudo apt install python3.11 python3.11-dev python3.11-venv`

### Data requirements

* A command to run, such as a `.mlir` file and a `iree-compile` command
* A known-working commit
* A known-broken commit

The commit range between known-working and known-broken should be as small as
possible to limit the number of test steps. The bisect algorithm is `O(log N)`
where `N` is the number of commits in the range, but wider windows have larger
risk of something in the test environment breaking (e.g. breaking API changes,
serialized `.mlir` files not being stable, etc.).

## Usage

### Example

Let's try to find the culprit commit for issue
<https://github.com/iree-org/iree/issues/18879>. Thanks to the detailed issue
description, we have all the data we need to run a bisect already.

To run the bisect tool:

1. Setup the test case by saving the input `.mlir` file and noting the test
   command:

    ```mlir
    // /tmp/issue_18879.mlir

    // This is the input program to the test command.
    module {
      func.func @main_graph(
        // ...
    ```

    ```bash
    # This is the command that will be tested at each commit.
    # The command should succeed (return 0) at and prior to the `--good-ref`
    # commit and should fail (return non-0) at the `--bad-ref` commit.

    # Try to keep these test commands (or scripts) minimal. If the failure is
    # in an earlier phase of the compiler (e.g. 'Flow' or 'Stream'), consider
    # using a flag like `--compile-to=hal` to exit early on successful run
    # instead of spending all the time to serialize an output `.vmfb` file.
    # https://iree.dev/developers/general/developer-tips/#compiling-phase-by-phase

    iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu -o /dev/null /tmp/issue_18879.mlir
    ```

    If the test command spans multiple lines, you can put it in an executable
    `.sh` file and use the `--test-script` option instead of `--test-command`.

2. Run the bisect tool, under Python 3.11:

    ```bash
    # Ensure 'python' is Python 3.11, e.g. using venv
    # (https://docs.python.org/3/library/venv.html):
    python3.11 -m venv .venv && source .venv/bin/activate
    # OR using pyenv (https://github.com/pyenv/pyenv):
    # pyenv shell 3.11
    python --version
    # Python 3.11.10

    ./bisect_packages.py \
      --good-ref=f9fa934c649749b30fc4be05d9cef78eb043f0e9 \
      --bad-ref=05bbcf1385146d075829cd940a52bf06961614d0 \
      --test-command="iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu -o /dev/null /tmp/issue_18879.mlir"

    # 206b60ca59c9dbbca5769694df4714c38cecaced is the first bad commit
    ```

    As expected, the bisect agrees with the culprit mentioned on the issue:
    <https://github.com/iree-org/iree/issues/18879#issuecomment-2435531655>.

    Note that any git ref can be used, so we can use tags too:

    ```bash
    ./bisect_packages.py \
      --good-ref=candidate-20241016.1048 \
      --bad-ref=candidate-20241017.1049 \
      --test-script=/tmp/issue_18879.sh
    ```

## How the tool works

1. The [`bisect_packages.py`](./bisect_packages.py) script is the main entry
   point which sets things up and runs `git bisect` commands.
2. The bisect operation starts with
   `git bisect start --no-checkout --first-parent` (flags to avoid modifying
   the working tree and traversing merge commits) and then calls to
   specify the commit range with `git bisect good` and `git bisect bad`.
3. The script injects wrapper code around the provided `--test-script`. First,
   the wrapper script calls
   [`install_packages_for_commit.py`](./install_packages_for_commit.py) to
   download IREE packages built at the test commit (marked by `BISECT_HEAD`) and
   install those packages into an isolated virtual environment. The wrapper
   script then puts that environment at the front of `PATH`, runs the original
   script, and finally forwards the original script's exit code to `git bisect`.
4. The script kicks off a `git bisect run` using the generated wrapper script,
   which then proceeds to test commits between `--good-ref` and `--bad-ref`,
   looking for when the test script switched from succeeding to failing.
5. After the script, the logs can be analyzed and `git bisect log` can be run
   from the repository root.

### Working directory cache

Downloaded files and virtual environments are cached at `~/.iree/bisect`
(this path can be changed using the `--work-dir` option). Each commit tested
gets its own subfolder that contains the downloaded release artifacts and a
Python venv with those packages installed in it:

```bash
$ tree -a ~/.iree/bisect -L 2
/home/nod/.iree/bisect
├── 099ffd556bc5d35efcca32af51cccc061a273a91
│   ├── iree_base_compiler-3.1.0.dev0+099ffd556bc5d35efcca32af51cccc061a273a91-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
│   ├── iree_base_runtime-3.1.0.dev0+099ffd556bc5d35efcca32af51cccc061a273a91-cp311-cp311-manylinux_2_28_x86_64.whl
│   └── .venv
├── 15006418ceb03023e8887cba87e93b499f669ad7
│   ├── iree_compiler-0.dev1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
│   ├── iree_runtime-0.dev1-cp311-cp311-manylinux_2_28_x86_64.whl
│   └── .venv
├── 206b60ca59c9dbbca5769694df4714c38cecaced
│   ├── iree_compiler-0.dev1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
│   ├── iree_runtime-0.dev1-cp311-cp311-manylinux_2_28_x86_64.whl
│   └── .venv
├── 23c32c633c01e0237cf5f3815b6647cf01827832
│   ├── iree_base_compiler-3.1.0.dev0+23c32c633c01e0237cf5f3815b6647cf01827832-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
│   ├── iree_base_runtime-3.1.0.dev0+23c32c633c01e0237cf5f3815b6647cf01827832-cp311-cp311-manylinux_2_28_x86_64.whl
│   └── .venv
```

### Wrapper script

Here is an example of a script that wraps the original `--test-script`. This is
what gets passed to `git bisect run`:

```bash
#!/bin/bash

#########################################
###### BISECT RELEASE SCRIPT SETUP ######
#########################################

set -xeuo pipefail

REF_HASH=$(git rev-parse BISECT_HEAD)
"/home/nod/.pyenv/shims/python3.11" /home/nod/dev/projects/iree/build_tools/pkgci/bisect/../setup_venv.py /home/nod/.iree/bisect/${REF_HASH}/.venv --artifact-path=/home/nod/.iree/bisect/${REF_HASH}  --fetch-git-ref=${REF_HASH}
PATH="/home/nod/.iree/bisect/$REF_HASH/.venv/bin:$PATH"

set +e

#########################################
############ ORIGINAL SCRIPT ############
#########################################

iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu -o /dev/null /home/nod/.iree/bisect/issue_18879.mlir

#########################################
##### BISECT RELEASE SCRIPT CLEANUP #####
#########################################

RET_VALUE=$?
if [ $RET_VALUE -ne 0 ]; then
    exit 1
fi
```

### Example annotated logs

Raw logs here: <https://gist.github.com/ScottTodd/cff468a50df63b65e5c5f449fabab6af>

```bash
$ ./bisect_packages.py \
  --good-ref=candidate-20241016.1048 \
  --bad-ref=candidate-20241017.1049 \
  --test-script=/home/nod/.iree/bisect/issue_18879.sh

Welcome to bisect_packages.py!

------------------------------------------------------------------
--------- Configuration ------------------------------------------
------------------------------------------------------------------

  Searching range         : 'candidate-20241016.1048' - 'candidate-20241017.1049'
  Using working directory : '/home/nod/.iree/bisect'
  Using test script       : '/home/nod/.iree/bisect/issue_18879.sh'

------------------------------------------------------------------

------------------------------------------------------------------
--------- Running git bisect -------------------------------------
------------------------------------------------------------------

# --------------------------------------
# Here we start to test the first commit
# --------------------------------------

Bisecting: 5 revisions left to test after this (roughly 3 steps)
[c7213deeb5c7abb0843088815580793b282fdc34] Produce releases for Python 3.13. (#18799)
running  '/home/nod/.iree/bisect/bisect_run_script.sh'
++ git rev-parse BISECT_HEAD
+ REF_HASH=c7213deeb5c7abb0843088815580793b282fdc34
+ /home/nod/dev/projects/iree/build_tools/pkgci/setup_venv_for_ref.py c7213deeb5c7abb0843088815580793b282fdc34 --work-dir /home/nod/.iree/bisect
------------------------------------------------------------------
Installing packages for ref: c7213deeb5c7abb0843088815580793b282fdc34
  Using base working directory : '/home/nod/.iree/bisect'

# -----------------------------------------------------
# Here we download and install packages for that commit
# -----------------------------------------------------
Running command to list workflow runs:
  gh api -H Accept: application/vnd.github+json -H X-GitHub-Api-Version: 2022-11-28 /repos/iree-org/iree/actions/workflows/pkgci.yml/runs?head_sha=c7213deeb5c7abb0843088815580793b282fdc34
Found workflow run: https://github.com/iree-org/iree/actions/runs/11375010806
Found cached .whl files in artifacts dir, skipping download
Creating venv at '/home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/.venv'

Running command to install dependencies:
  /home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/.venv/bin/python -m pip install --quiet numpy sympy
Running command to install package:
  /home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/.venv/bin/python -m pip install --quiet /home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/iree_compiler-0.dev1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
Running command to install package:
  /home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/.venv/bin/python -m pip install --quiet /home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/iree_runtime-0.dev1-cp311-cp311-manylinux_2_28_x86_64.whl

Checking packages with 'pip freeze':
iree-compiler @ file:///home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/iree_compiler-0.dev1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=4078073daae1b706361091389753a4887bfa7d4797ea66dce1d0daaa5bffc58c
iree-runtime @ file:///home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/iree_runtime-0.dev1-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=564779699f560ba1da406c3d7d08fc75ba8b8eb2f6fc6e074e691a34bbb29bdf
mpmath==1.3.0
numpy==2.1.3
sympy==1.13.3
------------------------------------------------------------------

+ PATH=/home/nod/.iree/bisect/c7213deeb5c7abb0843088815580793b282fdc34/.venv/bin:/usr/lib/git-core:/usr/lib/git-core:/home/nod/.pyenv/libexec:/home/nod/.pyenv/plugins/python-build/bin:/home/nod/.pyenv/plugins/pyenv-virtualenv/bin:/home/nod/.pyenv/plugins/pyenv-update/bin:/home/nod/.pyenv/plugins/pyenv-doctor/bin:/home/nod/.pyenv/shims:/home/nod/.pyenv/bin:/home/nod/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
# -----------------------------------------------------
# Here we run the test script
# -----------------------------------------------------
+ set +e
+ iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu -o /dev/null /home/nod/.iree/bisect/issue_18879.mlir
/home/nod/.iree/bisect/issue_18879.mlir:17:11: error: operand #0 does not dominate this use
    %21 = torch.operator "onnx.Resize"(%20, %none, %1) {torch.onnx.coordinate_transformation_mode = "asymmetric", torch.onnx.cubic_coeff_a = -7.500000e-01 : f32, torch.onnx.mode = "nearest", torch.onnx.nearest_mode = "floor"} : (!torch.vtensor<[1,18,14,14],f32>, !torch.none, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,18,56,56],f32>
          ^
/home/nod/.iree/bisect/issue_18879.mlir:17:11: note: see current operation: %144 = "tensor.extract"(%32, %1, %129, %137, %143) : (tensor<1x18x14x14xf32>, index, index, index, index) -> f32
/home/nod/.iree/bisect/issue_18879.mlir:16:11: note: operand defined here (op in a parent region)
    %20 = torch.operator "onnx.Conv"(%arg2, %2, %3) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 1 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,72,14,14],f32>, !torch.vtensor<[18,72,1,1],f32>, !torch.vtensor<[18],f32>) -> !torch.vtensor<[1,18,14,14],f32>
          ^
+ RET_VALUE=1
+ '[' 1 -ne 0 ']'
+ exit 1
# --------------------------------------------------------------------
# The test script completed, so now we proceed to test the next commit
# --------------------------------------------------------------------
Bisecting: 2 revisions left to test after this (roughly 2 steps)
[8568efa3cceb6dbbd69e8b681436a17efcce1a74] [GPU] Adding support for opt pass plugins during AMDGPU executable serialization (#18347)

# --------------------------------------------------------------------
# (repeat the download packages --> run test script step for other commits)
# ... skipping ahead ...
# --------------------------------------------------------------------

# -----------------------------------------
# Bisecting finished. Here are the findings
# -----------------------------------------
206b60ca59c9dbbca5769694df4714c38cecaced is the first bad commit
commit 206b60ca59c9dbbca5769694df4714c38cecaced
Author: Ian Wood <75152913+IanWood1@users.noreply.github.com>
Date:   Wed Oct 16 10:52:47 2024 -0700

    [DispatchCreation] Extend multi-use producer fusion (#18551)

    Fuse even in cases where the most dominant op isn't fusable, but other operations would be legal to fuse. Do this by moving the fusable consumer and all transitive defs before all other consumers (if legal).

    ---------

    Signed-off-by: Ian Wood <ianwood2024@u.northwestern.edu>

 .github/workflows/pkgci_regression_test.yml        |  4 +-
 .../FuseHorizontalContractions.cpp                 | 61 ++---------------
 .../FuseMultiUseElementwiseProducer.cpp            | 76 +++++++++++++++++-----
 .../iree/compiler/DispatchCreation/FusionUtils.cpp | 33 ++++++++++
 .../iree/compiler/DispatchCreation/FusionUtils.h   | 44 +++++++++++++
 .../test/fuse_multiuse_elementwise_producer.mlir   | 25 +++++++
 6 files changed, 169 insertions(+), 74 deletions(-)
bisect found first bad commit
```

### Development notes

Testing bisect:

```bash
pyenv shell 3.11

./bisect_packages.py \
  --good-ref=iree-3.0.0 \
  --bad-ref=iree-3.1.0rc20241122 \
  --test-script=./bisect_example_timestamp.sh

# 5b0740c97a33edce29e753b14b9ff04789afcc53 is the first bad commit
```
