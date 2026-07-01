---
icon: octicons/git-merge-16
---

# Integrating newer LLVM in IREE

IREE tracks LLVM closely via a submodule at `third_party/llvm-project`. This page
walks through the process of integrating a newer LLVM revision into IREE,
including building and verifying the result and pushing any required adaptions to
IREE's forked submodules.

## General points

The commands shown on this page assume a Linux environment.

## Setup environment

Create a working directory and check out the IREE repo and the LLVM submodule:

```bash
mkdir work
cd work
export WORK_DIR=$PWD
git clone https://github.com/iree-org/iree
cd iree
git submodule update --init
```

Add the upstream LLVM repo to the LLVM submodule:

```bash
cd "$WORK_DIR/iree/third_party/llvm-project"
git remote add upstream https://github.com/llvm/llvm-project.git
git fetch -pP upstream
```

Create the virtual environment for IREE work:

```bash
cd "$WORK_DIR"
uv venv venv --python 3.11
uv pip install -r iree/runtime/bindings/python/iree/runtime/build_requirements.txt -p venv
uv pip install pyright -p venv
source venv/bin/activate
```

## Integrating LLVM upstream

Update the local IREE repo and create an integration branch:

```bash
cd "$WORK_DIR/iree"
git fetch -pP origin
git checkout main
git pull --ff-only
export DATE=$(date +%Y%m%d)
git checkout -b integrates/llvm-$DATE
git submodule update --init
```

Take a note of the git hash of the LLVM repository currently used by IREE.
Check if there are any fixup commit from the earlier integrations:

```bash
cd "$WORK_DIR/iree/third_party/llvm-project"
git status  # should show detached HEAD pointing to the commit used in IREE's submodule
export CURRENT_LLVM_HASH=$(git log --oneline -1 | cut -d ' ' -f 1)
git fetch -pP upstream
export BASE_LLVM_HASH=$(git merge-base $CURRENT_LLVM_HASH upstream/main)
git log --oneline $BASE_LLVM_HASH
```

If there are no commits shown, the current LLVM integration is clean. If any
commits are shown, those were needed in one of the previous integration to get
IREE to compile. Those commits should be checked in the ongoing integration
(i.e. in the following steps) if those are still needed. The goal should be to
drop those and go back to a clean LLVM integration. If that's not possible, it
is also acceptable to keep those commits.

Create an integration branch in the LLVM submodule and rebase it on LLVM
upstream `main`:

```bash
cd "$WORK_DIR/iree/third_party/llvm-project"
git checkout -b sm-iree-integrates/llvm-$DATE
git rebase upstream/main
```

This will move the fixes shown above (if any) tot the top of the current LLVM
upstream main branch. Several things can happen with each of those commits:

- The commit might be dropped automatically: This means the same fix has been
  applied upstream and git recognized it. All is good, nothing need to be done.
- The commit becomes empty: If the commit had fixed an issue in upstream LLVM
  during the last integration and the issue has been fixed in LLVM in the
  same way, the commit might become empty. The rebase commit might halt due to
  this, point out that the commit would be empty and suggest to use
  `git rebase --continue --allow-empty`. In this case, it is best to
  continue using `git rebase --skip`, because it is desired to get closer to
  a clean LLVM integration.
- The commit might stay: The commit might apply cleanly to the top of the
  upstream main branch. This can mean that the fixup is still needed. However,
  it can also mean that the issue has been fixed in some other way, so the fixup
  might be redundant. It is always good to try if the LLVM integration works
  without this commit now.
- The commit conflicts with the upstream main branch: It is hard to tell what
  this means in general. It might mean the issue got fixed in a different way.
  It might mean LLVM upstream moved further ahead and the fix does not work any
  more. It might also be something else. This needs to be investigated. An easy
  try is to drop the commit and try if the integrate works anyways. Otherwise,
  debugging is needed.

Verify the new LLVM version by doing a trial build and run a few tests. This
basically follows the instructions to get a build started in
[getting started](./getting-started.md), but activates a few more settings in
order to catch more integration issues. Address sanitizers are enabled to
catch bugs like a buffer overrun or a use-after-free. The Python bindings are
enabled to make sure those also still build fine after the integration.

```bash
cd "$WORK_DIR"
source venv/bin/activate
cd iree
cmake -G Ninja -B "$WORK_DIR/iree-build" -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DIREE_ENABLE_ASAN=ON \
      -DIREE_BUILD_PYTHON_BINDINGS=ON \
      -DPython3_EXECUTABLE="$WORK_DIR/venv/bin/python" \
      -DIREE_ENABLE_ASSERTIONS=ON \
      -DIREE_ENABLE_SPLIT_DWARF=ON \
      -DIREE_ENABLE_THIN_ARCHIVES=ON \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DIREE_ENABLE_LLD=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cd "$WORK_DIR/iree-build"
set -o pipefail
export WORKERS=12  # please adapt based on number of cores and amount of main memory
ninja -j $WORKERS -k 999 all |& tee ninja-all.log
ninja -j $WORKERS -k 999 iree-test-deps |& tee iree-test-deps.log
```

Verify that the Bazel build works:

```bash
cd "$WORK_DIR/iree"
python configure_bazel.py
bazel build -j $WORKERS //tools/...
```

## Applying fixes

If anything goes wrong, try to fix it.

If there is a build failure in an IREE file, it is usually using a type or a
function from LLVM/MLIR that has just changes upstream. Try to identify the
type or function and look up its definition in the LLVM repository. Then, find
out if the LLVM file that defines it has been updated in the additional LLVM
commits that are being added. It is usually helpful to check the new commits
from LLVM upstream if any of those touched the file defining the type or
function:

```bash
cd "$WORK_DIR/iree/third_party/llvm-project"
git log --oneline $BASE_LLVM_HASH.. -- relative/path/of/llvm/file.h
```

If any commits are listed here, those have usually changed a type of a function
in upstream LLVM. Try to understand the change and see what needs to be adapted
in IREE. In some cases, it might be helpful to utilize an AI tool and ask it to
analyze the surfaced commit and suggest the required changes to the IREE
codebase. Always double-check the suggestions, because it might go wrong and
make the problem worse.

Serval approaches can be used to make the LLVM integration continue:

- Update the code in the IREE repo to adapt to the changed type or function.
  This is the most desired outcome, because it allows a clean LLVM integration.
  This does not cause any extra work in the following LLVM integrations.
- Cherry-pick a revert or a fixup from LLVM upstream. LLVM is moving fast. By
  the time a problematic LLVM commit has been identified during a LLVM
  integration into IREE, the issue might have been identified in LLVM upstream
  already. If so, there might already be a revert of the faulty commit or a
  fixup commit of the issue available upstream. If so, cherry-pick it.
- Decide to revert the offending LLVM upstream commit in IREE's LLVM fork in the
  integration branch. This may be needed in special cases, if it is impossible
  to adapt the IREE code base ad-hoc. However, it will cause further work in
  every following LLVM integration. So if this happens, there is urgent need to
  plan for adapting IREE to get rid of the revert soon after the completion of
  the ongoing LLVM integration.

If there are changes needed to other submodules forked by IREE, e.g. stablehlo
or torch-mlir, push the changes to branches there. (See
[Pushing changes to other forked submodules](#pushing-changes-to-other-forked-submodules)
below for copy-and-paste-able commands.)

Please re-run all the verification steps in order to make sure that the LLVM
integration works.

If all the verification steps are successful, submit the branches to the IREE
upstream repos:

```bash
cd "$WORK_DIR/iree/third_party/llvm-project"
git push origin sm-iree-integrates/llvm-$DATE:sm-iree-integrates/llvm-$DATE
export NEW_LLVM_HASH=$(git log --oneline -1 | cut -d ' ' -f 1)
cd "$WORK_DIR"
source venv/bin/activate
cd iree
git add third_party/llvm-project
git commit -s -m "Integrate LLVM to llvm/llvm-project@$NEW_LLVM_HASH"
git commit --amend  # add notes to body of commit message if any fixes / cherry-picks / reverts had been needed, mention LLVM commits that caused the need for adaptions
git push origin integrates/llvm-$DATE:integrates/llvm-$DATE
```

Open a PR to IREE upstream `main`. Get it through CI. Get it merged.

## Pushing changes to other forked submodules

Due to different LLVM integration cadences used by other submodules, e.g.
stablehlo, it is currently recommended to patch the IREE forks of those
submodules if needed during an LLVM integration into IREE.

Here are the example commands, using stablehlo as an example, for pushing some
required changes to an integration branch in the IREE fork:

```bash
cd "$WORK_DIR/iree/third_party/stablehlo"
git checkout -b sm-iree-integrates/llvm-$DATE
git add -p  # confirm changes
git commit -s -m "Adaptions to llvm/llvm-project@$NEW_LLVM_HASH for integrating LLVM into IREE"
git commit --amend  # add notes to body of commit message, mention what has been fixed and why, mention LLVM commits that caused the need for adaptions
git push origin integrates/llvm-$DATE:integrates/llvm-$DATE
```

If such a patch to a submodule is required, a follow-up needs to be done once
the LLVM integration into IREE is completed. It shall be checked in the upstream
of the respective project, if it has already integrated a new LLVM version
and potentially adapted to the issue that required patching already. If so, the
submodule shall be updated to a more recent upstream version. The process is
similar as described above for LLVM integrations. In general, going to a clean
upstream version for the submodule is most desirable. Cherry-picking a fix from
upstream or reverting a single commit is more desirable than a custom fix.

If a custom fix is needed because the submodule project has not integrated with
a recent LLVM version yet, it is recommended to open an LLVM integration PR for
the submodule project and push the developed fix to the project's upstream in
order to support the community and help IREE's fork to stay aligned with
upstream.

## After the integrate

If the integration required reverting a commit or implementing a custom fix in
LLVM or any other submodule, the issue needs to be analyzed as a follow-up. It
is possible that LLVM or other submodules have bugs, but it might also be that
a change in LLVM or a submodule surfaced a bug in IREE. Once the root-cause for
the need of the revert or fix is understood, different actions have to be taken
depending on if the issue is in IREE or in the upstream of the submodule.

If the bug is in IREE, either submit a pull request fixing it or open an issue
in the IREE project. In either case, please point out that the issue got
surfaced by a newer submodule version and that there currently is a revert or
fixup commit in IREE's fork of the submodule. In case of a fix PR, please
consider performing another submodule integration after the fix PR is merged.

If the bug is in the upstream of the submodule, please report the issue in the
submodule' upstream github project. If you can fix the issue, please open a pull
request to the submodule's upstream.
