# Scripts for Integrating Changes from Third Party Dependencies

This directory contains scripts for managing some of our more storied third
party dependencies (which are submodules in the project):

* llvm-project
* stablehlo

Depending on your activity, please refer to the appropriate script,
which has comments at the top on how to use it:

* `bump_llvm.py` : Bumping LLVM and related sub-projects to a new commit
* `patch_module.py` : Push local patches (cherry-picks) from a submodule to
  a mirror for use by others.

NOTE: If needing to make changes to these scripts, DO NOT make changes while
using them to perform one of the above tasks (i.e. your changes will just
get bundled into an unrelated patch unless if very careful). In this rare
case, just copy the contents of this directory to a temporary location and
edit/run from there. When done, submit the changes to main.

What follows are some rambly notes on how to do an LLVM integrate. The integrator
needs to look at the current LLVM pinned commit and figure out which are
cherry-picked. If any commit that is cherry picked is not part of the integrate,
the integrator needs to cherry pick it again.

TODO: Refactor these based on the common procedure we actually use.

## Cookbook

### Making an immediate fix to LLVM

Generally, we only cherry-pick committed changes to LLVM into IREE. This is
great when someone else has been nice enough to have already landed such a
change. However, when something in LLVM is breaking IREE, it can be helpful
to develop the change within IREE's third_party/llvm-project submodule and
land it from there. Because this is often done as part of an integrate, you
will typically be on the integrate branch for this procedure.

General procedure:

* Make changes in third_party/llvm-project as needed and get IREE build/test
  passing locally.
* Rebase the change onto LLVM head, get reviewed and land.
* Cherry-pick the landed change into IREE's llvm fork.

This should not be common. Reserve it for NFC, reverts and obvious bug fixes to
LLVM. Most LLVM changes will take time and should not be rushed or be done
outside of the bounds of the Development Policy. If you don't know if what you
are doing is appropriate, ask on discord or ask an experienced committer for
a second opinion.

Roughly (note that this is a development procedure, not a copy-paste exercise:
know what you are doing and adapt):

```
(iree)$ ... Make changes ...
(iree/third_party/llvm-project)$ cd third_party/llvm-project
# May need to delete the 'land' branch if re-using: git branch -D land
(iree/third_party/llvm-project)$ git checkout -b land
# Make sure you have an upstream COMMIT remote (or equiv)
(iree/third_party/llvm-project)$ git remote add COMMIT git@github.com:llvm/llvm-project.git
# Rebase your 'land' branch to upstream main
(iree/third_party/llvm-project)$ git fetch COMMIT && git pull --rebase COMMIT main

# To request a code-review.
(iree/third_party/llvm-project)$ arc diff
# To land the change (double check, etc -- follow upstream policies).
(iree/third_party/llvm-project)$ git push COMMIT HEAD:main
# Run "git log" and note that commit hash you will be cherry-picking.
# Assume this is $COMMIT_HASH below

# Move back to IREE's concept of LLVM head and cherry-pick as normal.
(iree/third_party/llvm-project)$ (cd ../.. && git submodule update third_party/llvm-project)
(iree/third_party/llvm-project)$ git cherry-pick $COMMIT_HASH
(iree/third_party/llvm-project)$ cd ../..
(iree)$ ./build_tools/scripts/integrate/patch_module.py --module=llvm-project
```

## Notes on integrating LLVM from the OSS side

This is a work in progress guide on how to bump versions of LLVM and related
dependencies. In the recent past, we did this in a different system and this
is just to get us by until we get it better scripted/automated.

In this guide, we reference this directory as `$SCRIPTS`.

### Advancing the mainline branch in forks

The [shark-infra](https://github.com/shark-infra) org maintains forks of
key repositories for which we may need to carry local patches. Anyone who
contributes to the project can request access to create patch branches here.

* https://github.com/shark-infra/llvm-project (`main` branch)
* https://github.com/shark-infra/stablehlo (`main` branch)

The [fork-roller](https://github.com/shark-infra/fork-roller) repository has an
action named [Advance Upstream Forks](https://github.com/shark-infra/fork-roller/blob/main/.github/workflows/advance_forks.yml)
to update the forks. Just select `Run Workflow` on that action and give it a
minute. You should see the fork repository mainline branch move forward. This
action runs hourly. If needing up to the minute changes, you may need to trigger
it manually.

### Bumping LLVM and Dependent Projects

#### Strategy 1: Bump third_party/llvm-project in isolation

It is very common to only bump llvm-project and not sync to new versions of
stablehlo and tensorflow. However, as we need to periodically integrate those
as well, if the Google repositories are up to date and you have the option
to integrate to a common LLVM commit, bringing stablehlo and tensorflow up
to date as well, it can save some cycles overall.

In order to bump to the current ToT commit, simply run:

```
$SCRIPTS/bump_llvm.py
```

This will create a branch in the main repository like `bump-llvm-YYYYMMDD`.
If you need to specify a branch name, use the `--branch-name=` argument.
A specific upstream commit can be selected with `--llvm-commit=`.

It will print what it is doing, and at the end will show the usual GitHub
"create a PR" banner for the branch created. Open that, create a PR, patch
until green and merge. If it takes a long time and `main` has moved forward,
you likely need to pull updates:

```
git pull --rebase origin main
git push -f
```

If you have sharded out integrate work, coordinate with others on the #builds
channel before force pushing a rebase.

Actually applying necessary patches to upgrade the project is an art form.
It is usually reasonable to focus on build breaks first, and starting with
Bazel can help, especially for catching nit-picky strict things:

```
bazel build tools:iree-compile
bazel test compiler/...
```

Once Bazel is good, remember to run
`./build_tools/bazel_to_cmake/bazel_to_cmake.py` and keep working on the
rest of the build.

For easy integrates, it can sometimes be easy enough to just spot the issue on
the CI and fix it directly. But if dealing with some large/breaking changes,
be prepared to settle in for a bit and play a triage role, working to get things
minimally to a point that you can shard failures to others.

Note that if not bumping stablehlo, then it is likely that you will hit a
compiler error in stablehlo at some point and will need to fix it. Advancing
it to HEAD is always an option, if that contains the fix, but this dependency
is unstable and should be maintained version locked with the integrations
directory. It is possible to advance it, but only if integrations tests pass,
and even then there is the chance for untested compatibility issues.

Typically, for the parts of stablehlo that we use, changes can be trivial (a
few lines, likely that you have already patched something similar in IREE).
Just make the changes in your submodule, commit them and push to a patch
branch with:

```
$SCRIPTS/patch_module.py --module_name=stablehlo
```

You can just do this in your integrate branch and incorporate the changes.
Typically, you will want to hold off on running `patch_module` until you are
ready to push the overall integrate branch and have some confidence that your
local fixes are sound. See the "Cherry-Picking" section below for more
details.

Good luck!

### Update C-API exported

If a new symbol needs to be export in the C-API run this [script](https://github.com/openxla/iree/blob/main/compiler/src/iree/compiler/API/generate_exports.py)
from IREE root directory:

```
python compiler/src/iree/compiler/API/generate_exports.py
```

Missing symbols would usually cause the following kind of errors in python builf kind of error:
```
ImportError: /work/full-build-dir/compiler/bindings/python/iree/compiler/_mlir_libs/_mlir.cpython-37m-x86_64-linux-gnu.so: undefined symbol: mlirLocationFromAttribute
```

### Cherry-picking

Please add the integrator to reviewers in the cherry-pick PR, so the integrator
won't miss the commits when bumping submodules. If you don't know who is the
integrator, you can reach out to @hanchung on discord or add hanhanW as a reviewer.

We support cherry-picking specific commits in to both llvm-project and stablehlo.
This should only ever be done to incorporate patches that enable further
development and which will resolve automatically as part of a future
integrate of the respective module: make sure that you are only cherry-picking
changes that have been (or will be immediately) committed upstream. For
experimental changes, feel free to push a personal branch to the fork repo
with such changes, which will let you use the CI -- but please do not commit
experimental, non-upstream committed commits to the main project.

The process for cherry-picking into llvm-project or stablehlo uses the same
script. The first step is to prepare a patch branch and reset your local
submodule to track it:

```
$SCRIPTS/patch_module.py --module={llvm-project|stablehlo}
```

If successful, this will allocate a new branch in the fork repo with a name
like `patched-llvm-project-YYYYMMDD[.index]`, push the current submodule
commit to it and switch your submodule to a local copy of the branch, setup
for pushing.

If you have already committed changes to your branch, it should be fine to
run the script and they will be incorporated into your new patch branch.

Push any changes in your submodule as needed, and then when ready, create
a patch in the main IREE project which includes the submodule updates. In the
GitHub UI, it should show a nice drill-down of the submodule with the changes.

Note that cherry-picking is a racey process: if two people cross each other,
one of them will conflict and then Master-Git will likely need to be called.
Coordinate on cherry-picks on the #builds channel.

A submodule can receive an arbitrary number of cherry-picks. This will just
yield more patch branches to track the reachable set. (TODO: this is not
strictly true with the current script -- it will extend the current patch
branch. But we should change it to just always create a new branch as there
is less room for error).

Undoing a sequence of cherry-picks is done by integrating to a new upstream
version, presumably one that includes the commits.

## Tips for reproducing failures locally

We can run the CI pipelines either with local setup or under docker. To run it
under docker, we can find the hash from CI log.

An example from a log:

```
[18:30:23 UTC] docker run --volume=/tmpfs/src/github/iree:/tmpfs/src/github/iree --workdir=/tmpfs/src/github/iree --rm --user=1003:1004 --volume=/tmpfs/fake_etc/group:/etc/group:ro --volume=/tmpfs/fake_etc/passwd:/etc/passwd:ro --volume=/tmpfs/fake_home:/home/kbuilder --volume=/home/kbuilder/.config/gcloud:/home/kbuilder/.config/gcloud:ro gcr.io/iree-oss/frontends-swiftshader@sha256:1d2424dc512545a32b68e3f6b839541832fa24b5fce78cb253b3a4cd4592d9b2 build_tools/kokoro/gcp_ubuntu/bazel/linux/x86-swiftshader/core/build.sh
Unable to find image 'gcr.io/iree-oss/frontends-swiftshader@sha256:1d2424dc512545a32b68e3f6b839541832fa24b5fce78cb253b3a4cd4592d9b2' locally
sha256:aeb8de9fb7af3913d385ec6b274320197d61aa7bc51a6e8bc0deba644da3e405: Pulling from iree-oss/frontends-swiftshader
```

You can find the hash tag from log and run the below command. It makes sure that
you have the enviroment as same as CI bot and requires less local setup.

```
docker run --interactive --tty --rm --volume=$PWD:/src/iree --workdir=/src/iree gcr.io/iree-oss/frontends-swiftshader@sha256:1d2424dc512545a32b68e3f6b839541832fa24b5fce78cb253b3a4cd4592d9b2
```

To repro failures in `iree/e2e/`:

```bash
cmake --build . --target iree-test-deps
ctest -R iree/e2e
```

To run all the tests in `llvm-external-projects/iree-dialects`:

```bash
cmake --build . --target check-iree-dialects
```

To triage IREE gcc build issues, we can look into logs in the CI gcc job and get the gcc version. E.g.,

```bash
export CC=gcc-9
export CXX=g++-9
mkdir build && cd build

# Note that the below command disable cuda backend.
cmake -G Ninja \
  -DIREE_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_BUILD_DOCS=ON \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DIREE_HAL_DRIVER_CUDA=OFF \
  -DIREE_TARGET_BACKEND_CUDA=OFF \
  ..
```

To repro failures in CI `bazel_linux_x86-swiftshader_core`, we can follow the
[developer doc](https://iree.dev/developers/bazel) to build IREE using bazel.
E.g.,

```bash
export CC=clang
export CXX=clang++
python configure_bazel.py
cd integrations/tensorflow
bazel test ...
```
