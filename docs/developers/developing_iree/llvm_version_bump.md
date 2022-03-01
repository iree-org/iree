# Notes on integrating LLVM from the OSS side

This is a work in progress guide on how to bump versions of LLVM and related
dependencies. In the recent past, we did this in a different system and this
is just to get us by until we get it better scripted/automated.

Note that scripts referenced in this guide are temporarily hosted in the
[iree-samples repository](https://github.com/google/iree-samples/tree/main/scripts/integrate).
This is because it is very non-user friendly to have branch and submodule
management scripts in the repository being managed, and we don't have an
immediately better place. In this guide, we reference this location as
`$SCRIPTS`.

## Advancing the `mainline` branch in forks

The IREE team maintains fork repositories for both llvm-project and mlir-hlo,
allowing them to be patched out of band. These repositories are:

* https://github.com/google/iree-llvm-fork (`main` branch)
* https://github.com/google/iree-mhlo-fork (`mainline` branch - TODO fix this)

By the time you read this, they may be on a cron to advance automatically, but
even so, it is a good idea to advance them prior to any integrate activities
so that you have freshest commits available. Iree repository has an
action named [Advance Upstream Forks](https://github.com/google/iree/actions/workflows/advance_upstream_forks.yml)
to update llvm fork. Just select `Run Workflow` on that action and give it a
minute. You should see the fork repository `main` branch move forward.
This currently doesn't update mhlo repository. You need to use the equivalent
[action](https://github.com/google/iree-mhlo-fork/actions/workflows/advance_mainline.yaml)
in mhlo fork to update this repository.



## Bumping LLVM and Dependent Projects

### Strategy 1: Bump third_party/llvm-project in isolation

It is very common to only bump llvm-project and not sync to new versions of
mlir-hlo and tensorflow. However, as we need to periodically integrate those
as well, if the Google repositories are up to date and you have the option
to integrate to a common LLVM commit, bringing mlir-hlo and tensorflow up
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
bazel build iree/tools:iree-translate
bazel test iree/compiler/...
```

Once Bazel is good, remember to run
`./build_tools/bazel_to_cmake/bazel_to_cmake.py` and keep working on the
rest of the build.

For easy integrates, it can sometimes be easy enough to just spot the issue on
the CI and fix it directly. But if dealing with some large/breaking changes,
be prepared to settle in for a bit and play a triage role, working to get things
minimally to a point that you can shard failures to others.

Note that if not bumping mlir-hlo, then it is likely that you will hit a
compiler error in mlir-hlo at some point and will need to fix it. Advancing
it to HEAD is always an option, if that contains the fix, but this dependency
is unstable and should be maintained version locked with the integrations
directory. It is possible to advance it, but only if integrations tests pass,
and even then there is the chance for untested compatibility issues.

Typically, for the parts of mlir-hlo that we use, changes can be trivial (a
few lines, likely that you have already patched something similar in IREE).
Just make the changes in your submodule, commit them and push to a patch
branch with:

```
$SCRIPTS/patch_module.py --module_name=mlir-hlo
```

You can just do this in your integrate branch and incorporate the changes.
Typically, you will want to hold off on running `patch_module` until you are
ready to push the overall integrate branch and have some confidence that your
local fixes are sound. See the "Cherry-Picking" section below for more
details.

Good luck!

### Strategy 2: Sync everything to a Google/TensorFlow commit

TODO: Add a script for this. Also note that there is a forked copy of
`iree-dialects` in integrations/tensorflow. When bumping that dependency,
the main-project version should be copied over the integrations version.

```
cd ~/src
git clone https://github.com/tensorflow/tensorflow.git
git clone https://github.com/tensorflow/mlir-hlo.git
```

Get MHLO's published version:

We use this one because it is the easiest to get at and most of the
activity is LLVM integrates.

```
cat mlir-hlo/build_tools/llvm_version.txt
```

Or `git log` to find a commit like:

```
commit f9f696890acbe198b6164a7ca43523e2bddd630a (HEAD -> master, origin/master, origin/HEAD)
Author: Stephan Herhut <herhut@google.com>
Date:   Wed Jan 12 08:00:24 2022 -0800

    Integrate LLVM at llvm/llvm-project@c490f8feb71e

    Updates LLVM usage to match
    [c490f8feb71e](https://github.com/llvm/llvm-project/commit/c490f8feb71e)

    PiperOrigin-RevId: 421298939
```

You can correlate this with a tensorflow commit by searching TensorFlow commits
for the `PiperOrigin-RevId`, which is shared between them. While not strictly
necessary to keep all of this in sync, if doing so, it will yield fewer
surprises.

An example of a corresponding commit in the tensorflow repo:

```
commit a20bfc24dfbc34ef4de644e6bf46b41e6e57b878
Author: Stephan Herhut <herhut@google.com>
Date:   Wed Jan 12 08:00:24 2022 -0800

    Integrate LLVM at llvm/llvm-project@c490f8feb71e

    Updates LLVM usage to match
    [c490f8feb71e](https://github.com/llvm/llvm-project/commit/c490f8feb71e)

    PiperOrigin-RevId: 421298939
    Change-Id: I7e6c1c25d42f6936f626550930957f5ee522b645
```

From this example:

```
LLVM_COMMIT="c490f8feb71e"
MHLO_COMMIT="f9f696890acbe198b6164a7ca43523e2bddd630a"
TF_COMMIT="a20bfc24dfbc34ef4de644e6bf46b41e6e57b878"
```

Apply:

```
cd ~/src/iree
git fetch
(cd third_party/llvm-project && git checkout $LLVM_COMMIT)
(cd third_party/mlir-hlo && git checkout $MHLO_COMMIT)
sed -i "s/^TENSORFLOW_COMMIT = .*$/TENSORFLOW_COMMIT = \"$TF_COMMIT\"/" integrations/tensorflow/WORKSPACE

# git status should show:
#        modified:   integrations/tensorflow/WORKSPACE
#        modified:   third_party/llvm-project (new commits)
#        modified:   third_party/mlir-hlo (new commits)
```

Make a patch:

```
git add -A
git commit
# Message like:
#    Integrate llvm-project and bump dependencies.
#
#    * llvm-project: c490f8feb71e
#    * mlir-hlo: f9f696890acbe198b6164a7ca43523e2bddd630a
#    * tensorflow: a20bfc24dfbc34ef4de644e6bf46b41e6e57b878
```

Either Yolo and send a PR to have the CI run it or do a local build. I will
typically only build integrations/tensorflow if the CI indicates there is an
issue.

```
# Push to the main repo so that we can better collaboratively apply fixes.
# If there are failures, feel free to call people in who know areas for help.
git push origin HEAD:llvm-bump
```

Either fix any issues or get people to do so and land patches until the
PR is green.


## Cherry-picking

We support cherry-picking specific commits in to both llvm-project and mlir-hlo.
This should only ever be done to incorporate patches that enable further
development and which will resolve automatically as part of a future
integrate of the respective module: make sure that you are only cherry-picking
changes that have been (or will be immediately) committed upstream. For
experimental changes, feel free to push a personal branch to the fork repo
with such changes, which will let you use the CI -- but please do not commit
experimental, non-upstream committed commits to the main project.

The process for cherry-picking into llvm-project or mlir-hlo uses the same
script. The first step is to prepare a patch branch and reset your local
submodule to track it:

```
$SCRIPTS/patch_module.py --module={llvm-project|mlir-hlo}
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
