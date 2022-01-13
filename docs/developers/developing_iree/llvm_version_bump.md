# Notes on integrating LLVM from the OSS side

This is a work in progress guide on how to bump versions of LLVM and related
dependencies. In the recent past, we did this in a different system and this
is just to get us by until we get it better scripted/automated.

## Strategy 1: Sync everything to a Google/TensorFlow commit

```
cd ~/src
git clone git clone https://github.com/tensorflow/tensorflow.git
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


## Strategy 2: Bump individual pieces

There are fragile interfaces between the frontends and backends that are not
guaranteed across patches to llvm-project/tensorflow/mlir-hlo; however, in
reality, these things are often quite stable.

At present, it is possible to bump any project independently. If it passes
tests, it should be fine. If it fails, it likely indicates an incompatibility.
The choice in such situation is to either wait and choose a better sync point
later or carry local patches. We are still working on setting up repo mirrors
for local patches to dependent projects.
