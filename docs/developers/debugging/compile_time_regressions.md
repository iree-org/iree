# Debugging Compile Time Regressions

So the IREE compiler used to compile a program quickly, but it is now slower.
What do you do?

## Initial information gathering

Try to answer as many of these questions as you can:

* **When did compilation get slower?**

  A specific git commit is ideal, but "sometime in the last week" is a good
  starting point. You'll ultimately want to find a culprit release or git
  commit that changed the compiler code.

* **How much slower did compilation get?**

  Be specific - did it jump from 1 minute to 2 minutes, or 1 minute to 1 hour?
  Identifying the scale of the regression can help set the priority to
  investigate it.

* **What is the full compile command?**

  Try to extract the input program and full list of flags passed to the
  compiler binary so that others can reproduce what you're seeing. Try to
  distill this as much as possible to using just native tools (no Python or
  other framework layers).

* **What environment is the compiler running in?**

  Are you using a `Debug` build, or a release build? What operating system and
  size machine is running the compiler (e.g. Linux developer machine, or a
  smaller system)?

## Culprit finding and bisecting

If you only have a rough idea of when something changed and want to narrow that
down to a specific code change, bisecting can help.

### Running `git bisect`

Building the compiler from source and using
[`git bisect`](https://git-scm.com/docs/git-bisect) will let you pinpoint
specific commits in IREE, though it typically won't let you step through changes
in submodules (e.g. MLIR updates in `third_party/llvm-project/`).

__Tip__: [Configure ccache](../developing_iree/ccache.md) if you'll be rebuilding the compiler while bisecting

A manual workflow with `git bisect` looks like this:

```bash
git bisect start --first-parent
git bisect good [<rev>]
git bisect bad [<rev>]

# Read the prompts from the command as it runs
# At each step, test the compiler:
#   git submodule update
#   cmake --build build/ --target iree-compile
#   ./build/tools/iree-compile <args>
#       attach Tracy, observe timing, print IR, etc. to determine if fast or slow
#       if fast, `git bisect good`
#       if slow, `git bisect bad`
#   repeat
```

An automated workflow can use `git bisect run` and a script:

```shell
# run_bisect.sh
git submodule update
cmake --build build/ --target iree-compile
# Other logic here
```

```bash
git bisect start --first-parent
git bisect good [<rev>]
git bisect bad [<rev>]
git bisect run run_bisect.sh
```

Other sample scripts:

<details><summary>Compile executable sources individually with a timeout:</summary>

```bash
#!/bin/bash

set -xeuo pipefail

# --------------------------------------------------------------------------- #
# Settings                                                                    #
# --------------------------------------------------------------------------- #

INPUT_FILE_PATH="/path/to/program.mlirbc"
TMP_DIR="../iree-tmp"

declare -a COMPILER_FLAGS=(
  "--iree-input-type=stablehlo"
  "--iree-hal-target-backends=cuda"
  "--iree-hal-cuda-llvm-target-arch=sm_80"
)

TIMEOUT_SECONDS_FOR_COMPILING_EACH_SOURCE=10

# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #

# Call to have `git bisect` skip this commit (don't mark as good _or_ bad)
# https://git-scm.com/docs/git-bisect#_bisect_run
skip_on_error() {
  >&2 echo "** Skipping due to error: $1 **"
  exit 125  # Special exit code for `git bisect skip`
}

# --------------------------------------------------------------------------- #
# Main script                                                                 #
# --------------------------------------------------------------------------- #

# Store git version hash, so we can dump artifacts to unique directories later.
GIT_SHA="$(git rev-parse --short HEAD)"

echo "** Building iree-compile at ${GIT_SHA} **"

# The `git bisect` command only checks out a commit, so update submodules.
git submodule update

# Build the compiler. You'll want ccache configured to make this fast!
cmake --build ../iree-build/ --target iree-compile || skip_on_error "CMake build failed"

# Run the compiler, dumping executable sources and stopping.
SOURCES_DIR="${TMP_DIR}/sources-${GIT_SHA}"
echo "** Running iree-compile at ${GIT_SHA}, dumping sources to ${SOURCES_DIR} **"
../iree-build/tools/iree-compile \
    ${INPUT_FILE_PATH} \
    ${COMPILER_FLAGS[@]} \
    --iree-hal-dump-executable-sources-to=${SOURCES_DIR} \
    --compile-to=executable-sources \
    -o /dev/null

# Run the compiler again on each executable individually.
echo "** Running iree-compile at ${GIT_SHA} for each executable source **"
SOURCES=($(ls -1 ${SOURCES_DIR}))
for SOURCE in "${SOURCES[@]}"; do
  echo "  * Compiling: ${SOURCE} *"
  timeout --verbose ${TIMEOUT_SECONDS_FOR_COMPILING_EACH_SOURCE} \
   ../iree-build/tools/iree-compile ${SOURCES_DIR}/${SOURCE} \
    ${COMPILER_FLAGS[@]} \
    --compile-mode=hal-executable \
    -o /dev/null
done
```

</details>

## Profiling and tracing

If you want to understand _why_ the compiler is fast or slow, or if you want to
compare performance in detail between two versions, consider these profiling
options.

### MLIR pass timing

The `-mlir-timing` flag enables
[Pass Timing](https://mlir.llvm.org/docs/PassManagement/#pass-timing)
instrumentation. Once the compiler finishes running, this prints a report like
```shell
===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0203 seconds

   ---Wall Time---  --- Name ---
   0.0047 ( 55.9%)  Canonicalizer
   0.0019 ( 22.2%)  VerifierPass
   0.0016 ( 18.5%)  LLVMLoweringPass
   0.0003 (  3.4%)  CSE
   0.0002 (  1.9%)  (A) DominanceInfo
   0.0084 (100.0%)  Total
```

This is easy data to collect, especially remotely over SSH, but it might not
paint a complete picture and requires waiting for compilation to finish.

### Using Tracy

See our documentation on
[profiling with Tracy](../developing_iree/profiling_with_tracy.md). For compile
time regressions, pay particular attention to the different compilation phases
(Flow/Stream/HAL), how many times `TranslateExecutablesPass` runs, and if there
are outlier passes that take significantly longer to run than others.

Here are some previous analyses for inspiration:

* https://github.com/openxla/iree/issues/12033
* https://github.com/openxla/iree/issues/12035
* https://github.com/openxla/iree/issues/12183
* https://github.com/openxla/iree/issues/13189

Example slow trace:

![slow trace](https://user-images.githubusercontent.com/4010439/233436147-2fa0fbb3-80cd-474c-bfff-3441c2d8f8fc.png)

Example fast trace:

![fast trace](https://user-images.githubusercontent.com/4010439/233455673-7469066b-2b0d-4462-b6a5-3af4a502e591.png)

Example sampling statistics showing 10s of minutes in LLVM codegen:

![slow LLVM codegen](https://user-images.githubusercontent.com/4010439/233441298-3c4f5afa-d1cc-43b3-8900-58652f295fe2.png)

## Stepping through compiler IR

Debugging an MLIR-based compiler like IREE usually involves reading IR at some
point. For compile time regressions, it helps to snapshot the IR at a few key
phases and look for differences between fast compilation and slow compilation.

Here is one useful flag combination:
```shell
--mlir-disable-threading \
--mlir-elide-elementsattrs-if-larger=8 \
--mlir-print-ir-after=iree-hal-materialize-interfaces
```
