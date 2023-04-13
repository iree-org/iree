# Profiling with Tracy

[Tracy](https://github.com/wolfpld/tracy) is a profiler that puts together in a
single view both instrumentation and system profiling (sampling, systrace). It's
key to understand the nuance here.
* *Instrumentation* is code built into the process being profiled, collecting
  timestamps at the start and end of "zones". Once it's enabled at build time,
  it typically just works &mdash; it is a part of our application logic just
  like anything else, so there's no reason why it would not work.
* *Sampling* and *SysTrace* rely on specific
  system features to collect information on what is *actually* running. These
  rely on OS and binary (ELF) file features, so they can take a bit more care to
  get to work properly.

There are two components to Tracy. They communicate over a TCP socket.

*   The "client" is the program being profiled.
*   The "server" is:
    *   Either the Tracy profiler UI (which we build as `iree-tracy-profiler`),
    *   Or the Tracy command-line capture tool (`iree-tracy-capture`) that can
        save a trace for later loading in the Tracy profiler UI.

## The Tracy manual

The primary source of Tracy documentation, including for build instructions, is
a PDF manual that's part of each numbered release.
[Download](https://github.com/wolfpld/tracy/releases/latest/download/tracy.pdf)
or
[view in browser](https://docs.google.com/viewer?url=https://github.com/wolfpld/tracy/releases/latest/download/tracy.pdf).

## Overview

We will go through each steps below, but here is an overview. It highlights the simpler subset of instructions when only instrumentation is needed, vs. the additional steps needed when Sampling is also wanted.

Component | Instrumentation only | Instrumentation and Sampling
--- | --- | ---
Build Tracy capture (`iree-tracy-capture`) | Base instructions below for [dependencies](#install-dependencies) and [build](#build-the-tracy-tools) | Same
Build Tracy profiler (`iree-tracy-profiler`) | Base instructions below for [dependencies](#install-dependencies) and [build](#build-the-tracy-tools) | Same plus [`capstone-next` instructions](#do-you-need-capstone-next) for CPU disassembly to work
Build the IREE compiler (`iree-compile`) for profiling your own modules  | [Nothing particular](#build-the-iree-compiler-iree-compile) | Same
Build the IREE compiler (`iree-compile`) for profiling the compiler itself | [Also need](#build-the-iree-compiler-iree-compile) CMake setting: `IREE_ENABLE_COMPILER_TRACING` | Same
Compile your IREE module (run `iree-compile`) | [Nothing particular](#compile-your-iree-module-run-iree-compile) | [Also need](#additional-steps-for-sampling) to pass `--iree-llvmcpu-link-embedded=false` (and also, for `llvm-cpu` backend, pass `--iree-llvmcpu-debug-symbols=true`, but that is currently default).
Build IREE device binaries (`iree-run-module` etc) | [Base instructions below](#build-iree-device-binaries-with-tracy-instrumentation-clients) (CMake: set `IREE_ENABLE_RUNTIME_TRACING`) | [Also need](#additional-steps-for-sampling-1) debug information (Set `CMAKE_BUILD_TYPE` to `RelWithDebInfo`).
Run IREE device binaries loading your modules | [Nothing particular](#running-the-profiled-program) (May need to set the environment variable `TRACY_NO_EXIT=1` for short-running benchmarks) | [Also need](#additional-steps-for-sampling-2) to set the environment variable `IREE_PRESERVE_DYLIB_TEMP_FILES` and adjust device security settings or run as root depending on OS.
Run Tracy capture (`iree-tracy-capture`) to collect the trace | If device!=host (e.g. Android), [set up TCP port forwarding](#running-the-tracy-capture-cli-connecting-and-saving-profiles). | Same
Build IREE's own tests and benchmark suites with Tracy instrumentation | [As above](#build-iree-device-binaries-with-tracy-instrumentation-clients), CMake: set `IREE_ENABLE_RUNTIME_TRACING`. | [Also need](#additional-steps-for-sampling) the CMake setting `IREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER` so that `--iree-llvmcpu-link-embedded=false` will be passed to `iree-compile`.

## Install dependencies

### Do you need capstone-next?

You can skip this section if you don't need disassembly of CPU code.

[Capstone](https://github.com/capstone-engine/capstone) is the disassembly
framework used by Tracy. The default branch, which is what OS packages still
distribute, is running a few years behind current CPU architectures.

Newer CPU architectures such as RISC-V, or newer extensions of existing
architectures (e.g. new SIMD instructions in the ARM architecture) are typically
only supported in the
[`next`](https://github.com/capstone-engine/capstone/tree/next) branch. If you
need that support, check out and build that branch. Consider uninstalling any OS
package for `capstone` or otherwise ensure that your IREE build will pick up
your `next` branch build.

### Linux

If you haven't opted to build `capstone-next` (see above section), install the
OS package for `capstone` now (Debian-based distributions):

```shell
sudo apt install libcapstone-dev
```

Install other dependencies:

```shell
sudo apt install libtbb-dev libzstd-dev libglfw3-dev libfreetype6-dev libgtk-3-dev
```

If you only build the command-line tool `iree-tracy-capture` and not the
graphical `iree-tracy-profiler`, you can install only:

```shell
sudo apt install libtbb-dev libzstd-dev
```

The zstd version on Ubuntu 18.04 is old. You will need to install it from source
from https://github.com/facebook/zstd.git

### Mac

If you haven't opted to build `capstone-next` (see above section), install the
system `capstone` now:

```shell
brew install capstone
```

Install other dependencies:

```shell
brew install glfw freetype
```

## Build the Tracy tools

A CMake-based build system for Tracy is maintained as part of IREE. In your IREE
desktop build directory, set the following CMake option:

```shell
$ cmake -DIREE_BUILD_TRACY=ON -DIREE_ENABLE_LLD=ON .
```

That enables building the Tracy server tools, `iree-tracy-profiler` and
`iree-tracy-capture`, introduced above. It also enables building the tool
`iree-tracy-csvexport` which can be used to export a captured trace as a
CSV file (see Section 6 "Exporting zone statistics to CSV" in the Tracy manual).

If profiling on Android/ARM, you might need the patch discussed in the next
paragraph.

Consider building **without** assertions (`cmake -DIREE_ENABLE_ASSERTIONS=OFF`).
At least `iree-tracy-profiler` has some
[faulty assertions](https://github.com/wolfpld/tracy/pull/382) that can cause
the profiler UI to crash during normal usage.

Rebuild, either everything or just these specific targets:

```shell
cmake --build . --target iree-tracy-profiler iree-tracy-capture iree-tracy-csvexport
```

This should have created the `iree-tracy-profiler`, `iree-tracy-capture`, and
`iree-tracy-csvexport` binaries:

```shell
$ find . -name iree-tracy-*
./tracy/iree-tracy-profiler
./tracy/iree-tracy-capture
./tracy/iree-tracy-csvexport
```

## Build the IREE compiler (`iree-compile`)

Most people don't need to rebuild `iree-compile` at all for Tracy and can skip this section.

If you want to profile `iree-compile` itself as opposed to just profiling modules compiled with it, then rebuild it with the CMake setting `IREE_ENABLE_COMPILER_TRACING` set to `ON`.

## Compile your IREE module (run `iree-compile`)

If you only want Instrumentation and not Sampling then you don't need anything particular here. Just run `iree-compile` as usual.

### Additional steps for Sampling

In order for Sampling to work with your compiled modules, add this flag to your `iree-compile` command line: `--iree-llvmcpu-link-embedded=false`.

For the `llvm-cpu` target backend, sampling features also rely on debug information in the compiled module, enabled by `--iree-llvmcpu-debug-symbols=true`, but that is currently the default.

When building IREE's own test and benchmark suites, if Tracy Sampling support is wanted, set the CMake setting `IREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER` to `ON`. It has the effect of passing that `--iree-llvmcpu-link-embedded=false` when compiling test/benchmark modules.

## Build IREE device binaries with Tracy instrumentation ("clients")

Set the CMake setting `IREE_ENABLE_RUNTIME_TRACING` to `ON` and rebuild IREE device binaries, e.g.

```
cd iree-device-build-dir
cmake -DIREE_ENABLE_RUNTIME_TRACING=ON .
cmake --build .
```

### Additional steps for Sampling

In order for Sampling features to work, make sure that binaries contain debug information. That usually means changing the `CMAKE_BUILD_TYPE` to `RelWithDebInfo` instead of `Release`.

In your IREE device build directory, set the following CMake options:

```
cd iree-device-build-dir
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .
```

## Running the profiled program

The basic recipe is to just run your program as usual on the device and, while it is running, run `iree-tracy-capture` on the host to connect to it.

In the typical case of a short-running benchmark, one usually runs with the environment variable `TRACY_NO_EXIT` defined so that the benchmark does not exit until `iree-tracy-capture` has connected to it.

Example:

```shell
TRACY_NO_EXIT=1 /data/local/tmp/iree-benchmark-module ... (usual flags)
```

### Additional steps for Sampling

In order for Sampling to work, the IREE compiled module code mapping must still be
accessible by the time Tracy tries to read symbols code. This requires setting the environment variable `IREE_PRESERVE_DYLIB_TEMP_FILES`. It is easiest to set it to `1` but one may also set it to an explicit path where to preserve the temporary files.

Example:

```shell
TRACY_NO_EXIT=1 IREE_PRESERVE_DYLIB_TEMP_FILES=1 /data/local/tmp/iree-benchmark-module ... (usual flags)
```

Tracing doesn't work properly on VMs (see "Problematic Platforms / Virtual
Machines" section 2.1.6.4 of the [manual](#the-tracy-manual)). To get sampling,
you should run the profiled program on bare metal.

## Operating system settings required for Sampling and SysTrace

### Desktop Linux

On desktop Linux, the profiled application must be run as root, e.g. with
`sudo`. Otherwise, profile data will lack important components.

### Android

When profiling on an Android device, in order to get the most useful information
in the trace, tweak system permissions as follows before profiling. This needs
to be done again after every reboot of the Android device.

From your desktop, get a shell on the Android device:

```shell
adb shell
```

The following commands are meant to be run from that Android device shell.
First, get root access for this shell:

```shell
$ su
#
```

Now run the following commands as root on the Android device:

```
setenforce 0
mount -o remount,hidepid=0 /proc
echo 0 > /proc/sys/kernel/perf_event_paranoid
echo 0 > /proc/sys/kernel/kptr_restrict
```

Note: in order for this to work, the device needs to be *rooted*, which means
that the above `su` command must succeed. This is sometimes confused with the
`adb root` command, but that's not the same. `adb root` restarts the `adbd`
daemon as root, which causes device shells to be root shells by default. This is
unnecessary here and we don't recommend it: real Android applications *never*
run as root, so Tracy/Android *has* to support running benchmarks as regular
user and it's best to stick to this for the sake of realistic benchmarks.
Internally, Tracy executes `su` commands to perform certain actions, so it too
relies on the device being *rooted* without relying on the benchmark process
being run as root.

### "RESOURCE_EXHAUSTED; failed to open file" issue

This is a
[known issue with how tracy operates](https://github.com/wolfpld/tracy/issues/512).
One way to workaround it is to manually increase the total number of files
that can be kept opened simultaneously and run the benchmark command with that
setting:
```
sudo sh -c "ulimit -n <bigNum> && <myTracyInstrumentedProgram>"
```

**Explanation:**

Tracy keeps a number of file descriptors open that, depending on the machine and
its settings, may exceed the limit allowed by the system resulting in `iree`
to fail to open more files.
In particular, it is commom to have a relatively low limit when running
with `sudo`.

## Running the Tracy Capture CLI, connecting and saving profiles

While the program that you want to profile is still running (thanks to
`TRACY_NO_EXIT=1`), start the Tracy capture tool in another terminal. From the
IREE build directory:

```shell
tracy/iree-tracy-capture -o myprofile.tracy
Connecting to 127.0.0.1:8086...
```

It should connect to the IREE client and save the output to myprofile.tracy that
can be visualized by the client below. You can start the capture tool first to
make sure you don't miss any capture events.

Note that the connection uses TCP port 8086. If the Tracy-instrumented program
is running on a separate machine, this port needs to be forwarded. In
particular, when benchmarking on Android, this is needed:

```shell
adb forward tcp:8086 tcp:8086
```

## Running the Tracy profiler UI, connecting and visualizing

If you have previously captured a tracy file (previous section), this command
should succeed loading it (from the IREE build directory):

```shell
tracy/iree-tracy-profiler myprofile.tracy
```

Alternatively, while the program that you want to profile is still running
(possibly thanks to `TRACY_NO_EXIT=1`), the Tracy profiler can connect to it
directly (so it is not required to capture the trace into a file): just running

```shell
tracy/iree-tracy-profiler
```

should show a dialog offering to connect to a client i.e. a profiled program:

![Tracy connection dialog](https://gist.github.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_connect.jpeg)

If connecting doesn't work:

*   If the profiled program is on a separate machine, make sure you've correctly
    set up port forwarding.
*   On Android, the `adb forward` may need to be run again.
*   Make sure that the profiled program is still running. Do you need
    `TRACY_NO_EXIT=1`?
*   Kill the profiled program and restart it.

You should now start seeing a profile. The initial view should look like this:

![Tracy initial view, normal case](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_initial_view.jpeg)

Before going further, take a second to check that your recorded profile data has
all the data that it should have. Permissions issues, as discussed above, could
cause it to lack "sampling" or "CPU data" information, particularly on Android.
For example, here is what he initial view looks like when one forgot to run the
profiled program as root on Desktop Linux (where running as root is required, as
explained above):

![Tracy initial view, permissions issue](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_permissions_issue.jpeg)

Notice how the latter screenshot is lacking the following elements:

*   No 'CPU data' header on the left side, with the list of all CPU cores. The
    'CPU usage' graph is something else.
*   No 'ghost' icon next to the 'Main thread' header.

Click the 'Statistics' button at the top. It will open a window like this:

![Tracy statistics window](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_statistics.jpeg)

See how the above screenshot has two radio buttons at the top: 'Instrumentation'
and 'Sampling'. At this point, if you don't see the 'Sampling' radio button, you
need to resolve that first, as discussed above about possible permissions
issues.

These 'Instrumentation' and 'Sampling' statistics correspond the two kinds of
data that Tracy collects about your program. In the Tracy main view, they
correspond, respectively, to 'instrumentation' and 'ghost' zones. Refer to the
[Tracy PDF manual](#the-tracy-manual) for a general introduction to these
concepts. For each thread, the ghost icon toggles the view between these two
kinds of zones.

Back to the main view, look for the part of the timeline that is of interest to
you. Your area of interest might not be on the Main thread. In fact, it might be
on a thread that's not visible in the initial view at all. To pan around with
the mouse, hold the **right mouse button** down (or its keyboard equivalent on
macOS). Alternatively, look for the 'Frame' control at the top of the Tracy
window. Use the 'next frame' arrow button until more interesting threads appear.

IREE module code tends to run on a thread whose name contains the word `worker`.

Once you have identified the thread of interest, you typically want to click its
ghost icon to view its "ghost" (i.e. sampling) zones.

Here is what you should get when clicking on a ghost zone:

![ghost zone source view](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_source_view.jpeg)

The percentages column to the left of the disassembly shows where time is being
spent. This is unique to the sampling data (ghost zones) and has no equivalent
in the instrumentation data (instrumentation zones). Here is what we get
clicking on the corresponding instrumentation zone:

![instrumentation zone source view](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_normal_zone_info.jpeg)

This still has a 'Source' button but that only shows the last C++ caller that
had explicit Tracy information, so here we see a file under `iree/hal` whereas
the Ghost zone saw into the IREE compiled module that that calls into, with the
source view pointing to the `.mlir` file.

## Configuring Tracy instrumentation

Set IREE's `IREE_TRACING_MODE` value (defined in
[iree/base/tracing.h](https://github.com/openxla/iree/blob/main/iree/base/tracing.h))
to adjust which tracing features, such as allocation tracking and callstacks,
are enabled.
