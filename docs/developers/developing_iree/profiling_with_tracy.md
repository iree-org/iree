# Profiling with Tracy

[Tracy](https://github.com/wolfpld/tracy) is a profiler that puts together in a
single view:

*   Both CPU and GPU profiling.
*   Both sampling and instrumentation.
*   Both specifics of our own process, and whole-system profiling a la
    "systrace".

Since Tracy relies on instrumentation, it requires IREE binaries to be built
with a special flag to enable it.

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
$ cmake -DIREE_BUILD_TRACY=ON .
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

## Build IREE binaries with Tracy instrumentation ("clients")

In your IREE device build directory, set the following CMake options:

```shell
$ cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=ON \
  .
```

The `IREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER` option is only needed for
Tracy to see into IREE CPU codegen module code in any IREE benchmark or test
that involves such modules. Its effect is to pass
`--iree-llvm-link-embedded=false` to the compiler, so when you build CPU-codegen
modules by manually invoking `iree-compile`, you also need to pass that flag in
order for the resulting code to be transparent to Tracy. You can omit that if
you only need Tracy to see into the IREE runtime, leaving IREE CPU codegen
modules opaque.

For tracing the compiler, additionally set `IREE_ENABLE_COMPILER_TRACING` to
`ON`. Compiler tracing is less stable, particularly on Linux with MLIR threading
enabled (https://github.com/iree-org/iree/issues/6404).

Once done configuring CMake, proceed to rebuild, e.g.

```shell
cmake --build .
```

Or if interested in running the benchmark suites,

```shell
cmake --build . --target iree-benchmark-suites
```

## Running the profiled program

There are platform-specific additional prerequisites to get sampling to work,
but we will get to that below, focusing for now on the basic recipe:

Run the instrumented program as usual, but with the following environment
variables set:

*   `TRACY_NO_EXIT=1`
*   `IREE_PRESERVE_DYLIB_TEMP_FILES=1`

Example:

```shell
TRACY_NO_EXIT=1 IREE_PRESERVE_DYLIB_TEMP_FILES=1 \
  /data/local/tmp/iree-benchmark-module \
    --driver=local-task \
    --module=/data/local/tmp/android_module.fbvm \
    --function=serving_default \
    --input=1x384xi32
```

Explanation:

*   `TRACY_NO_EXIT=1` ensures that your program does not exit until a Tracy
    server (either `iree-tracy-capture` or `iree-tracy-profiler`) has connected
    to it and obtained the trace.
*   `IREE_PRESERVE_DYLIB_TEMP_FILES=1` is only needed if you want Tracy to see
    into IREE CPU codegen module code. It is also possible to pass an explicit
    path, e.g. `IREE_PRESERVE_DYLIB_TEMP_FILES=/tmp/iree-tmpfiles` (make sure to
    create that directory), to better control proliferation of temporary files.

Tracing doesn't work properly on VMs (see "Problematic Platforms / Virtual
Machines" section 2.1.6.4 of the [manual](#the-tracy-manual)). To get sampling,
you should run the profiled program on bare metal.

### Permissions issues on desktop Linux

On desktop Linux, the profiled application must be run as root, e.g. with
`sudo`. Otherwise, profile data will lack important components.

### Permissions issues on Android

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

## "RESOURCE_EXHAUSTED; failed to open file" issue

This is a
[known issue with how tracy operates](https://github.com/wolfpld/tracy/issues/512).
One way to workaround it is to manually increase the total number of files
that can be kept opened simultanously and run the benchmark command with that
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
[iree/base/tracing.h](https://github.com/iree-org/iree/blob/main/iree/base/tracing.h))
to adjust which tracing features, such as allocation tracking and callstacks,
are enabled.
