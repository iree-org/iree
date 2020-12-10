---
layout: default
permalink: developing-iree/profiling-with-tracy
title: Profiling with Tracy
parent: Developing IREE
---

# Profiling with Tracy
{: .no_toc }

[Tracy](https://github.com/wolfpld/tracy) is a profiler that puts together in a
single view:
* Both CPU and GPU profiling.
* Both sampling and instrumentation.
* Both specifics of our own process, and whole-system profiling a la "systrace".

Since Tracy relies on instrumentation, it requires IREE binaries to be built
with a special flag to enable it.

There are two components to Tracy. They communicate over a TCP socket.
* The "client" is the program being profiled.
* The "server" is the Tracy profiler UI.

## The Tracy manual

The primary source of Tracy documentation, including for build instructions, is
a PDF manual that's part of each numbered release. To find the latest one,
navigate [here](https://github.com/wolfpld/tracy/releases) and search for
`tracy.pdf`.

## Building the Tracy UI (the "server")

This is explained in section 2.3 of the [manual](#the-tracy-manual) for Windows
and Linux. Here we give some more detailed instructions for some systems.

The IREE repository contains its own clone of the Tracy repository in
`third_party/tracy`, so there is no need to make a separate clone of it. You can
use one if you want, but be aware that the Tracy client/server protocol gets
updated sometimes. Building both sides from the same `iree/third_party/tracy`
lowers the risk of running into a protocol version mismatch.

### Linux

Install dependencies (Debian-based distributions):
```
sudo apt install libcapstone-dev libtbb-dev libglfw3-dev libfreetype6-dev libgtk-3-dev
```

Build (from your `iree/` clone root directory):
```
make -C third_party/tracy/profiler/build/unix -j12 release
```

### Mac

TODO write this (Kojo?)

## Building IREE with Tracy instrumentation (the "client")

IREE needs to be build with Tracy instrumentation enabled. This enables both the
collection of data, and its streaming to the Tracy server over a socket.

This is only supported in the CMake build system of IREE, not in Bazel.

In the initial CMake configuration command:
*   Set `IREE_ENABLE_RUNTIME_TRACING` to `ON`.
*   Use the `RelWithDebInfo` build type.

For example:

```shell
$ cmake \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  ... # other cmake arguments as usual
```
## Permissions issues

The profiled application (i.e. the Tracy client) needs to have appropriate
permissions so perform the special I/O required to collect the profile
information. This is OS-specific.

### Desktop Linux

On desktop Linux, the Tracy client must be run as root, e.g. with `sudo`.
Otherwise, profile data will lack important components.

### Android

On Android it is not necessary to run as root and in fact, Android graphical
applications never run as root, so it's advisable to run all programs as
non-root for consistency.

The Android device must be prepared as follows to enable Tracy profiling.
* The device must be rooted.
  * That means that in `adb shell`, the command `su` must succeed.
  * That does NOT mean doing `adb root`. The effect of `adb root` is to have the
    `adbd` daemon itself run as root, which causes `adb shell` to give you a
    root shell by default. If you are in that case, consider doing `adb unroot`
    to restart the `adbd` server as non-root. Not mandatory, but again, running
    anything as root on Android is a deviation from normal user conditions.
* Execute the following commands in a root shell on the device (i.e. `adb
  shell`, then `su`, then the following commands). These are from the
  [manual](#the-tracy-manual), but hard to find there, and copy-pasting from PDF
  introduces unwanted whitespace. These settings normally persist until the next
  reboot of the device.
  * `setenforce 0`
  * `mount -o remount,hidepid=0 /proc`
  * `echo 0 > /proc/sys/kernel/perf_event_paranoid`

## Port forwarding

The Tracy client and server communicate by default over port `8086`. When they
run on different machines, e.g. with embedded/Android profiling or remote
profiling, port forwarding must be set up.

### Between a computer and a local Android device connected to it by USB

Run this command. You might need to run it again more a little frequently than
you reboot the device. When experiencing connection issues, try that first.

```shell
adb forward tcp:8086 tcp:8086
```

### Between two computers over the network

TODO write this (`ssh` stuff...)

## Running the profiled program

Run your IREE workload as you normally would: now that it's been built with
Tracy instrumentation enabled, it should do all the right things automatically.

The only change that you are likely to need in your command line is to set the
`TRACY_NO_EXIT=1` environment variable. This ensures that your program does not
exit until the Tracy server (the UI) has connected to it and finished uploading
the profile data.

Typically, `TRACY_NO_EXIT=1` is needed when profiling `iree-benchmark-module`.
It wouldn't be needed when profiling a real user-facing application.

Example:

```shell
TRACY_NO_EXIT=1 /data/local/tmp/iree-benchmark-module \
  --driver=dylib \
  --function_inputs='1x384xi32,1x384xi32,1x384xi32' \
  --module_file=/data/local/tmp/android_module.fbvm \
  --entry_function=serving_default
```

## Running the Tracy profiler UI, connecting and visualizing

While the program that you want to profile is still running (possibly thanks to
`TRACY_NO_EXIT=1`), start the Tracy profiler UI which we had built above. From
the IREE root directory:
```shell
./third_party/tracy/profiler/build/unix/Tracy-release
```

It should show a dialog offering to connect to a client i.e. a profiled program:

![Tracy connection
dialog](https://gist.github.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_connect.jpeg)

If connecting doesn't work:
* If the profiled program is on a separate machine, make sure you've correctly
  set up port forwarding.
  * On Android, the `adb forward` many need to be run again.
* Make sure that the profiled program is still running. Do you need
  `TRACY_NO_EXIT=1`?
* Kill the profiled program and restart it.

You should then start seeing a profile. The initial view should look like this:

![Tracy initial view, normal
case](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_initial_view.jpeg)

Before going further, take a second to check that your recorded profile data has
all the data that it should have. Permissions issues, as discussed above, could
cause it to lack "sampling" or "CPU data" information. For example, here is what
he initial view looks like when one forgot to run the profiled program as root
on Desktop Linux (where running as root is required, as explained above):

![Tracy initial view, permissions
issue](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_permissions_issue.jpeg)

Notice how the latter screenshot is lacking the following elements:
* No 'CPU data' header on the left side, with the list of all CPU cores. The
  'CPU usage' graph is something else.
* No 'ghost' icon next to the 'Main thread' header.

When running into any of the above issues, refer to the above Permissions
section. Look for any interesting `stderr` message (in the profiled program's
terminal).

Click the 'Statistics' button at the top. It will open a window like this:

![Tracy statistics
window](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_statistics.jpeg)

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
on a thread that's not visible in the initial view at all. Indeed, the initial
view tends to be zoomed-in a lot. Either use the mouse directly to zoom out and
navigate, to look for the 'Frame' control at the top of the Tracy window. Use
the 'next frame' arrow button until more interesting threads appear. Typically,
IREE generated code tends to run on a thread named `cpu0`, which is actually a
thread name and unrelated to `CPU0` from the systrace view.

Once you have identified the thread of interest, use its ghost icon to toggle
between instrumentation and ghost zones, and zoom until you have found the zone
of interest.

Here is what you should get when clicking on a ghost zone:

![ghost zone source
view](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_source_view.jpeg)

The percentages column to the left of the disassembly shows where time is being
spent. This is unique to the sampling data (ghost zones) and has no equivalent
in the instrumentation data (instrumentation zones). Here is what we get
clicking on the corresponding instrumentation zone:

![instrumentation zone source
view](https://gist.githubusercontent.com/bjacob/ff7dec20c1dfc7d0fc556cc7275bca9a/raw/fe4e22ca0301ebbfd537c47332a4a2c300a417b3/tracy_normal_zone_info.jpeg)

This still has a 'Source' button but that only shows the last C++ caller that
had explicit Tracy information, so here we see a file under `iree/hal` whereas
the Ghost zone saw into the IREE compiled module that that calls into, with the
source view pointing to the `.mlir` file.

## Configuring Tracy instrumentation

Set IREE's `IREE_TRACING_MODE` value (defined in
[iree/base/tracing.h](https://github.com/google/iree/blob/main/iree/base/tracing.h))
to adjust which tracing features, such as allocation tracking and callstacks,
are enabled.