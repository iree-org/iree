# Profiling

IREE [benchmarking](./benchmarking.md) gives us an accurate and reproducible
view of program performance at specific levels of granularity. To analyze
system behavior in more depth, there are various ways to
[profile](https://en.wikipedia.org/wiki/Profiling_(computer_programming))
IREE.

## Whole-system Profiling with Tracy

IREE uses Tracy as the main tool to perform whole-system profiling.
[Tracy](https://github.com/wolfpld/tracy) is a real-time, nanosecond resolution,
remote telemetry, hybrid frame and sampling profiler. Tracy can profile CPU,
GPU, memory, locks, context switches, and much more.

### Building Tracy

To use tracing in IREE, you need to build IREE with following requirements:

*   Set `IREE_ENABLE_RUNTIME_TRACING` to `ON`.
*   Use Release/RelWithDebInfo build.

For example:

```shell
$ export IREE_DEFAULT_COPTS='-DNDEBUG'
$ cmake -B build/ \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

The above compiles IREE with Tracy APIs so that IREE will stream profiling data
back to Tracy when running. To collect and analyze these data, you can either
use GUI or CLI tools. Tracy profiler is the GUI tool. You can find the
Tracy manual on its [releases page](https://github.com/wolfpld/tracy/releases)
for more details on Tracy itself.

#### Building on Linux

To build the profiler on Linux, you may need to install some external
libraries. Some Linux distributions will require you to add a `lib` prefix and a
`-dev`, or `-devel` postfix to library names. For example, you might see the
error:

```
Package glfw3 was not found in the pkg-config search path.
```

and then you could try to install `libglfw3-dev`.

Instructions to build Tracy profiler:

```shell
$ cd third_party/tracy/profiler/build/unix
$ make release
```

### Using Tracy

Launch the profiler UI and click connect to start waiting for a traced program
to running. Now you can launch the IREE binary you want to trace and Tracy
should connect automatically and stream data. For example:

Compile a .mlir file using `iree-translate`:

```shell
$ build/iree/tools/iree-translate \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=vmla \
  $PWD/iree/tools/test/simple.mlir \
  -o /tmp/simple.vmfb
```

Run a compiled module once:

```shell
$ build/iree/tools/iree-run-module \
  --module_file=/tmp/simple.vmfb \
  --driver=vmla \
  --entry_function=abs \
  --function_inputs="i32=-2"
```

Benchmark a compiled module, running it many times:

```shell
$ build/iree/tools/iree-benchmark-module \
  --module_file=/tmp/simple.vmfb \
  --driver=vmla \
  --entry_function=abs \
  --function_inputs="i32=-2"
```

> Note:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;IREE binaries may finish running before even
> connecting to Tracy. For such cases, you can set `TRACY_NO_EXIT=1` in the
> environment to keep the IREE binary alive until Tracy connects to it.

### Configuring Tracy

Set IREE's `IREE_TRACING_MODE` value (defined in
[iree/base/tracing.h](https://github.com/google/iree/blob/main/iree/base/tracing.h))
to adjust which tracing features, such as allocation tracking and callstacks,
are enabled.

In order for Tracy to record detailed statistics via sampling, the program
collecting data must be run using elevated permissions (Administrator on Windows,
root on Linux, rooted Android device). See Tracy's user manual for more
information.

## Vulkan GPU Profiling

Tracy offers great insights into CPU/GPU interactions and Vulkan API usage
details. However, information at a finer granularity, especially inside a
particular shader dispatch, is missing. To supplement general purpose tools
like Tracy, vendor-specific tools can be used.

(TODO: add some pictures for each tool)

### Android GPUs

There are multiple GPU vendors for the Android platforms, each offering their
own tools. [Android GPU Inspector](https://gpuinspector.dev/)
(AGI) provides a cross-vendor solution. See the
[documentation](https://gpuinspector.dev/docs/) for more details.

#### Build Android app to run IREE

In order to perform capture and analysis with AGI, you will need a full Android
app. In IREE we have a simple Android native app wrapper to help package
IREE core libraries together with a specific VM bytecode invocation into an
Android app. The wrapper and its documentation are placed at
[`iree/tools/android/run_module_app/`](https://github.com/google/iree/tree/main/iree/tools/android/run_module_app).

For example, to package a module compiled from the following `mhlo-dot.mlir` as
an Android app:

```mlir
func @dot(%lhs: tensor<2x4xf32>, %rhs: tensor<4x2xf32>) -> tensor<2x2xf32>
  attributes { iree.vmfb.export } {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
```

```shell
# First translate into a VM bytecode module
$ /path/to/iree/build/iree/tools/iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vulkan \
  /path/to/mhlo-dot.mlir \
  -o /tmp/mhlo-dot.vmfb

# Then package the Android app
$ /path/to/iree/source/iree/tools/android/run_module_app/build_apk.sh \
  ./build-apk \
  --module_file /tmp/mhlo-dot.vmfb \
  --entry_function dot \
  --function_inputs_file /path/to/inputs/file \
  --driver vulkan
```

Where `/path/to/input/file` is a file containing inputs to `dot`, for example:

```
2x4xf32=[[1.0 2.0 3.0 4.0][5.0 6.0 7.0 8.0]]
4x2xf32=[[9.0 10.0][11.0 12.0][13.0 14.0][15.0 16.0]]
```

The above will build an `iree-run-module.apk` under the `./build-apk/`
directory, which you can then install via `adb install`.

`build_apk.sh` needs the Android SDK and NDK internally, an easy way to manage
them is by installing [Android Studio](https://developer.android.com/studio).
After installation, you will need to set up a few environment variables, which
are printed at the beginning of `build_apk.sh` invocation.

#### Capture and analyze with AGI

You can follow AGI's
[Getting Started](https://gpuinspector.dev/docs/getting-started) page to learn
how to use it. In general the steps are:

* Install the latest AGI from https://github.com/google/agi/releases and launch.
* Fill in the "Application" field by searching the app. The line should read
  like `android.intent.action.MAIN:com.google.iree.run_module/android.app.NativeActivity`.
* Select start at beginning and choose a proper duration.
* Configure system profile to include all GPU counters.
* Start capture.

Generated traces are in the [perfetto](https://perfetto.dev/) format. They can
be viewed directly within AGI and also online in a browser at
https://ui.perfetto.dev/, without needing an Android device.

### Desktop GPUs

Vulkan supports both graphics and compute, but most tools in the Vulkan
ecosystem focus on graphics. As a result, some Vulkan profiling tools expect
commands to correspond to a sequence of frames presented to displays via
framebuffers. This means additional steps for IREE and other Vulkan
applications that solely rely on headless compute. For graphics-focused tools,
we need to wrap IREE's logic inside a dummy rendering loop in order to provide
the necessary markers for these tools to perform capture and analysis.

IREE provides an `iree-run-module-vulkan-gui` binary that can invoke a specific
bytecode module within a proper GUI application. The graphics side is leveraging
[Dear ImGui](https://github.com/ocornut/imgui); it calls into IREE
synchronously during rendering each frame and prints the bytecode invocation
results to the screen.

To build `iree-run-module-vulkan-gui`:

```shell
# Using Bazel
$ bazel build //iree/testing/vulkan:iree-run-module-vulkan-gui

# Using CMake
$ cmake --build /path/to/build/dir --target iree-run-module-vulkan-gui
```

The generated binary should be invoked in a console environment and it takes
the same command-line options as the main
[`iree-run-module`](./developer-overview.md#iree-run-module), except the
`--driver` option. You can use `--help` to learn them all. The binary will
launch a GUI window for use with Vulkan tools.

#### AMD

For AMD GPUs, [Radeon GPU Profiler](https://gpuopen.com/rgp/) (RGP) is the tool
to understand fine details of how IREE GPU performs. See the
[documentation](https://radeon-gpuprofiler.readthedocs.io/en/latest/) for
details. In general the steps to get started are:

* Download and install AMD RGP from https://gpuopen.com/rgp/.
* Compile `iree-run-module-vulkan-gui` as said in the above.
* Open "Radeon Developer Panel" and connect to the local
  "Radeon Developer Service".
* Start `iree-run-module-vulkan-gui` from console with proper VM bytecode module
  invocation.
* You should see it in the "Applications" panel of "Radeon Developer Panel".
  Click "Capture profile" to capture.

Afterwards you can analyze the profile with RGP. Viewing the profile does not
need the GPU anymore; it can be opened by a RGP application installed anywhere.

#### NVIDIA

For NVIDIA GPUs, [NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics)
is the tool to understand fine details of how IREE GPU performs. See the
[documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/index.html)
for details. In general the steps to get started are:

* Download and install NVIDIA Nsight Graphics from https://developer.nvidia.com/nsight-graphics.
* Compile `iree-run-module-vulkan-gui` as said in the above.
* Open NVIDIA Nsight Graphics, select "Quick Launch" on the welcome page.
* Fill out the "Application Executable" and "Command Line Arguments" to point
  to `iree-run-module-vulkan-gui` and a specific VM bytecode module and its
  invocation information.
* Select an "Activity" ("Frame Profiler" and "GPU Trace" are particularly
  interesting) and launch.
* Capture any frame to perform analysis.
