# Profiling

IREE [benchmarking](./benchmarking.md) gives us an overall view of the
performance at a specific level of granularity. To understand performance
details of the IREE system, one would need to
[profile](https://en.wikipedia.org/wiki/Profiling_(computer_programming))
IREE, in order to know how components at different levels function and interact.

## Whole-system Profiling with Tracy

IREE uses Tracy as the main tool to perform whole-system profiling.
[Tracy](https://github.com/wolfpld/tracy) is a real-time, nanosecond resolution,
remote telemetry, hybrid frame and sampling profiler. It can profile CPU, GPU,
memory, locks, context switches, and many more.

### Building Tracy

To use tracing in IREE, you need to build IREE with following requirements:

*   Turn `IREE_ENABLE_RUNTIME_TRACING` on.
*   Add `-DNDEBUG` to `IREE_DEFAULT_COPTS`.
*   Use Release/RelWithDebInfo build.

For example:

```shell
export IREE_DEFAULT_COPTS='-DNDEBUG'
cmake -B build/ \
      -DIREE_ENABLE_RUNTIME_TRACING=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

The above compiles IREE with Tracy APIs so that IREE will stream profiling data
back to Tracy when running. To be able to collect and analyze these data, you
can either use GUI or CLI tools. Tracy profiler is the GUI tool. You can find
the
[Tracy manual](https://github.com/wolfpld/tracy/releases/download/v0.6.3/tracy.pdf)
for more details on Tracy itself.

To build the profiler on Linux, you will need to install some external
libraries. Some Linux distributions will require you to add a `lib` prefix and a
`-dev`, or `-devel` postfix to library names. For example, you might see the
error:

```
Package glfw3 was not found in the pkg-config search path.
```

and then you could try to install `libglfw3-dev`.

Instructions to build Tracy profiler:

```shell
cd third_party/tracy/profiler/build/unix
make release
```

### Using Tracy

Launch the profiler UI, and click connect. Then the server will wait for the
connection. Now you can launch the IREE binary you want to trace, it should
connect automatically and stream data. For example:

Prepare the module to profile:

```shell
build/iree/tools/iree-benchmark-module \
  --module_file=/tmp/module.fb \
  --driver=vmla \
  --entry_function=abs \
  --function_inputs="i32=-2"
```

Run the module:

```shell
build/iree/tools/iree-run-module \
  --module_file=/tmp/module.fb \
  --driver=vmla \
  --entry_function=abs \
  --function_inputs="i32=-2"
```

## Vulkan GPU Profiling

Tracy gives us great insights over CPU/GPU interactions and Vulkan API usage
details. However, information at a finer granularity, especially inside a
particular shader dispatch, is missing. To supplement, one would typically need
to use other third-party or vendor-specific tools.

### Android GPUs

There are multiple GPU vendors for the Android platforms. One can use tools
provided by the GPU vendor. [Android GPU Inspector](https://gpuinspector.dev/)
(AGI) provides a cross-vendor solution. See the
[documentation](https://gpuinspector.dev/docs/) for more details.

#### Build Android app to run IREE

In order to perform capture and analysis with AGI, one will need a full Android
app. In IREE we have a simple Android native app wrapper to help package
IREE core libraries together with a specific VM bytecode invocation into an
Android app. The wrapper and its documentation is placed at
[`iree/tools/android/run_module_app/`](https://github.com/google/iree/tree/main/iree/tools/android/run_module_app).

For example, to package `iree/tools/test/simple.mlir` as an Android app:

```shell
# First translate into VM bytecode module
/path/to/iree/build/iree/tools/iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vmla \
  /path/to/iree/source/iree/tools/test/simple.mlir \
  -o /tmp/simple.vmfb

# Then package the Android app
/path/to/iree/source/iree/tools/android/run_module_app/build_apk.sh \
  ./build-apk \
  --module_file ./iree/tools/test/simple.mlir \
  --entry_function abs \
  --function_inputs_file /path/to/inputs/file \
  --driver vulkan
```

Where `/path/to/input/file` is a file containing inputs to `abs`, for example,
`i32=-2`.

The above will build an `iree-run-module.apk` under the `./build-apk/`
directory. One can then install via `adb install`.

`build_apk.sh` needs Android SDK and NDK internally. And easy way to manage
them is by installing the [Android Studio](https://developer.android.com/studio).
After installation, you will need to set up a few environment variables, which
are printed at the beginning of `build_apk.sh` invocation.

#### Capture and analyze with AGI

You can follow AGI's [get started](https://gpuinspector.dev/docs/getting-started)
page to learn how to use it. In general the steps are:

* Install the latest AGI from https://github.com/google/agi/releases and launch.
* Fill in the "Application" field by searching the app. The line should read
  like `android.intent.action.MAIN:com.google.iree.run_module/android.app.NativeActivity`.
* Select start at beginning and choose a proper duration.
* Configure system profile to include all GPU counters.
* Start capture.

The generated trace is in perfetto format. To view the trace does not need the
device anymore. It can be viewed directly with AGI and also online in a browswer
at https://ui.perfetto.dev/.

### Desktop GPUs

Vulkan is traditionally used for grahpics rendering. So Vulkan profiling tools
at the moment typically have such assumption and require a rendering boundary
marked by framebuffer presentation. This means additional steps for IREE and
other Vulkan applications that solely rely on headless compute. We need to wrap
the core IREE logic inside a dummy rendering loop in order to provide tools the
necessary markers to perform capture and analysis.

IREE provides an `iree-run-module-vulkan-gui` binary that can invoke a specific
bytecode module within a proper GUI application. The graphics side is leveraging
[Dear ImGui](https://github.com/ocornut/imgui); it invokes IREE core
synchronously during rendering each frame and prints the bytecode invoation
result to the screen.

To build `iree-run-module-vulkan-gui`:

```shell
# Using Bazel
bazel build //iree/testing/vulkan:iree-run-module-vulkan-gui

# Using CMake
cmake --build /path/to/build/dir --target iree-run-module-vulkan-gui
```

The generated binary should be invoked in a console environment and it takes
the same command-line options as the main
[`iree-run-module`](./developer-overview.md#iree-run-module), except the
`--driver` option. You can use `--help` to learn them all. The binary will
invoke a GUI window to let one to use Vulkan tools.

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

For NVIDIA GPUs, [NVIDIA Nsight Grahpics](https://developer.nvidia.com/nsight-graphics)
is the tool to understand fine details of how IREE GPU performs. See the
[documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/index.html)
for details. In general the steps to get started are:

* Download and install NVIDIA Nsight Grahpics from https://developer.nvidia.com/nsight-graphics.
* Compile `iree-run-module-vulkan-gui` as said in the above.
* Open NVIDIA Nsight Grahpics, select "Quick Launch" on the welcome page.
* Fill out the "Application Executable" and "Command Line Arguments" to point
  to `iree-run-module-vulkan-gui` and a specific VM bytecode module and its
  invocation information.
* Select an "Activity" ("Frame Profiler" and "GPU Trace" are particularly
  intereting) and launch.
* Capture any frame to perform analysis.