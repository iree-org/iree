# Profiling GPUs using Vulkan

[Tracy](./profiling_with_tracy.md) offers great insights into CPU/GPU
interactions and Vulkan API usage
details. However, information at a finer granularity, especially inside a
particular shader dispatch, is missing. To supplement general purpose tools
like Tracy, vendor-specific tools can be used.

(TODO: add some pictures for each tool)

## RenderDoc

Support for [RenderDoc](https://github.com/baldurk/renderdoc) can be enabled by
configuring cmake with `-DIREE_ENABLE_RENDERDOC_PROFILING=ON`. When built in to
IREE the profiling functionality is available for programmatic use via the
`iree_hal_device_profiling_begin` and `iree_hal_device_profiling_end` APIs.

When using one of the standard IREE tools (`iree-run-module`,
`iree-benchmark-module`, etc) the `--device_profiling_mode=queue` flag can be
passed to enable capture around the entire invocation (be careful when
benchmarking as the recordings can be quite large!). The default capture file
name can be specified with `--device_profiling_file=foo.rdc`.

Capturing in the RenderDoc UI can be done by specifying the IREE tool or
embedding application (`iree-run-module`, etc) as the launch executable and
adding all arguments as normal.

Capturing from the command line can be done using `renderdoccmd` with the
specified file appearing (by default) in the executable directory:

```shell
renderdoccmd capture tools/iree-run-module --device_profiling_mode=queue --device_profiling_file=foo.rdc ...
stat tools/foo.rdc
renderdoccmd capture tools/iree-run-module --device_profiling_mode=queue --device_profiling_file=/some/path/foo.rdc ...
stat /some/path/foo.rdc
```

## Android GPUs

There are multiple GPU vendors for the Android platforms, each offering their
own tools. [Android GPU Inspector](https://gpuinspector.dev/)
(AGI) provides a cross-vendor solution. See the
[documentation](https://gpuinspector.dev/docs/) for more details.

### Build Android app to run IREE

In order to perform capture and analysis with AGI, you will need a full Android
app. In IREE we have a simple Android native app wrapper to help package
IREE core libraries together with a specific VM bytecode invocation into an
Android app. The wrapper and its documentation are placed at
[`tools/android/run_module_app/`](https://github.com/openxla/iree/tree/main/tools/android/run_module_app).

For example, to package a module compiled from the following
`stablehlo-dot.mlir` as an Android app:

```mlir
func @dot(%lhs: tensor<2x4xf32>, %rhs: tensor<4x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
```

```shell
# First compile into a VM bytecode module
$ /path/to/iree/build/tools/iree-compile -- \
  --iree-input-type=stablehlo \
  --iree-hal-target-backends=vulkan-spirv \
  /path/to/stablehlo-dot.mlir \
  -o /tmp/stablehlo-dot.vmfb

# Then package the Android app
$ /path/to/iree/source/tools/android/run_module_app/build_apk.sh \
  ./build-apk \
  --device vulkan \
  --module /tmp/stablehlo-dot.vmfb \
  --function dot \
  --input=...
```

Where `/path/to/input/file` is a file containing inputs to `dot`, for example:

``` text
2x4xf32=[[1.0 2.0 3.0 4.0][5.0 6.0 7.0 8.0]]
4x2xf32=[[9.0 10.0][11.0 12.0][13.0 14.0][15.0 16.0]]
```

The above will build an `iree-run-module.apk` under the `./build-apk/`
directory, which you can then install via `adb install`.

`build_apk.sh` needs the Android SDK and NDK internally, an easy way to manage
them is by installing [Android Studio](https://developer.android.com/studio).
After installation, you will need to set up a few environment variables, which
are printed at the beginning of `build_apk.sh` invocation.

### Capture and analyze with AGI

You can follow AGI's
[Getting Started](https://gpuinspector.dev/docs/getting-started) page to learn
how to use it. In general the steps are:

* Install the latest AGI from <https://github.com/google/agi/releases> and launch.
* Fill in the "Application" field by searching the app. The line should read
  like `android.intent.action.MAIN:dev.iree.run_module/android.app.NativeActivity`.
* Select start at beginning and choose a proper duration.
* Configure system profile to include all GPU counters.
* Start capture.

Generated traces are in the [perfetto](https://perfetto.dev/) format. They can
be viewed directly within AGI and also online in a browser at
<https://ui.perfetto.dev/>, without needing an Android device.

## Desktop GPUs

Vulkan supports both graphics and compute, but most tools in the Vulkan
ecosystem focus on graphics. As a result, some Vulkan profiling tools expect
commands to correspond to a sequence of frames presented to displays via
framebuffers. This means additional steps for IREE and other Vulkan
applications that solely rely on headless compute. For graphics-focused tools,
we need to wrap IREE's logic inside a dummy rendering loop in order to provide
the necessary markers for these tools to perform capture and analysis.

### AMD

For AMD GPUs, [Radeon GPU Profiler](https://gpuopen.com/rgp/) (RGP) is the tool
to understand fine details of how IREE GPU performs. See the
[documentation](https://radeon-gpuprofiler.readthedocs.io/en/latest/) for
details.

### NVIDIA

For NVIDIA GPUs, [NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics)
is the tool to understand fine details of how IREE GPU performs. See the
[documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/index.html)
for details.
