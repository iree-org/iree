---
hide:
  - tags
tags:
  - GPU
  - Vulkan
icon: material/chart-line
---

# Profiling GPUs using Vulkan

[Tracy](./profiling-with-tracy.md) offers great insights into CPU/GPU
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
[documentation](https://developer.android.com/agi) for more details.

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
