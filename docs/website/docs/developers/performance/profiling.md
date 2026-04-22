---
icon: material/chart-line
---

# Profiling overview

IREE [benchmarking](./benchmarking.md) gives us an accurate and reproducible
view of program performance at specific levels of granularity. To analyze system
behavior in more depth, there are various ways to
[profile](https://en.wikipedia.org/wiki/Profiling_(computer_programming)) IREE.

## Device profiling and replay

[Device profiling](./device-profiling.md) captures HAL-native `.ireeprof`
bundles from the devices executing a workload. Use it to inspect queue
operations, dispatch timings, memory events, executable metadata, device
metrics, and backend-specific counters or traces.

[Device replay](./device-replay.md) captures HAL work into `.ireereplay`
reproducers that can be run, benchmarked, profiled, and inspected without
re-running the original application. Replay is especially useful when you need
to capture a workload once and repeatedly benchmark or profile the same device
operation stream.

## CPU cache and other CPU event profiling

For some advanced CPU profiling needs such as querying CPU cache and other
events, one may need to use some OS-specific profilers. See
[Profiling CPUs](./profiling-cpu-events.md).

## Vulkan GPU Profiling

[Tracy](./profiling-with-tracy.md) offers great insights into CPU/GPU
interactions and Vulkan API usage details. However, information at a finer
granularity, especially inside a particular shader dispatch, is missing. To
supplement general purpose tools like Tracy, vendor-specific tools can be used.
Refer to [Profiling GPUs using Vulkan](./profiling-gpu-vulkan.md).

## Tracy

Tracy is a profiler that's been used for a wide range of profiling tasks on
IREE. Refer to [Profiling with Tracy](./profiling-with-tracy.md).
