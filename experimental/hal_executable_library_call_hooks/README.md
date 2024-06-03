# A `hal_executable_library_call` hook to study CPU event counts on Linux.

To use this, build IREE with:
1. `cmake -DCMAKE_C_FLAGS=-DIREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK .` to enable the hooks in the IREE runtime. This enables using hooks by `LD_PRELOAD=...some_hooks.so`
2. `cmake -DIREE_BUILD_EXPERIMENTAL_HAL_EXECUTABLE_LIBRARY_CALL_HOOKS=ON .` to enable building this directory, which provides such a hooks `.so` implementation.

Example:

Suppose that we have a program like this `matmul.mlir`:

```mlir
func.func @matmul_dynamic(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}
```

Compile it like usual, but just make sure that we dump the actual function names of each dispatch function, so that we will be able to filter for it:

```
tools/iree-compile ~/matmul.mlir -o /tmp/matmul.vmfb \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=znver4 \
  --iree-llvmcpu-enable-ukernels=all \
  --iree-hal-dump-executable-intermediates-to=/tmp
```

Thanks to the dumped intermediates in `/tmp` we see, for instance, that the interesting function for us is named `matmul_dynamic_dispatch_3_mmt4d_DxDxDx16x16x1_f32`.

So we run like this:

```
IREE_HOOK_FILTER_NAME=matmul_dynamic_dispatch_3_mmt4d_DxDxDx16x16x1_f32 \
IREE_HOOK_PERF_EVENT_TYPES=ls_any_fills_from_sys.all \
LD_PRELOAD=/home/benoit/iree-build/experimental/hal_executable_library_call_hooks/libiree_experimental_hal_executable_library_call_hooks_hook_cpu_events_linux.so \
tools/iree-benchmark-module --device_allocator=caching --module=/tmp/matmul.vmfb --function=matmul_dynamic --input=4000x4000xf32 --input=4000x4000xf32 --input=4000x4000xf32
```

> [!NOTE]
> This tool relies on the `perf_event_open` system call. Most Linux systems do not give sufficient permissions by default and need to be overridden by writing `0` to the file `/proc/sys/kernel/perf_event_paranoid`

We get output like this:

```
Statistics for thread iree-worker-15:
  15536 matching calls, of which:
    15536 calls on cpu 31
  duration_ms:
    mean: 52.7
    16-ile means: 39.6 44.8 46.5 47.6 48.4 49.1 49.7 50.3 50.8 51.4 52 52.8 53.9 55.4 59 92.6
  ls_any_fills_from_sys.all_dram_io:
    mean: 1.57e+03
    16-ile means: 942 1.24e+03 1.35e+03 1.44e+03 1.5e+03 1.54e+03 1.58e+03 1.61e+03 1.63e+03 1.66e+03 1.68e+03 1.71e+03 1.73e+03 1.77e+03 1.81e+03 2.02e+03
  correlation of duration_ms vs. ls_any_fills_from_sys.all_dram_io: 0.46
  conditional probability of duration_ms 16-ile (↓) given ls_any_fills_from_sys.all_dram_io 16-ile (→):
          0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
       0 ▆▆ ▃▃ ▂▁ ▁▁ ▁_ __ __ _  _  _        _  _  _  _
       1 ▃▃ ▄▄ ▃▃ ▂▂ ▁▁ ▁_ __ __ __ __ _  __ _  _     __
       2 ▂▁ ▃▃ ▃▃ ▃▂ ▂▂ ▂▁ ▁_ ▁_ ▁_ __ __ __ __ _  _  __
       3 ▁_ ▂▂ ▂▂ ▂▂ ▂▂ ▂▁ ▂▁ ▂▁ ▁▁ ▁▁ ▁▁ ▁_ ▁_ __ __ __
       4 __ ▁▁ ▂▁ ▂▂ ▂▁ ▂▂ ▂▁ ▂▁ ▁▁ ▁▁ ▁▁ ▁▁ ▁▁ ▁_ __ __
       5 _  ▁_ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▁▁ ▂▁ ▁▁ ▁_ __
       6 _  __ ▁▁ ▁▁ ▁▁ ▂▁ ▂▁ ▁▁ ▂▁ ▂▁ ▂▁ ▂▁ ▁▁ ▂▁ ▂▁ __
       7 _  __ ▁_ ▁▁ ▁▁ ▁▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▂ ▂▁ ▂▁ ▂▁ ▁▁ ▁_
       8 _  __ ▁_ ▁_ ▁▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▁▁
       9 _  _  __ ▁_ ▁▁ ▁▁ ▁▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▁▁
       a    _  _  ▁_ ▁▁ ▁▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▂ ▂▂ ▂▁
       b _  _  __ ▁_ ▁▁ ▁▁ ▁▁ ▁▁ ▂▁ ▁▁ ▂▁ ▂▁ ▂▁ ▂▂ ▂▂ ▂▁
       c    _  _  ▁_ ▁▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁
       d _  _  _  ▁_ ▁▁ ▁▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▁ ▂▂ ▂▁ ▂▁ ▂▁
       e _  __ __ ▁_ ▁_ ▁▁ ▁▁ ▂▁ ▂▁ ▂▁ ▁▁ ▂▁ ▂▁ ▂▂ ▂▁ ▂▂
       f __ _  _  __ ▁_ ▁_ ▁_ ▁_ ▁_ ▁_ ▁▁ ▁▁ ▂▁ ▂▁ ▂▂ ▅▄
```

As this is a LD_PRELOAD hook, this can't take command-line arguments, so all the settings are controlled by environment variables:
* `IREE_HOOK_FILTER_NAME`: If specified, will filter executable library calls for this specific function name. Otherwise will gather all calls, which would typically make for hard-to-interpret results. One almost always wants to specify this.
* `IREE_HOOK_SKIP_START_MS`: How many milliseconds to skip initially before recording data. Think of it as warm-up. Default 0.
* `IREE_HOOK_PERF_EVENT_TYPES`: Comma-separated list of events to count. The available event names are a subset of the ones available in Linux's `perf`. The exact list is what is dumped by `IREE_HOOK_LIST_EVENT_TYPES=1`. If multiple event types are specified, cross-event conditional probability tables will be printed for each pair of event, so this grows quadratically. In general, one will pass only one event type unless specifically interested in correlating two events. There may also be CPU-specific overhead or limits associated with querying multiple event types simultaneously.
* `IREE_HOOK_LIST_EVENT_TYPES`: if defined, will dump a list of event type to use in `IREE_HOOK_PERF_EVENT_TYPES`, and exit. Note that some are AMD-specific (indicated by `[AMD]`) while others are generic.
* `IREE_HOOK_OUTPUT_CSV`: if defined, must point to an existing directory (e.g. `/tmp/csv`) where to dump raw CSV data files. Otherwise, no CSV will be dumped, just overall stats.
* `IREE_HOOK_BUCKET_COUNT`: controls the number of buckets (i.e. percentiles) to distinguish when printing stats. Higher values result in more detailed but heavier output. Default 16.
* `IREE_HOOK_NO_PROBABILITY_TABLE`: if defined, skips printing probability tables. Their size is (bucket_count) x (bucket_count).
* `IREE_HOOK_GAMMA`: gamma-correction factor in the semigraphical probability-table rendering. Default 0.5.
