---
icon: material/chart-line
---

# Device profiling

IREE device profiling captures HAL-native records from the devices that execute
your workload. A `.ireeprof` file can contain queue submissions, command buffer
metadata, dispatch timings, host execution spans, memory lifecycle events,
periodic device metrics, executable metadata, and backend-specific counter or
trace artifacts.

Use device profiling when you need to answer questions below the VM invocation
level:

* Which HAL queue operations did this invocation issue?
* Which dispatches, transfers, or allocations dominate device time?
* Which executable export name should be optimized?
* Did a benchmark replay spend time in useful HAL work, setup, copies, or host
  execution?
* What hardware counters or executable trace artifacts did a backend collect
  for a selected dispatch?

Device profiling complements [benchmarking](./benchmarking.md) and
[Tracy](./profiling-with-tracy.md). Benchmarks tell you how long a workload
takes. Tracy shows process-wide runtime behavior and CPU/GPU API interactions.
Device profiling records the HAL/device work in a structured format that can be
queried with `iree-profile` or exported for other tooling.

## Capture with IREE tools

Tools that create HAL devices accept the `--device_profiling_*` flags. For
example, capture queue and device queue events while running a module:

```shell
iree-run-module \
  --device=amdgpu \
  --module=/tmp/model.vmfb \
  --function=main \
  --input=@/tmp/inputs.txt \
  --device_profiling_mode=queue-events,device-queue-events \
  --device_profiling_output=/tmp/model.ireeprof
```

The same flags work with `iree-benchmark-module`:

```shell
iree-benchmark-module \
  --device=amdgpu \
  --module=/tmp/model.vmfb \
  --function=main \
  --benchmark_min_time=20x \
  --device_profiling_mode=queue-events,device-queue-events \
  --device_profiling_output=/tmp/model.ireeprof
```

When profiling a benchmark, prefer fixed iteration counts such as
`--benchmark_min_time=20x` while first validating a workflow. This makes the
capture easier to compare across runs and avoids accidentally collecting a very
large profile from a short microbenchmark.

The benchmark replay tool also accepts the same flags. That is the usual way to
capture a profile from an already-recorded HAL workload:

```shell
iree-benchmark-replay \
  --device=amdgpu \
  --benchmark_min_time=50x \
  --device_profiling_mode=queue-events,device-queue-events \
  --device_profiling_output=/tmp/model-replay.ireeprof \
  /tmp/model.ireereplay
```

`iree-benchmark-replay` flushes profile data outside the timed benchmark
iteration. The `.ireeprof` still describes the replayed HAL work, but profile
serialization is not charged to the benchmark timing.

## Profiling modes

`--device_profiling_mode` is a comma-separated list of HAL profiling data
families. The selected HAL driver decides which families it supports and must
fail loudly for unsupported requested data.

| Mode | Records requested |
| --- | --- |
| `queue-events` | Host-timestamped queue operation records, dependency strategy, operation counts, and transfer byte totals. |
| `host-execution` | Host execution spans, such as local dispatch bodies or host-side command buffer replay. |
| `device-queue-events` | Device-timestamped queue operation spans. |
| `dispatch-events` | Device-timestamped dispatch execution records. |
| `memory-events` | HAL memory allocation, reservation, pool, and buffer lifecycle records. |
| `device-metrics` | Periodic device metrics such as clocks, temperature, power, memory occupancy, utilization, or bandwidth when supported. |
| `counters` | Explicitly selected hardware or software counter samples. Requires one or more `--device_profiling_counter=` flags. |
| `executable-metadata` | Executable, code object, and export metadata needed for offline analysis. |
| `executable-traces` | Heavy executable trace artifacts such as AMDGPU ATT/SQTT traces for selected operations. |

Use the narrowest mode set that captures the data you need. Some modes are
cheap metadata streams; others can insert device packets, allocate large trace
buffers, or perturb the workload enough that the resulting timing should not be
treated as ordinary benchmark data.

## Useful capture flags

Common filters select which operations emit expensive artifacts while leaving
cheap metadata available for decoding:

```shell
--device_profiling_filter_export='*matmul*'
--device_profiling_filter_command_buffer=3
--device_profiling_filter_command_index=12
--device_profiling_filter_physical_device=0
--device_profiling_filter_queue=1
```

For counter capture, select the `counters` mode and pass backend-specific
counter names:

```shell
iree-benchmark-module \
  --device=amdgpu \
  --module=/tmp/model.vmfb \
  --function=main \
  --benchmark_min_time=20x \
  --device_profiling_mode=counters \
  --device_profiling_counter=SQ_WAVES \
  --device_profiling_counter=SQ_BUSY_CYCLES \
  --device_profiling_filter_export='*matmul*' \
  --device_profiling_output=/tmp/model-counters.ireeprof
```

Long-running workloads can request periodic flushes:

```shell
--device_profiling_flush_interval_ms=1000
```

Use periodic flushing only when the backend documents that in-flight snapshots
are safe for the selected data families. A flush may be a no-op for producers
that do not buffer completed records.

For a quick aggregate report without writing a `.ireeprof`, use:

```shell
--print_device_statistics
```

This starts the backend's lightweight statistics mode and prints aggregate
device statistics at shutdown. It cannot be combined with
`--device_profiling_output`.

External profiler/tool capture flags are separate from HAL-native `.ireeprof`
output:

```shell
--device_capture_tool=renderdoc
--device_capture_file=/tmp/frame.rdc
--device_capture_label=warmup-frame
```

These flags control provider-specific artifacts such as RenderDoc captures or
Metal GPU traces. Success does not imply that a `.ireeprof` bundle was written.

## Capture from C

Embedding applications can capture the same profile bundles directly through
the HAL API. Create a sink, begin profiling on the devices you want to observe,
run the workload, and end profiling.

```c
#include "iree/hal/api.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_handle.h"

iree_io_file_handle_t* file_handle = NULL;
IREE_RETURN_IF_ERROR(iree_io_file_handle_create(
    IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN |
        IREE_IO_FILE_MODE_SHARE_READ,
    IREE_SV("/tmp/model.ireeprof"), /*initial_size=*/0, host_allocator,
    &file_handle));

iree_hal_profile_sink_t* sink = NULL;
iree_status_t status =
    iree_hal_profile_file_sink_create(file_handle, host_allocator, &sink);
iree_io_file_handle_release(file_handle);
IREE_RETURN_IF_ERROR(status);

iree_hal_device_profiling_options_t options = {0};
options.data_families =
    IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
    IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS;
options.sink = sink;

status = iree_hal_device_profiling_begin(device, &options);
if (iree_status_is_ok(status)) {
  /* Run the workload while profiling is active. */
  status = iree_hal_device_profiling_end(device);
}
iree_hal_profile_sink_release(sink);
IREE_RETURN_IF_ERROR(status);
```

For multi-device applications, begin profiling on each HAL device that
participates in the workload and end profiling after all captured work has
completed. Unless the selected backend documents dynamic toggling support,
externally serialize `iree_hal_device_profiling_begin`,
`iree_hal_device_profiling_flush`, and `iree_hal_device_profiling_end` with
queue submission and command buffer recording on the same device. In practice,
start profiling before issuing the workload and end it after the device is idle
or after the application-visible synchronization point you care about.

## Inspect with iree-profile

`iree-profile` is the command line reader for `.ireeprof` bundles:

```shell
iree-profile summary /tmp/model.ireeprof
```

Example summary output:

```text
IREE HAL profile summary
records: file=8 session_begin=1 chunks=6 session_end=1 non_ok_session_end=0 unknown=0
chunks: devices=1 queues=1 executables=1 executable_exports=1 command_buffers=0
event_records: queue_events=3 host_execution_events=4 host_execution_duration_ns=68574
devices:
  device[0]: device_records=1 queues=1/1
    dispatches=0 valid=0 invalid=0
```

Use projection commands to enter the profile from the object you care about:

```shell
iree-profile explain /tmp/model.ireeprof
iree-profile dispatch /tmp/model.ireeprof
iree-profile queue --format=jsonl /tmp/model.ireeprof
iree-profile command --id=3 --format=jsonl --dispatch_events \
  /tmp/model.ireeprof
iree-profile memory --format=jsonl /tmp/model.ireeprof
iree-profile executable --filter='*matmul*' /tmp/model.ireeprof
```

`statistics` is useful for compact scripts:

```shell
iree-profile statistics /tmp/model.ireeprof
```

Example statistics output:

```text
IREE HAL device statistics:
  aggregate_rows=10
  dispatch_export_total=0 ns
  host_execution_queue_total=68.574 us
  host_queue p=0 q=0 alloca  count=1 total=24.027 us avg=24.027 us operations=1 payload=8B
  host_execute abs_dispatch_0_elementwise_2_f32  count=1 total=681 ns avg=681 ns
```

For automation, prefer JSONL projections:

```shell
iree-profile dispatch --format=jsonl /tmp/model.ireeprof | \
  jq 'select(.type=="dispatch_group") | {key,avg_ns,count}'

iree-profile queue --format=jsonl /tmp/model.ireeprof | \
  jq 'select(.type=="queue_event" or .type=="queue_submission")'

iree-profile export --format=ireeperf-jsonl \
  --output=/tmp/model.ireeperf.jsonl /tmp/model.ireeprof
```

Report JSONL rows are keyed by `type` and are intended for command-local
drilldown. `export --format=ireeperf-jsonl` emits a schema-versioned
interchange stream keyed by `record_type`; use that for long-lived downstream
adapters and telemetry imports.

## Compose with replay

Device profiling and [device replay](./device-replay.md) are intentionally
separate:

* `.ireereplay` says what HAL work to run.
* `.ireeprof` says what happened while a run executed.

A common workflow is:

```shell
iree-run-module \
  --device=amdgpu \
  --module=/tmp/model.vmfb \
  --function=main \
  --input=@/tmp/inputs.txt \
  --device_replay_output=/tmp/model.ireereplay

iree-benchmark-replay \
  --device=amdgpu \
  --benchmark_min_time=50x \
  --device_profiling_mode=queue-events,device-queue-events \
  --device_profiling_output=/tmp/model-replay.ireeprof \
  /tmp/model.ireereplay

iree-profile explain /tmp/model-replay.ireeprof
```

This captures the application once, then profiles deterministic replayed HAL
work as many times as needed while iterating on drivers, devices, or executable
substitution.

## Appendix: ATT and executable traces

When built with AMDGPU profiling support, `iree-profile att` can decode
AMDGPU ATT/SQTT trace artifacts embedded in a `.ireeprof` bundle:

```shell
iree-benchmark-module \
  --device=amdgpu \
  --module=/tmp/model.vmfb \
  --function=main \
  --benchmark_min_time=20x \
  --device_profiling_mode=executable-traces,dispatch-events,executable-metadata \
  --device_profiling_filter_export='*matmul*' \
  --device_profiling_output=/tmp/model-att.ireeprof

iree-profile att \
  --rocm_library_path=/opt/rocm/lib \
  --filter='*matmul*' \
  /tmp/model-att.ireeprof
```

Executable traces are heavy and should be captured with a narrow filter. If
`--rocm_library_path` is omitted, the tool falls back to
`IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH`, `IREE_HAL_AMDGPU_LIBHSA_PATH`, and then
the system dynamic library search path.

## Appendix: Agent-oriented help

The profile tool can print a compact Markdown playbook for humans or agents:

```shell
iree-profile --agents_md
```

Use this when building scripts around `iree-profile` or when you need the
current command list, JSONL row families, and cross-reference recipes from the
exact binary in your build tree.
