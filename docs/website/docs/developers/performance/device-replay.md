---
icon: material/replay
---

# Device replay

IREE device replay records the HAL work issued by an application into a
`.ireereplay` file. The replay can then be executed, benchmarked, profiled, or
dumped without running the original application or VM invocation.

Replay is a HAL-level reproducer. It is not a module-level recording of VM
inputs and outputs, and it is not a profiler by itself. It records enough of
the HAL resource graph and operation stream to reproduce useful device work:

* the retained `iree_hal_device_group_t` topology;
* devices, allocators, executable caches, executables, command buffers,
  semaphores, buffers, and files;
* host-visible byte payloads written through HAL map, unmap, update, and file
  read operations;
* executable payloads and executable metadata;
* command buffer operations, direct queue operations, and synchronization
  edges;
* external file references for large stable inputs such as parameter archives.

Replay and [device profiling](./device-profiling.md) compose cleanly:
`.ireereplay` says what HAL work to run, and `.ireeprof` says what happened
while a run executed. A common performance loop is "capture once, replay many
times, profile the replay."

## Capture with IREE tools

`iree-run-module` and `iree-benchmark-module` can wrap the resolved HAL device
group and write a replay stream:

```shell
iree-run-module \
  --device=amdgpu \
  --module=/tmp/model.vmfb \
  --function=main \
  --input=@/tmp/inputs.txt \
  --parameters=model=/models/llama.irpa \
  --device_replay_output=/tmp/model.ireereplay \
  --device_replay_file_policy=reference
```

The replay recorder is shared by all devices in the selected device group, so
multi-device host calls are emitted into one ordered stream. Device-visible
ordering is still the recorded semaphore, event, command buffer, and barrier
graph; replay does not infer FIFO ordering from host record order.

Capture from `iree-benchmark-module` the same way:

```shell
iree-benchmark-module \
  --device=amdgpu \
  --module=/tmp/model.vmfb \
  --function=main \
  --benchmark_min_time=20x \
  --device_replay_output=/tmp/model.ireereplay \
  --device_replay_file_policy=reference
```

The capture includes the HAL work issued by the benchmark run. The recorder is
closed after the tool's HAL work completes, so the file header contains the
final logical length when the process exits successfully.

## File policies

Large models often use external parameter files. The replay recorder must avoid
turning every capture into a copy of a 40 GB or 1 TB parameter archive, while
still making it clear what storage the replay depends on.

`--device_replay_file_policy=` controls imported fd-backed HAL files:

| Policy | Behavior | Use when |
| --- | --- | --- |
| `reference` | Record the external path and validation metadata. Do not copy file bytes. | The referenced file will be preserved beside the capture. This is the default and the right policy for large `.irpa` files. |
| `capture-ranges` | Embed only byte ranges read through HAL `queue_read` operations. Replay substitutes those reads with queue updates. | You need a hermetic correctness replay and do not need to benchmark storage-backed reads. |
| `capture-all` | Embed every byte of each fd-backed file. | Files are small, or the external files cannot be preserved. This can make captures enormous. |
| `fail` | Reject fd-backed files. Host-allocation-backed files are still embedded inline. | Tests must prove they do not depend on external files. |

`--device_replay_file_validation=` controls validation for referenced files:

| Validation | Behavior | Cost |
| --- | --- | --- |
| `identity` | Record cheap platform identity metadata such as file length, device, inode, and modification time. | Default. Does not scan file contents. |
| `digest` | Record and validate a content digest. | Opt-in only. Reads every byte during capture and replay. |

Use `digest` only when referenced files will be copied or staged to a different
filesystem and platform identity cannot be preserved. For very large parameter
files, digest validation can dominate capture and replay setup time.

## Capture from C

Applications that already work through the HAL should wrap their retained
`iree_hal_device_group_t`. The wrapped group preserves topology order and
contains replacement devices that record operations before forwarding to the
real devices.

```c
#include "iree/hal/replay/recorder.h"
#include "iree/io/file_handle.h"

iree_io_file_handle_t* file_handle = NULL;
IREE_RETURN_IF_ERROR(iree_io_file_handle_create(
    IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN |
        IREE_IO_FILE_MODE_SHARE_READ,
    IREE_SV("/tmp/model.ireereplay"), /*initial_size=*/0, host_allocator,
    &file_handle));

iree_hal_replay_recorder_options_t options =
    iree_hal_replay_recorder_options_default();
options.external_file_policy =
    IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_REFERENCE;
options.external_file_validation =
    IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_IDENTITY;

iree_hal_replay_recorder_t* recorder = NULL;
iree_status_t status = iree_hal_replay_recorder_create(
    file_handle, &options, host_allocator, &recorder);
iree_io_file_handle_release(file_handle);
IREE_RETURN_IF_ERROR(status);

iree_hal_device_group_t* replay_group = NULL;
status = iree_hal_replay_wrap_device_group(recorder, base_group,
                                           host_allocator, &replay_group);
if (iree_status_is_ok(status)) {
  /* Use replay_group wherever the application would normally use base_group. */
  iree_hal_device_group_release(replay_group);
}

status = iree_status_join(status, iree_hal_replay_recorder_close(recorder));
iree_hal_replay_recorder_release(recorder);
IREE_RETURN_IF_ERROR(status);
```

Close the recorder after all HAL work and host-visible writes have reached
their HAL boundaries. Closing writes the final file length and reports any
terminal recorder failure instead of silently producing a partial replay.

## Run a replay

Use `iree-run-replay` to execute a capture once:

```shell
iree-run-replay --device=amdgpu /tmp/model.ireereplay
```

The target device group must match the captured topology closely enough for the
recorded HAL operations. A mismatch in device count or unsupported operation is
a hard failure. That is intentional: silently skipping unsupported HAL work
would produce a misleading reproducer.

If the capture references files from a different mount root, remap the prefix
before replay opens them:

```shell
iree-run-replay \
  --device=amdgpu \
  --replay_file_remap=/mnt/capture=/mnt/replay \
  /tmp/model.ireereplay
```

The remapped file must still satisfy the recorded validation metadata.

## Benchmark and profile a replay

Use `iree-benchmark-replay` to measure deterministic replayed HAL work:

```shell
iree-benchmark-replay \
  --device=amdgpu \
  --benchmark_min_time=50x \
  /tmp/model.ireereplay
```

Example benchmark output:

```text
-------------------------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------
BM_replay/process_time/real_time      0.138 ms        0.138 ms            3 items_per_second=7.23988k/s
```

Add profiling flags to collect a `.ireeprof` bundle from the benchmarked replay
iterations:

```shell
iree-benchmark-replay \
  --device=amdgpu \
  --benchmark_min_time=50x \
  --device_profiling_mode=queue-events,device-queue-events \
  --device_profiling_output=/tmp/model-replay.ireeprof \
  /tmp/model.ireereplay

iree-profile explain /tmp/model-replay.ireeprof
```

`iree-benchmark-replay` performs profile flushes outside the timed region. The
profiling output describes the useful replay work without charging profile
serialization to each benchmark iteration.

## Dump and query a replay

Use text mode for a human-readable summary and record stream:

```shell
iree-dump-replay --format=text /tmp/model.ireereplay
```

Example output:

```text
IREE HAL replay v1.0
file_length: 9107
header_length: 24
summary:
  hermetic: yes
  environment_referenced: no
  strict_replay_supported: yes
  records: total=42 objects=12 operations=29 unsupported=0
  files: total=0 external=0 inline=0 ranges=0 unknown=0
records:
  @408 #4 operation   dev=1 obj=1 rel=3 thread=0 status=OK object=device(1) op=device.create_executable_cache(9)
  @920 #9 operation   dev=1 obj=3 rel=4 thread=0 status=OK object=executable_cache(6) op=executable_cache.prepare_executable(402)
  @5115 #14 operation dev=1 obj=5 rel=0 thread=0 status=OK object=command_buffer(5) op=command_buffer.dispatch(313)
```

Use JSONL for scripts and agent workflows:

```shell
iree-dump-replay --format=jsonl /tmp/model.ireereplay | \
  jq 'select(.kind=="operation" and .operation=="device.queue_execute")'
```

Example JSONL row:

```json
{"kind":"operation","sequence_ordinal":27,"object_type":"device","operation":"device.queue_execute","payload":{"command_buffer_id":5,"wait_semaphores":[{"semaphore_id":7,"value":1}],"signal_semaphores":[{"semaphore_id":9,"value":1}]}}
```

The dumper reports blob payloads as byte ranges in the original replay file.
That keeps large captures queryable and lets generated projections refer to the
capture without emitting a second giant sidecar.

## Substitute executables

Replay can replace captured executable payloads at execution time. This is
useful when iterating on generated kernels while preserving the same captured
HAL workload, inputs, synchronization, and benchmark harness.

First find executable ids:

```shell
iree-dump-replay --format=jsonl /tmp/model.ireereplay | \
  jq 'select(.operation=="executable_cache.prepare_executable") |
      {executable_id:.related_object_id, format:.payload.format_range}'
```

Then substitute a replacement:

```shell
iree-run-replay \
  --device=amdgpu \
  --replay_executable_substitution=4=/tmp/new-kernel.hsaco \
  /tmp/model.ireereplay
```

If the target executable cache needs an explicit format, include it in the
selector:

```shell
iree-benchmark-replay \
  --device=amdgpu \
  --benchmark_min_time=50x \
  --replay_executable_substitution=4@amdgcn-amd-amdhsa--gfx1100=/tmp/new-kernel.hsaco \
  /tmp/model.ireereplay
```

For captures that should use one replacement for every executable, use the
`all` selector:

```shell
iree-run-replay \
  --device=amdgpu \
  --replay_executable_substitution=all=/tmp/new-kernel.bin \
  /tmp/model.ireereplay
```

Substitution is strict. Replay validates available executable metadata, export
counts, reflected ABI shape, constants, bindings, and workgroup information
before dispatching a replacement. An ABI mismatch fails before the replacement
can silently benchmark a different program.

## Current fidelity contracts

Replay should fail loudly when it cannot reproduce the captured HAL work. These
failures are part of the contract:

* Missing or identity-mismatched external files mean replay might point at the
  wrong parameter archive. Fix the path with `--replay_file_remap` or restore
  the referenced file.
* Persistent host write maps without an observable flush or unmap boundary are
  rejected because replay cannot see the final byte contents.
* Host calls, channels, collectives, allocator import/export, and opaque
  external handles are visible in dumps and fail in strict execution until they
  have replay semantics.
* Imported or exported external buffers are not replayed as best-effort
  snapshots because the application can mutate them outside observable HAL map,
  flush, or update operations.
* Target topology matters. Select a device group whose device count and
  capabilities match the captured workload.

These constraints keep replay useful for correctness and performance work: a
successful replay should mean the HAL stream was actually reproduced, not that
unsupported operations were skipped.

## Workflow: large parameter files

For a normal model serving workflow that uses a large `.irpa` parameter file,
capture by reference:

```shell
iree-run-module \
  --device=amdgpu \
  --module=/tmp/model.vmfb \
  --function=main \
  --parameters=model=/data/weights/model.irpa \
  --input=@/tmp/prompt.txt \
  --device_replay_output=/tmp/model.ireereplay \
  --device_replay_file_policy=reference \
  --device_replay_file_validation=identity
```

Move or copy the replay and keep the parameter file available. If the replay
host uses a different mount root:

```shell
iree-benchmark-replay \
  --device=amdgpu \
  --benchmark_min_time=20x \
  --replay_file_remap=/data/weights=/mnt/replay/weights \
  /tmp/model.ireereplay
```

If platform identity cannot be preserved across staging, recapture with
`--device_replay_file_validation=digest`. Do not enable digest validation for
terabyte-scale files unless the full scan is acceptable.

## Workflow: hermetic correctness capture

For a small test or a bug report where external files should not be required,
use range capture:

```shell
iree-run-module \
  --device=local-sync \
  --module=/tmp/model.vmfb \
  --function=main \
  --parameters=model=/tmp/fixture.irpa \
  --input=@/tmp/inputs.txt \
  --device_replay_output=/tmp/model-hermetic.ireereplay \
  --device_replay_file_policy=capture-ranges
```

Replay substitutes captured file reads with queue updates. That preserves the
bytes consumed by the HAL stream, but it is not a storage benchmark for the
original file read path.

## Appendix: Programmatic replay execution

Embedding applications can execute a replay directly with
`iree_hal_replay_execute_file`:

```c
#include "iree/hal/replay/execute.h"
#include "iree/io/file_contents.h"

iree_io_file_contents_t* replay_contents = NULL;
iree_status_t status = iree_io_file_contents_map(
    IREE_SV("/tmp/model.ireereplay"), IREE_IO_FILE_ACCESS_READ,
    host_allocator, &replay_contents);

iree_hal_replay_file_path_remap_t remaps[] = {
    {IREE_SV("/mnt/capture"), IREE_SV("/mnt/replay")},
};
iree_hal_replay_execute_options_t options =
    iree_hal_replay_execute_options_default();
options.file_path_remap_count = IREE_ARRAYSIZE(remaps);
options.file_path_remaps = remaps;

if (iree_status_is_ok(status)) {
  status = iree_hal_replay_execute_file(replay_contents->const_buffer,
                                        device_group, &options,
                                        host_allocator);
}
iree_io_file_contents_free(replay_contents);
IREE_RETURN_IF_ERROR(status);
```

Executable substitution is exposed as a callback on
`iree_hal_replay_execute_options_t`, allowing callers to decide per captured
executable:

```c
static iree_status_t substitute_executable(
    void* user_data,
    const iree_hal_replay_executable_substitution_request_t* request,
    iree_hal_replay_executable_substitution_t* out_substitution) {
  const replacement_library_t* library = (const replacement_library_t*)user_data;
  if (request->executable_id != library->target_executable_id) {
    return iree_ok_status();
  }
  out_substitution->substitute = true;
  out_substitution->source = library->path;
  out_substitution->executable_format = library->format;
  out_substitution->executable_data = library->data;
  return iree_ok_status();
}

options.executable_substitution_callback.fn = substitute_executable;
options.executable_substitution_callback.user_data = &replacement_library;
```

Replacement data only needs to remain valid for the prepare call made by replay.

## Appendix: C projection

`iree-dump-replay --format=c` is reserved for a generated C reproducer
projection. Current builds accept the flag so scripts can probe support, but
return `UNIMPLEMENTED` instead of emitting a partial C file.

The intended projection will reference byte ranges in the original
`.ireereplay` file for large payloads, matching the JSONL dumper's range-based
model.

## Appendix: Agent-oriented help

Replay tools can print a compact Markdown playbook from the exact binary in
your build tree:

```shell
iree-run-module --agents_md
iree-run-replay --agents_md
iree-benchmark-replay --agents_md
iree-dump-replay --agents_md
```

Use this when integrating replay into scripts, CI reproducers, or agent
workflows that need the current flag list and diagnostics without reading the
source tree.
