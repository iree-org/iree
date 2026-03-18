# Remote HAL Demo (WIP)

Status: demo works end-to-end on epoll. Two P0 blockers remain before
this can ship without shame (bd-2hsw gate).

## What Works

- `iree-serve-device` starts, binds TCP, accepts connections
- Remote client driver registered, creates device, auto-connects
- Full HAL command buffer recording + dispatch proven via CTS (ASAN clean)
- All queue operations (fill, copy, update, alloca, dealloca, dispatch)
- Executable upload, buffer map/unmap, semaphore wait
- **End-to-end demo**: compile → serve → remote dispatch → correct results

## Running the Demo

### 1. Compile a test module

```bash
iree-bazel-run //tools:iree-compile -- /tmp/simple_add.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=generic \
  -o /tmp/simple_add.vmfb
```

Where `simple_add.mlir` is:
```mlir
func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

### 2. Build the tools

io_uring has known proactor bugs that block the demo (bd-2v9n). Use the
`IREE_ASYNC_FORCE_POSIX_PROACTOR` compile flag to force the epoll backend:

```bash
iree-bazel-build --copt=-DIREE_ASYNC_FORCE_POSIX_PROACTOR=1 \
  //tools:iree-serve-device //tools:iree-run-module
```

Without this flag, the default behavior tries io_uring first and falls
back to epoll if unavailable. The flag skips the io_uring attempt entirely.

### 3. Start the server (in its own terminal)

```bash
iree-bazel-run --copt=-DIREE_ASYNC_FORCE_POSIX_PROACTOR=1 \
  //tools:iree-serve-device -- \
  --device=local-task \
  --bind=tcp://127.0.0.1:5000
```

Expected output:
```
Serving on tcp://127.0.0.1:5000 (Ctrl+C to stop)
```

### 4. Connect with iree-run-module (in a second terminal)

```bash
iree-bazel-run --copt=-DIREE_ASYNC_FORCE_POSIX_PROACTOR=1 \
  //tools:iree-run-module -- \
  --device=remote-tcp://127.0.0.1:5000 \
  --module=/tmp/simple_add.vmfb \
  --function=add \
  --input="4xf32=1,2,3,4" \
  --input="4xf32=10,20,30,40"
```

Expected output:
```
EXEC @add
result[0]: hal.buffer_view
4xf32=11 22 33 44
```

## Architecture

```
iree-run-module                          iree-serve-device
  |                                        |
  v                                        v
remote client device  <-- TCP -->  remote server
  |                                        |
  v                                        v
(proxy buffers,                    local-task device
 command recording,                  (real execution)
 executable upload)
```

## Blockers (bd-2hsw gate)

- **bd-qs4j** (P0): Server-side command failures must reach client
  semaphores. Currently the server has no way to propagate errors through
  ADVANCE frames — the client either deadlocks or sees silent success.
  Fix: add status_code to advance_payload_t (reserved0 field), call
  frontier_tracker_fail_axis on the client. ~50 lines across 3 files.

- **bd-2yio** (P0): Resource releases arrive on the control channel,
  queue commands on the queue channel — no ordering guarantee. A release
  can race ahead of a COMMAND referencing the same resource. The
  workaround (skip all releases, leak until session close) was
  deliberately not landed. Need to either defer releases until pending
  queue commands drain, or unify the channels.

## Other Open Issues

- **bd-2v9n** (P1): io_uring NOPs submitted during CQE callback
  processing don't generate visible CQEs under DEFER_TASKRUN. Blocks
  the demo on io_uring (session bootstrap stalls). Use
  `--copt=-DIREE_ASYNC_FORCE_POSIX_PROACTOR=1` to force epoll.

- **bd-1257** (P1): io_uring multishot recv without PBUF_RING requires
  emulation (resubmit single-shot on completion). The submit/completion
  dual-path and WRITE→NONE slab fallback are in place; the resubmit
  loop is not yet implemented.

- **bd-2bpo** (P2): Client teardown race — bootstrap timer callback
  accesses freed frontier_tracker after device release.

## CTS Test Status

```bash
# Force epoll (recommended until io_uring bugs are resolved):
iree-bazel-test --config=asan --copt=-DIREE_ASYNC_FORCE_POSIX_PROACTOR=1 \
  //runtime/src/iree/hal/remote/cts:buffer_tests \
  //runtime/src/iree/hal/remote/cts:core_tests \
  //runtime/src/iree/hal/remote/cts:command_buffer_tests \
  //runtime/src/iree/hal/remote/cts:queue_tests

# Also works on io_uring (4/4 ASAN clean):
iree-bazel-test --config=asan \
  //runtime/src/iree/hal/remote/cts:buffer_tests \
  //runtime/src/iree/hal/remote/cts:core_tests \
  //runtime/src/iree/hal/remote/cts:command_buffer_tests \
  //runtime/src/iree/hal/remote/cts:queue_tests
```
