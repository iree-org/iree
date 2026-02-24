# iree/async/cts/ -- Conformance Test Suite

The CTS validates all proactor backends against a shared set of test suites
and benchmarks. Tests are written once and run against every registered backend
configuration.

## Architecture

CTS uses link-time composition:

- **Test suites** (`core/`, `socket/`, `event/`, `buffer/`, `sync/`, `futex/`)
  self-register via `CTS_REGISTER_TEST_SUITE()`. Each suite is a library, not
  a runnable binary.
- **Backends** register via `CtsRegistry::RegisterBackend()` in per-backend
  `backends.cc` files (e.g., `platform/io_uring/cts/backends.cc`,
  `platform/posix/cts/backends.cc`).
- **Runnable binaries** are assembled at the backend level by linking test
  suite libraries + backend registration + `test_main.cc` (or
  `benchmark_main.cc`). See `platform/{io_uring,posix}/cts/BUILD.bazel`.
- At startup, `main()` calls `CtsRegistry::InstantiateAll()` before
  `RUN_ALL_TESTS()`. This creates gtest parameterized test instances for each
  suite x backend pair that passes tag filtering.

## Backend Configurations

**io_uring** (5 configurations):
- `io_uring`: Full capabilities (zerocopy, multishot, MSG_RING)
- `io_uring_no_zerocopy`: Multishot only, tests non-zerocopy send path
- `io_uring_no_multishot`: Zerocopy only, tests single-shot accept/recv
- `io_uring_no_messaging`: No MSG_RING, tests fallback message passing
- `io_uring_minimal`: No optional features, tests base io_uring path

**POSIX** (1-3 configurations depending on platform):
- `posix_poll`: Always available (all POSIX)
- `posix_epoll`: Linux only
- `posix_kqueue`: macOS/BSD only

## Tag Filtering

Test suites declare required and excluded tags. Backends declare capability
tags. A suite only instantiates against backends whose tags match:

```cpp
// Runs against all backends:
CTS_REGISTER_TEST_SUITE(LifecycleTest);

// Requires zerocopy tag:
CTS_REGISTER_TEST_SUITE_WITH_TAGS(ZeroCopyTest, {"zerocopy"}, {});

// Runs against all backends EXCEPT those tagged "portable":
CTS_REGISTER_TEST_SUITE_WITH_TAGS(AdvancedTest, {}, {"portable"});
```

## Test Categories

| Directory | Contents |
|-----------|----------|
| `core/` | Lifecycle, nop, timer, sequence, error propagation, resource exhaustion |
| `socket/` | TCP/UDP lifecycle, transfer, multishot, send flags, zero-copy, messaging |
| `event/` | Event create/set/wait, event source monitoring, event pool |
| `buffer/` | Buffer pool, registration |
| `sync/` | Semaphore sync/async, cancellation, notification, relay, fence, signal |
| `futex/` | Futex wait/wake operations |
| `util/` | Registry, test base, benchmark infrastructure |

## Running

CTS test targets live in the backend directories, not here. The suites in
this directory are libraries linked into the backend binaries.

```bash
# All CTS tests across all backends:
iree-bazel-test --config=asan //runtime/src/iree/async/platform/io_uring/cts:...
iree-bazel-test --config=asan //runtime/src/iree/async/platform/posix/cts:...

# Specific test suite for one backend:
iree-bazel-test --config=asan //runtime/src/iree/async/platform/io_uring/cts:core_tests
iree-bazel-test --config=asan //runtime/src/iree/async/platform/posix/cts:socket_tests

# CTS benchmarks (no ASAN for meaningful numbers):
iree-bazel-test --compilation_mode=opt \
    //runtime/src/iree/async/platform/io_uring/cts:core_benchmarks_test
iree-bazel-test --compilation_mode=opt \
    //runtime/src/iree/async/platform/posix/cts:core_benchmarks_test
```
