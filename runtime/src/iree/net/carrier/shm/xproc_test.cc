// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process SHM carrier tests.
//
// Uses the coordinated test harness to spawn actual separate processes that
// establish SHM carrier pairs via handshake over a local channel. Validates
// the full cross-process path: handle exchange, SHM mapping in separate
// address spaces, shared wake notifications, MPSC ring operation, and direct
// read/write across processes.
//
// The in-process tests (carrier_test.cc, handshake_test.cc, CTS) exercise
// the carrier API thoroughly but use socketpair() or create_pair() — both
// within a single process. These tests verify that the handshake protocol
// actually works when the fd/handle passing, SHM mapping, and notification
// primitives cross the process boundary.
//
// Under thread sanitizer (TSAN), each child process carries ~10-15x overhead.
// Timeouts are scaled accordingly to avoid false failures.
//
// Platform channels:
//   POSIX:   Unix domain socket in the temp directory.
//   Windows: Named pipe (\\.\pipe\<name>) derived from the temp directory.

#include "iree/base/api.h"  // Must precede platform checks for IREE_PLATFORM_*.

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cstring>
#include <functional>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/shm/carrier.h"
#include "iree/net/carrier/shm/handshake.h"
#include "iree/net/carrier/shm/shared_wake.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Cross-process tests spawn child processes that both carry sanitizer overhead.
// TSAN adds ~10-15x latency per process, so round-trip timeouts must scale.
#if defined(IREE_SANITIZER_THREAD)
static constexpr int64_t kPollTimeoutMs = 30000;
#else
static constexpr int64_t kPollTimeoutMs = 5000;
#endif

//===----------------------------------------------------------------------===//
// Error reporting for child processes
//===----------------------------------------------------------------------===//

// Role functions run in child processes without gtest. This macro prints the
// failing expression and status, then returns 1 (failure exit code).
#define XPROC_CHECK_OK(expr)                                   \
  do {                                                         \
    iree_status_t xproc_status__ = (expr);                     \
    if (!iree_status_is_ok(xproc_status__)) {                  \
      fprintf(stderr, "XPROC_CHECK_OK failed: %s\n  ", #expr); \
      iree_status_fprint(stderr, xproc_status__);              \
      fprintf(stderr, "\n");                                   \
      iree_status_free(xproc_status__);                        \
      return 1;                                                \
    }                                                          \
  } while (0)

#define XPROC_CHECK(cond, ...)                           \
  do {                                                   \
    if (!(cond)) {                                       \
      fprintf(stderr, "XPROC_CHECK failed: %s ", #cond); \
      fprintf(stderr, __VA_ARGS__);                      \
      fprintf(stderr, "\n");                             \
      return 1;                                          \
    }                                                    \
  } while (0)

//===----------------------------------------------------------------------===//
// RAII context for cross-process carrier setup
//===----------------------------------------------------------------------===//

// Manages proactor, shared_wake, carrier, and server handle lifetime.
// Used by role functions for automatic cleanup on all exit paths.
struct XProcContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_net_shm_shared_wake_t* shared_wake = nullptr;
  iree_net_carrier_t* carrier = nullptr;

  // Platform-specific server handle. The server binds/creates the channel;
  // the handshake consumes (and closes) the connected handle, so this may
  // be invalidated before the destructor runs.
#if defined(IREE_PLATFORM_WINDOWS)
  HANDLE pipe_handle = INVALID_HANDLE_VALUE;
#else
  int listen_fd = -1;
#endif

  // Recv capture: accumulates received data for verification.
  std::vector<uint8_t> recv_buffer;
  std::atomic<iree_host_size_t> recv_total_bytes{0};

  // Completion tracking: counts send completions.
  std::atomic<int> completion_count{0};
  std::atomic<iree_host_size_t> completion_bytes{0};

  ~XProcContext() {
    if (carrier) {
      // Set a null recv handler to avoid callbacks during teardown.
      iree_net_carrier_set_recv_handler(carrier, NullRecvHandler());
      DeactivateAndDrain();
      iree_net_carrier_release(carrier);
    }
    if (shared_wake) iree_net_shm_shared_wake_release(shared_wake);
    if (proactor) iree_async_proactor_release(proactor);
#if defined(IREE_PLATFORM_WINDOWS)
    if (pipe_handle != INVALID_HANDLE_VALUE) CloseHandle(pipe_handle);
#else
    if (listen_fd >= 0) close(listen_fd);
#endif
  }

  // Static recv handler that captures data into this context.
  static iree_status_t RecvHandler(void* user_data, iree_async_span_t data,
                                   iree_async_buffer_lease_t* lease) {
    auto* context = static_cast<XProcContext*>(user_data);
    uint8_t* ptr = iree_async_span_ptr(data);
    context->recv_buffer.insert(context->recv_buffer.end(), ptr,
                                ptr + data.length);
    context->recv_total_bytes.fetch_add(data.length, std::memory_order_relaxed);
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  iree_net_carrier_recv_handler_t AsRecvHandler() {
    return {RecvHandler, this};
  }

  // Static completion callback that tracks send completions.
  static void CompletionCallback(void* callback_user_data,
                                 uint64_t operation_user_data,
                                 iree_status_t status,
                                 iree_host_size_t bytes_transferred,
                                 iree_async_buffer_lease_t* recv_lease) {
    auto* context = static_cast<XProcContext*>(callback_user_data);
    context->completion_count.fetch_add(1, std::memory_order_relaxed);
    context->completion_bytes.fetch_add(bytes_transferred,
                                        std::memory_order_relaxed);
    iree_status_ignore(status);
  }

  iree_net_carrier_callback_t AsCallback() {
    return {CompletionCallback, this};
  }

  // Null recv handler for teardown.
  static iree_status_t NullRecvFn(void* user_data, iree_async_span_t data,
                                  iree_async_buffer_lease_t* lease) {
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  static iree_net_carrier_recv_handler_t NullRecvHandler() {
    return {NullRecvFn, nullptr};
  }

  // Polls the proactor until |condition| returns true or timeout expires.
  bool PollUntil(std::function<bool()> condition,
                 int64_t timeout_ms = kPollTimeoutMs) {
    iree_time_t deadline_ns =
        iree_time_now() + iree_make_duration_ms(timeout_ms);
    iree_timeout_t timeout = iree_make_deadline(deadline_ns);
    while (!condition()) {
      if (iree_time_now() >= deadline_ns) return false;
      iree_host_size_t completed = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor, timeout, &completed);
      iree_status_ignore(status);
    }
    return true;
  }

  // Deactivates the carrier and drains remaining operations.
  void DeactivateAndDrain() {
    if (!carrier) return;
    iree_net_carrier_state_t state = iree_net_carrier_state(carrier);
    if (state == IREE_NET_CARRIER_STATE_CREATED ||
        state == IREE_NET_CARRIER_STATE_DEACTIVATED) {
      return;
    }
    if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
      iree_status_t status =
          iree_net_carrier_deactivate(carrier, nullptr, nullptr);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        return;
      }
    }
    iree_time_t deadline_ns = iree_time_now() + iree_make_duration_ms(5000);
    while (iree_net_carrier_state(carrier) !=
           IREE_NET_CARRIER_STATE_DEACTIVATED) {
      if (iree_time_now() >= deadline_ns) break;
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor, iree_make_timeout_ms(100), &completed);
      iree_status_ignore(status);
    }
  }
};

//===----------------------------------------------------------------------===//
// Channel and handshake helpers
//===----------------------------------------------------------------------===//

// Creates the proactor and shared_wake (shared mode for cross-process).
static iree_status_t SetupProactor(XProcContext* context) {
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
      iree_async_proactor_options_default(), iree_allocator_system(),
      &context->proactor));
  return iree_net_shm_shared_wake_create_shared(
      context->proactor, iree_allocator_system(), &context->shared_wake);
}

// Builds a platform-appropriate channel address from the temp directory.
//
// POSIX:   Unix domain socket path in the temp directory.
// Windows: Named pipe path (\\.\pipe\<basename>) derived from the temp
//          directory name, which is unique per test run.
static void MakeAddress(const char* temp_directory, char* out_address,
                        size_t capacity) {
#if defined(IREE_PLATFORM_WINDOWS)
  const char* basename = temp_directory;
  for (const char* p = temp_directory; *p; ++p) {
    if (*p == '/' || *p == '\\') basename = p + 1;
  }
  snprintf(out_address, capacity, "\\\\.\\pipe\\%s", basename);
#else
  snprintf(out_address, capacity, "%s/c.sock", temp_directory);
#endif
}

#if defined(IREE_PLATFORM_WINDOWS)

// Converts a narrow (UTF-8) string to a wide string for Windows APIs.
static iree_status_t NarrowToWide(const char* narrow, WCHAR* wide,
                                  int wide_capacity) {
  int length = MultiByteToWideChar(CP_UTF8, 0, narrow, -1, wide, wide_capacity);
  if (length <= 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "MultiByteToWideChar failed: %lu",
                            (unsigned long)GetLastError());
  }
  return iree_ok_status();
}

// Server: creates a named pipe and stores the handle in the context.
// The pipe is created with FILE_FLAG_OVERLAPPED because the handshake uses
// overlapped ReadFile/WriteFile with manual-reset events.
static iree_status_t ServerBind(const char* address, XProcContext* context) {
  WCHAR wide_path[MAX_PATH + 1];
  IREE_RETURN_IF_ERROR(NarrowToWide(address, wide_path, MAX_PATH + 1));

  HANDLE pipe =
      CreateNamedPipeW(wide_path, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                       PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                       1,     // Single instance (one client per test scenario).
                       4096,  // Output buffer size.
                       4096,  // Input buffer size.
                       5000,  // Default timeout (milliseconds).
                       NULL);  // Default security.
  if (pipe == INVALID_HANDLE_VALUE) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CreateNamedPipeW failed: %lu",
                            (unsigned long)GetLastError());
  }
  context->pipe_handle = pipe;
  return iree_ok_status();
}

// Server: waits for a client to connect to the pipe, then runs the server-side
// handshake. Creates a carrier from the handshake result.
//
// Uses overlapped ConnectNamedPipe with a blocking wait via event. The pipe
// handle is consumed by the handshake (which closes it on return).
static iree_status_t ServerAcceptAndHandshake(
    XProcContext* context, iree_net_carrier_callback_t callback) {
  // Wait for client connection via overlapped ConnectNamedPipe.
  HANDLE event = CreateEventW(NULL, /*bManualReset=*/TRUE,
                              /*bInitialState=*/FALSE, NULL);
  if (!event) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CreateEvent for ConnectNamedPipe failed: %lu",
                            (unsigned long)GetLastError());
  }

  OVERLAPPED overlapped;
  memset(&overlapped, 0, sizeof(overlapped));
  overlapped.hEvent = event;

  BOOL connected = ConnectNamedPipe(context->pipe_handle, &overlapped);
  if (!connected) {
    DWORD error = GetLastError();
    if (error == ERROR_IO_PENDING) {
      // Normal case: waiting for client. Block until it arrives.
      DWORD wait_result = WaitForSingleObject(event, 30000);
      if (wait_result == WAIT_TIMEOUT) {
        CancelIoEx(context->pipe_handle, &overlapped);
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
        return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                                "timed out waiting for client connection");
      }
      if (wait_result != WAIT_OBJECT_0) {
        CloseHandle(event);
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "WaitForSingleObject failed: %lu",
                                (unsigned long)GetLastError());
      }
      DWORD bytes_transferred = 0;
      if (!GetOverlappedResult(context->pipe_handle, &overlapped,
                               &bytes_transferred, FALSE)) {
        CloseHandle(event);
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "ConnectNamedPipe overlapped result failed: %lu",
            (unsigned long)GetLastError());
      }
    } else if (error == ERROR_PIPE_CONNECTED) {
      // Client connected between CreateNamedPipeW and ConnectNamedPipe.
    } else {
      CloseHandle(event);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "ConnectNamedPipe failed: %lu",
                              (unsigned long)error);
    }
  }
  CloseHandle(event);

  // Pass the connected pipe to the handshake, which takes ownership and
  // closes it on return (both success and failure).
  iree_async_primitive_t channel =
      iree_async_primitive_from_win32_handle((uintptr_t)context->pipe_handle);
  context->pipe_handle = INVALID_HANDLE_VALUE;

  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_server(
      channel, context->shared_wake, iree_net_shm_carrier_options_default(),
      context->proactor, iree_allocator_system(), &handshake_result));

  iree_status_t status =
      iree_net_shm_carrier_create(&handshake_result.carrier_params, callback,
                                  iree_allocator_system(), &context->carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_shm_xproc_context_release(handshake_result.context);
  }
  return status;
}

// Client: opens the named pipe and runs the client-side handshake.
// Creates a carrier from the handshake result.
static iree_status_t ClientConnectAndHandshake(
    const char* address, XProcContext* context,
    iree_net_carrier_callback_t callback) {
  WCHAR wide_path[MAX_PATH + 1];
  IREE_RETURN_IF_ERROR(NarrowToWide(address, wide_path, MAX_PATH + 1));

  // CreateFileW on a named pipe connects synchronously.
  HANDLE pipe = CreateFileW(wide_path, GENERIC_READ | GENERIC_WRITE,
                            0,     // No sharing.
                            NULL,  // Default security.
                            OPEN_EXISTING,
                            FILE_FLAG_OVERLAPPED,  // Required by handshake.
                            NULL);                 // No template.
  if (pipe == INVALID_HANDLE_VALUE) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CreateFileW (pipe connect) failed: %lu",
                            (unsigned long)GetLastError());
  }

  // The handshake takes ownership of the channel (closes it on return).
  iree_async_primitive_t channel =
      iree_async_primitive_from_win32_handle((uintptr_t)pipe);

  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_client(
      channel, context->shared_wake, context->proactor, iree_allocator_system(),
      &handshake_result));

  iree_status_t status =
      iree_net_shm_carrier_create(&handshake_result.carrier_params, callback,
                                  iree_allocator_system(), &context->carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_shm_xproc_context_release(handshake_result.context);
  }
  return status;
}

#else  // POSIX

// Server: creates a Unix domain socket, binds, and listens. The socket path
// is stored in the temp directory and serves as the rendezvous address for
// the client.
static iree_status_t ServerBind(const char* address, XProcContext* context) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "socket(AF_UNIX) failed: %s",
                            strerror(errno));
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t path_length = strlen(address);
  if (path_length >= sizeof(addr.sun_path)) {
    close(fd);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "socket path too long (%zu >= %zu)", path_length,
                            sizeof(addr.sun_path));
  }
  memcpy(addr.sun_path, address, path_length + 1);

  if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    close(fd);
    return iree_make_status(IREE_STATUS_INTERNAL, "bind(%s) failed: %s",
                            address, strerror(errno));
  }
  if (listen(fd, 1) != 0) {
    close(fd);
    return iree_make_status(IREE_STATUS_INTERNAL, "listen(%s) failed: %s",
                            address, strerror(errno));
  }

  context->listen_fd = fd;
  return iree_ok_status();
}

// Server: accepts one connection and runs the server-side handshake.
// Creates a carrier from the handshake result.
static iree_status_t ServerAcceptAndHandshake(
    XProcContext* context, iree_net_carrier_callback_t callback) {
  int client_fd = accept(context->listen_fd, nullptr, nullptr);
  if (client_fd < 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "accept() failed: %s",
                            strerror(errno));
  }

  // The handshake takes ownership of the fd (closes it on return).
  iree_async_primitive_t channel = iree_async_primitive_from_fd(client_fd);

  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_server(
      channel, context->shared_wake, iree_net_shm_carrier_options_default(),
      context->proactor, iree_allocator_system(), &handshake_result));

  iree_status_t status =
      iree_net_shm_carrier_create(&handshake_result.carrier_params, callback,
                                  iree_allocator_system(), &context->carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_shm_xproc_context_release(handshake_result.context);
  }
  return status;
}

// Client: connects to the server's Unix domain socket and runs the
// client-side handshake. Creates a carrier from the handshake result.
static iree_status_t ClientConnectAndHandshake(
    const char* address, XProcContext* context,
    iree_net_carrier_callback_t callback) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "socket(AF_UNIX) failed: %s",
                            strerror(errno));
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t path_length = strlen(address);
  if (path_length >= sizeof(addr.sun_path)) {
    close(fd);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "socket path too long");
  }
  memcpy(addr.sun_path, address, path_length + 1);

  if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    close(fd);
    return iree_make_status(IREE_STATUS_INTERNAL, "connect(%s) failed: %s",
                            address, strerror(errno));
  }

  // The handshake takes ownership of the fd.
  iree_async_primitive_t channel = iree_async_primitive_from_fd(fd);

  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_client(
      channel, context->shared_wake, context->proactor, iree_allocator_system(),
      &handshake_result));

  iree_status_t status =
      iree_net_shm_carrier_create(&handshake_result.carrier_params, callback,
                                  iree_allocator_system(), &context->carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_shm_xproc_context_release(handshake_result.context);
  }
  return status;
}

#endif  // IREE_PLATFORM_WINDOWS

// Sends a message via the carrier. Convenience wrapper around the scatter-
// gather send API.
static iree_status_t SendMessage(iree_net_carrier_t* carrier, const void* data,
                                 iree_host_size_t length) {
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<void*>(data), length);
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  params.user_data = 0;
  return iree_net_carrier_send(carrier, &params);
}

//===----------------------------------------------------------------------===//
// Test 1: Handshake and carrier creation
//===----------------------------------------------------------------------===//

static int handshake_server_role(int argc, char** argv,
                                 const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  iree_net_carrier_callback_t no_callback = {nullptr, nullptr};
  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, no_callback));
  XPROC_CHECK(context.carrier != nullptr, "carrier is null");

  // Activate and immediately deactivate to verify the carrier is functional.
  iree_net_carrier_set_recv_handler(context.carrier,
                                    XProcContext::NullRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  return 0;
}

static int handshake_client_role(int argc, char** argv,
                                 const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));

  iree_net_carrier_callback_t no_callback = {nullptr, nullptr};
  XPROC_CHECK_OK(ClientConnectAndHandshake(address, &context, no_callback));
  XPROC_CHECK(context.carrier != nullptr, "carrier is null");

  iree_net_carrier_set_recv_handler(context.carrier,
                                    XProcContext::NullRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  return 0;
}

//===----------------------------------------------------------------------===//
// Test 2: Send/recv round trip
//===----------------------------------------------------------------------===//

static const char kClientMessage[] = "hello from client";
static const char kServerMessage[] = "hello from server";

static int sendrecv_server_role(int argc, char** argv,
                                const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, context.AsCallback()));

  // Activate and wait for the client's message.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kClientMessage);
  }),
              "timed out waiting for client message");

  XPROC_CHECK(context.recv_buffer.size() == strlen(kClientMessage),
              "expected %zu bytes, got %zu", strlen(kClientMessage),
              context.recv_buffer.size());
  XPROC_CHECK(memcmp(context.recv_buffer.data(), kClientMessage,
                     strlen(kClientMessage)) == 0,
              "client message mismatch");

  // Send response.
  XPROC_CHECK_OK(
      SendMessage(context.carrier, kServerMessage, strlen(kServerMessage)));

  return 0;
}

static int sendrecv_client_role(int argc, char** argv,
                                const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(
      ClientConnectAndHandshake(address, &context, context.AsCallback()));

  // Activate and send our message.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK_OK(
      SendMessage(context.carrier, kClientMessage, strlen(kClientMessage)));

  // Wait for the server's response.
  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kServerMessage);
  }),
              "timed out waiting for server response");

  XPROC_CHECK(context.recv_buffer.size() == strlen(kServerMessage),
              "expected %zu bytes, got %zu", strlen(kServerMessage),
              context.recv_buffer.size());
  XPROC_CHECK(memcmp(context.recv_buffer.data(), kServerMessage,
                     strlen(kServerMessage)) == 0,
              "server response mismatch");

  return 0;
}

//===----------------------------------------------------------------------===//
// Test 3: Direct write with signaling
//===----------------------------------------------------------------------===//

// Known pattern written by the client at a fixed SHM offset.
static const iree_host_size_t kDirectWriteOffset = 0x1000;
static const iree_host_size_t kDirectWriteLength = 64;

static void FillPattern(uint8_t* buffer, iree_host_size_t length,
                        uint8_t seed) {
  for (iree_host_size_t i = 0; i < length; ++i) {
    buffer[i] = (uint8_t)(seed + i);
  }
}

static int dwrite_server_role(int argc, char** argv,
                              const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, context.AsCallback()));

  // Activate and wait for the signaling direct_write to arrive via recv
  // handler. The REFERENCE entry resolves to the SHM data.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= kDirectWriteLength;
  }),
              "timed out waiting for direct_write data");

  // Verify the received data matches the expected pattern.
  uint8_t expected[kDirectWriteLength];
  FillPattern(expected, kDirectWriteLength, 0xAB);

  XPROC_CHECK(context.recv_buffer.size() == kDirectWriteLength,
              "expected %zu bytes, got %zu", kDirectWriteLength,
              context.recv_buffer.size());
  XPROC_CHECK(
      memcmp(context.recv_buffer.data(), expected, kDirectWriteLength) == 0,
      "direct_write data mismatch");

  return 0;
}

static int dwrite_client_role(int argc, char** argv,
                              const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(
      ClientConnectAndHandshake(address, &context, context.AsCallback()));

  iree_net_carrier_set_recv_handler(context.carrier,
                                    XProcContext::NullRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  // Prepare the source data and write it at kDirectWriteOffset in region 0
  // with SIGNAL_RECEIVER flag so the server's recv handler fires.
  uint8_t source_data[kDirectWriteLength];
  FillPattern(source_data, kDirectWriteLength, 0xAB);

  iree_net_direct_write_params_t params = {};
  params.local = iree_async_span_from_ptr(source_data, sizeof(source_data));
  params.remote = iree_net_remote_handle_t{{0, kDirectWriteOffset}};
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER;
  params.immediate = 0;
  params.user_data = 0;
  XPROC_CHECK_OK(iree_net_carrier_direct_write(context.carrier, &params));

  // Wait for the send completion.
  XPROC_CHECK(
      context.PollUntil([&] { return context.completion_count.load() >= 1; }),
      "timed out waiting for direct_write completion");

  return 0;
}

//===----------------------------------------------------------------------===//
// Test 4: Direct read across processes
//===----------------------------------------------------------------------===//

// The server writes data at this offset; the client reads it.
static const iree_host_size_t kDirectReadOffset = 0x2000;
static const iree_host_size_t kDirectReadLength = 48;
static const char kDataReadyMarker[] = "ready";
static const char kDataReadyAck[] = "ack";

static int dread_server_role(int argc, char** argv,
                             const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, context.AsCallback()));

  // Write known data directly into the SHM region at the agreed offset.
  // The server is the creator of the SHM region, so region 0 is its mapping.
  iree_net_shm_region_info_t region_info = {};
  XPROC_CHECK_OK(
      iree_net_shm_carrier_query_region(context.carrier, 0, &region_info));

  XPROC_CHECK(kDirectReadOffset + kDirectReadLength <= region_info.size,
              "offset+length exceeds region size");

  uint8_t write_data[kDirectReadLength];
  FillPattern(write_data, kDirectReadLength, 0xCD);
  memcpy((uint8_t*)region_info.base_ptr + kDirectReadOffset, write_data,
         kDirectReadLength);

  // Activate and tell the client the data is ready.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK_OK(
      SendMessage(context.carrier, kDataReadyMarker, strlen(kDataReadyMarker)));

  // Wait for the client's ack.
  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kDataReadyAck);
  }),
              "timed out waiting for client ack");

  return 0;
}

static int dread_client_role(int argc, char** argv,
                             const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(
      ClientConnectAndHandshake(address, &context, context.AsCallback()));

  // Activate and wait for the server's "data ready" marker.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kDataReadyMarker);
  }),
              "timed out waiting for data-ready marker");

  // Direct read from the agreed offset. The client's region 0 is its mapping
  // of the same SHM object the server created.
  uint8_t read_buffer[kDirectReadLength];
  memset(read_buffer, 0, sizeof(read_buffer));

  iree_net_direct_read_params_t params = {};
  params.local = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  params.remote = iree_net_remote_handle_t{{0, kDirectReadOffset}};
  params.user_data = 0;
  XPROC_CHECK_OK(iree_net_carrier_direct_read(context.carrier, &params));

  // Verify the data matches what the server wrote.
  uint8_t expected[kDirectReadLength];
  FillPattern(expected, kDirectReadLength, 0xCD);
  XPROC_CHECK(memcmp(read_buffer, expected, kDirectReadLength) == 0,
              "direct_read data mismatch");

  // Send ack so the server can exit cleanly.
  XPROC_CHECK_OK(
      SendMessage(context.carrier, kDataReadyAck, strlen(kDataReadyAck)));

  return 0;
}

//===----------------------------------------------------------------------===//
// Test configs
//===----------------------------------------------------------------------===//

static const iree_test_role_t kHandshakeRoles[] = {
    {"handshake_server", handshake_server_role, /*signals_ready=*/true},
    {"handshake_client", handshake_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kHandshakeConfig = {
    /*.roles=*/kHandshakeRoles,
    /*.role_count=*/2,
    /*.timeout_ms=*/30000,
};

static const iree_test_role_t kSendRecvRoles[] = {
    {"sendrecv_server", sendrecv_server_role, /*signals_ready=*/true},
    {"sendrecv_client", sendrecv_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kSendRecvConfig = {
    /*.roles=*/kSendRecvRoles,
    /*.role_count=*/2,
    /*.timeout_ms=*/30000,
};

static const iree_test_role_t kDirectWriteRoles[] = {
    {"dwrite_server", dwrite_server_role, /*signals_ready=*/true},
    {"dwrite_client", dwrite_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kDirectWriteConfig = {
    /*.roles=*/kDirectWriteRoles,
    /*.role_count=*/2,
    /*.timeout_ms=*/30000,
};

static const iree_test_role_t kDirectReadRoles[] = {
    {"dread_server", dread_server_role, /*signals_ready=*/true},
    {"dread_client", dread_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kDirectReadConfig = {
    /*.roles=*/kDirectReadRoles,
    /*.role_count=*/2,
    /*.timeout_ms=*/30000,
};

// Combined config with all roles for child dispatch. The coordinated_test_main
// uses this to find the right entry function when --iree_test_role is set.
static const iree_test_role_t kAllRoles[] = {
    {"handshake_server", handshake_server_role, /*signals_ready=*/true},
    {"handshake_client", handshake_client_role, /*signals_ready=*/false},
    {"sendrecv_server", sendrecv_server_role, /*signals_ready=*/true},
    {"sendrecv_client", sendrecv_client_role, /*signals_ready=*/false},
    {"dwrite_server", dwrite_server_role, /*signals_ready=*/true},
    {"dwrite_client", dwrite_client_role, /*signals_ready=*/false},
    {"dread_server", dread_server_role, /*signals_ready=*/true},
    {"dread_client", dread_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kAllRolesConfig = {
    /*.roles=*/kAllRoles,
    /*.role_count=*/8,
    /*.timeout_ms=*/30000,
};
IREE_COORDINATED_TEST_REGISTER(kAllRolesConfig);

//===----------------------------------------------------------------------===//
// Proactor availability check
//===----------------------------------------------------------------------===//

// Checks that a platform proactor can be created. If unavailable (e.g.,
// missing kernel support), the test is skipped. Called in the launcher
// process before spawning children.
static bool ProactorAvailable() {
  iree_async_proactor_t* proactor = nullptr;
  iree_status_t status =
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    return false;
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return false;
  }
  iree_async_proactor_release(proactor);
  return true;
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST(XProcCarrier, HandshakeAndCreate) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kHandshakeConfig));
}

TEST(XProcCarrier, SendRecvRoundTrip) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kSendRecvConfig));
}

TEST(XProcCarrier, DirectWriteSignaling) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kDirectWriteConfig));
}

TEST(XProcCarrier, DirectReadAcrossProcesses) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kDirectReadConfig));
}

}  // namespace
