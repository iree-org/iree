// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_BASE_WAIT_HANDLE_H_
#define IREE_BASE_WAIT_HANDLE_H_

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/base/time.h"

namespace iree {

// Interfaces for waitable objects that can produce WaitHandles.
// WaitableObjects are much like ::thread::Selectable, only they support both
// the classic locking style as well as file descriptors for use with select().
//
// Usage:
//  class MyWaitableObject : public WaitableObject {
//   public:
//    std::string DebugString() const override { return "something useful"; }
//    WaitHandle OnAsyncTask() {
//      return WaitHandle(retain_ref(this));
//    }
//   private:
//    StatusOr<std::pair<FdType, int>> AcquireFdForWait(
//        Time deadline_ns) override {
//      // If blocking traditionally do so now and then return this:
//      return std::make_pair(FdType::kPermanent, kSignaledFd);
//      // Otherwise, see ManualResetEvent for an example using fds.
//    }
//    StatusOr<bool> TryResolveWakeOnFd(int fd) override {
//      // Return true iff the object is really acquired, such as the semaphore
//      // being decremented.
//      return true;
//    }
//  };
class WaitableObject : public RefObject<WaitableObject> {
 public:
  // Indicates that a file descriptor is invalid. It will not block when waited
  // upon.
  constexpr static int kInvalidFd = -1;
  // Indicates that a file descriptor should be treated as signaled.
  // Waiting on this fd should return as if it has already been signaled.
  constexpr static int kSignaledFd = -2;

  // Defines the type of the native handle used for synchronization.
  enum class FdType : uint16_t {
    // Event has no handle and should be treated as permanently signaled.
    kPermanent,

    // Android/Linux/iOS-compatible POSIX pipe handle.
    // Two handles are generated: one for transmitting and one for receiving.
    //
    // More information:
    // http://man7.org/linux/man-pages/man2/pipe.2.html
    kPipe,

    // Android/Linux eventfd handle.
    // These are akin to pipe() but require only a single handle and have
    // significantly lower overhead (equivalent if not slightly better than
    // pthreads condvars).
    //
    // eventfds support acting as both semaphores and auto reset events.
    //
    // More information:
    // http://man7.org/linux/man-pages/man2/eventfd.2.html
    kEventFd,

    // Android/Linux sync_file handle (aka 'sync fence').
    // The handle is allocated indirectly by the device driver via the
    // <linux/sync_file.h> API. It may be waited upon with poll(), select(), or
    // epoll() and must be closed with close() when no longer required. If
    // waiting on multiple sync_files the caller should first merge them
    // together.
    //
    // A sync_file must only be used as fences (one-shot manual reset events).
    //
    // More information:
    // https://www.kernel.org/doc/Documentation/sync_file.txt
    // https://lwn.net/Articles/702339/
    // https://source.android.com/devices/graphics/implement-vsync#explicit_synchronization
    kSyncFile,
  };

  virtual ~WaitableObject() = default;

  // Returns a string representing the object, either specified as a debug_name
  // or a unique ID.
  virtual std::string DebugString() const = 0;

  // Attempts to acquire a file descriptor for the waitable objects by the given
  // |deadline|. In many cases this will return immediately with a valid fd.
  //
  // In cases where the file descriptor may not be available the call may block
  // until either it is available or the |deadline| has elapsed. Use
  // InfinitePast() to prevent blocking.
  //
  // Returns a valid file descriptor or kInvalidFd as an indication that the
  // object should not be waited on (already signaled, etc). Can return
  // kSignaledFd to indicate that it's already known that the handle has been
  // signaled and the caller should resolve as if it caused a wake normally.
  virtual StatusOr<std::pair<FdType, int>> AcquireFdForWait(
      Time deadline_ns) = 0;

  // Tries to resolve the object with the given |fd|.
  // In many cases this will no-op, however some types may require additional
  // checks to ensure that the wait operation succeeded (such as semaphores
  // that may need to query a count). If resolution fails the waitable object
  // must not be considered signaled. This call will never block.
  virtual StatusOr<bool> TryResolveWakeOnFd(int fd) = 0;
};

// Handle to waitable objects.
// WaitHandles are created by a particular synchronization primitive, such as
// Fence, as a way for one or more observers to poll or wait for notification.
//
// External synchronization primitives can be wrapped in WaitHandles to enable
// other libraries or languages to be waited on alongside WaitHandles created
// by the IREE primitives like Fence. See the notes on WaitHandleType for a list
// of handle types that are supported.
//
// Wait handles are thread-safe in that multiple threads may be waiting on them
// concurrently.
class WaitHandle {
 public:
  // Returns a WaitHandle that when waited on will never block.
  static WaitHandle AlwaysSignaling();

  // Returns a WaitHandle that when waited on will always fail.
  static WaitHandle AlwaysFailing();

  using WaitHandleSpan = absl::Span<WaitHandle* const>;

  // Blocks the caller until all passed |wait_handles| are signaled or the
  // |deadline| elapses.
  //
  // Returns success if the wait is successful and all events have been
  // signaled.
  //
  // Returns DEADLINE_EXCEEDED if the |deadline| elapses without all handles
  // having been signaled. Note that a subset of the |wait_handles| may have
  // been signaled and each can be queried to see which one.
  static Status WaitAll(WaitHandleSpan wait_handles, Time deadline_ns);
  static Status WaitAll(WaitHandleSpan wait_handles, Duration timeout_ns) {
    return WaitAll(wait_handles, RelativeTimeoutToDeadlineNanos(timeout_ns));
  }
  static Status WaitAll(WaitHandleSpan wait_handles) {
    return WaitAll(wait_handles, InfiniteFuture());
  }

  // Tries waiting on the handles and returns immediately if it would have
  // blocked. The caller will not be blocked even if a handle has not yet been
  // signaled.
  //
  // Returns true if all handles have been signaled.
  static StatusOr<bool> TryWaitAll(WaitHandleSpan wait_handles);

  // Blocks the caller until at least one of the |wait_handles| is signaled or
  // the |deadline| elapses.
  //
  // Returns the index into |wait_handles| of a handle that was signaled. Note
  // that more than one handle may have been signaled and all of the other
  // |wait_handles| should be queried or waited on again until waits for them
  // succeed.
  //
  // Returns DEADLINE_EXCEEDED if the |deadline| elapses without any handles
  // having been signaled.
  static StatusOr<int> WaitAny(WaitHandleSpan wait_handles, Time deadline_ns);
  static StatusOr<int> WaitAny(WaitHandleSpan wait_handles,
                               Duration timeout_ns) {
    return WaitAny(wait_handles, RelativeTimeoutToDeadlineNanos(timeout_ns));
  }
  static StatusOr<int> WaitAny(WaitHandleSpan wait_handles) {
    return WaitAny(wait_handles, InfiniteFuture());
  }

  // Tries waiting for at least one handle to complete and returns immediately
  // if none have been. The caller will not be blocked even if a handle has not
  // yet been signaled.
  //
  // Returns the index into |wait_handles| of a handle that was signaled. Note
  // that more than one handle may have been signaled and all of the other
  // |wait_handles| should be queried or waited on again until waits for them
  // succeed.
  //
  // Returns -1 if no handles were signaled.
  static StatusOr<int> TryWaitAny(WaitHandleSpan wait_handles);

  // Default constructor creates a permanently signaled handle.
  // Waiting on this handle will never block.
  WaitHandle() = default;

  // Wraps an existing sync file descriptor.
  // Ownership of the file descriptor is transferred to the WaitHandle and must
  // be duplicated by the caller if they want to continue using it.
  explicit WaitHandle(ref_ptr<WaitableObject> object);

  ~WaitHandle();

  // Copying not supported. Create a new WaitHandle from the source.
  WaitHandle(const WaitHandle&) = delete;
  WaitHandle& operator=(const WaitHandle&) = delete;

  // Moving supported; sync primitive ownership is transferred.
  WaitHandle(WaitHandle&& other);
  WaitHandle& operator=(WaitHandle&& other);

  // Unique ID for the WaitHandle instance.
  // Two wait handles, even if waiting on the same underlying primitive, will
  // have differing unique_ids. This can be used for deduping the handles or
  // storing handles in a map.
  uint64_t unique_id() const { return unique_id_; }

  // Returns a unique string representing the handle.
  std::string DebugString() const;

  // Blocks the caller until the handle is signaled or the |deadline| elapses.
  //
  // If waiting on multiple wait handles use WaitAll or WaitAny instead of
  // multiple calls to Wait as they can significantly reduce overhead.
  //
  // Returns success if the wait is successful and the |wait_handle| was
  // signaled. Returns DEADLINE_EXCEEDED if the timeout elapses without the
  // handle having been signaled.
  Status Wait(Time deadline_ns) { return WaitAll({this}, deadline); }
  Status Wait(Duration timeout_ns) {
    return WaitAll({this}, RelativeTimeoutToDeadlineNanos(timeout_ns));
  }
  Status Wait() { return WaitAll({this}, InfiniteFuture()); }

  // Tries waiting on the handle and returns immediately if it would have
  // waited. The caller will not be blocked even if the handle has not yet been
  // signaled.
  //
  // Returns true if the handle has been signaled.
  StatusOr<bool> TryWait();

  // These accessors should generally be considered opaque but may be useful to
  // code trying to interop with other runtimes.
  const ref_ptr<WaitableObject>& object() const { return object_; }

 private:
  // Disposes the handle by closing the fd and issuing callbacks.
  void Dispose();

  static std::atomic<uint64_t> next_unique_id_;

  uint64_t unique_id_ = 0;
  ref_ptr<WaitableObject> object_;
};

// A manually-resettable event primitive.
// Effectively a binary semaphore with a maximum_count of 1 when running in
// auto-reset mode but also provides a sticky manual reset mode.
class ManualResetEvent : public WaitableObject {
 public:
  explicit ManualResetEvent(const char* debug_name = nullptr);

  ~ManualResetEvent() override;

  // Copying not supported.
  ManualResetEvent(const ManualResetEvent&) = delete;
  ManualResetEvent& operator=(const ManualResetEvent&) = delete;

  // Moving supported; sync primitive ownership is transferred.
  ManualResetEvent(ManualResetEvent&& other);
  ManualResetEvent& operator=(ManualResetEvent&& other);

  std::string DebugString() const override;

  // Sets the specified event object to the signaled state.
  // The event stays signaled until Reset is called. Multiple waiters will be
  // woken.
  Status Set();

  // Resets the specified event object to the nonsignaled state.
  // Resetting an event that is already reset has no effect.
  Status Reset();

  // Returns a WaitHandle that will be signaled when the event is set.
  WaitHandle OnSet();

 protected:
  void Initialize();
  void Dispose();

  StatusOr<std::pair<FdType, int>> AcquireFdForWait(Time deadline_ns) override {
    return std::make_pair(fd_type_, fd_);
  }
  StatusOr<bool> TryResolveWakeOnFd(int fd) override { return true; }

  FdType fd_type_ = FdType::kPermanent;
  int fd_ = kInvalidFd;
  int write_fd_ = kInvalidFd;  // Used only for fd_type_ == kPipe.
  const char* debug_name_ = nullptr;
};

}  // namespace iree

#endif  // IREE_BASE_WAIT_HANDLE_H_
