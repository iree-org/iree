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

#include "iree/base/wait_handle.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <unistd.h>

#include <type_traits>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "iree/base/status.h"

// TODO(benvanik): organize these macros - they are terrible.

#if !defined(__ANDROID__) && !defined(OS_IOS) && !defined(__EMSCRIPTEN__)
#define IREE_HAS_PPOLL 1
#endif  // !__ANDROID__  && !__EMSCRIPTEN__
#define IREE_HAS_POLL 1

#if !defined(OS_IOS) && !defined(OS_MACOSX) && !defined(__EMSCRIPTEN__)
#define IREE_HAS_EVENTFD 1
#endif
#define IREE_HAS_PIPE 1
// #define IREE_HAS_SYNC_FILE 1

#if defined(IREE_HAS_EVENTFD)
#include <sys/eventfd.h>
#endif  // IREE_HAS_EVENTFD

namespace iree {

namespace {

constexpr int kInvalidFd = WaitableObject::kInvalidFd;
constexpr int kSignaledFd = WaitableObject::kSignaledFd;

// Retries a syscall until it succeeds or fails for a real reason.
template <typename SyscallT, typename... ParamsT>
StatusOr<typename std::result_of<SyscallT(ParamsT...)>::type> Syscall(
    SyscallT syscall, ParamsT&&... params) {
  while (true) {
    const auto rv = syscall(std::forward<ParamsT>(params)...);
    if (rv >= 0) return rv;
    if (errno == EINTR) {
      // Retry on EINTR.
      continue;
    } else {
      return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC);
    }
  }
}

#if defined(IREE_HAS_PPOLL)

// ppoll(), present on Linux.
// ppoll is preferred as it has a much better timing mechanism; poll can have a
// large slop on the deadline.
// Documentation: https://linux.die.net/man/2/poll
StatusOr<int> SystemPoll(absl::Span<pollfd> poll_fds, Time deadline_ns) {
  // Convert the deadline into a tmo_p struct for ppoll that controls whether
  // the call is blocking or non-blocking. Note that we must do this every
  // iteration of the loop as a previous ppoll may have taken some of the
  // time.
  //
  // See the ppoll docs for more information as to what the expected value is:
  // http://man7.org/linux/man-pages/man2/poll.2.html
  timespec timeout_spec;
  timespec* tmo_p;
  if (deadline == InfinitePast()) {
    // 0 for non-blocking.
    timeout_spec = {0};
    tmo_p = &timeout_spec;
  } else if (deadline == InfiniteFuture()) {
    // nullptr to ppoll() to block forever.
    tmo_p = nullptr;
  } else {
    // Wait only for as much time as we have before the deadline is exceeded.
    absl::Duration remaining_time = deadline - Now();
    if (remaining_time < absl::ZeroDuration()) {
      // Note: we likely have already bailed before getting here with a negative
      // duration.
      return DeadlineExceededErrorBuilder(IREE_LOC);
    }
    timeout_spec = absl::ToTimespec(remaining_time);
    tmo_p = &timeout_spec;
  }
  return Syscall(::ppoll, poll_fds.data(), poll_fds.size(), tmo_p, nullptr);
}

#elif defined(IREE_HAS_POLL)

// poll(), present pretty much everywhere.
// Documentation: https://linux.die.net/man/2/poll
StatusOr<int> SystemPoll(absl::Span<pollfd> poll_fds, Time deadline_ns) {
  int timeout;
  if (deadline == InfinitePast()) {
    // Don't block.
    timeout = 0;
  } else if (deadline == InfiniteFuture()) {
    // Block forever.
    timeout = -1;
  } else {
    absl::Duration remaining_time = deadline - Now();
    if (remaining_time < absl::ZeroDuration()) {
      return DeadlineExceededErrorBuilder(IREE_LOC);
    }
    timeout = static_cast<int>(absl::ToInt64Milliseconds(remaining_time));
  }
  return Syscall(::poll, poll_fds.data(), poll_fds.size(), timeout);
}

#else
#error "No SystemPoll implementation"
#endif  // IREE_HAS_PPOLL / IREE_HAS_POLL / etc

// Builds the list of pollfds to for ppoll wait on and will perform any
// required wait handle callbacks.
//
// The provided deadline will be observed if any of the wait handles needs to
// block for acquiring an fd.
StatusOr<absl::FixedArray<pollfd>> AcquireWaitHandles(
    WaitHandle::WaitHandleSpan wait_handles, Time deadline_ns) {
  absl::FixedArray<pollfd> poll_fds{wait_handles.size()};
  for (int i = 0; i < wait_handles.size(); ++i) {
    poll_fds[i].events = POLLIN | POLLPRI | POLLERR | POLLHUP | POLLNVAL;
    poll_fds[i].revents = 0;
    // NOTE: poll will ignore any negative fds and our kInvalidFd == -1 so we
    // can still put them in the list and it'll just skip them.
    if (!wait_handles[i] || !wait_handles[i]->object()) {
      poll_fds[i].fd = kInvalidFd;
      continue;
    }

    // Acquire the file descriptor for waiting.
    // This may block (if |deadline| allows it) if the fd is not yet available.
    // This is like a pre-wait for the actual poll operation. It can be bad with
    // WaitAny, though we could handle that better here.
    IREE_ASSIGN_OR_RETURN(
        auto fd_info, wait_handles[i]->object()->AcquireFdForWait(deadline));
    poll_fds[i].fd = fd_info.second;

    // Abort if deadline exceeded.
    if (deadline != InfinitePast() && deadline < Now()) {
      return DeadlineExceededErrorBuilder(IREE_LOC)
             << "Deadline exceeded acquiring for fds";
    }
  }
  return poll_fds;
}

Status ClearFd(WaitableObject::FdType fd_type, int fd) {
  // Read in a loop until the read would block.
  // Depending on how the users setup the fd the act of reading may reset the
  // entire handle (such as with the default eventfd mode) or multiple reads
  // may be required (such as with semaphores).
  while (true) {
#if defined(IREE_HAS_EVENTFD)
    eventfd_t val = 0;
    int rv = ::eventfd_read(fd, &val);
#elif defined(IREE_HAS_PIPE)
    char buf;
    int rv = ::read(fd, &buf, 1);
#else
    return UnimplementedErrorBuilder(IREE_LOC) << "fd_type cannot be cleared";
#endif  // IREE_HAS_EVENTFD
    if (rv != -1) {
      // Success! Keep going.
      continue;
    } else {
      if (errno == EWOULDBLOCK) {
        // The read would have blocked meaning that we've hit the end and
        // successfully cleared the fd.
        return OkStatus();
      } else if (errno == EINTR) {
        // Retry.
        continue;
      } else {
        return ErrnoToCanonicalStatusBuilder(errno, IREE_LOC)
               << "ClearFd failed";
      }
    }
  }
}

// Performs a single poll on multiple fds and returns information about the
// signaled fds, if any.
Status MultiPoll(WaitHandle::WaitHandleSpan wait_handles,
                 absl::Span<pollfd> poll_fds, Time deadline_ns,
                 int* out_any_signaled_index, int* out_unsignaled_count) {
  *out_any_signaled_index = -1;
  *out_unsignaled_count = 0;

  // poll has a nasty behavior where it allows -1 for fds... except for at [0].
  // To keep the rest of the code sane we correct for that here as epoll doesn't
  // have that behavior and we may want to special case this later.
  bool any_valid_fds = true;
  int swapped_zero_index = -1;
  if (poll_fds[0].fd < 0) {
    // Find a valid handle.
    for (int i = 1; i < poll_fds.size(); ++i) {
      if (poll_fds[i].fd > 0) {
        swapped_zero_index = i;
        std::swap(poll_fds[0], poll_fds[i]);
        break;
      }
    }
    if (swapped_zero_index == -1) {
      // No valid handles found, meaning that all handles are invalid.
      // We'll skip the wait below so we can share the processing code for any
      // fds that may be kSignaledFd.
      any_valid_fds = false;
    }
  }

  // Pass handles to ppoll.
  // http://man7.org/linux/man-pages/man2/poll.2.html
  if (any_valid_fds) {
    IREE_ASSIGN_OR_RETURN(int rv, SystemPoll(poll_fds, deadline));
    if (rv == 0) {
      // Call timed out and no descriptors were ready.
      // If this was just a poll then that's fine.
      return DeadlineExceededErrorBuilder(IREE_LOC);
    }
  }

  // If we had swapped fds[0] above we need to correct for that now.
  if (swapped_zero_index != -1) {
    std::swap(poll_fds[0], poll_fds[swapped_zero_index]);
  }

  // |rv| denotes the number of fds that were ready. Run through the list and
  // find the ones that were ready and mark them as completed.
  for (int i = 0; i < poll_fds.size(); ++i) {
    if (poll_fds[i].fd == kSignaledFd || poll_fds[i].revents == POLLIN) {
      // First attempt any resolve actions. If these fail we can't consider the
      // fd as having been signaled.
      IREE_ASSIGN_OR_RETURN(
          bool resolved,
          wait_handles[i]->object()->TryResolveWakeOnFd(poll_fds[i].fd));
      if (!resolved) {
        ++(*out_unsignaled_count);
        continue;
      }

      // Successful wait. Kill the fd so it is ignored on the next poll.
      poll_fds[i].fd = kInvalidFd;
      *out_any_signaled_index = i;
    } else if (poll_fds[i].revents) {
      if (poll_fds[i].revents & POLLERR) {
        return InternalErrorBuilder(IREE_LOC);
      } else if (poll_fds[i].revents & POLLHUP) {
        return CancelledErrorBuilder(IREE_LOC);
      } else if (poll_fds[i].revents & POLLNVAL) {
        return InvalidArgumentErrorBuilder(IREE_LOC);
      } else {
        return UnknownErrorBuilder(IREE_LOC);
      }
    } else if (poll_fds[i].fd != kInvalidFd) {
      ++(*out_unsignaled_count);
    }
  }

  return OkStatus();
}

}  // namespace

// static
std::atomic<uint64_t> WaitHandle::next_unique_id_{1};

// static
WaitHandle WaitHandle::AlwaysSignaling() {
  class AlwaysSignalingObject : public WaitableObject {
   public:
    std::string DebugString() const override { return "signal"; }
    StatusOr<std::pair<FdType, int>> AcquireFdForWait(
        Time deadline_ns) override {
      return std::make_pair(FdType::kPermanent, kSignaledFd);
    }
    StatusOr<bool> TryResolveWakeOnFd(int fd) override { return true; }
  };
  static auto* obj = new AlwaysSignalingObject();
  return WaitHandle(add_ref(obj));
}

// static
WaitHandle WaitHandle::AlwaysFailing() {
  class AlwaysFailingObject : public WaitableObject {
   public:
    std::string DebugString() const override { return "fail"; }
    StatusOr<std::pair<FdType, int>> AcquireFdForWait(
        Time deadline_ns) override {
      return InternalErrorBuilder(IREE_LOC) << "AlwaysFailingObject";
    }
    StatusOr<bool> TryResolveWakeOnFd(int fd) override {
      return InternalErrorBuilder(IREE_LOC) << "AlwaysFailingObject";
    }
  };
  static auto* obj = new AlwaysFailingObject();
  return WaitHandle(add_ref(obj));
}

// static
Status WaitHandle::WaitAll(WaitHandleSpan wait_handles, Time deadline_ns) {
  if (wait_handles.empty()) return OkStatus();

  // Build the list of pollfds to wait on.
  IREE_ASSIGN_OR_RETURN(auto poll_fds,
                        AcquireWaitHandles(wait_handles, deadline));

  // Loop until all handles have been signaled or the deadline is exceeded.
  int unsignaled_count = 0;
  do {
    int any_signaled_index = 0;
    IREE_RETURN_IF_ERROR(MultiPoll(wait_handles, absl::MakeSpan(poll_fds),
                                   deadline, &any_signaled_index,
                                   &unsignaled_count));
  } while (unsignaled_count > 0 && Now() < deadline);

  if (unsignaled_count == 0) {
    // All waits resolved.
    return OkStatus();
  } else {
    // One or more were unsignaled.
    return DeadlineExceededErrorBuilder(IREE_LOC);
  }
}

// static
StatusOr<bool> WaitHandle::TryWaitAll(WaitHandleSpan wait_handles) {
  auto status = WaitAll(wait_handles, InfinitePast());
  if (status.ok()) {
    return true;
  } else if (IsDeadlineExceeded(status)) {
    return false;
  }
  return status;
}

// static
StatusOr<int> WaitHandle::WaitAny(WaitHandleSpan wait_handles,
                                  Time deadline_ns) {
  if (wait_handles.empty()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "At least one wait handle is required for WaitAny";
  }

  // Build the list of pollfds to wait on.
  IREE_ASSIGN_OR_RETURN(auto poll_fds,
                        AcquireWaitHandles(wait_handles, deadline));

  // Poll once; this makes a WaitAny just a WaitMulti that doesn't loop.
  int any_signaled_index = -1;
  int unsignaled_count = 0;
  IREE_RETURN_IF_ERROR(MultiPoll(wait_handles, absl::MakeSpan(poll_fds),
                                 deadline, &any_signaled_index,
                                 &unsignaled_count));
  if (any_signaled_index == -1) {
    // No wait handles were valid. Pretend 0 was signaled.
    return 0;
  }
  return any_signaled_index;
}

// static
StatusOr<int> WaitHandle::TryWaitAny(WaitHandleSpan wait_handles) {
  auto status_or = WaitAny(wait_handles, InfinitePast());
  return IsDeadlineExceeded(status_or.status()) ? -1 : status_or;
}

// Storage for static class variables; these won't be needed when we can use
// c++17 everywhere.
constexpr int WaitableObject::kInvalidFd;
constexpr int WaitableObject::kSignaledFd;

WaitHandle::WaitHandle(ref_ptr<WaitableObject> object)
    : unique_id_(++next_unique_id_), object_(std::move(object)) {}

WaitHandle::~WaitHandle() { Dispose(); }

void WaitHandle::Dispose() { object_.reset(); }

WaitHandle::WaitHandle(WaitHandle&& other)
    : unique_id_(other.unique_id_), object_(std::move(other.object_)) {
  other.unique_id_ = 0;
}

WaitHandle& WaitHandle::operator=(WaitHandle&& other) {
  if (this != std::addressof(other)) {
    // Close current handle.
    Dispose();

    // Take ownership of handle and resources.
    object_ = std::move(other.object_);

    other.unique_id_ = ++next_unique_id_;
  }
  return *this;
}

std::string WaitHandle::DebugString() const {
  return object_ ? object_->DebugString() : absl::StrCat("wh_", unique_id_);
}

StatusOr<bool> WaitHandle::TryWait() {
  auto status = WaitAll({this}, InfinitePast());
  if (status.ok()) {
    return true;
  } else if (IsDeadlineExceeded(status)) {
    return false;
  }
  return status;
}

ManualResetEvent::ManualResetEvent(const char* debug_name)
    : debug_name_(debug_name) {
  Initialize();
}

ManualResetEvent::~ManualResetEvent() { Dispose(); }

void ManualResetEvent::Initialize() {
#if defined(IREE_HAS_EVENTFD)
  // Create with an eventfd by default when we support it.
  // eventfd has lower overhead than pipes (the syscalls are cheap).
  // This usually will only fail if the system is completely out of handles.
  //
  // Docs: http://man7.org/linux/man-pages/man2/eventfd.2.html
  fd_type_ = FdType::kEventFd;
  fd_ = Syscall(::eventfd, 0, EFD_CLOEXEC | EFD_NONBLOCK).value();
#elif defined(IREE_HAS_PIPE)
  // Android/Linux/iOS-compatible POSIX pipe handle.
  // Two handles are generated: one for transmitting and one for receiving.
  //
  // Docs: http://man7.org/linux/man-pages/man2/pipe.2.html
  fd_type_ = FdType::kPipe;
  int pipefd[2];
  Syscall(::pipe, pipefd).value();
  Syscall(::fcntl, pipefd[0], F_SETFL, O_NONBLOCK).value();
  fd_ = pipefd[0];
  write_fd_ = pipefd[1];
#else
// NOTE: sync_file does not use Notifier as they come from the kernel.
#error "No fd-based sync primitive on this platform"
#endif  // IREE_HAS_EVENTFD / IREE_HAS_PIPE / etc
}

void ManualResetEvent::Dispose() {
  if (fd_ != kInvalidFd) {
    // Always signal, as we need to ensure waiters are woken.
    IREE_CHECK_OK(Set());
    Syscall(::close, fd_).value();
    fd_ = kInvalidFd;
  }
  if (write_fd_ != kInvalidFd) {
    Syscall(::close, write_fd_).value();
    write_fd_ = kInvalidFd;
  }
}

ManualResetEvent::ManualResetEvent(ManualResetEvent&& other)
    : fd_type_(other.fd_type_),
      fd_(other.fd_),
      write_fd_(other.write_fd_),
      debug_name_(other.debug_name_) {
  other.fd_type_ = FdType::kPermanent;
  other.fd_ = kInvalidFd;
  other.write_fd_ = kInvalidFd;
  other.debug_name_ = nullptr;
}

ManualResetEvent& ManualResetEvent::operator=(ManualResetEvent&& other) {
  if (this != std::addressof(other)) {
    Dispose();
    fd_type_ = other.fd_type_;
    fd_ = other.fd_;
    write_fd_ = other.write_fd_;
    debug_name_ = other.debug_name_;
    other.fd_type_ = FdType::kPermanent;
    other.fd_ = kInvalidFd;
    other.write_fd_ = kInvalidFd;
    other.debug_name_ = nullptr;
    other.Initialize();
  }
  return *this;
}

std::string ManualResetEvent::DebugString() const {
  if (debug_name_) {
    return debug_name_;
  }
#if defined(IREE_HAS_EVENTFD)
  return absl::StrCat("eventfd_", fd_);
#elif defined(IREE_HAS_PIPE)
  return absl::StrCat("pipe_", fd_, "_", write_fd_);
#else
  return absl::StrCat("unknown_", fd_, "_", write_fd_);
#endif  // IREE_HAS_EVENTFD / IREE_HAS_PIPE
}

Status ManualResetEvent::Set() {
#if defined(IREE_HAS_EVENTFD)
  return Syscall(::eventfd_write, fd_, 1ull).status();
#elif defined(IREE_HAS_PIPE)
  char buf = '\n';
  return Syscall(::write, write_fd_, &buf, 1).status();
#else
  return UnimplementedErrorBuilder(IREE_LOC)
         << "No fd-based sync primitive on this platform";
#endif  // IREE_HAS_EVENTFD / IREE_HAS_PIPE
}

Status ManualResetEvent::Reset() { return ClearFd(fd_type_, fd_); }

WaitHandle ManualResetEvent::OnSet() { return WaitHandle(add_ref(this)); }

}  // namespace iree
