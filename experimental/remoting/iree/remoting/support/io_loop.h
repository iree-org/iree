// Copyright 2020 Google LLC
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

#ifndef IREE_REMOTING_SUPPORT_IO_LOOP_H_
#define IREE_REMOTING_SUPPORT_IO_LOOP_H_

#include <functional>
#include <memory>
#include <type_traits>

#include "experimental/remoting/iree/remoting/support/io_buffer.h"
#include "experimental/remoting/iree/remoting/support/socket.h"
#include "iree/base/api.h"

namespace iree {
namespace remoting {

class IoLoop;
class IoRequest;

// Base class for IO requests.
class IoRequest {
 public:
  enum class Type {
    kAccept,
    kSocketShutdown,
    kSocketClose,
    kSocketConnect,
    // TODO: Split this in to kSocketVecWrite, kSocketVecRead and kSocketRecv.
    kSocketVec,
  };
  using CompletionHandlerTy = void (*)(IoRequest *request,
                                       void *completion_data);
  IoRequest(Type type) : type_(type) {}
  ~IoRequest() { ClearStatus(); }
  Type type() const { return type_; }
  IoLoop *io_loop() { return io_loop_; }

  template <typename RequestTy>
  struct Deleter {
    void operator()(RequestTy *request) {
      request->template Release<RequestTy>();
    }
  };

  // Sets the untyped completion handler (also can be set in the constructor).
  void SetCompletionHandler(void *completion_data,
                            CompletionHandlerTy handler) {
    completion_handler_ = handler;
    completion_data_ = completion_data;
  }

  // Triggers the completion handler. Intended to be called by IoLoop
  // implementations.
  void HandleCompletion() {
    if (completion_handler_) completion_handler_(this, completion_data_);
  }

  // Status handling.
  bool ok() { return iree_status_is_ok(status_); }
  // Consumes the status from the request, resetting it to ok and returning
  // the prior. Caller now has responsibility for handling status.
  iree_status_t ConsumeStatus() {
    iree_status_t ret_status = status_;
    status_ = iree_ok_status();
    return ret_status;
  }
  // Clears any status that this request contains.
  void ClearStatus() {
    if (IREE_UNLIKELY(!iree_status_is_ok(status_))) {
      status_ = iree_status_ignore(status_);
    }
  }
  // Sets a new status on the request. Request takes ownership of the status.
  void set_status(iree_status_t status) {
    ClearStatus();
    status_ = status;
  }

 protected:
  void Retain() { ref_count_ += 1; }
  template <typename RequestTy>
  void Release();

 private:
  int ref_count_ = 1;
  Type type_;
  IoLoop *io_loop_ = nullptr;
  // Because the completion handler is type erased and provided by the derived
  // class, it is necessary that upon construction, an appropriate function
  // pointer be installed to signal completion.
  CompletionHandlerTy completion_handler_ = nullptr;
  void *completion_data_ = nullptr;
  iree_status_t status_ = iree_ok_status();

  friend class IoLoop;
};

// A pointer to an IoRequest which delegates deletion to the IoLoop.
template <typename T>
using IoRequestPtr = std::unique_ptr<T, IoRequest::Deleter<T>>;

// An IO event loop designed to be backed by queue-based kernel IO interfaces
// (and emulated elsewhere).
class IoLoop {
 public:
  enum class ImplType {
    kUring,
  };
  virtual ~IoLoop() = default;

  // The type of the implementation.
  ImplType impl_type() const { return impl_type_; }

  // Creates an IoLoop appropriate for this platform and features available.
  static iree_status_t Create(std::unique_ptr<IoLoop> &created);

  // Runs the IoLoop until drained.
  using KeepRunningPredicate = std::function<bool()>;
  virtual void Run(KeepRunningPredicate keep_running_predicate = nullptr) = 0;

  // Custom allocator for IoRequests.
  // TODO: Have a non-heap allocator.
  template <typename RequestTy, typename... Args>
  IoRequestPtr<RequestTy> NewRequest(Args &&... args) {
    static_assert(std::is_convertible<RequestTy *, IoRequest *>::value,
                  "Request must be a subclass of IoRequest");
    RequestTy *request = AllocateRequest<RequestTy>();
    new (request) RequestTy(std::forward<Args>(args)...);
    request->io_loop_ = this;
    return IoRequestPtr<RequestTy>(request);
  }

  // Submits a request. This will always result in the request's callback being
  // invoked at a later date (not within the scope of this call).
  template <typename RequestTy>
  void Submit(IoRequestPtr<RequestTy> request) {
    SubmitGeneric(request.release());
  }

  // Combines NewRequest and Submit into one call.
  template <typename RequestTy, typename... Args>
  void SubmitNew(Args &&... args) {
    Submit(NewRequest<RequestTy>(std::forward<Args>(args)...));
  }

 protected:
  int &inflight_count() { return inflight_count_; }

 private:
  // Forward-declare implementation types.
  class UringImpl;

  ImplType impl_type_;

  // Number of requests in flight.
  int inflight_count_ = 0;

  // Initializes the loop with implementation specific information including:
  //   - impl_type: The implementation type (closed hierarchy)
  IoLoop(ImplType impl_type) : impl_type_(impl_type) {}

  // Allocates uninitialized storage for the given request.
  // TODO: Have a non-heap allocator.
  template <typename RequestTy>
  RequestTy *AllocateRequest() {
    return static_cast<RequestTy *>(::operator new(sizeof(RequestTy)));
  }

  // Frees storage for a request pointer
  // TODO: Have a non-heap allocator.
  void FreeRequest(void *request) { ::operator delete(request); }

  // Submits a request as a raw pointer, transferring ownership to this
  // function. Eventually, request->HandleComplete() must be called.
  void SubmitGeneric(IoRequest *request);

  friend class IoRequest;
};

template <typename RequestTy>
inline void IoRequest::Release() {
  if (--ref_count_ == 0) {
    RequestTy *typed_this = static_cast<RequestTy *>(this);
    IoLoop *local_loop = this->io_loop();
    typed_this->~RequestTy();
    local_loop->FreeRequest(typed_this);
  }
}

//------------------------------------------------------------------------------
// Specific IO request types.
//------------------------------------------------------------------------------

// CRTP base class for derived IoRequest classes, implementing additional
// mechanics.
template <typename DerivedTy>
class DerivedIoRequest : public IoRequest {
 public:
  // TODO: Unify the type-erased fp callback type with this std::function
  // based one.
  using CallbackTy = std::function<void(IoRequestPtr<DerivedTy> request)>;

  DerivedIoRequest(Type type, CallbackTy on_complete = nullptr)
      : IoRequest(type), on_complete_(std::move(on_complete)) {
    if (on_complete_) {
      SetCompletionHandler(nullptr,
                           &DerivedIoRequest::DerivedCompletionHandler);
    }
  }

 protected:
  CallbackTy on_complete_;

  static void DerivedCompletionHandler(IoRequest *base_request,
                                       void *unused_data) {
    // The completion handler always owns the IoRequest, so promote it back
    // into a smart pointer. If there is a user level on_complete callback,
    // it will be transferred there, and that code will decide whether it
    // lives or dies.
    DerivedTy *derived_request = static_cast<DerivedTy *>(base_request);
    IoRequestPtr<DerivedTy> owned_request(derived_request);

    if (derived_request->on_complete_) {
      // Make sure the request survives for the duration of the callback,
      // even if the callback drops the unique_ptr on the floor.
      derived_request->Retain();
      derived_request->on_complete_(std::move(owned_request));
      derived_request->template Release<DerivedTy>();
    }
  }

  friend class IoLoop;
};

// An IO request to accept a connection from a socket.
class IoAcceptRequest final : public DerivedIoRequest<IoAcceptRequest> {
 public:
  using Ptr = IoRequestPtr<IoAcceptRequest>;
  IoAcceptRequest(socket_t listen_fd, CallbackTy on_complete = nullptr)
      : DerivedIoRequest(Type::kAccept, std::move(on_complete)),
        listen_fd_(listen_fd) {}

  socket_t listen_fd() { return listen_fd_; }
  struct sockaddr_storage &client_addr() {
    return client_addr_;
  }
  socklen_t &client_addr_size() { return client_addr_size_; }
  socket_t &client_fd() { return client_fd_; }

 private:
  socket_t listen_fd_;
  struct sockaddr_storage client_addr_;
  socklen_t client_addr_size_ = sizeof(struct sockaddr_storage);
  socket_t client_fd_ = -1;
};

// Closes a socket. This is separate from close() because on Windows, the two
// are distinct.
class IoCloseSocketRequest final
    : public DerivedIoRequest<IoCloseSocketRequest> {
 public:
  using Ptr = IoRequestPtr<IoCloseSocketRequest>;
  IoCloseSocketRequest(socket_t fd, CallbackTy on_complete = nullptr)
      : DerivedIoRequest(Type::kSocketClose, std::move(on_complete)), fd_(fd) {}

  socket_t &fd() { return fd_; }

 private:
  socket_t fd_;
};

// Connects a socket.
class IoConnectSocketRequest final
    : public DerivedIoRequest<IoConnectSocketRequest> {
 public:
  using Ptr = IoRequestPtr<IoConnectSocketRequest>;
  IoConnectSocketRequest(socket_t fd, SocketAddress addr,
                         CallbackTy on_complete = nullptr)
      : DerivedIoRequest(Type::kSocketConnect, std::move(on_complete)),
        fd_(fd),
        addr_(addr) {}

  socket_t &fd() { return fd_; }
  SocketAddress &addr() { return addr_; }

 private:
  socket_t fd_;
  SocketAddress addr_;
};

class IoSocketShutdownRequest final
    : public DerivedIoRequest<IoSocketShutdownRequest> {
 public:
  using Ptr = IoRequestPtr<IoSocketShutdownRequest>;
  IoSocketShutdownRequest(socket_t fd, int how = SHUT_WR,
                          CallbackTy on_complete = nullptr)
      : DerivedIoRequest(Type::kSocketShutdown, std::move(on_complete)),
        fd_(fd),
        how_(how) {}

  socket_t &fd() { return fd_; }
  int &how() { return how_; }

 private:
  socket_t fd_;
  int how_;
};

// Vectored socket read or write request.
// Note that this is mainly useful for write as the behavior on read is somewhat
// under-documented and platform dependent (i.e. it can differ based on whether
// the underylying syscall blocks to attempt to read the full vector or
// returns when some data is available).
class IoSocketVecRequest final : public DerivedIoRequest<IoSocketVecRequest> {
 public:
  class Flags {
   public:
    bool is_write() { return is_write_; }
    bool use_readv() { return use_readv_; }

   private:
    Flags() = default;
    bool is_write_ : 1 = false;
    bool use_readv_ : 1 = false;
    friend class IoSocketVecRequest;
  };

  static Flags ForWrite() {
    Flags flags;
    flags.is_write_ = true;
    flags.use_readv_ = false;
    return flags;
  }

  static Flags ForReadV() {
    Flags flags;
    flags.is_write_ = false;
    flags.use_readv_ = true;
    return flags;
  }

  static Flags ForRecv() {
    Flags flags;
    flags.is_write_ = false;
    flags.use_readv_ = false;
    return flags;
  }

  using Ptr = IoRequestPtr<IoSocketVecRequest>;
  IoSocketVecRequest(Flags flags, socket_t fd, IoBufferVec::Ptr iovec,
                     CallbackTy on_complete = nullptr)
      : DerivedIoRequest(Type::kSocketVec, std::move(on_complete)),
        iovec_(std::move(iovec)),
        fd_(fd),
        flags_(flags) {
    if (flags_.is_write()) {
      assert(flags_.use_readv() || iovec_->size() == 1);
    }
  }
  Flags flags() { return flags_; }

  IoBufferVec::Ptr &iovec() { return iovec_; }
  socket_t &fd() { return fd_; }
  size_t &complete_bytes() { return complete_bytes_; }

 private:
  IoBufferVec::Ptr iovec_;
  socket_t fd_;
  size_t complete_bytes_;
  Flags flags_;
};

}  // namespace remoting
}  // namespace iree

#endif  // IREE_REMOTING_SUPPORT_IO_LOOP_H_
