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

namespace iree {
namespace remoting {

class IoLoop::UringImpl : public IoLoop {
 public:
  static constexpr unsigned kQueueEntries = 16;
  UringImpl() : IoLoop(ImplType::kUring) { memset(&params, 0, sizeof(params)); }
  ~UringImpl() override {
    if (alloced) {
      io_uring_queue_exit(&ring);
    }
  }

  static iree_status_t TryCreateSpecific(std::unique_ptr<IoLoop> &created) {
    auto self = std::make_unique<UringImpl>();
    if (io_uring_queue_init_params(kQueueEntries, &self->ring, &self->params) <
        0) {
      return StatusFromErrno("io_uring_queue_init_params failed", errno);
    }
    self->alloced = true;

    if ((self->params.features & IORING_FEAT_FAST_POLL)) {
      self->has_fast_poll = true;
    } else {
      IREE_DVLOG(1)
          << "IORING_FEAT_FAST_POLL not available (needs kernel>=5.6)";
    }

    // Children should not inherit the ring.
    io_uring_ring_dontfork(&self->ring);

    created = std::move(self);
    return iree_ok_status();
  }

  void SubmitSpecific(IoRequest *request) {
    IREE_DCHECK(request != nullptr) << "Cannot submit null IoRequest";
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    // TODO: Something better?
    IREE_CHECK(sqe != nullptr) << "Insufficient uring queue entries";
    IREE_DVLOG(1) << "queueing request type "
                  << static_cast<int>(request->type()) << " onto ring";
    inflight_count() += 1;
    PrepareRequest(request, sqe);
    io_uring_sqe_set_flags(sqe, 0);
    memcpy(&sqe->user_data, &request, sizeof(IoRequest *));

    // TODO: There are a lot of ways to optimize syscall count here so that
    // number of Submit() < number of syscalls.
    int count = io_uring_submit(&ring);
    IREE_DVLOG(1) << "io_uring_submit: count=" << count;
  }

  void Run(KeepRunningPredicate keep_running_predicate) override {
    while (inflight_count() > 0) {
      bool should_block =
          keep_running_predicate ? keep_running_predicate() : true;
      // Process completions.
      struct io_uring_cqe *cqe;

      int wait_count = should_block ? 1 : 0;
      IREE_DVLOG(1) << "io_uring_wait_cqe_nr(): inflight=" << inflight_count()
                    << ", nr=" << wait_count;
      int rc = io_uring_wait_cqe_nr(&ring, &cqe, wait_count);
      IREE_DVLOG(1) << "returned io_uring_wait_cqe_nr(): rc=" << rc;
      if (rc == -EAGAIN) {
        assert(!should_block && "got EAGAIN on blocking operation");
        return;
      }
      IREE_CHECK(rc == 0);  // TODO: Better error handling.

      unsigned head;
      unsigned count = 0;
      io_uring_for_each_cqe(&ring, head, cqe) {
        ++count;
        IoRequest *request;
        memcpy(&request, &cqe->user_data, sizeof(IoRequest *));
        IREE_DCHECK(request != nullptr)
            << "IoRequest in io_uring_cqe userdata is null";
        FillCompletedRequest(request, cqe);
        request->HandleCompletion();
        inflight_count() -= 1;
      }
      io_uring_cq_advance(&ring, count);
    }
  }

  void PrepareRequest(IoRequest *request, struct io_uring_sqe *sqe) {
    using Type = IoRequest::Type;
    auto t = request->type();
    switch (t) {
      case Type::kAccept: {
        auto *accept = static_cast<IoAcceptRequest *>(request);
        io_uring_prep_accept(
            sqe, accept->listen_fd(),
            reinterpret_cast<struct sockaddr *>(&accept->client_addr()),
            &accept->client_addr_size(), /*flags=*/0);
        return;
      }
      case Type::kSocketClose: {
        auto *close = static_cast<IoCloseSocketRequest *>(request);
        io_uring_prep_close(sqe, close->fd());
        return;
      }
      case Type::kSocketConnect: {
        auto *connect = static_cast<IoConnectSocketRequest *>(request);
        io_uring_prep_connect(sqe, connect->fd(), connect->addr().sockaddr(),
                              connect->addr().addr_len());
        return;
      }
      case Type::kSocketShutdown: {
        auto *shutdown_req = static_cast<IoSocketShutdownRequest *>(request);
        io_uring_prep_shutdown(sqe, shutdown_req->fd(), shutdown_req->how());
        return;
      }
      case Type::kSocketVec: {
        auto *socket_vec = static_cast<IoSocketVecRequest *>(request);
        if (socket_vec->flags().is_write()) {
          io_uring_prep_writev(sqe, socket_vec->fd(),
                               socket_vec->iovec()->front(),
                               socket_vec->iovec()->size(), /*offset=*/0);
        } else {
          if (socket_vec->flags().use_readv()) {
            io_uring_prep_readv(sqe, socket_vec->fd(),
                                socket_vec->iovec()->front(),
                                socket_vec->iovec()->size(), /*offset=*/0);
          } else {
            void *buf = socket_vec->iovec()->iov_base(0);
            size_t len = socket_vec->iovec()->iov_len(0);
            io_uring_prep_recv(sqe, socket_vec->fd(), buf, len, /*flags=*/0);
          }
        }
        return;
      }
      default:
        IREE_CHECK(false) << "Unhandled Uring IoRequest::Type "
                          << static_cast<int>(t);
    }
  }

  void FillCompletedRequest(IoRequest *request, struct io_uring_cqe *cqe) {
    int result = cqe->res;
    if (result < 0) {
      request->set_status(StatusFromErrno("syscall failed", -result));
    } else {
      request->ClearStatus();
    }

    using Type = IoRequest::Type;
    auto t = request->type();
    switch (t) {
      case Type::kAccept: {
        auto *accept = static_cast<IoAcceptRequest *>(request);
        accept->client_fd() = result;
        return;
      }
      case Type::kSocketClose: {
        return;
      }
      case Type::kSocketConnect: {
        return;
      }
      case Type::kSocketShutdown: {
        if (result == -EINVAL) {
          // Support for the shutdown op was recent. If not supported, just
          // use the blocking version.
          auto *shutdown_req = static_cast<IoSocketShutdownRequest *>(request);
          IREE_DVLOG(1) << "Kernel does not support IORING_OP_SHUTDOWN: "
                           "calling blocking version";
          int rc = ::shutdown(shutdown_req->fd(), shutdown_req->how());
          if (rc == 0) {
            request->ClearStatus();
          }
        }
        return;
      }
      case Type::kSocketVec: {
        auto *socket_vec = static_cast<IoSocketVecRequest *>(request);
        socket_vec->complete_bytes() = result >= 0 ? result : 0;
        return;
      }
    }
    IREE_CHECK(false) << "Unhandled Uring IoRequest::Type "
                      << static_cast<int>(t);
  }

 private:
  static iree_status_t StatusFromErrno(const char *message,
                                       int explicit_errno) {
    char msg_buf[128];
    msg_buf[0] = 0;
    char *actual_msg_buf;
    auto code = iree_status_code_from_errno(explicit_errno);
#ifdef _GNU_SOURCE
    actual_msg_buf = strerror_r(explicit_errno, msg_buf, sizeof(msg_buf));
#else
    actual_msg_buf = msg_buf;
    strerror_r(current_errno, msg_buf, sizeof(msg_buf))
#endif
    IREE_DLOG(WARNING) << message << ": " << actual_msg_buf;
    return iree_make_status(code, "%s: %s", message, actual_msg_buf);
  }

  struct io_uring ring;
  struct io_uring_params params;
  bool alloced : 1 = false;
  bool has_fast_poll : 1 = false;

  friend class IoLoop;
};

}  // namespace remoting
}  // namespace iree
