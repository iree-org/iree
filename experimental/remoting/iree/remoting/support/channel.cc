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

#include "experimental/remoting/iree/remoting/support/channel.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

using namespace std::placeholders;

namespace iree {
namespace remoting {

//===----------------------------------------------------------------------===//
// Channel base classes
//===----------------------------------------------------------------------===//

void Channel::Write(IoBuffer::Ptr buffer, int32_t length, int32_t offset,
                    bool flush) {
  auto iovec = NewIoVec();
  iovec->add(buffer->data_bytes() + offset, length, std::move(buffer));
  Write(std::move(iovec), /*flush=*/flush);
}

void Channel::Write(const void *data, size_t length, bool flush) {
  auto iovec = NewIoVec();
  iovec->add(const_cast<void *>(data), length);
  Write(std::move(iovec), /*flush=*/flush);
}

//===----------------------------------------------------------------------===//
// SocketChannel
//===----------------------------------------------------------------------===//

SocketChannel::SocketChannel(IoLoop &io_loop, IoBufferPool &buffer_pool,
                             IoBufferVec::Pool &iovec_pool, socket_t fd)
    : Channel(buffer_pool, iovec_pool), io_loop_(io_loop), fd_(fd) {}

SocketChannel::~SocketChannel() {
  IREE_DVLOG(1) << "~SocketChannel";
  iree_status_ignore(read_error_);
}

void SocketChannel::Close() {
  if (close_initiated_) return;
  close_initiated_ = true;
  // Note that outstanding reads and writes are not necessarily cancelled by
  // a close, but a socket shutdown can cause an orderly shutdown such that
  // outstanding requests are cancelled prior to an actual close.
  io_loop_.SubmitNew<IoSocketShutdownRequest>(
      fd(), SHUT_RDWR,
      std::bind(&SocketChannel::HandleShutdownOnClose, this, _1));
}

void SocketChannel::HandleShutdownOnClose(IoSocketShutdownRequest::Ptr r) {
  IREE_DVLOG(1) << "SocketChannel::HandleShutdownOnClose";
  io_loop_.SubmitNew<IoCloseSocketRequest>(
      fd(), std::bind(&SocketChannel::HandleCloseComplete, this, _1));
}

void SocketChannel::HandleCloseComplete(IoCloseSocketRequest::Ptr r) {
  IREE_DVLOG(1) << "SocketChanne::HandleCloseComplete";
  close_complete_ = true;
  if (on_close_) {
    on_close_(r->ConsumeStatus());
  }
  SchedQuiescent();
}

void SocketChannel::Write(IoBufferVec::Ptr io_vector, bool flush) {
  if (IREE_LIKELY(io_vector)) {
    if (write_incoming_.call_count() == 0) {
      // First write in this batch.
      write_incoming_.accum() = std::move(io_vector);
    } else {
      // Subsequent write.
      write_incoming_.accum()->append(*io_vector);
    }
    write_incoming_.call_count() += 1;
    if (flush) {
      write_incoming_.flushed() = true;
      SchedWrite();
    }
  } else if (flush && !io_vector && write_incoming_.call_count() > 0) {
    // Stand-alone flush with no io_vector.
    write_incoming_.flushed() = true;
    SchedWrite();
  }
}

void SocketChannel::HandleWriteComplete(IoSocketVecRequest::Ptr r) {
  IREE_DVLOG(1) << "SocketChannel::HandleWriteComplete: completed_bytes="
                << r->complete_bytes();
  WriteBuffer current;
  current.TakeFrom(write_outgoing_);
  assert(current.call_count() > 0 &&
         "HandleWriteComplete with no active calls");

  if (IREE_LIKELY(on_write_)) {
    auto status = r->ConsumeStatus();
    // Calls 1..N get a clone.
    for (int i = 1; i < current.call_count(); ++i) {
      on_write_(iree_status_clone(status));
    }
    // And 0 gets the original.
    on_write_(status);
  }

  SchedWrite();
  SchedQuiescent();
}

void SocketChannel::SchedWrite() {
  if (write_incoming_.call_count() == 0 || write_outgoing_.call_count() > 0) {
    // No pending write or write in progress. Do nothing.
    return;
  }
  if (IREE_UNLIKELY(close_initiated_)) {
    IREE_LOG(WARNING) << "SocketChannel: Dropping Write() after Close()";
    return;
  }
  IREE_DVLOG(1) << "SocketChannel::SchedWrite: call_count="
                << write_incoming_.call_count();
  write_outgoing_.TakeFrom(write_incoming_);
  assert(write_incoming_.call_count() == 0);
  io_loop_.SubmitNew<IoSocketVecRequest>(
      IoSocketVecRequest::ForWrite(), fd(), std::move(write_outgoing_.accum()),
      std::bind(&SocketChannel::HandleWriteComplete, this, _1));
}

void SocketChannel::Read(int32_t max_size, bool read_fully) {
  assert(max_size >= 0 && "Read size must be >= 0");
  read_outgoing_.emplace_back(max_size, read_fully);
  read_outgoing_bytes_ += max_size;
  read_needs_sched_ = true;
  if (!callback_critical_section_) {
    SchedRead();
  }
}

void SocketChannel::HandleReadComplete(IoSocketVecRequest::Ptr r) {
  IREE_DVLOG(1) << "[ENTER] SocketChannel::HandleReadComplete(): inflight="
                << read_inflight_ << ", received=" << r->complete_bytes()
                << ", read_incoming.size=" << read_incoming_.size();

  read_inflight_ -= 1;
  // Handle error (either this request error'd or already in an error state).
  if (IREE_UNLIKELY(!r->ok() || r->complete_bytes() == 0)) {
    read_error_ = r->ConsumeStatus();
    if (iree_status_is_ok(read_error_)) {
      read_error_ = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
      IREE_DVLOG(1) << "SocketChannel::HandleReadComplete(): eof";
    } else {
      IREE_DVLOG(1) << "SocketChannel::HandleReadComplete(): read error = "
                    << read_error_;
    }
  } else {
    read_incoming_bytes_ += r->complete_bytes();
    read_incoming_.emplace_back(std::move(r->iovec()), r->complete_bytes());
  }

  do {
    callback_critical_section_ = true;
    SchedRead();
    callback_critical_section_ = false;
  } while (read_needs_sched_);
  SchedQuiescent();
  IREE_DVLOG(1) << "[EXIT] SocketChannel::HandleReadComplete(): inflight="
                << read_inflight_
                << ", read_incoming.size=" << read_incoming_.size();
}

void SocketChannel::SchedRead() {
  read_needs_sched_ = false;
  // Service any client reads that can be fulfilled.
  while (!read_outgoing_.empty()) {
    ReadOutgoing op = read_outgoing_.front();
    if (IREE_LIKELY(!op.fully() || read_incoming_bytes_ >= op.size())) {
      // Op can be fulfilled.
      IREE_DVLOG(1) << "SocketChannel::SchedRead(): Fulfilling read "
                       "operation (outstanding="
                    << read_outgoing_.size() << ")";
      read_outgoing_.pop_front();
      FulfillReadOp(op);
    } else {
      // Insufficient received to fulfill.
      if (IREE_UNLIKELY(!iree_status_is_ok(read_error_) &&
                        !read_outgoing_.empty())) {
        // Call back with the read error.
        read_outgoing_.pop_front();
        if (on_read_) {
          on_read_(iree_status_clone(read_error_), nullptr);
        }
      } else {
        // Channel is not in error - wait for more data.
        break;
      }
    }
  }

  // Schedule any new transport reads needed.
  // TODO: This can be loosened up significantly in order to keep a healthy
  // backlog of read requests pending.
  if (!read_outgoing_.empty() && read_inflight_ == 0) {
    if (IREE_UNLIKELY(close_initiated_)) {
      IREE_LOG(WARNING) << "SocketChannel: Dropping Read() after Close()";
      return;
    }
    IREE_DVLOG(1) << "SocketChannel::SchedRead(): Schedule vector read";
    IoBuffer::Ptr buffer;
    // TODO: Harvest any partially read buffer and reschedule in order to
    // avoid excess buffer use on a trickling socket.
    buffer = NewBuffer();
    auto size = buffer->size();
    int32_t offset = 0;

    auto iovec = NewIoVec();
    iovec->add(buffer->data(), size, std::move(buffer));
    read_inflight_ += 1;
    io_loop_.SubmitNew<IoSocketVecRequest>(
        IoSocketVecRequest::ForRecv(), fd(), std::move(iovec),
        std::bind(&SocketChannel::HandleReadComplete, this, _1));
  } else {
    IREE_DVLOG(1) << "SocketChannel::SchedRead(): "
                  << "Not scheduling new read (read_outgoing.size="
                  << read_outgoing_.size()
                  << ", read_incoming.size=" << read_incoming_.size()
                  << ", read_inflight=" << read_inflight_ << ")";
  }
}

void SocketChannel::FulfillReadOp(ReadOutgoing op) {
  auto client_iovec = NewIoVec();
  int32_t complete_bytes = 0;
  int32_t remain_bytes = op.size();
  // Always transfer at least one (in the case of !fully()).
  do {
    if (read_incoming_.empty()) break;
    ReadIncoming &incoming = read_incoming_.front();
    int32_t transfer_size_bytes = incoming.iovec()->TransferTo(
        *client_iovec, std::min(remain_bytes, incoming.available_bytes()));
    incoming.available_bytes() -= transfer_size_bytes;
    read_incoming_bytes_ -= transfer_size_bytes;
    complete_bytes += transfer_size_bytes;
    if (incoming.available_bytes() == 0) {
      IREE_DVLOG(1) << "Recycling drained iovec";
      read_incoming_.pop_front();
    } else {
      IREE_DVLOG(1) << "Not recycling iovec (available_bytes="
                    << incoming.available_bytes() << ")";
    }
    remain_bytes -= transfer_size_bytes;
  } while (remain_bytes > 0 && read_incoming_bytes_ > 0);

  if (op.fully()) {
    assert(remain_bytes == 0 &&
           "Short read of FulfillReadOp for requested full read");
  }

  if (on_read_) {
    on_read_(iree_ok_status(), std::move(client_iovec));
  }
}

void SocketChannel::SchedQuiescent() {
  if (IREE_LIKELY(!close_complete_ || quiescent_)) return;
  if (write_outgoing_.call_count() == 0 && read_inflight_ == 0) {
    IREE_DVLOG(1) << "SocketChannel: Enter quiescent state";
    quiescent_ = true;
    if (on_quiescent_) {
      on_quiescent_();
    }
  }
}

}  // namespace remoting
}  // namespace iree
