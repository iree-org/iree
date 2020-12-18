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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <type_traits>

#include "experimental/remoting/iree/remoting/support/io_buffer.h"
#include "experimental/remoting/iree/remoting/support/io_loop.h"
#include "experimental/remoting/iree/remoting/support/socket.h"
#include "iree/base/api.h"

#ifndef IREE_REMOTING_SUPPORT_CHANNEL_H_
#define IREE_REMOTING_SUPPORT_CHANNEL_H_

namespace iree {
namespace remoting {

//===----------------------------------------------------------------------===//
// Channel Base Classes
//
// Channels present a view over indeterminate streams, allowing higher-level
// operations such as full reads/writes and vector read/writes that allow
// for simplified ergonomics for systems wishing to minimize downstream copies.
//===----------------------------------------------------------------------===//

// Base class shared between Input and Output channels.
// Provides common definitions and shared facilities.
class Channel {
 public:
  virtual ~Channel() = default;

  // A callback that only receives a status.
  using StatusCallback = std::function<void(iree_status_t status)>;

  // A callback that received a status and vectorized results in the form of
  // an IoVec. If the status is failing, |result| will be nullptr.
  using VectorCallback =
      std::function<void(iree_status_t status, IoBufferVec::Ptr result)>;

  IoBufferPool &buffer_pool() { return buffer_pool_; }
  IoBufferVec::Pool &iovec_pool() { return iovec_pool_; }

  // Gets a fresh IoBufferVec from the pool.
  IoBufferVec::Ptr NewIoVec() { return iovec_pool_.Get(); }

  // Gets a fresh buffer from the pool.
  IoBuffer::Ptr NewBuffer() { return buffer_pool_.Get(); }

  // The optimal buffer size for this channel.
  size_t buffer_size() { return buffer_pool_.buffer_data_size(); }

  //===---------------------------------------------------------------------===/
  // Close/quiesce operations
  //===---------------------------------------------------------------------===/

  // Asynchronously closes the channel. For bidi channels (i.e. Sockets), both
  // sides will close the same underlying entity and only one should be
  // closed. No matter how many times Close() is invoked, only one OnClose()
  // will be called back.
  void OnClose(StatusCallback on_close) { on_close_ = std::move(on_close); }
  virtual void Close() = 0;

  // Callback for when the channel becomes quiescent.
  // Unless if an outside actor manipulates the channel, no further requests
  // or callbacks will take place. This happens at some point after OnClose
  // is called.
  using QuiescentCallback = std::function<void()>;
  void OnQuiescent(QuiescentCallback on_quiescent) {
    on_quiescent_ = std::move(on_quiescent);
  }

  // Whether the channel is quiescent.
  virtual bool is_quiescent() = 0;

  //===---------------------------------------------------------------------===/
  // Write operations
  //===---------------------------------------------------------------------===/

  // Writes the given IO vector to the channel, calling back with a status.
  // Does not call back if nullptr. All backing buffers and data are treated
  // as constant.
  void OnWrite(StatusCallback on_write) { on_write_ = std::move(on_write); }
  virtual void Write(IoBufferVec::Ptr io_vector, bool flush = true) = 0;

  // Short-cut to perform a non-vectored write of a single buffer.
  void Write(IoBuffer::Ptr buffer, int32_t offset, int32_t length,
             bool flush = true);

  // Short-cut to perform a non-vectored write of a single wrapped data pointer.
  // Per the contract with IoBufferVec, the data pointer must be
  // malloc-aligned.
  void Write(const void *data, size_t length, bool flush = true);

  // Null write whose only purpose is to flush.
  void Flush() { Write(/*io_vector=*/IoBufferVec::Ptr(), /*flush=*/true); }

  //===---------------------------------------------------------------------===/
  // Read operations
  //===---------------------------------------------------------------------===/

  // Reads up to |max_size| bytes from the channel, invoking the given callback
  // with the status and results. If |read_fully| is true, then exactly
  // |max_size| bytes will be read (or a failing status will be returned).
  void OnRead(VectorCallback on_read) { on_read_ = std::move(on_read); }
  virtual void Read(int32_t max_size, bool read_fully) = 0;

 protected:
  Channel(IoBufferPool &buffer_pool, IoBufferVec::Pool &iovec_pool)
      : buffer_pool_(buffer_pool), iovec_pool_(iovec_pool) {}
  StatusCallback on_close_;
  QuiescentCallback on_quiescent_;
  StatusCallback on_write_;
  VectorCallback on_read_;

 private:
  IoBufferPool &buffer_pool_;
  IoBufferVec::Pool &iovec_pool_;
};

//===----------------------------------------------------------------------===//
// Socket channels
//===----------------------------------------------------------------------===//

// An asynchronous bidi channel that wraps a socket via an IoLoop.
// TODO: Move this to the Channel class as it describes the general contract.
//
// Writes:
// -------
// Writes are semi-buffered and sequenced, requiring a write with flush=true
// in order to commit them to the medium. Note that |flush| is an advisory
// signal: the implementation may choose to flush more aggressively.
// There is also a facility for performing un-buffered writes, which send
// immediately, but this is retained for cases that need it vs the default
// because it requires great care in use.
//
// Reads:
// ------
// Reads are semi-buffered in order to service the read_fully case. Reads are
// always scheduled for over-read at a buffer granularity and consumer
// callbacks are serviced from these read buffers, allowing spanning reads of
// sizes up to what an IoBufferVec can handle. Note that this is still zero
// copy capable since results are delivered to consumers as an IoBufferVec that
// may span multiple physical buffers. As a consequence, calls to Read() will
// produce a sequence of callbacks matching the original calls regardless of
// whether chained via callbacks or not.
//
// Reads should be submitted un-chained in order to provide more read buffer
// depth to the transport layer (i.e. queue larger/fewer spanning reads for
// higher throughput).
//
// Possible improvements in the future:
//   - Uses std::deque in a few places for queues. Should be replaced with
//     something with better memory ergonomics (i.e. deque can allocate 4KiB
//     minimum on some standard libraries).
//   - Additional read throughput can likely be had via read ahead, but this
//     needs to be done carefully (and is different between server and client
//     needs).
class SocketChannel final : public Channel {
 public:
  SocketChannel(IoLoop &io_loop, IoBufferPool &buffer_pool,
                IoBufferVec::Pool &iovec_pool, socket_t fd);
  ~SocketChannel();

  void Close() override;
  void Read(int32_t max_size, bool read_fully) override;
  using Channel::Read;
  void Write(IoBufferVec::Ptr io_vector, bool flush = true) override;
  using Channel::Flush;
  using Channel::Write;

  // The backing socket file descriptor.
  socket_t fd() { return fd_; }
  IoLoop &io_loop() { return io_loop_; }

  // Whether the channel is quiescent.
  bool is_quiescent() override { return quiescent_; }

 private:
  class ReadOutgoing {
   public:
    ReadOutgoing(int32_t size, bool fully) : size_(fully ? size : -size) {}
    int32_t size() { return size_ >= 0 ? size_ : -size_; }
    bool fully() { return size_ >= 0; }

   private:
    int32_t size_;
  };
  class ReadIncoming {
   public:
    ReadIncoming(IoBufferVec::Ptr iovec, int32_t available_bytes)
        : iovec_(std::move(iovec)), available_bytes_(available_bytes) {}

    IoBufferVec::Ptr &iovec() { return iovec_; }
    int32_t &available_bytes() { return available_bytes_; }

   private:
    IoBufferVec::Ptr iovec_;
    int32_t available_bytes_;
  };
  class WriteBuffer {
   public:
    WriteBuffer() = default;
    WriteBuffer(IoBufferVec::Ptr accum) : accum_(std::move(accum)) {}
    WriteBuffer(WriteBuffer &&other) = delete;

    void TakeFrom(WriteBuffer &other) {
      accum_ = std::move(other.accum_);
      call_count_ = other.call_count_;
      other.call_count_ = 0;
      flushed_ = other.flushed_;
      other.flushed_ = false;
    }

    IoBufferVec::Ptr &accum() { return accum_; }
    int &call_count() { return call_count_; }
    bool &flushed() { return flushed_; }

   private:
    IoBufferVec::Ptr accum_;
    int call_count_ = 0;
    bool flushed_ = false;
  };

  // Note that these persistent callbacks will always be std::bind()'d to
  // |this|, which is a fast-path. Therefore we don't go to the trouble of
  // pre-binding them on construction. Separate read and write completions
  // are maintained to give the branch predictor a better time.
  void HandleShutdownOnClose(IoSocketShutdownRequest::Ptr r);
  void HandleCloseComplete(IoCloseSocketRequest::Ptr r);
  void HandleReadComplete(IoSocketVecRequest::Ptr r);
  void HandleWriteComplete(IoSocketVecRequest::Ptr r);

  // Schedules any needed backing reads and notifies clients of any progress.
  void SchedRead();
  void FulfillReadOp(ReadOutgoing op);
  void SchedWrite();
  void SchedQuiescent();

  IoLoop &io_loop_;
  socket_t fd_;

  // Sum of all bytes available in |read_incoming_|.
  uint64_t read_incoming_bytes_ = 0;

  // Sum of all bytes for outstanding reads in |read_outgoing_|. Used to
  // manage the depth of the read request queue sent to the kernel.
  uint64_t read_outgoing_bytes_ = 0;

  // FIFO of data that has been received from the socket and is ready for
  // consumption by the client.
  std::deque<ReadIncoming> read_incoming_;

  // FIFO of outstanding read requests waiting to be serviced.
  std::deque<ReadOutgoing> read_outgoing_;

  // Write requests being accumulated from the caller.
  WriteBuffer write_incoming_;

  // Write request that has been sent to the transport and is awaiting
  // completion.
  WriteBuffer write_outgoing_;

  // If the channel is in read-error, remember that so all future reads can
  // be failed.
  iree_status_t read_error_ = iree_ok_status();

  // Number of underlying read requests in-flight.
  int32_t read_inflight_ = 0;

  // Whether we are in a critical section that cannot be re-entered via
  // a callback.
  bool callback_critical_section_ : 1 = false;

  // Whether read requests have been made that require scheduling.
  bool read_needs_sched_ : 1 = false;

  // Whether a close request has been initiated.
  bool close_initiated_ : 1 = false;

  // Whether a close request has been completed.
  bool close_complete_ : 1 = false;

  // Whether we have entered the quiescent state.
  bool quiescent_ : 1 = false;
};  // namespace remoting

}  // namespace remoting
}  // namespace iree

#endif  // IREE_REMOTING_SUPPORT_CHANNEL_H_
