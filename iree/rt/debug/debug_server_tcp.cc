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

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <exception>
#include <thread>  // NOLINT

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "flatbuffers/flatbuffers.h"
#include "iree/base/status.h"
#include "iree/rt/debug/debug_server.h"
#include "iree/rt/debug/debug_service.h"
#include "iree/rt/debug/debug_tcp_util.h"
#include "iree/schemas/debug_service_generated.h"

namespace iree {
namespace rt {
namespace debug {
namespace {

// Writes the given typed response message to the given fd by wrapping it in
// a size-prefixed rpc::Request union.
//
// Example:
//  ::flatbuffers::FlatBufferBuilder fbb;
//  rpc::SuspendInvocationResponseBuilder response(fbb);
//  RETURN_IF_ERROR(WriteResponse(fd_, response.Finish(), std::move(fbb)));
template <typename T>
Status WriteResponse(int fd, ::flatbuffers::Offset<T> message_offs,
                     ::flatbuffers::FlatBufferBuilder fbb) {
  rpc::ResponseBuilder response_builder(fbb);
  response_builder.add_message_type(rpc::ResponseUnionTraits<T>::enum_value);
  response_builder.add_message(message_offs.Union());
  auto response_offs = response_builder.Finish();
  rpc::ServicePacketBuilder packet_builder(fbb);
  packet_builder.add_response(response_offs);
  fbb.FinishSizePrefixed(packet_builder.Finish());
  return tcp::WriteBuffer(fd, fbb.Release());
}

class TcpDebugSession : public DebugSession {
 public:
  using ClosedCallback =
      std::function<void(TcpDebugSession* session, Status status)>;

  static StatusOr<std::unique_ptr<TcpDebugSession>> Accept(
      DebugService* debug_service, int client_fd,
      ClosedCallback closed_callback) {
    VLOG(2) << "Client " << client_fd << ": Setting up socket options...";
    // Disable Nagel's algorithm to ensure we have low latency.
    RETURN_IF_ERROR(tcp::ToggleSocketNagelsAlgorithm(client_fd, false));
    // Enable keepalive assuming the client is local and this high freq is ok.
    RETURN_IF_ERROR(tcp::ToggleSocketLocalKeepalive(client_fd, true));
    // Linger around for a bit to flush all data.
    RETURN_IF_ERROR(tcp::ToggleSocketLinger(client_fd, true));

    return absl::make_unique<TcpDebugSession>(debug_service, client_fd,
                                              std::move(closed_callback));
  }

  TcpDebugSession(DebugService* debug_service, int client_fd,
                  ClosedCallback closed_callback)
      : debug_service_(debug_service),
        client_fd_(client_fd),
        closed_callback_(std::move(closed_callback)) {
    CHECK_OK(debug_service_->RegisterDebugSession(this));
    session_thread_ = std::thread([this]() { SessionThread(); });
  }

  ~TcpDebugSession() override {
    CHECK_OK(debug_service_->UnregisterDebugSession(this));
    VLOG(2) << "Client " << client_fd_ << ": Shutting down session socket...";
    ::shutdown(client_fd_, SHUT_RD);
    if (session_thread_.joinable() &&
        session_thread_.get_id() != std::this_thread::get_id()) {
      VLOG(2) << "Client " << client_fd_ << ": Joining socket thread...";
      session_thread_.join();
      VLOG(2) << "Client " << client_fd_ << ": Joined socket thread!";
    } else {
      VLOG(2) << "Client " << client_fd_ << ": Detaching socket thread...";
      session_thread_.detach();
    }
    VLOG(2) << "Client " << client_fd_ << ": Closing session socket...";
    ::close(client_fd_);
    VLOG(2) << "Client " << client_fd_ << ": Closed session socket!";
    client_fd_ = -1;
  }

  Status OnServiceShutdown() {
    VLOG(2) << "Client " << client_fd_ << ": Post OnServiceShutdown()";
    ::flatbuffers::FlatBufferBuilder fbb;
    rpc::ServiceShutdownEventBuilder event(fbb);
    return PostEvent(event.Finish(), std::move(fbb));
  }

  Status OnContextRegistered(Context* context) override {
    VLOG(2) << "Client " << client_fd_ << ": Post OnContextRegistered("
            << context->id() << ")";
    ::flatbuffers::FlatBufferBuilder fbb;
    rpc::ContextRegisteredEventBuilder event(fbb);
    event.add_context_id(context->id());
    return PostEvent(event.Finish(), std::move(fbb));
  }
  Status OnContextUnregistered(Context* context) override {
    VLOG(2) << "Client " << client_fd_ << ": Post OnContextUnregistered("
            << context->id() << ")";
    ::flatbuffers::FlatBufferBuilder fbb;
    rpc::ContextUnregisteredEventBuilder event(fbb);
    event.add_context_id(context->id());
    return PostEvent(event.Finish(), std::move(fbb));
  }

  Status OnModuleLoaded(Context* context, Module* module) override {
    VLOG(2) << "Client " << client_fd_ << ": Post OnModuleLoaded("
            << context->id() << ", " << module->name() << ")";
    ::flatbuffers::FlatBufferBuilder fbb;
    auto module_name_offs =
        fbb.CreateString(module->name().data(), module->name().size());
    rpc::ModuleLoadedEventBuilder event(fbb);
    event.add_context_id(context->id());
    event.add_module_name(module_name_offs);
    return PostEvent(event.Finish(), std::move(fbb));
  }

  Status OnInvocationRegistered(Invocation* invocation) override {
    VLOG(2) << "Client " << client_fd_ << ": Post OnInvocationRegistered("
            << invocation->id() << ")";
    ::flatbuffers::FlatBufferBuilder fbb;
    rpc::InvocationRegisteredEventBuilder event(fbb);
    event.add_invocation_id(invocation->id());
    return PostEvent(event.Finish(), std::move(fbb));
  }
  Status OnInvocationUnregistered(Invocation* invocation) override {
    VLOG(2) << "Client " << client_fd_ << ": Post OnInvocationUnregistered("
            << invocation->id() << ")";
    ::flatbuffers::FlatBufferBuilder fbb;
    rpc::InvocationUnregisteredEventBuilder event(fbb);
    event.add_invocation_id(invocation->id());
    return PostEvent(event.Finish(), std::move(fbb));
  }

  Status OnBreakpointResolved(const rpc::BreakpointDefT& breakpoint,
                              Context* context) override {
    VLOG(2) << "Client " << client_fd_ << ": Post OnBreakpointResolved("
            << breakpoint.breakpoint_id << ", " << context->id() << ", "
            << breakpoint.function_ordinal << ")";
    rpc::BreakpointResolvedEventT event;
    event.breakpoint = absl::make_unique<rpc::BreakpointDefT>();
    *event.breakpoint = breakpoint;
    event.context_id = context->id();
    ::flatbuffers::FlatBufferBuilder fbb;
    return PostEvent(rpc::BreakpointResolvedEvent::Pack(fbb, &event),
                     std::move(fbb));
  }

  Status OnBreakpointHit(int breakpoint_id,
                         const Invocation& invocation) override {
    VLOG(2) << "Client " << client_fd_ << ": Post OnBreakpointHit("
            << breakpoint_id << ", " << invocation.id() << ")";
    ::flatbuffers::FlatBufferBuilder fbb;
    ASSIGN_OR_RETURN(auto invocation_offs,
                     debug_service_->SerializeInvocation(invocation, &fbb));
    rpc::BreakpointHitEventBuilder event(fbb);
    event.add_breakpoint_id(breakpoint_id);
    event.add_invocation(invocation_offs);
    return PostEvent(event.Finish(), std::move(fbb));
  }

 private:
  void SessionThread() {
    VLOG(2) << "Client " << client_fd_ << ": Thread entry";
    Status session_status = OkStatus();
    while (session_status.ok()) {
      auto buffer_or = tcp::ReadBuffer<rpc::Request>(client_fd_);
      if (!buffer_or.ok()) {
        if (IsCancelled(buffer_or.status())) {
          // Graceful shutdown.
          VLOG(2) << "Client " << client_fd_ << ": Graceful shutdown requested";
          break;
        }
        // Error reading.
        session_status = std::move(buffer_or).status();
        LOG(ERROR) << "Client " << client_fd_
                   << ": Error reading request buffer: " << session_status;
        break;
      }
      auto request_buffer = std::move(buffer_or).ValueOrDie();
      session_status = DispatchRequest(request_buffer.GetRoot());
      if (!session_status.ok()) {
        LOG(ERROR) << "Client " << client_fd_
                   << ": Error dispatching request: " << session_status;
        break;
      }
    }
    VLOG(2) << "Client " << client_fd_ << ": Thread exit";
    AbortSession(session_status);
  }

  void AbortSession(Status status) {
    if (status.ok()) {
      VLOG(2) << "Debug client disconnected";
    } else {
      LOG(ERROR) << "Debug session aborted; " << status;
      ::flatbuffers::FlatBufferBuilder fbb;
      auto message_offs =
          fbb.CreateString(status.message().data(), status.message().size());
      rpc::StatusBuilder status_builder(fbb);
      status_builder.add_code(static_cast<int>(status.code()));
      status_builder.add_message(message_offs);
      auto status_offs = status_builder.Finish();
      rpc::ResponseBuilder response(fbb);
      response.add_status(status_offs);
      fbb.FinishSizePrefixed(response.Finish());
      tcp::WriteBuffer(client_fd_, fbb.Release()).IgnoreError();
    }
    closed_callback_(this, std::move(status));
  }

  template <typename T>
  Status PostEvent(::flatbuffers::Offset<T> event_offs,
                   ::flatbuffers::FlatBufferBuilder fbb) {
    rpc::ServicePacketBuilder packet_builder(fbb);
    packet_builder.add_event_type(rpc::EventUnionTraits<T>::enum_value);
    packet_builder.add_event(event_offs.Union());
    fbb.FinishSizePrefixed(packet_builder.Finish());
    return tcp::WriteBuffer(client_fd_, fbb.Release());
  }

  Status DispatchRequest(const rpc::Request& request) {
    ::flatbuffers::FlatBufferBuilder fbb;
    switch (request.message_type()) {
#define DISPATCH_REQUEST(method_name)                                          \
  case rpc::RequestUnion::method_name##Request: {                              \
    VLOG(2) << "Client " << client_fd_                                         \
            << ": DispatchRequest(" #method_name ")...";                       \
    ASSIGN_OR_RETURN(auto response_offs,                                       \
                     debug_service_->method_name(                              \
                         *request.message_as_##method_name##Request(), &fbb)); \
    return WriteResponse(client_fd_, response_offs, std::move(fbb));           \
  }
      DISPATCH_REQUEST(MakeReady);
      DISPATCH_REQUEST(GetStatus);
      DISPATCH_REQUEST(ListContexts);
      DISPATCH_REQUEST(GetModule);
      DISPATCH_REQUEST(GetFunction);
      DISPATCH_REQUEST(ListInvocations);
      DISPATCH_REQUEST(SuspendInvocations);
      DISPATCH_REQUEST(ResumeInvocations);
      DISPATCH_REQUEST(StepInvocation);
      DISPATCH_REQUEST(GetInvocationLocal);
      DISPATCH_REQUEST(SetInvocationLocal);
      DISPATCH_REQUEST(ListBreakpoints);
      DISPATCH_REQUEST(AddBreakpoint);
      DISPATCH_REQUEST(RemoveBreakpoint);
      DISPATCH_REQUEST(StartProfiling);
      DISPATCH_REQUEST(StopProfiling);
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unimplemented debug service request: "
               << static_cast<int>(request.message_type());
    }
  }

  DebugService* debug_service_;
  int client_fd_;
  ClosedCallback closed_callback_;
  std::thread session_thread_;
};

class TcpDebugServer final : public DebugServer {
 public:
  static StatusOr<std::unique_ptr<TcpDebugServer>> Listen(int port) {
    // We support both IPv4 and IPv6 by using the IN6ADDR_ANY. This requires
    // that we setup the socket as INET6 and enable reuse (so the same port can
    // be bound for both IPv4 and IPv6).
    int listen_fd = ::socket(AF_INET6, SOCK_STREAM, 0);
    RETURN_IF_ERROR(tcp::ToggleSocketAddressReuse(listen_fd, true));

    struct sockaddr_in6 socket_addr = {0};
    socket_addr.sin6_family = AF_INET6;
    socket_addr.sin6_port = htons(port);
    socket_addr.sin6_addr = in6addr_any;
    if (::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&socket_addr),
               sizeof(socket_addr)) < 0) {
      return AlreadyExistsErrorBuilder(IREE_LOC)
             << "Unable to bind socket to port " << port << ": (" << errno
             << ") " << ::strerror(errno);
    }
    if (::listen(listen_fd, 1)) {
      ::close(listen_fd);
      return AlreadyExistsErrorBuilder(IREE_LOC)
             << "Unable to listen on port " << port << ": (" << errno << ") "
             << ::strerror(errno);
    }
    return absl::make_unique<TcpDebugServer>(listen_fd);
  }

  TcpDebugServer(int listen_fd) : listen_fd_(listen_fd) {
    server_thread_ = std::thread([this]() { ListenThread(); });
  }

  ~TcpDebugServer() ABSL_LOCKS_EXCLUDED(mutex_) override {
    absl::ReleasableMutexLock lock(&mutex_);
    LOG(INFO) << "Shutting down debug server...";

    // Notify all sessions.
    for (auto& session : sessions_) {
      session->OnServiceShutdown().IgnoreError();
    }

    // Shut down listen socket first so that we can't accept new connections.
    VLOG(2) << "Shutting down listen socket...";
    ::shutdown(listen_fd_, SHUT_RDWR);
    if (server_thread_.joinable()) {
      VLOG(2) << "Joining listen thread...";
      server_thread_.join();
      VLOG(2) << "Joined listen thread!";
    }
    VLOG(2) << "Closing listen socket...";
    ::close(listen_fd_);
    listen_fd_ = -1;
    VLOG(2) << "Closed listen socket!";

    // Kill all active sessions. Note that we must do this outside of our lock.
    std::vector<std::unique_ptr<TcpDebugSession>> sessions =
        std::move(sessions_);
    std::vector<std::function<void()>> at_exit_callbacks =
        std::move(at_exit_callbacks_);
    lock.Release();
    VLOG(2) << "Clearing live sessions...";
    sessions.clear();
    VLOG(2) << "Calling AtExit callbacks...";
    for (auto& callback : at_exit_callbacks) {
      callback();
    }
    LOG(INFO) << "Debug server shutdown!";
  }

  DebugService* debug_service() { return &debug_service_; }

  Status AcceptNewSession(int client_fd) {
    LOG(INFO) << "Accepting new client session as " << client_fd;
    ASSIGN_OR_RETURN(auto session,
                     TcpDebugSession::Accept(
                         &debug_service_, client_fd,
                         [this](TcpDebugSession* session, Status status) {
                           absl::MutexLock lock(&mutex_);
                           for (auto it = sessions_.begin();
                                it != sessions_.end(); ++it) {
                             if (it->get() == session) {
                               sessions_.erase(it);
                               break;
                             }
                           }
                           return OkStatus();
                         }));

    absl::MutexLock lock(&mutex_);
    sessions_.push_back(std::move(session));
    return OkStatus();
  }

  void AtExit(std::function<void()> callback) override {
    absl::MutexLock lock(&mutex_);
    at_exit_callbacks_.push_back(std::move(callback));
  }

  Status WaitUntilSessionReady() override {
    return debug_service_.WaitUntilAllSessionsReady();
  }

 protected:
  Status RegisterContext(Context* context) override {
    return debug_service_.RegisterContext(context);
  }
  Status UnregisterContext(Context* context) override {
    return debug_service_.UnregisterContext(context);
  }
  Status RegisterContextModule(Context* context, Module* module) override {
    return debug_service_.RegisterContextModule(context, module);
  }
  Status RegisterInvocation(Invocation* invocation) override {
    return debug_service_.RegisterInvocation(invocation);
  }
  Status UnregisterInvocation(Invocation* invocation) override {
    return debug_service_.UnregisterInvocation(invocation);
  }

 private:
  void ListenThread() {
    VLOG(2) << "Listen thread entry";
    while (true) {
      struct sockaddr_in accept_socket_addr;
      socklen_t accept_socket_addr_length = sizeof(accept_socket_addr);
      int accepted_fd = ::accept(
          listen_fd_, reinterpret_cast<struct sockaddr*>(&accept_socket_addr),
          &accept_socket_addr_length);
      if (accepted_fd < 0) {
        if (errno == EINVAL) {
          // Shutting down gracefully.
          break;
        }
        // We may be able to recover from some of these cases, but... shrug.
        LOG(FATAL) << "Failed to accept client socket: (" << errno << ") "
                   << ::strerror(errno);
        break;
      }
      auto accept_status = AcceptNewSession(accepted_fd);
      if (!accept_status.ok()) {
        LOG(ERROR) << "Failed to accept incoming debug client: "
                   << accept_status;
      }
    }
    VLOG(2) << "Listen thread exit";
  }

  int listen_fd_;
  std::thread server_thread_;

  absl::Mutex mutex_;
  std::vector<std::unique_ptr<TcpDebugSession>> sessions_
      ABSL_GUARDED_BY(mutex_);
  std::vector<std::function<void()>> at_exit_callbacks_ ABSL_GUARDED_BY(mutex_);

  DebugService debug_service_;
};

}  // namespace

// static
StatusOr<std::unique_ptr<DebugServer>> DebugServer::Create(int listen_port) {
  ASSIGN_OR_RETURN(auto debug_server, TcpDebugServer::Listen(listen_port));
  LOG(INFO) << "Debug server listening on localhost:" << listen_port;
  return debug_server;
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
