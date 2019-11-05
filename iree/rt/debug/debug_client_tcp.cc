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

#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "flatbuffers/base.h"
#include "flatbuffers/flatbuffers.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/rt/debug/debug_client.h"
#include "iree/rt/debug/debug_tcp_util.h"
#include "iree/rt/module.h"
#include "iree/schemas/debug_service_generated.h"
#include "iree/schemas/module_def_generated.h"

namespace iree {
namespace rt {
namespace debug {
namespace {

using ::flatbuffers::FlatBufferBuilder;

// Parses a host:port address, with support for the RFC 3986 IPv6 [host]:port
// format. Returns a pair of (hostname, port), with port being 0 if none was
// specified.
//
// Parses:
//   foo (port 0)     / foo:123
//   1.2.3.4 (port 0) / 1.2.3.4:123
//   [foo] (port 0)   / [foo]:123
//   [::1] (port 0)   / [::1]:123
StatusOr<std::pair<std::string, int>> ParseAddress(absl::string_view address) {
  address = absl::StripAsciiWhitespace(address);
  absl::string_view hostname;
  absl::string_view port_str;
  size_t bracket_loc = address.find_last_of(']');
  if (bracket_loc != std::string::npos) {
    // Has at least a ]. Let's assume it's mostly right.
    if (address.find('[') != 0) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Mismatched brackets in address: " << address;
    }
    hostname = address.substr(1, bracket_loc - 1);
    port_str = address.substr(bracket_loc + 1);
    if (port_str.find(':') == 0) {
      port_str.remove_prefix(1);
    }
  } else {
    size_t colon_loc = address.find_last_of(':');
    if (colon_loc != std::string::npos) {
      hostname = address.substr(0, colon_loc);
      port_str = address.substr(colon_loc + 1);
    } else {
      hostname = address;
      port_str = "";
    }
  }
  int port = 0;
  if (!port_str.empty() && !absl::SimpleAtoi(port_str, &port)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Unable to parse port '" << port_str << "' from " << address;
  }
  return std::make_pair(std::string(hostname), port);
}

class TcpDebugClient final : public DebugClient {
 public:
  class TcpRemoteBreakpoint : public RemoteBreakpoint {
   public:
    TcpRemoteBreakpoint(int id, Type type, TcpDebugClient* client)
        : RemoteBreakpoint(id, type) {}

    const std::string& module_name() const override { return def_.module_name; }
    const std::string& function_name() const override {
      return def_.function_name;
    }
    int function_ordinal() const override { return def_.function_ordinal; }
    int bytecode_offset() const override { return def_.bytecode_offset; }

    Status MergeFrom(const rpc::BreakpointDef& breakpoint_def) {
      breakpoint_def.UnPackTo(&def_);
      return OkStatus();
    }

   private:
    rpc::BreakpointDefT def_;
  };

  class TcpRemoteFunction final : public RemoteFunction {
   public:
    TcpRemoteFunction(RemoteModule* module, int function_ordinal,
                      const FunctionDef* function_def, TcpDebugClient* client)
        : RemoteFunction(module, function_ordinal),
          def_(function_def),
          client_(client) {
      name_ = def_->name() ? std::string(WrapString(def_->name())) : "";
    }

    const std::string& name() const override { return name_; }

    const FunctionDef& def() override { return *def_; }

    bool is_loaded() const override {
      return contents_.flatbuffers_buffer.size() > 0;
    }

    bool CheckLoadedOrRequest() override {
      if (!is_loaded()) {
        DemandContents();
      }
      return is_loaded();
    }

    void WhenLoaded(LoadCallback callback) override {
      if (is_loaded()) {
        callback(this);
        return;
      }
      load_callbacks_.push_back(std::move(callback));
    }

    const BytecodeDef* bytecode() override {
      CHECK(is_loaded());
      return contents_.bytecode_def;
    }

   private:
    void DemandContents() {
      if (!has_requested_contents_) {
        VLOG(2) << "Client " << client_->fd() << ": GetFunction("
                << module()->context_id() << ", " << module()->name() << ", "
                << ordinal() << ")";
        FlatBufferBuilder fbb;
        rpc::GetFunctionRequestT request;
        request.session_id = client_->session_id();
        request.context_id = module()->context_id();
        request.module_name = module()->name();
        request.function_ordinal = ordinal();
        auto status =
            client_->IssueRequest<rpc::GetFunctionRequest,
                                  rpc::ResponseUnion::GetFunctionResponse>(
                rpc::GetFunctionRequest::Pack(fbb, &request), std::move(fbb),
                [this](Status status,
                       const rpc::Response& response_union) -> Status {
                  if (!status.ok()) return status;
                  const auto& response =
                      *response_union.message_as_GetFunctionResponse();
                  VLOG(2) << "Client " << client_->fd() << ": GetFunction("
                          << module()->context_id() << ", " << module()->name()
                          << ", " << ordinal() << ") = ...";
                  RETURN_IF_ERROR(MergeFrom(response));
                  for (auto& callback : load_callbacks_) {
                    callback(this);
                  }
                  load_callbacks_.clear();
                  return OkStatus();
                });
        if (!status.ok()) {
          LOG(ERROR) << "Failed to request module: " << status;
          return;
        }
        has_requested_contents_ = true;
      }
    }

    Status MergeFrom(const rpc::GetFunctionResponse& response) {
      // Clone and retain the contents.
      // TODO(benvanik): find a way to steal to avoid the reserialization.
      BytecodeDefT bytecode_def_storage;
      response.bytecode()->UnPackTo(&bytecode_def_storage);
      ::flatbuffers::FlatBufferBuilder fbb;
      fbb.Finish(response.bytecode()->Pack(fbb, &bytecode_def_storage));
      contents_.flatbuffers_buffer = fbb.Release();
      contents_.bytecode_def = ::flatbuffers::GetRoot<BytecodeDef>(
          contents_.flatbuffers_buffer.data());
      return OkStatus();
    }

    const FunctionDef* def_;
    TcpDebugClient* client_;
    std::string name_;
    bool has_requested_contents_ = false;
    std::vector<LoadCallback> load_callbacks_;
    struct {
      ::flatbuffers::DetachedBuffer flatbuffers_buffer;
      const BytecodeDef* bytecode_def = nullptr;
    } contents_;
  };

  class TcpRemoteModule final : public RemoteModule {
   public:
    TcpRemoteModule(int context_id, std::string module_name,
                    TcpDebugClient* client)
        : RemoteModule(context_id, std::move(module_name)), client_(client) {}

    const ModuleDef& def() override {
      CHECK(is_loaded());
      return *module_file_->root();
    }

    bool is_loaded() const override { return module_file_ != nullptr; }

    bool CheckLoadedOrRequest() override {
      if (!is_loaded()) {
        DemandModuleDef();
      }
      return is_loaded();
    }

    void WhenLoaded(LoadCallback callback) override {
      if (is_loaded()) {
        callback(this);
        return;
      }
      load_callbacks_.push_back(std::move(callback));
    }

    absl::Span<RemoteFunction*> functions() override {
      auto* module_def = DemandModuleDef();
      if (!module_def) return {};
      return {reinterpret_cast<RemoteFunction**>(functions_.data()),
              functions_.size()};
    }

   private:
    const ModuleDef* DemandModuleDef() {
      if (module_file_) {
        return module_file_->root();
      }
      if (!has_requested_module_def_) {
        VLOG(2) << "Client " << client_->fd() << ": GetModule(" << context_id()
                << ", " << name() << ")";
        FlatBufferBuilder fbb;
        rpc::GetModuleRequestT request;
        request.session_id = client_->session_id();
        request.context_id = context_id();
        request.module_name = name();
        auto status =
            client_->IssueRequest<rpc::GetModuleRequest,
                                  rpc::ResponseUnion::GetModuleResponse>(
                rpc::GetModuleRequest::Pack(fbb, &request), std::move(fbb),
                [this](Status status,
                       const rpc::Response& response_union) -> Status {
                  if (!status.ok()) return status;
                  const auto& response =
                      *response_union.message_as_GetModuleResponse();
                  VLOG(2) << "Client " << client_->fd() << ": GetModule("
                          << context_id() << ", " << name() << ") = ...";
                  RETURN_IF_ERROR(MergeFrom(response));
                  for (auto& callback : load_callbacks_) {
                    callback(this);
                  }
                  load_callbacks_.clear();
                  return OkStatus();
                });
        if (!status.ok()) {
          LOG(ERROR) << "Failed to request module: " << status;
          return nullptr;
        }
        has_requested_module_def_ = true;
      }
      return nullptr;
    }

    Status MergeFrom(const rpc::GetModuleResponse& response) {
      // Clone and retain the module.
      // TODO(benvanik): find a way to steal to avoid the reserialization.
      ModuleDefT module_def_storage;
      response.module_()->UnPackTo(&module_def_storage);
      FlatBufferBuilder fbb;
      auto module_offs = response.module_()->Pack(fbb, &module_def_storage);
      FinishModuleDefBuffer(fbb, module_offs);
      ASSIGN_OR_RETURN(auto module_file,
                       ModuleFile::CreateWithBackingBuffer(fbb.Release()));

      const auto& module_def = module_file->root();
      const auto& function_table = *module_def->function_table();
      functions_.reserve(function_table.functions()->size());
      for (int i = 0; i < function_table.functions()->size(); ++i) {
        const auto* function_def = function_table.functions()->Get(i);
        functions_.push_back(absl::make_unique<TcpRemoteFunction>(
            this, i, function_def, client_));
      }

      module_file_ = std::move(module_file);
      return OkStatus();
    }

    TcpDebugClient* client_;
    bool has_requested_module_def_ = false;
    std::vector<LoadCallback> load_callbacks_;
    ref_ptr<ModuleFile> module_file_;
    std::vector<std::unique_ptr<RemoteFunction>> functions_;
  };

  class TcpRemoteContext final : public RemoteContext {
   public:
    TcpRemoteContext(int context_id, TcpDebugClient* client)
        : RemoteContext(context_id), client_(client) {}

    absl::Span<RemoteModule* const> modules() const override {
      return absl::MakeConstSpan(modules_);
    }

    Status AddModule(std::unique_ptr<TcpRemoteModule> module) {
      modules_.push_back(module.get());
      module_map_.insert({module->name(), std::move(module)});
      return OkStatus();
    }

    Status MergeFrom(const rpc::ContextDef& context_def) { return OkStatus(); }

   private:
    TcpDebugClient* client_;
    std::vector<RemoteModule*> modules_;
    absl::flat_hash_map<std::string, std::unique_ptr<TcpRemoteModule>>
        module_map_;
  };

  class TcpRemoteInvocation final : public RemoteInvocation {
   public:
    TcpRemoteInvocation(int invocation_id, TcpDebugClient* client)
        : RemoteInvocation(invocation_id), client_(client) {}

    const rpc::InvocationDefT& def() const override { return def_; }

    Status MergeFrom(const rpc::InvocationDef& invocation_def) {
      invocation_def.UnPackTo(&def_);
      return OkStatus();
    }

   private:
    TcpDebugClient* client_;
    rpc::InvocationDefT def_;
  };

  static StatusOr<std::unique_ptr<TcpDebugClient>> Create(int fd,
                                                          Listener* listener) {
    VLOG(2) << "Client " << fd << ": Setting up socket options...";
    // Disable Nagel's algorithm to ensure we have low latency.
    RETURN_IF_ERROR(tcp::ToggleSocketNagelsAlgorithm(fd, false));
    // Enable keepalive assuming the client is local and this high freq is ok.
    RETURN_IF_ERROR(tcp::ToggleSocketLocalKeepalive(fd, true));
    // Linger around for a bit to flush all data.
    RETURN_IF_ERROR(tcp::ToggleSocketLinger(fd, true));
    // Disable blocking as we are poll based.
    RETURN_IF_ERROR(tcp::ToggleSocketBlocking(fd, false));

    auto client = absl::make_unique<TcpDebugClient>(fd, listener);
    RETURN_IF_ERROR(client->Refresh());
    return client;
  }

  TcpDebugClient(int fd, Listener* listener) : fd_(fd), listener_(listener) {}

  ~TcpDebugClient() override {
    VLOG(2) << "Client " << fd_ << ": Shutting down session socket...";
    ::shutdown(fd_, SHUT_WR);
    VLOG(2) << "Client " << fd_ << ": Closing session socket...";
    ::close(fd_);
    VLOG(2) << "Client " << fd_ << ": Closed session socket!";
    fd_ = -1;
  }

  int fd() const { return fd_; }
  int session_id() const { return session_id_; }

  absl::Span<RemoteContext* const> contexts() const override {
    return absl::MakeConstSpan(contexts_);
  }

  absl::Span<RemoteInvocation* const> invocations() const override {
    return absl::MakeConstSpan(invocations_);
  }

  absl::Span<RemoteBreakpoint* const> breakpoints() const override {
    return absl::MakeConstSpan(breakpoints_);
  }

  // Writes the given typed request message to the given fd by wrapping it in
  // a size-prefixed rpc::Request union.
  //
  // Example:
  //  FlatBufferBuilder fbb;
  //  rpc::SuspendInvocationRequestBuilder request(fbb);
  //  RETURN_IF_ERROR(WriteRequest(fd_, request.Finish(), std::move(fbb)));
  template <typename T>
  Status WriteRequest(int fd, ::flatbuffers::Offset<T> request_offs,
                      FlatBufferBuilder fbb) {
    rpc::RequestBuilder request_builder(fbb);
    request_builder.add_message_type(rpc::RequestUnionTraits<T>::enum_value);
    request_builder.add_message(request_offs.Union());
    fbb.FinishSizePrefixed(request_builder.Finish());
    auto write_status = tcp::WriteBuffer(fd, fbb.Release());
    if (shutdown_pending_ && IsUnavailable(write_status)) {
      return OkStatus();
    }
    return write_status;
  }

  Status ResolveFunction(
      std::string module_name, std::string function_name,
      std::function<void(StatusOr<int> function_ordinal)> callback) override {
    VLOG(2) << "Client " << fd_ << ": ResolveFunction(" << module_name << ", "
            << function_name << ")";
    FlatBufferBuilder fbb;
    rpc::ResolveFunctionRequestT request;
    request.session_id = session_id_;
    request.module_name = module_name;
    request.function_name = function_name;
    return IssueRequest<rpc::ResolveFunctionRequest,
                        rpc::ResponseUnion::ResolveFunctionResponse>(
        rpc::ResolveFunctionRequest::Pack(fbb, &request), std::move(fbb),
        [this, module_name, function_name, callback](
            Status status, const rpc::Response& response_union) -> Status {
          if (status.ok()) {
            const auto& response =
                *response_union.message_as_ResolveFunctionResponse();
            VLOG(2) << "Client " << fd_ << ": ResolveFunction(" << module_name
                    << ", " << function_name
                    << ") = " << response.function_ordinal();
            callback(response.function_ordinal());
          } else {
            callback(std::move(status));
          }
          return OkStatus();
        });
  }

  Status GetFunction(std::string module_name, int function_ordinal,
                     std::function<void(StatusOr<RemoteFunction*> function)>
                         callback) override {
    // See if we have the module already. If not, we'll fetch it first.
    RemoteModule* target_module = nullptr;
    for (auto* context : contexts_) {
      for (auto* module : context->modules()) {
        if (module->name() == module_name) {
          target_module = module;
          break;
        }
      }
      if (target_module) break;
    }
    if (!target_module) {
      // TODO(benvanik): fetch contexts first.
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Demand fetch contexts not yet implemented";
    }
    // Found at least one module with the right name.
    if (target_module->is_loaded()) {
      callback(target_module->functions()[function_ordinal]);
      return OkStatus();
    } else {
      // Wait until the module completes loading.
      target_module->WhenLoaded(
          [callback, function_ordinal](StatusOr<RemoteModule*> module_or) {
            if (!module_or.ok()) {
              callback(module_or.status());
              return;
            }
            callback(module_or.ValueOrDie()->functions()[function_ordinal]);
          });
      return OkStatus();
    }
  }

  Status AddFunctionBreakpoint(
      std::string module_name, std::string function_name, int offset,
      std::function<void(const RemoteBreakpoint& breakpoint)> callback)
      override {
    VLOG(2) << "Client " << fd_ << ": AddFunctionBreakpoint(" << module_name
            << ", " << function_name << ", " << offset << ")";
    FlatBufferBuilder fbb;

    auto breakpoint = absl::make_unique<rpc::BreakpointDefT>();
    breakpoint->module_name = module_name;
    breakpoint->function_name = function_name;
    breakpoint->function_ordinal = -1;
    breakpoint->bytecode_offset = offset;
    rpc::AddBreakpointRequestT request;
    request.session_id = session_id_;
    request.breakpoint = std::move(breakpoint);
    return IssueRequest<rpc::AddBreakpointRequest,
                        rpc::ResponseUnion::AddBreakpointResponse>(
        rpc::AddBreakpointRequest::Pack(fbb, &request), std::move(fbb),
        [this, callback](Status status,
                         const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          const auto& response =
              *response_union.message_as_AddBreakpointResponse();
          RETURN_IF_ERROR(RegisterBreakpoint(*response.breakpoint()));
          if (callback) {
            ASSIGN_OR_RETURN(
                auto breakpoint,
                GetBreakpoint(response.breakpoint()->breakpoint_id()));
            callback(*breakpoint);
          }
          return OkStatus();
        });
  }

  Status RemoveBreakpoint(const RemoteBreakpoint& breakpoint) override {
    VLOG(2) << "Client " << fd_ << ": RemoveBreakpoint(" << breakpoint.id()
            << ")";
    int breakpoint_id = breakpoint.id();
    ASSIGN_OR_RETURN(auto* breakpoint_ptr, GetBreakpoint(breakpoint_id));
    RETURN_IF_ERROR(UnregisterBreakpoint(breakpoint_ptr));
    FlatBufferBuilder fbb;
    rpc::RemoveBreakpointRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    request.add_breakpoint_id(breakpoint_id);
    return IssueRequest<rpc::RemoveBreakpointRequest,
                        rpc::ResponseUnion::RemoveBreakpointResponse>(
        request.Finish(), std::move(fbb),
        [](Status status, const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          // No non-error status.
          return OkStatus();
        });
  }

  Status MakeReady() override {
    FlatBufferBuilder fbb;
    rpc::MakeReadyRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    return IssueRequest<rpc::MakeReadyRequest,
                        rpc::ResponseUnion::MakeReadyResponse>(
        request.Finish(), std::move(fbb),
        [](Status status, const rpc::Response& response_union) {
          return status;
        });
  }

  Status SuspendAllInvocations() override {
    VLOG(2) << "Client " << fd_ << ": SuspendAllInvocations()";
    FlatBufferBuilder fbb;
    rpc::SuspendInvocationsRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    return IssueRequest<rpc::SuspendInvocationsRequest,
                        rpc::ResponseUnion::SuspendInvocationsResponse>(
        request.Finish(), std::move(fbb),
        [this](Status status, const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          return RefreshInvocations();
        });
  }

  Status ResumeAllInvocations() override {
    VLOG(2) << "Client " << fd_ << ": ResumeAllInvocations()";
    FlatBufferBuilder fbb;
    rpc::ResumeInvocationsRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    return IssueRequest<rpc::ResumeInvocationsRequest,
                        rpc::ResponseUnion::ResumeInvocationsResponse>(
        request.Finish(), std::move(fbb),
        [this](Status status, const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          return RefreshInvocations();
        });
  }

  Status SuspendInvocations(
      absl::Span<RemoteInvocation*> invocations) override {
    VLOG(2) << "Client " << fd_ << ": SuspendInvocations(...)";
    FlatBufferBuilder fbb;
    auto invocation_ids_offs = fbb.CreateVector<int32_t>(
        invocations.size(),
        [&invocations](size_t i) { return invocations[i]->id(); });
    rpc::SuspendInvocationsRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    request.add_invocation_ids(invocation_ids_offs);
    return IssueRequest<rpc::SuspendInvocationsRequest,
                        rpc::ResponseUnion::SuspendInvocationsResponse>(
        request.Finish(), std::move(fbb),
        [this](Status status, const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          return RefreshInvocations();
        });
  }

  Status ResumeInvocations(absl::Span<RemoteInvocation*> invocations) override {
    VLOG(2) << "Client " << fd_ << ": ResumeInvocations(...)";
    FlatBufferBuilder fbb;
    auto invocation_ids_offs = fbb.CreateVector<int32_t>(
        invocations.size(),
        [&invocations](size_t i) { return invocations[i]->id(); });
    rpc::ResumeInvocationsRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    request.add_invocation_ids(invocation_ids_offs);
    return IssueRequest<rpc::ResumeInvocationsRequest,
                        rpc::ResponseUnion::ResumeInvocationsResponse>(
        request.Finish(), std::move(fbb),
        [this](Status status, const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          return RefreshInvocations();
        });
  }

  Status StepInvocation(const RemoteInvocation& invocation,
                        std::function<void()> callback) override {
    int step_id = next_step_id_++;
    VLOG(2) << "Client " << fd_ << ": StepInvocation(" << invocation.id()
            << ") as step_id=" << step_id;
    rpc::StepInvocationRequestT step_request;
    step_request.step_id = step_id;
    step_request.invocation_id = invocation.id();
    step_request.step_mode = rpc::StepMode::STEP_ONCE;
    return StepInvocation(&step_request, std::move(callback));
  }

  Status StepInvocationToOffset(const RemoteInvocation& invocation,
                                int bytecode_offset,
                                std::function<void()> callback) override {
    int step_id = next_step_id_++;
    VLOG(2) << "Client " << fd_ << ": StepInvocationToOffset("
            << invocation.id() << ", " << bytecode_offset
            << ") as step_id=" << step_id;
    rpc::StepInvocationRequestT step_request;
    step_request.step_id = step_id;
    step_request.invocation_id = invocation.id();
    step_request.step_mode = rpc::StepMode::STEP_TO_OFFSET;
    step_request.bytecode_offset = bytecode_offset;
    return StepInvocation(&step_request, std::move(callback));
  }

  Status Poll() override {
    while (true) {
      // If nothing awaiting then return immediately.
      if (!tcp::CanReadBuffer(fd_)) {
        break;
      }

      // Read the pending response and dispatch.
      auto packet_buffer_or = tcp::ReadBuffer<rpc::ServicePacket>(fd_);
      if (!packet_buffer_or.ok()) {
        if (shutdown_pending_ && IsUnavailable(packet_buffer_or.status())) {
          // This is a graceful close.
          return CancelledErrorBuilder(IREE_LOC) << "Service shutdown";
        }
        return packet_buffer_or.status();
      }
      const auto& packet = packet_buffer_or.ValueOrDie().GetRoot();
      if (packet.response()) {
        RETURN_IF_ERROR(DispatchResponse(*packet.response()));
      }
      if (packet.event()) {
        RETURN_IF_ERROR(DispatchEvent(packet));
      }
    }
    return OkStatus();
  }

  using ResponseCallback =
      std::function<Status(Status status, const rpc::Response& response)>;

  template <typename T, rpc::ResponseUnion response_type>
  Status IssueRequest(::flatbuffers::Offset<T> request_offs,
                      FlatBufferBuilder fbb, ResponseCallback callback) {
    RETURN_IF_ERROR(WriteRequest(fd_, request_offs, std::move(fbb)));
    pending_responses_.push({response_type, std::move(callback)});
    return OkStatus();
  }

 private:
  Status Refresh() {
    RETURN_IF_ERROR(RefreshContexts());
    RETURN_IF_ERROR(RefreshInvocations());
    RETURN_IF_ERROR(RefreshBreakpoints());
    return OkStatus();
  }

  Status RefreshContexts() {
    VLOG(2) << "Request contexts refresh...";
    FlatBufferBuilder fbb;
    rpc::ListContextsRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    return IssueRequest<rpc::ListContextsRequest,
                        rpc::ResponseUnion::ListContextsResponse>(
        request.Finish(), std::move(fbb),
        [this](Status status, const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          VLOG(2) << "Refreshing contexts...";
          const auto& response =
              *response_union.message_as_ListContextsResponse();
          for (auto* context_def : *response.contexts()) {
            auto context_or = GetContext(context_def->context_id());
            if (!context_or.ok()) {
              // Not found; add new.
              RETURN_IF_ERROR(RegisterContext(context_def->context_id()));
              context_or = GetContext(context_def->context_id());
            }
            RETURN_IF_ERROR(context_or.status());
            RETURN_IF_ERROR(context_or.ValueOrDie()->MergeFrom(*context_def));
          }
          VLOG(2) << "Refreshed contexts!";
          return OkStatus();
        });
  }

  Status RefreshInvocations() {
    VLOG(2) << "Request invocation states refresh...";
    FlatBufferBuilder fbb;
    rpc::ListInvocationsRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    return IssueRequest<rpc::ListInvocationsRequest,
                        rpc::ResponseUnion::ListInvocationsResponse>(
        request.Finish(), std::move(fbb),
        [this](Status status, const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          VLOG(2) << "Refreshing invocation states...";
          const auto& response =
              *response_union.message_as_ListInvocationsResponse();
          for (auto* invocation_def : *response.invocations()) {
            auto invocation_or = GetInvocation(invocation_def->invocation_id());
            if (!invocation_or.ok()) {
              // Not found; add new.
              RETURN_IF_ERROR(
                  RegisterInvocation(invocation_def->invocation_id()));
              invocation_or = GetInvocation(invocation_def->invocation_id());
            }
            RETURN_IF_ERROR(invocation_or.status());
            RETURN_IF_ERROR(
                invocation_or.ValueOrDie()->MergeFrom(*invocation_def));
          }
          // TODO(benvanik): handle removals/deaths.
          VLOG(2) << "Refreshed invocation states!";
          return OkStatus();
        });
  }

  Status RefreshBreakpoints() {
    VLOG(2) << "Requesting breakpoint refresh...";
    FlatBufferBuilder fbb;
    rpc::ListBreakpointsRequestBuilder request(fbb);
    request.add_session_id(session_id_);
    return IssueRequest<rpc::ListBreakpointsRequest,
                        rpc::ResponseUnion::ListBreakpointsResponse>(
        request.Finish(), std::move(fbb),
        [this](Status status, const rpc::Response& response_union) -> Status {
          if (!status.ok()) return status;
          VLOG(2) << "Refreshing breakpoints...";
          const auto& response =
              *response_union.message_as_ListBreakpointsResponse();
          for (auto* breakpoint_def : *response.breakpoints()) {
            auto breakpoint_or = GetBreakpoint(breakpoint_def->breakpoint_id());
            if (!breakpoint_or.ok()) {
              // Not found; add new.
              RETURN_IF_ERROR(RegisterBreakpoint(*breakpoint_def));
              breakpoint_or = GetBreakpoint(breakpoint_def->breakpoint_id());
            }
            RETURN_IF_ERROR(breakpoint_or.status());
            RETURN_IF_ERROR(
                breakpoint_or.ValueOrDie()->MergeFrom(*breakpoint_def));
          }
          // TODO(benvanik): handle removals/deaths.
          VLOG(2) << "Refreshed breakpoints!";
          return OkStatus();
        });
  }

  Status DispatchResponse(const rpc::Response& response) {
    if (pending_responses_.empty()) {
      return FailedPreconditionErrorBuilder(IREE_LOC)
             << "Response received but no request is pending";
    }
    auto type_callback = std::move(pending_responses_.front());
    pending_responses_.pop();

    if (response.status()) {
      const auto& status = *response.status();
      Status client_status =
          StatusBuilder(static_cast<StatusCode>(status.code()), IREE_LOC)
          << "Server request failed: " << WrapString(status.message());
      return type_callback.second(std::move(client_status), response);
    }

    if (!response.message()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Response contains no message body";
    }

    if (response.message_type() != type_callback.first) {
      return DataLossErrorBuilder(IREE_LOC)
             << "Out of order response (mismatch pending)";
    }
    return type_callback.second(OkStatus(), response);
  }

  Status DispatchEvent(const rpc::ServicePacket& packet) {
    switch (packet.event_type()) {
#define DISPATCH_EVENT(event_name)                                 \
  case rpc::EventUnion::event_name##Event: {                       \
    VLOG(2) << "EVENT: " << #event_name;                           \
    return On##event_name(*packet.event_as_##event_name##Event()); \
  }
      DISPATCH_EVENT(ServiceShutdown);
      DISPATCH_EVENT(ContextRegistered);
      DISPATCH_EVENT(ContextUnregistered);
      DISPATCH_EVENT(ModuleLoaded);
      DISPATCH_EVENT(InvocationRegistered);
      DISPATCH_EVENT(InvocationUnregistered);
      DISPATCH_EVENT(BreakpointResolved);
      DISPATCH_EVENT(BreakpointHit);
      DISPATCH_EVENT(StepCompleted);
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unimplemented debug service event: "
               << static_cast<int>(packet.event_type());
    }
  }

  StatusOr<TcpRemoteContext*> GetContext(int context_id) {
    auto it = context_map_.find(context_id);
    if (it == context_map_.end()) {
      return NotFoundErrorBuilder(IREE_LOC) << "Context was never registered";
    }
    return it->second.get();
  }

  Status OnServiceShutdown(const rpc::ServiceShutdownEvent& event) {
    LOG(INFO) << "Service is shutting down; setting pending shutdown flag";
    shutdown_pending_ = true;
    return OkStatus();
  }

  Status RegisterContext(int context_id) {
    auto context = absl::make_unique<TcpRemoteContext>(context_id, this);
    VLOG(2) << "RegisterContext(" << context_id << ")";
    auto context_ptr = context.get();
    context_map_.insert({context_id, std::move(context)});
    contexts_.push_back(context_ptr);
    return listener_->OnContextRegistered(*context_ptr);
  }

  Status OnContextRegistered(const rpc::ContextRegisteredEvent& event) {
    VLOG(2) << "OnContextRegistered(" << event.context_id() << ")";
    auto it = context_map_.find(event.context_id());
    if (it != context_map_.end()) {
      return FailedPreconditionErrorBuilder(IREE_LOC)
             << "Context already registered";
    }
    return RegisterContext(event.context_id());
  }

  Status OnContextUnregistered(const rpc::ContextUnregisteredEvent& event) {
    VLOG(2) << "OnContextUnregistered(" << event.context_id() << ")";
    auto it = context_map_.find(event.context_id());
    if (it == context_map_.end()) {
      return FailedPreconditionErrorBuilder(IREE_LOC)
             << "Context was never registered";
    }
    auto context = std::move(it->second);
    context_map_.erase(it);
    auto list_it = std::find(contexts_.begin(), contexts_.end(), context.get());
    contexts_.erase(list_it);
    return listener_->OnContextUnregistered(*context);
  }

  Status OnModuleLoaded(const rpc::ModuleLoadedEvent& event) {
    VLOG(2) << "OnModuleLoaded(" << event.context_id() << ", "
            << WrapString(event.module_name()) << ")";
    ASSIGN_OR_RETURN(auto* context, GetContext(event.context_id()));
    auto module_name = WrapString(event.module_name());
    auto module = absl::make_unique<TcpRemoteModule>(
        event.context_id(), std::string(module_name), this);
    auto* module_ptr = module.get();
    RETURN_IF_ERROR(context->AddModule(std::move(module)));
    return listener_->OnModuleLoaded(*context, *module_ptr);
  }

  StatusOr<TcpRemoteInvocation*> GetInvocation(int invocation_id) {
    auto it = invocation_map_.find(invocation_id);
    if (it == invocation_map_.end()) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Invocation was never registered";
    }
    return it->second.get();
  }

  Status RegisterInvocation(int invocation_id) {
    VLOG(2) << "RegisterInvocation(" << invocation_id << ")";
    auto invocation =
        absl::make_unique<TcpRemoteInvocation>(invocation_id, this);
    auto invocation_ptr = invocation.get();
    invocation_map_.insert({invocation_id, std::move(invocation)});
    invocations_.push_back(invocation_ptr);
    RETURN_IF_ERROR(RefreshInvocations());
    return listener_->OnInvocationRegistered(*invocation_ptr);
  }

  Status OnInvocationRegistered(const rpc::InvocationRegisteredEvent& event) {
    VLOG(2) << "OnInvocationRegistered(" << event.invocation_id() << ")";
    auto it = invocation_map_.find(event.invocation_id());
    if (it != invocation_map_.end()) {
      return FailedPreconditionErrorBuilder(IREE_LOC)
             << "Invocation already registered";
    }
    return RegisterInvocation(event.invocation_id());
  }

  Status OnInvocationUnregistered(
      const rpc::InvocationUnregisteredEvent& event) {
    VLOG(2) << "OnInvocationUnregistered(" << event.invocation_id() << ")";
    auto it = invocation_map_.find(event.invocation_id());
    if (it == invocation_map_.end()) {
      return FailedPreconditionErrorBuilder(IREE_LOC)
             << "Invocation was never registered";
    }
    auto invocation = std::move(it->second);
    invocation_map_.erase(it);
    auto list_it =
        std::find(invocations_.begin(), invocations_.end(), invocation.get());
    invocations_.erase(list_it);
    return listener_->OnInvocationUnregistered(*invocation);
  }

  StatusOr<TcpRemoteBreakpoint*> GetBreakpoint(int breakpoint_id) {
    auto it = breakpoint_map_.find(breakpoint_id);
    if (it == breakpoint_map_.end()) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Breakpoint " << breakpoint_id << " was never registered";
    }
    return it->second.get();
  }

  Status RegisterBreakpoint(const rpc::BreakpointDef& breakpoint_def) {
    auto it = breakpoint_map_.find(breakpoint_def.breakpoint_id());
    if (it != breakpoint_map_.end()) {
      VLOG(2) << "RegisterBreakpoint(" << breakpoint_def.breakpoint_id()
              << ") (update)";
      return it->second->MergeFrom(breakpoint_def);
    }

    VLOG(2) << "RegisterBreakpoint(" << breakpoint_def.breakpoint_id() << ")";
    auto breakpoint = absl::make_unique<TcpRemoteBreakpoint>(
        breakpoint_def.breakpoint_id(),
        static_cast<RemoteBreakpoint::Type>(breakpoint_def.breakpoint_type()),
        this);
    RETURN_IF_ERROR(breakpoint->MergeFrom(breakpoint_def));
    breakpoints_.push_back(breakpoint.get());
    breakpoint_map_.insert({breakpoint->id(), std::move(breakpoint)});
    return OkStatus();
  }

  Status UnregisterBreakpoint(RemoteBreakpoint* breakpoint) {
    VLOG(2) << "UnregisterBreakpoint(" << breakpoint->id() << ")";
    auto it = breakpoint_map_.find(breakpoint->id());
    if (it == breakpoint_map_.end()) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Breakpoint was never registered";
    }
    breakpoint_map_.erase(it);
    auto list_it =
        std::find(breakpoints_.begin(), breakpoints_.end(), breakpoint);
    breakpoints_.erase(list_it);
    return OkStatus();
  }

  Status OnBreakpointResolved(const rpc::BreakpointResolvedEvent& event) {
    VLOG(2) << "OnBreakpointResolved(" << event.breakpoint()->breakpoint_id()
            << ")";
    auto it = breakpoint_map_.find(event.breakpoint()->breakpoint_id());
    if (it == breakpoint_map_.end()) {
      RETURN_IF_ERROR(RegisterBreakpoint(*event.breakpoint()));
    } else {
      RETURN_IF_ERROR(it->second->MergeFrom(*event.breakpoint()));
    }
    return OkStatus();
  }

  Status OnBreakpointHit(const rpc::BreakpointHitEvent& event) {
    VLOG(2) << "OnBreakpointHit(" << event.breakpoint_id() << ")";
    ASSIGN_OR_RETURN(auto* breakpoint, GetBreakpoint(event.breakpoint_id()));
    auto* invocation_def = event.invocation();
    auto invocation_or = GetInvocation(invocation_def->invocation_id());
    if (!invocation_or.ok()) {
      // Not found; add new.
      RETURN_IF_ERROR(RegisterInvocation(invocation_def->invocation_id()));
      invocation_or = GetInvocation(invocation_def->invocation_id());
    }
    RETURN_IF_ERROR(invocation_or.status());
    RETURN_IF_ERROR(invocation_or.ValueOrDie()->MergeFrom(*invocation_def));
    return listener_->OnBreakpointHit(*breakpoint, *invocation_or.ValueOrDie());
  }

  Status StepInvocation(rpc::StepInvocationRequestT* step_request,
                        std::function<void()> callback) {
    FlatBufferBuilder fbb;
    auto status = IssueRequest<rpc::StepInvocationRequest,
                               rpc::ResponseUnion::StepInvocationResponse>(
        rpc::StepInvocationRequest::Pack(fbb, step_request), std::move(fbb),
        [](Status status, const rpc::Response& response_union) -> Status {
          return status;
        });
    RETURN_IF_ERROR(status);
    pending_step_callbacks_[step_request->step_id] = std::move(callback);
    return OkStatus();
  }

  Status OnStepCompleted(const rpc::StepCompletedEvent& event) {
    VLOG(2) << "OnStepCompleted(" << event.step_id() << ")";

    // Update all invocation states that are contained.
    // This may only be a subset of relevant states.
    for (auto* invocation_def : *event.invocations()) {
      ASSIGN_OR_RETURN(auto invocation,
                       GetInvocation(invocation_def->invocation_id()));
      RETURN_IF_ERROR(invocation->MergeFrom(*invocation_def));
    }

    // Dispatch step callback. Note that it may have been cancelled and that's
    // ok. We'll just make ready to resume execution.
    auto it = pending_step_callbacks_.find(event.step_id());
    if (it != pending_step_callbacks_.end()) {
      it->second();
      pending_step_callbacks_.erase(it);
    } else {
      LOG(WARNING) << "Step " << event.step_id()
                   << " not found; was cancelled?";
      RETURN_IF_ERROR(MakeReady());
    }
    return OkStatus();
  }

  int session_id_ = 123;

  int fd_ = -1;
  Listener* listener_;
  bool shutdown_pending_ = false;
  std::queue<std::pair<rpc::ResponseUnion, ResponseCallback>>
      pending_responses_;

  std::vector<RemoteContext*> contexts_;
  absl::flat_hash_map<int, std::unique_ptr<TcpRemoteContext>> context_map_;
  std::vector<RemoteInvocation*> invocations_;
  absl::flat_hash_map<int, std::unique_ptr<TcpRemoteInvocation>>
      invocation_map_;
  std::vector<RemoteBreakpoint*> breakpoints_;
  absl::flat_hash_map<int, std::unique_ptr<TcpRemoteBreakpoint>>
      breakpoint_map_;

  int next_step_id_ = 1;
  absl::flat_hash_map<int, std::function<void()>> pending_step_callbacks_;
};

}  // namespace

// static
StatusOr<std::unique_ptr<DebugClient>> DebugClient::Connect(
    absl::string_view service_address, Listener* listener) {
  // Parse address into hostname and port.
  ASSIGN_OR_RETURN(auto hostname_port, ParseAddress(service_address));
  if (hostname_port.second == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "No port specified in service address; port must match the "
              "server: "
           << service_address;
  }

  // Attempt to resolve the address.
  // Note that if we only wanted local debugging we could remove the dep on
  // getaddrinfo/having a valid DNS setup.
  addrinfo hints = {0};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* resolved_address = nullptr;
  auto port_str = std::to_string(hostname_port.second);
  int getaddrinfo_ret = ::getaddrinfo(
      hostname_port.first.c_str(), port_str.c_str(), &hints, &resolved_address);
  if (getaddrinfo_ret != 0) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Unable to resolve debug service address for " << service_address
           << ": (" << getaddrinfo_ret << ") "
           << ::gai_strerror(getaddrinfo_ret);
  }

  // Attempt to connect with each address returned from the query.
  int fd = -1;
  for (addrinfo* rp = resolved_address; rp != nullptr; rp = rp->ai_next) {
    fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) continue;
    if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
      break;  // Success!
    }
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(resolved_address);
  if (fd == -1) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Unable to connect to " << service_address << " on any address: ("
           << errno << ") " << ::strerror(errno);
  }

  LOG(INFO) << "Connected to debug service at " << service_address;

  return TcpDebugClient::Create(fd, listener);
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
