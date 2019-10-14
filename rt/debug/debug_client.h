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

#ifndef IREE_RT_DEBUG_DEBUG_CLIENT_H_
#define IREE_RT_DEBUG_DEBUG_CLIENT_H_

#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/schemas/debug_service_generated.h"

namespace iree {
namespace rt {
namespace debug {

// Remote breakpoint currently active on the server.
class RemoteBreakpoint {
 public:
  enum class Type {
    kBytecodeFunction = 0,
    kNativeFunction = 1,
  };

  virtual ~RemoteBreakpoint() = default;

  int id() const { return id_; }
  Type type() const { return type_; }

  virtual const std::string& module_name() const = 0;
  virtual const std::string& function_name() const = 0;
  virtual int function_ordinal() const = 0;
  virtual int bytecode_offset() const = 0;

 protected:
  explicit RemoteBreakpoint(int id, Type type) : id_(id), type_(type) {}

 private:
  int id_;
  Type type_;
};

class RemoteModule;

class RemoteFunction {
 public:
  virtual ~RemoteFunction() = default;

  RemoteModule* module() const { return module_; }
  int ordinal() const { return function_ordinal_; }
  virtual const std::string& name() const = 0;

  virtual const FunctionDef& def() = 0;

  virtual bool is_loaded() const = 0;
  virtual bool CheckLoadedOrRequest() = 0;

  using LoadCallback = std::function<void(StatusOr<RemoteFunction*>)>;
  virtual void WhenLoaded(LoadCallback callback) = 0;

  virtual const BytecodeDef* bytecode() = 0;

 protected:
  RemoteFunction(RemoteModule* module, int function_ordinal)
      : module_(module), function_ordinal_(function_ordinal) {}

  RemoteModule* module_;
  int function_ordinal_;
};

class RemoteModule {
 public:
  virtual ~RemoteModule() = default;

  int context_id() const { return context_id_; }
  const std::string& name() const { return name_; }

  virtual const ModuleDef& def() = 0;

  virtual bool is_loaded() const = 0;
  virtual bool CheckLoadedOrRequest() = 0;

  using LoadCallback = std::function<void(StatusOr<RemoteModule*>)>;
  virtual void WhenLoaded(LoadCallback callback) = 0;

  virtual absl::Span<RemoteFunction*> functions() = 0;

 protected:
  RemoteModule(int context_id, std::string name)
      : context_id_(context_id), name_(std::move(name)) {}

 private:
  int context_id_;
  std::string name_;
};

class RemoteContext {
 public:
  virtual ~RemoteContext() = default;

  int id() const { return id_; }

  virtual absl::Span<RemoteModule* const> modules() const = 0;

 protected:
  explicit RemoteContext(int id) : id_(id) {}

 private:
  int id_;
};

class RemoteInvocation {
 public:
  virtual ~RemoteInvocation() = default;

  int id() const { return id_; }
  const std::string& name() const { return name_; }

  virtual const rpc::InvocationDefT& def() const = 0;

 protected:
  explicit RemoteInvocation(int id)
      : id_(id), name_(absl::StrCat("Invocation ", id)) {}

 private:
  int id_;
  std::string name_;
};

// Debugger RPC server client.
// Statefully tracks a DebugServer to provide common client operations and
// memoized queries.
//
// Thread-compatible. Do not use the client from multiple threads concurrently.
// All remote updates of local state are performed by the Poll function. See
// Poll for more details.
class DebugClient {
 public:
  // Debug event listener interface.
  // Event methods will be called from within Poll calls (so on that thread).
  //
  // When the server posts an event it will mark the client as unready and
  // suspend execution of all invocations until MakeReady is used to indicate
  // that the client is ready for the server to resume. Each event needs a
  // matching MakeReady ack.
  //
  // Listeners can defer acking if they need to perform additional queries or
  // state changes to the server or wait for user interaction. Multiple events
  // may come in while unready if there was a series of events pending on the
  // server.
  class Listener {
   public:
    virtual ~Listener() = default;

    // Signals that a context has been registered on the server.
    virtual Status OnContextRegistered(const RemoteContext& context) = 0;
    virtual Status OnContextUnregistered(const RemoteContext& context) = 0;

    // Signals that a module has been loaded into a context on the server.
    virtual Status OnModuleLoaded(const RemoteContext& context,
                                  const RemoteModule& module) = 0;

    // Signals that a invocation has been registered on the server.
    virtual Status OnInvocationRegistered(
        const RemoteInvocation& invocation) = 0;
    virtual Status OnInvocationUnregistered(
        const RemoteInvocation& invocation) = 0;

    // Signals that a breakpoint has been hit by a invocation on the server.
    virtual Status OnBreakpointHit(const RemoteBreakpoint& breakpoint,
                                   const RemoteInvocation& invocation) = 0;
  };

  // Connects to a remote debug service at the provided IP:port.
  // The specified |listener| will receive async event notifications.
  static StatusOr<std::unique_ptr<DebugClient>> Connect(
      absl::string_view service_address, Listener* listener);

  virtual ~DebugClient() = default;

  // Returns true if the client is connected to a service.
  // virtual bool is_connected() const = 0;

  // A list of all contexts registered with the server.
  virtual absl::Span<RemoteContext* const> contexts() const = 0;

  // A list of all invocations registered with the server.
  virtual absl::Span<RemoteInvocation* const> invocations() const = 0;

  // A list of all breakpoints registered with the server.
  virtual absl::Span<RemoteBreakpoint* const> breakpoints() const = 0;

  // Resolves a function to a module ordinal.
  // This will occur asynchronously and the |callback| will be issued on the
  // polling thread.
  virtual Status ResolveFunction(
      std::string module_name, std::string function_name,
      std::function<void(StatusOr<int> function_ordinal)> callback) = 0;

  // Gets a function body instance.
  // The provided |callback| will be issued on the polling thread when the
  // function is available.
  virtual Status GetFunction(
      std::string module_name, int function_ordinal,
      std::function<void(StatusOr<RemoteFunction*> function)> callback) = 0;
  Status GetFunction(
      std::string module_name, std::string function_name,
      std::function<void(StatusOr<RemoteFunction*> function)> callback);

  // Adds a breakpoint for the given module:function:offset.
  // The breakpoint will apply to all contexts with the module loaded.
  virtual Status AddFunctionBreakpoint(
      std::string module_name, std::string function_name, int offset,
      std::function<void(const RemoteBreakpoint& breakpoint)> callback =
          nullptr) = 0;

  // Removes a breakpoint from the server.
  virtual Status RemoveBreakpoint(const RemoteBreakpoint& breakpoint) = 0;

  // Notifies the server that the debug session is ready to continue.
  // This must be called once on connection to and in acknowledgement to any
  // events posted by the server (read: any call to the Listener::On* methods).
  virtual Status MakeReady() = 0;

  // Suspends all invocations running on the server.
  virtual Status SuspendAllInvocations() = 0;

  // Resumes all invocations running on the server.
  virtual Status ResumeAllInvocations() = 0;

  // Suspends a list of invocations running on the server. Invocations not in
  // the provided list will not be suspended, such as new invocations created
  // while the request is pending.
  virtual Status SuspendInvocations(
      absl::Span<RemoteInvocation*> invocations) = 0;

  // Resumes a list of invocations running on the server.
  virtual Status ResumeInvocations(
      absl::Span<RemoteInvocation*> invocations) = 0;

  // Steps a invocation one bytecode operation.
  virtual Status StepInvocation(const RemoteInvocation& invocation,
                                std::function<void()> callback) = 0;
  // Steps a invocation over one bytecode operation, not stopping until it
  // completes.
  Status StepInvocationOver(const RemoteInvocation& invocation,
                            std::function<void()> callback);
  // Steps a invocation out of the current block.
  Status StepInvocationOut(const RemoteInvocation& invocation,
                           std::function<void()> callback);
  // Steps a invocation to a specific bytecode offset within the current
  // function.
  virtual Status StepInvocationToOffset(const RemoteInvocation& invocation,
                                        int bytecode_offset,
                                        std::function<void()> callback) = 0;

  // TODO(benvanik): profiling modes.

  // Polls for the current state of the debug service and processes incoming
  // responses. Must be called as frequently as the UI is desired to update.
  // Returns CancelledError when the service is being shutdown/disconnected.
  //
  // Events on the Listener will be called from within this method.
  virtual Status Poll() = 0;
};

}  // namespace debug
}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_DEBUG_DEBUG_CLIENT_H_
