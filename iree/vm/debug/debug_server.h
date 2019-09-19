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

#ifndef IREE_VM_DEBUG_DEBUG_SERVER_H_
#define IREE_VM_DEBUG_DEBUG_SERVER_H_

#include "iree/base/status.h"

namespace iree {
namespace vm {
class FiberState;
class SequencerContext;
class Module;
}  // namespace vm
}  // namespace iree

namespace iree {
namespace vm {
namespace debug {

// Runtime debugging server.
// Enabled only when compiled in (by defining IREE_DEBUG=1), this provides an
// RPC server that allows debuggers to attach, query, and manipulate contexts.
// This interface is used by various parts of the runtime such as dispatch to
// query the current debug state and signal events.
//
// Thread-safe. Contexts may be registered and unregistered from any thread.
class DebugServer {
 public:
  // Creates a new debug service listening on the provided |port|.
  // Even when disabled the device can still be created however it will not
  // perform any actual operations and act as if the debugger is not attached.
  static StatusOr<std::unique_ptr<DebugServer>> Create(int listen_port);

  // TODO(benvanik): ensure this gets optimized out when disabled.
  // Seems to be the case: https://gcc.godbolt.org/z/0zf-L4
  virtual ~DebugServer() = default;

  // Attaches a callback that will be made when the debug server is shutting
  // down. This can be used to keep resources alive that require the debugger.
  // The callback will be made from a random thread.
  virtual void AtExit(std::function<void()> callback) = 0;

  // Blocks the caller until a client session connects and resumes all fibers.
  // Returns AbortedError if a session connects/is connected but disconnects
  // during the wait.
  virtual Status WaitUntilSessionReady() = 0;

 protected:
  friend class ::iree::vm::SequencerContext;

  // Registers a context with the debug service.
  // Ownership remains with the caller and UnregisterContext must be called
  // prior to the context being destroyed.
  virtual Status RegisterContext(SequencerContext* context) = 0;
  virtual Status UnregisterContext(SequencerContext* context) = 0;

  // Registers a new module linked into an existing Context.
  virtual Status RegisterContextModule(SequencerContext* context,
                                       vm::Module* module) = 0;

  friend class ::iree::vm::FiberState;

  // Registers a fiber state with the debug service.
  // Ownership remains with the caller and UnregisterFiberState must be called
  // prior to the fiber state being destroyed.
  virtual Status RegisterFiberState(FiberState* fiber_state) = 0;
  virtual Status UnregisterFiberState(FiberState* fiber_state) = 0;
};

}  // namespace debug
}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_DEBUG_DEBUG_SERVER_H_
