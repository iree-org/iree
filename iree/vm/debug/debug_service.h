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

#ifndef IREE_VM_DEBUG_DEBUG_SERVICE_H_
#define IREE_VM_DEBUG_DEBUG_SERVICE_H_

#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "flatbuffers/flatbuffers.h"
#include "iree/base/status.h"
#include "iree/schemas/debug_service_generated.h"
#include "iree/vm/debug/debug_session.h"
#include "iree/vm/fiber_state.h"
#include "iree/vm/sequencer_context.h"

namespace iree {
namespace vm {
namespace debug {

// Debugging service used to implement the DebugService RPC methods in a
// transport-independent way. Specific DebugServer implementations can compose
// with a DebugService to avoid needing to maintain state themselves. Multiple
// DebugServer instances could share the same DebugService instance to ensure
// all clients - regardless of transport - share the same state.
//
// Thread-safe.
class DebugService {
 public:
  // Registers a context with the debug service.
  // Ownership remains with the caller and UnregisterContext must be called
  // prior to the context being destroyed.
  Status RegisterContext(SequencerContext* context);
  Status UnregisterContext(SequencerContext* context);

  // Registers a new module linked into an existing Context.
  Status RegisterContextModule(SequencerContext* context, vm::Module* module);

  // Registers a fiber state with the debug service.
  // Ownership remains with the caller and UnregisterFiberState must be called
  // prior to the fiber state being destroyed.
  Status RegisterFiberState(FiberState* fiber_state);
  Status UnregisterFiberState(FiberState* fiber_state);

  // Registers a debug session with the service.
  Status RegisterDebugSession(DebugSession* session);
  Status UnregisterDebugSession(DebugSession* session);

  // Blocks the caller until all sessions are ready.
  // Returns AbortedError if a session connects/is already connected but
  // disconnects during the wait.
  Status WaitUntilAllSessionsReady();

  StatusOr<::flatbuffers::Offset<rpc::MakeReadyResponse>> MakeReady(
      const rpc::MakeReadyRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);

  StatusOr<::flatbuffers::Offset<rpc::GetStatusResponse>> GetStatus(
      const rpc::GetStatusRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);

  StatusOr<::flatbuffers::Offset<rpc::ListContextsResponse>> ListContexts(
      const rpc::ListContextsRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);

  StatusOr<::flatbuffers::Offset<rpc::GetModuleResponse>> GetModule(
      const rpc::GetModuleRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::GetFunctionResponse>> GetFunction(
      const rpc::GetFunctionRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::ResolveFunctionResponse>> ResolveFunction(
      const rpc::ResolveFunctionRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);

  StatusOr<::flatbuffers::Offset<rpc::ListFibersResponse>> ListFibers(
      const rpc::ListFibersRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::SuspendFibersResponse>> SuspendFibers(
      const rpc::SuspendFibersRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::ResumeFibersResponse>> ResumeFibers(
      const rpc::ResumeFibersRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::StepFiberResponse>> StepFiber(
      const rpc::StepFiberRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::GetFiberLocalResponse>> GetFiberLocal(
      const rpc::GetFiberLocalRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::SetFiberLocalResponse>> SetFiberLocal(
      const rpc::SetFiberLocalRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);

  StatusOr<::flatbuffers::Offset<rpc::ListBreakpointsResponse>> ListBreakpoints(
      const rpc::ListBreakpointsRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::AddBreakpointResponse>> AddBreakpoint(
      const rpc::AddBreakpointRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::RemoveBreakpointResponse>>
  RemoveBreakpoint(const rpc::RemoveBreakpointRequest& request,
                   ::flatbuffers::FlatBufferBuilder* fbb);

  StatusOr<::flatbuffers::Offset<rpc::StartProfilingResponse>> StartProfiling(
      const rpc::StartProfilingRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);
  StatusOr<::flatbuffers::Offset<rpc::StopProfilingResponse>> StopProfiling(
      const rpc::StopProfilingRequest& request,
      ::flatbuffers::FlatBufferBuilder* fbb);

  // Serializes a fiber state and its stack frames.
  StatusOr<::flatbuffers::Offset<rpc::FiberStateDef>> SerializeFiberState(
      const FiberState& fiber_state, ::flatbuffers::FlatBufferBuilder* fbb);

 private:
  StatusOr<SequencerContext*> GetContext(int context_id) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  StatusOr<vm::Module*> GetModule(int context_id,
                                  absl::string_view module_name) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  StatusOr<FiberState*> GetFiberState(int fiber_id) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Suspends all fibers on all contexts. Returns only once all fibers have been
  // suspended successfully. Fails if any fiber fails to suspend.
  Status SuspendAllFibers() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Resumes all fibers on all contexts (the inverse of SuspendAllFibers).
  // Returns immediately.
  Status ResumeAllFibers() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Marks all sessions as unready.
  Status UnreadyAllSessions() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Attempts to re-register all breakpoints for a module.
  Status RegisterModuleBreakpoints(SequencerContext* context,
                                   vm::Module* module)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  Status RegisterFunctionBreakpoint(SequencerContext* context,
                                    vm::Module* module,
                                    rpc::BreakpointDefT* breakpoint)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  Status UnregisterFunctionBreakpoint(const rpc::BreakpointDefT& breakpoint)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  // Signals that the given breakpoint was hit by the specified fiber.
  // Called without the debug lock held.
  Status OnFunctionBreakpointHit(int breakpoint_id,
                                 const vm::Stack& fiber_state);

  absl::Mutex mutex_;
  std::vector<SequencerContext*> contexts_ ABSL_GUARDED_BY(mutex_);
  std::vector<FiberState*> fiber_states_ ABSL_GUARDED_BY(mutex_);
  std::vector<DebugSession*> sessions_ ABSL_GUARDED_BY(mutex_);
  int sessions_unready_ ABSL_GUARDED_BY(mutex_) = 0;
  int sessions_ready_ ABSL_GUARDED_BY(mutex_) = 0;

  std::vector<rpc::BreakpointDefT> breakpoints_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace debug
}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_DEBUG_DEBUG_SERVICE_H_
