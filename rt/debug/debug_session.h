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

#ifndef IREE_RT_DEBUG_DEBUG_SESSION_H_
#define IREE_RT_DEBUG_DEBUG_SESSION_H_

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "base/status.h"
#include "rt/context.h"
#include "rt/invocation.h"
#include "rt/module.h"
#include "schemas/debug_service_generated.h"

namespace iree {
namespace rt {
namespace debug {

// An active debugging session maintained by the DebugService.
// Each connected client gets a session and transport-specific implementations
// use the event methods to receive signals from the service.
//
// All methods are called only while the debug lock is held and may be called
// from any thread.
class DebugSession {
 public:
  virtual ~DebugSession() = default;

  // Session ID used in all RPCs related to this session.
  // This can be used for attributing RPCs to the originating session when
  // multiple sessions may be active at a time/over the same transport.
  int id() const { return session_id_; }

  // Returns true if the session has issued a MakeReady request and is ok if
  // execution resumes.
  bool is_ready() const;

  // Signals that the session has readied up and is now active.
  // Called with the global debug lock held.
  virtual Status OnReady();

  // Signals that the session has gone unready (from an event/etc) and the
  // service is now awaiting it to ready up.
  // Called with the global debug lock held.
  virtual Status OnUnready();

  // Signals that a context has been registered.
  // Called with the global debug lock held.
  virtual Status OnContextRegistered(Context* context) = 0;
  virtual Status OnContextUnregistered(Context* context) = 0;

  // Signals that a module has been loaded in a context.
  // Called with the global debug lock held.
  virtual Status OnModuleLoaded(Context* context, Module* module) = 0;

  // Signals that a invocation has been registered.
  // Called with the global debug lock held.
  virtual Status OnInvocationRegistered(Invocation* invocation) = 0;
  virtual Status OnInvocationUnregistered(Invocation* invocation) = 0;

  // Signals that a breakpoint has been resolved to a particular function in a
  // context.
  // Called with the global debug lock held.
  virtual Status OnBreakpointResolved(const rpc::BreakpointDefT& breakpoint,
                                      Context* context) = 0;

  // Signals that the given breakpoint has been hit during execution.
  // Called with the global debug lock held.
  virtual Status OnBreakpointHit(int breakpoint_id,
                                 const Invocation& invocation) = 0;

 private:
  mutable absl::Mutex mutex_;
  int session_id_ = 0;
  int ready_ ABSL_GUARDED_BY(mutex_) = -1;
};

}  // namespace debug
}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_DEBUG_DEBUG_SESSION_H_
