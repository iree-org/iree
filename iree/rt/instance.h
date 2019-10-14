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

#ifndef IREE_RT_INSTANCE_H_
#define IREE_RT_INSTANCE_H_

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/ref_ptr.h"
#include "iree/hal/device_manager.h"
#include "iree/rt/context.h"

namespace iree {
namespace rt {
namespace debug {
class DebugServer;
}  // namespace debug
}  // namespace rt
}  // namespace iree

namespace iree {
namespace rt {

// Shared runtime instance responsible for routing Context events, enumerating
// and creating hardware device interfaces, and managing thread pools.
//
// A single runtime instance can service multiple contexts and hosting
// applications should try to reuse instances as much as possible. This ensures
// that resource allocation across contexts is handled and extraneous device
// interaction is avoided. For devices that may have exclusive access
// restrictions it is mandatory to share instances, so plan accordingly.
//
// Thread-safe.
class Instance final : public RefObject<Instance> {
 public:
  // Creates an instance with an optional attached |debug_server|.
  explicit Instance(std::unique_ptr<debug::DebugServer> debug_server = nullptr);
  ~Instance();
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;

  // Optional debug server that has access to contexts in this instance.
  debug::DebugServer* debug_server() const { return debug_server_.get(); }

  // Device manager used to enumerate available devices.
  hal::DeviceManager* device_manager() const { return &device_manager_; }

 private:
  friend class Context;
  void RegisterContext(Context* context);
  void UnregisterContext(Context* context);

  std::unique_ptr<debug::DebugServer> debug_server_;
  mutable hal::DeviceManager device_manager_;

  absl::Mutex contexts_mutex_;
  IntrusiveList<Context, offsetof(Context, instance_list_link_)> contexts_
      ABSL_GUARDED_BY(contexts_mutex_);
};

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_INSTANCE_H_
