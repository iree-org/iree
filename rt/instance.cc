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

#include "rt/instance.h"

#include "base/tracing.h"
#include "rt/debug/debug_server.h"

namespace iree {
namespace rt {

Instance::Instance(std::unique_ptr<debug::DebugServer> debug_server)
    : debug_server_(std::move(debug_server)) {
  IREE_TRACE_SCOPE0("Instance::ctor");
}

Instance::~Instance() { IREE_TRACE_SCOPE0("Instance::dtor"); }

void Instance::RegisterContext(Context* context) {
  {
    absl::MutexLock lock(&contexts_mutex_);
    contexts_.push_back(context);
  }
  if (debug_server_) {
    CHECK_OK(debug_server_->RegisterContext(context));
  }
}

void Instance::UnregisterContext(Context* context) {
  if (debug_server_) {
    CHECK_OK(debug_server_->UnregisterContext(context));
  }
  {
    absl::MutexLock lock(&contexts_mutex_);
    contexts_.erase(context);
  }
}

}  // namespace rt
}  // namespace iree
