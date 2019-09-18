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

#include "third_party/mlir_edge/iree/vm/debug/debug_session.h"

#include "third_party/absl/types/source_location.h"
#include "third_party/mlir_edge/iree/base/status.h"

namespace iree {
namespace vm {
namespace debug {

bool DebugSession::is_ready() const {
  absl::MutexLock lock(&mutex_);
  return ready_ == 0;
}

Status DebugSession::OnReady() {
  absl::MutexLock lock(&mutex_);
  if (ready_ > 0) {
    return FailedPreconditionErrorBuilder(ABSL_LOC)
           << "Session has already readied up";
  }
  ++ready_;
  VLOG(2) << "Session " << id() << ": ++ready = " << ready_;
  return OkStatus();
}

Status DebugSession::OnUnready() {
  absl::MutexLock lock(&mutex_);
  --ready_;
  VLOG(2) << "Session " << id() << ": --ready = " << ready_;
  return OkStatus();
}

}  // namespace debug
}  // namespace vm
}  // namespace iree
