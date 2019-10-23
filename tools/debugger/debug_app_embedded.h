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

#ifndef IREE_TOOLS_DEBUGGER_DEBUG_APP_EMBEDDED_H_
#define IREE_TOOLS_DEBUGGER_DEBUG_APP_EMBEDDED_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "base/status.h"

namespace iree {
namespace rt {
namespace debug {

// RAII handle for keeping the debugger alive.
// When the instance is destroyed the debugger app will be closed.
class EmbeddedDebugger {
 public:
  virtual ~EmbeddedDebugger() = default;

  // Blocks the caller until the debugger is closed by the user.
  virtual Status AwaitClose() = 0;
};

// Launches the debugger app.
// Returns a handle that can be used to wait for the debugger to close or
// force it to close.
StatusOr<std::unique_ptr<EmbeddedDebugger>> LaunchDebugger();

// Launches the debugger app and attaches to the given server address.
// Returns a handle that can be used to wait for the debugger to close or
// force it to close.
StatusOr<std::unique_ptr<EmbeddedDebugger>> AttachDebugger(
    absl::string_view service_address);

}  // namespace debug
}  // namespace rt
}  // namespace iree

#endif  // IREE_TOOLS_DEBUGGER_DEBUG_APP_EMBEDDED_H_
