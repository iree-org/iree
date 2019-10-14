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

#include "iree/rt/debug/debug_client.h"

#include "iree/base/source_location.h"
#include "iree/base/status.h"

namespace iree {
namespace rt {
namespace debug {

Status DebugClient::GetFunction(
    std::string module_name, std::string function_name,
    std::function<void(StatusOr<RemoteFunction*> function)> callback) {
  return ResolveFunction(
      module_name, function_name,
      [this, module_name, callback](StatusOr<int> function_ordinal) {
        if (!function_ordinal.ok()) {
          callback(function_ordinal.status());
          return;
        }
        auto status =
            GetFunction(module_name, function_ordinal.ValueOrDie(), callback);
        if (!status.ok()) {
          callback(std::move(status));
        }
      });
}

Status DebugClient::StepInvocationOver(const RemoteInvocation& invocation,
                                       std::function<void()> callback) {
  // TODO(benvanik): implement bytecode stepping search.
  // int bytecode_offset = 0;
  // return StepInvocationToOffset(invocation, bytecode_offset,
  // std::move(callback));
  return UnimplementedErrorBuilder(IREE_LOC)
         << "StepInvocationOver not yet implemented";
}

Status DebugClient::StepInvocationOut(const RemoteInvocation& invocation,
                                      std::function<void()> callback) {
  // TODO(benvanik): implement bytecode stepping search.
  // int bytecode_offset = 0;
  // return StepInvocationToOffset(invocation, bytecode_offset,
  // std::move(callback));
  return UnimplementedErrorBuilder(IREE_LOC)
         << "StepInvocationOut not yet implemented";
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
