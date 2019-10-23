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

// Native (linux/etc) debug_app entry point.
// This should work on any platform with pthreads and SDL support.

#include "base/init.h"
#include "base/status.h"
#include "tools/debugger/debug_app_embedded.h"

namespace iree {
namespace rt {
namespace debug {

Status Run() {
  ASSIGN_OR_RETURN(auto handle, LaunchDebugger());
  RETURN_IF_ERROR(handle->AwaitClose());
  handle.reset();
  return OkStatus();
}

extern "C" int main(int argc, char** argv) {
  InitializeEnvironment(&argc, &argv);
  auto status = Run();
  if (!status.ok()) {
    LOG(ERROR) << "Debugger error: " << status;
    return 1;
  }
  return 0;
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
