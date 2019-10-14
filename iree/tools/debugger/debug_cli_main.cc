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

#include "absl/flags/flag.h"
#include "iree/base/init.h"
#include "iree/base/status.h"
#include "iree/tools/debugger/debug_prompt.h"

ABSL_FLAG(std::string, debug_service_uri, "0.0.0.0:6000",
          "IP/port of debug service to connect to.");

namespace iree {
namespace rt {
namespace debug {

Status Run() {
  // TODO(benvanik): retry until connected? would allow auto-build reconnects.
  return AttachDebugPrompt(absl::GetFlag(FLAGS_debug_service_uri));
}

extern "C" int main(int argc, char** argv) {
  InitializeEnvironment(&argc, &argv);
  CHECK_OK(Run());
  return 0;
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
