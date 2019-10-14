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

#include "iree/rt/debug/debug_server_flags.h"

#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "iree/base/memory.h"
#include "iree/base/status.h"

#if defined(IREE_DEBUG_EMBEDDED_APP_PRESENT)
#include "iree/tools/debugger/debug_app_embedded.h"
#endif  // IREE_DEBUG_EMBEDDED_APP_PRESENT

ABSL_FLAG(int32_t, iree_debug_service_port, 6000,
          "TCP port to listen for debug service connections.");
ABSL_FLAG(bool, iree_wait_for_debugger, false,
          "Waits until a debugger connects to continue startup.");
ABSL_FLAG(bool, iree_attach_debugger, false, "Attaches a debugger at startup.");

namespace iree {
namespace rt {
namespace debug {

StatusOr<std::unique_ptr<DebugServer>> CreateDebugServerFromFlags() {
  // Create the server based on whatever version is compiled in.
  // Note that this will return nullptr if no server is available.
  ASSIGN_OR_RETURN(
      auto debug_server,
      DebugServer::Create(absl::GetFlag(FLAGS_iree_debug_service_port)));
  if (!debug_server) {
    return nullptr;
  }

#if defined(IREE_DEBUG_EMBEDDED_APP_PRESENT)
  // If the embedded debug UI is present then we can launch that now.
  std::unique_ptr<EmbeddedDebugger> debugger;
  if (absl::GetFlag(FLAGS_iree_attach_debugger)) {
    LOG(INFO) << "Attaching debugger at startup...";
    ASSIGN_OR_RETURN(
        debugger,
        AttachDebugger(absl::StrCat(
            "localhost:", absl::GetFlag(FLAGS_iree_debug_service_port))));
    RETURN_IF_ERROR(debug_server->WaitUntilSessionReady());
    LOG(INFO) << "Debugger attached";
    // TODO(benvanik): C++14 to avoid this.
    auto debugger_baton = IreeMoveToLambda(debugger);
    debug_server->AtExit([debugger_baton]() { debugger_baton.value.reset(); });
  }
#else
  if (absl::GetFlag(FLAGS_iree_attach_debugger)) {
    LOG(WARNING) << "--iree_attach_debugger specified but no embedded debugger "
                    "is present. Build with --define=IREE_DEBUG=1.";
  }
#endif  // IREE_DEBUG_EMBEDDED_APP_PRESENT

  // Wait for a debugger to connect.
  if (absl::GetFlag(FLAGS_iree_wait_for_debugger)) {
    LOG(INFO) << "Waiting for a debugger to connect...";
    RETURN_IF_ERROR(debug_server->WaitUntilSessionReady());
    LOG(INFO) << "Debugger ready, resuming...";
  }

  return std::move(debug_server);
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
