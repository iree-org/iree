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

#include "iree/tools/debugger/debug_prompt.h"

#include "iree/base/status.h"
#include "iree/rt/debug/debug_client.h"

namespace iree {
namespace rt {
namespace debug {
namespace {

class DebugPrompt : private DebugClient::Listener {
 public:
  Status Connect(absl::string_view debug_service_uri) {
    // Connect to the debug service.
    ASSIGN_OR_RETURN(debug_client_,
                     DebugClient::Connect(debug_service_uri, this));
    return OkStatus();
  }

  Status Run() {
    // Query commands, transmit requests, and dispatch responses.
    while (true) {
      RETURN_IF_ERROR(debug_client_->Poll());

      // TODO(benvanik): ask for a command.
      // TODO(benvanik): display stuff.
    }
  }

 private:
  Status OnContextRegistered(const RemoteContext& context) override {
    // Ack.
    return debug_client_->MakeReady();
  }

  Status OnContextUnregistered(const RemoteContext& context) override {
    // Ack.
    return debug_client_->MakeReady();
  }

  Status OnModuleLoaded(const RemoteContext& context,
                        const RemoteModule& module) override {
    // Ack.
    return debug_client_->MakeReady();
  }

  Status OnInvocationRegistered(const RemoteInvocation& invocation) override {
    // Ack.
    return debug_client_->MakeReady();
  }

  Status OnInvocationUnregistered(const RemoteInvocation& invocation) override {
    // Ack.
    return debug_client_->MakeReady();
  }

  Status OnBreakpointHit(const RemoteBreakpoint& breakpoint,
                         const RemoteInvocation& invocation) override {
    // Ack.
    return debug_client_->MakeReady();
  }

  std::unique_ptr<DebugClient> debug_client_;
};

}  // namespace

Status AttachDebugPrompt(absl::string_view debug_service_uri) {
  DebugPrompt debug_prompt;
  RETURN_IF_ERROR(debug_prompt.Connect(debug_service_uri));
  return debug_prompt.Run();
}

}  // namespace debug
}  // namespace rt
}  // namespace iree
