// Copyright 2020 Google LLC
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
//
#include "iree/hal/llvmjit/llvmjit_command_processor.h"

#include "iree/base/tracing.h"
#include "iree/hal/llvmjit/llvmjit_executable.h"

namespace iree {
namespace hal {
namespace llvmjit {

LLVMJITCommandProcessor::LLVMJITCommandProcessor(
    Allocator* allocator, CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories)
    : HostLocalCommandProcessor(allocator, mode, command_categories) {}

LLVMJITCommandProcessor::~LLVMJITCommandProcessor() = default;

Status LLVMJITCommandProcessor::Dispatch(
    const DispatchRequest& dispatch_request) {
  IREE_TRACE_SCOPE0("LLVMJITCommandProcessor::Dispatch");
  return OkStatus();
}
}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
