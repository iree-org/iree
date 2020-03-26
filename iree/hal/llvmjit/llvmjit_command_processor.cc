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

#include <memory>
#include <vector>

#include "iree/base/tracing.h"
#include "iree/hal/buffer.h"
#include "iree/hal/llvmjit/llvmjit_executable.h"
#include "iree/hal/llvmjit/memref_runtime.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

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
  auto* executable =
      static_cast<LLVMJITExecutable*>(dispatch_request.executable);

  llvm::SmallVector<UnrankedMemRefType<uint32_t>*, 4> descriptors(
      dispatch_request.bindings.size());
  llvm::SmallVector<void*, 4> args(dispatch_request.bindings.size());
  for (size_t i = 0; i < dispatch_request.bindings.size(); ++i) {
    ASSIGN_OR_RETURN(
        auto memory,
        dispatch_request.bindings.at(i).buffer->MapMemory<uint32_t>(
            MemoryAccessBitfield::kWrite));
    auto data = memory.mutable_data();
    descriptors[i] = allocUnrankedDescriptor<uint32_t>(data);
    args[i] = &descriptors[i]->descriptor;
  }
  auto status = executable->Invoke(dispatch_request.entry_point, args);

  for (int i = 0; i < descriptors.size(); ++i) {
    freeUnrankedDescriptor(descriptors[i]);
  }

  return status;
}

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
