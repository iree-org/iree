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

namespace iree {
namespace hal {
namespace llvmjit {

LLVMJITCommandProcessor::LLVMJITCommandProcessor(
    Allocator* allocator, CommandCategoryBitfield command_categories)
    : HostLocalCommandProcessor(allocator, command_categories) {}

LLVMJITCommandProcessor::~LLVMJITCommandProcessor() = default;

Status LLVMJITCommandProcessor::DispatchInline(
    Executable* executable, int32_t entry_point,
    std::array<uint32_t, 3> workgroups, const PushConstantBlock& push_constants,
    absl::Span<const absl::Span<const DescriptorSet::Binding>> set_bindings) {
  IREE_TRACE_SCOPE0("LLVMJITCommandProcessor::DispatchInline");
  auto* llvmjit_executable = static_cast<LLVMJITExecutable*>(executable);

  llvm::SmallVector<UnrankedMemRefType<uint32_t>*, 4> descriptors;
  llvm::SmallVector<void*, 4> args;
  for (size_t set = 0; set < set_bindings.size(); ++set) {
    for (size_t binding = 0; binding < set_bindings[set].size(); ++binding) {
      const auto& io_binding = set_bindings[set][binding];
      ASSIGN_OR_RETURN(auto memory, io_binding.buffer->MapMemory<uint8_t>(
                                        MemoryAccessBitfield::kWrite,
                                        io_binding.offset, io_binding.length));
      auto data = memory.mutable_data();
      auto descriptor = allocUnrankedDescriptor<uint32_t>(data);
      descriptors.push_back(descriptor);
      args.push_back(&descriptor->descriptor);
    }
  }
  auto push_constants_descriptor = allocUnrankedDescriptor<uint32_t>(
      (void*)push_constants.values.data(),
      {static_cast<long>(push_constants.values.size())});
  descriptors.push_back(push_constants_descriptor);
  args.push_back(&push_constants_descriptor->descriptor);

  auto status = llvmjit_executable->Invoke(entry_point, args);

  for (int i = 0; i < descriptors.size(); ++i) {
    freeUnrankedDescriptor(descriptors[i]);
  }

  return status;
}

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
