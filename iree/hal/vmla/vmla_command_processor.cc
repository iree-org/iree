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

#include "iree/hal/vmla/vmla_command_processor.h"

#include "iree/base/api_util.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/host/host_buffer.h"
#include "iree/hal/vmla/vmla_executable.h"
#include "iree/hal/vmla/vmla_module.h"
#include "iree/vm/invocation.h"
#include "iree/vm/stack.h"
#include "iree/vm/variant_list.h"

namespace iree {
namespace hal {
namespace vmla {

VMLACommandProcessor::VMLACommandProcessor(
    Allocator* allocator, CommandCategoryBitfield command_categories)
    : HostLocalCommandProcessor(allocator, command_categories) {}

VMLACommandProcessor::~VMLACommandProcessor() = default;

Status VMLACommandProcessor::DispatchInline(
    Executable* executable, int32_t entry_point,
    std::array<uint32_t, 3> workgroups, const PushConstantBlock& push_constants,
    absl::Span<const absl::Span<const DescriptorSet::Binding>> set_bindings) {
  IREE_TRACE_SCOPE0("VMLACommandProcessor::DispatchInline");

  auto* vmla_executable = static_cast<VMLAExecutable*>(executable);
  if (entry_point >= vmla_executable->entry_functions().size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Invalid entry point ordinal " << entry_point;
  }

  auto interface = vmla_executable->interface();
  RETURN_IF_ERROR(interface->SetConstants(push_constants.values));

  for (int set_ordinal = 0; set_ordinal < set_bindings.size(); ++set_ordinal) {
    for (const auto& binding : set_bindings[set_ordinal]) {
      // TODO(benvanik): plumb binding directly into VMLA to avoid this.
      void* data = static_cast<HostBuffer*>(binding.buffer->allocated_buffer())
                       ->mutable_data();
      data = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(data) +
                                     binding.buffer->byte_offset());
      ASSIGN_OR_RETURN(auto buffer,
                       Buffer::WrapMutable(data, binding.buffer->byte_length(),
                                           IREE_ALLOCATOR_NULL));
      RETURN_IF_ERROR(interface->SetBinding(set_ordinal, binding.binding,
                                            {std::move(buffer)}));
    }
  }

  auto status = FromApiStatus(
      iree_vm_invoke(vmla_executable->context(),
                     vmla_executable->entry_functions()[entry_point],
                     /*policy=*/nullptr, vmla_executable->interface_inputs(),
                     /*outputs=*/nullptr, IREE_ALLOCATOR_SYSTEM),
      IREE_LOC);

  interface->Reset();

  return std::move(status);
}

}  // namespace vmla
}  // namespace hal
}  // namespace iree
