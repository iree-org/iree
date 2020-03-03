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
#include "iree/vm/variant_list.h"

namespace iree {
namespace hal {
namespace vmla {

namespace {

Status MarshalIO(Interface* interface,
                 const DispatchRequest& dispatch_request) {
  IREE_TRACE_SCOPE0("VMLACommandProcessor::MarshalIO");

  for (int i = 0; i < dispatch_request.bindings.size(); ++i) {
    const auto& binding = dispatch_request.bindings[i];
    void* data = static_cast<HostBuffer*>(binding.buffer->allocated_buffer())
                     ->mutable_data();
    data = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(data) +
                                   binding.buffer->byte_offset());
    ASSIGN_OR_RETURN(auto buffer,
                     Buffer::WrapMutable(data, binding.buffer->byte_length(),
                                         IREE_ALLOCATOR_NULL));
    RETURN_IF_ERROR(interface->SetBinding(0, i, {std::move(buffer)}));
  }

  return OkStatus();
}

}  // namespace

VMLACommandProcessor::VMLACommandProcessor(
    Allocator* allocator, CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories)
    : HostLocalCommandProcessor(allocator, mode, command_categories) {}

VMLACommandProcessor::~VMLACommandProcessor() = default;

Status VMLACommandProcessor::Dispatch(const DispatchRequest& dispatch_request) {
  IREE_TRACE_SCOPE0("VMLACommandProcessor::Dispatch");

  auto* executable = static_cast<VMLAExecutable*>(dispatch_request.executable);
  if (dispatch_request.entry_point >= executable->entry_functions().size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Invalid entry point ordinal " << dispatch_request.entry_point;
  }

  RETURN_IF_ERROR(MarshalIO(executable->interface(), dispatch_request));
  return FromApiStatus(
      iree_vm_invoke(
          executable->context(),
          executable->entry_functions()[dispatch_request.entry_point],
          /*policy=*/nullptr, executable->interface_inputs(),
          /*outputs=*/nullptr, IREE_ALLOCATOR_SYSTEM),
      IREE_LOC);
}

}  // namespace vmla
}  // namespace hal
}  // namespace iree
