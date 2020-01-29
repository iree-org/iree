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
#include "iree/hal/vmla/vmla_executable.h"
#include "iree/vm/invocation.h"
#include "iree/vm/variant_list.h"

namespace iree {
namespace hal {
namespace vmla {

namespace {

using UniqueVariantList =
    std::unique_ptr<iree_vm_variant_list_t, void (*)(iree_vm_variant_list_t*)>;

StatusOr<UniqueVariantList> MarshalIO(const DispatchRequest& dispatch_request) {
  iree_vm_variant_list_t* io_list_ptr = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_variant_list_alloc(dispatch_request.bindings.size(),
                                 IREE_ALLOCATOR_SYSTEM, &io_list_ptr),
      IREE_LOC));
  auto io_list = UniqueVariantList(
      io_list_ptr,
      [](iree_vm_variant_list_t* ptr) { iree_vm_variant_list_free(ptr); });

  // TODO(benvanik): marshal buffers into byte buffers.

  return std::move(io_list);
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

  ASSIGN_OR_RETURN(UniqueVariantList io_list, MarshalIO(dispatch_request));
  return FromApiStatus(
      iree_vm_invoke(
          executable->context(),
          executable->entry_functions()[dispatch_request.entry_point],
          /*policy=*/nullptr, io_list.get(), /*outputs=*/nullptr,
          IREE_ALLOCATOR_SYSTEM),
      IREE_LOC);
}

}  // namespace vmla
}  // namespace hal
}  // namespace iree
