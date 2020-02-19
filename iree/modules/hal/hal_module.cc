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

#include "iree/modules/hal/hal_module.h"

#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/api_detail.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/device.h"
#include "iree/vm/module_abi_cc.h"

namespace iree {
namespace hal {
namespace {

// Pretty prints an array, e.g. [1, 2, 3, 4]
static std::string PrettyPrint(absl::Span<const int32_t> arr) {
  return "[" + absl::StrJoin(arr, ",") + "]";
}

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

static iree_vm_ref_type_descriptor_t iree_hal_allocator_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_view_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_command_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_descriptor_set_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_descriptor_set_layout_descriptor =
    {0};
static iree_vm_ref_type_descriptor_t iree_hal_device_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_layout_descriptor = {
    0};

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_module_register_types() {
  static bool has_registered = false;
  if (has_registered) return IREE_STATUS_OK;

  IREE_VM_REGISTER_CC_TYPE(Allocator, "hal.allocator",
                           iree_hal_allocator_descriptor);
  IREE_VM_REGISTER_CC_TYPE(Buffer, "hal.buffer", iree_hal_buffer_descriptor);
  IREE_VM_REGISTER_CC_TYPE(iree_hal_buffer_view, "hal.buffer_view",
                           iree_hal_buffer_view_descriptor);
  IREE_VM_REGISTER_CC_TYPE(CommandBuffer, "hal.command_buffer",
                           iree_hal_command_buffer_descriptor);
  IREE_VM_REGISTER_CC_TYPE(DescriptorSet, "hal.descriptor_set",
                           iree_hal_descriptor_set_descriptor);
  IREE_VM_REGISTER_CC_TYPE(DescriptorSetLayout, "hal.descriptor_set_layout",
                           iree_hal_descriptor_set_layout_descriptor);
  IREE_VM_REGISTER_CC_TYPE(Device, "hal.device", iree_hal_device_descriptor);
  IREE_VM_REGISTER_CC_TYPE(Executable, "hal.executable",
                           iree_hal_executable_descriptor);
  IREE_VM_REGISTER_CC_TYPE(ExecutableLayout, "hal.executable_layout",
                           iree_hal_executable_layout_descriptor);

  has_registered = true;
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// Type wrappers
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_allocator, iree_hal_allocator_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer_view, iree_hal_buffer_view_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_command_buffer,
                             iree_hal_command_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set,
                             iree_hal_descriptor_set_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set_layout,
                             iree_hal_descriptor_set_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_device, iree_hal_device_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable_layout,
                             iree_hal_executable_layout_t);

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

class HALModuleState final {
 public:
  HALModuleState(iree_allocator_t allocator, ref_ptr<Device> shared_device,
                 ref_ptr<ExecutableCache> executable_cache)
      : allocator_(allocator),
        shared_device_(std::move(shared_device)),
        executable_cache_(std::move(executable_cache)) {}

  ~HALModuleState() {
    for (auto& ref : deferred_releases_) {
      iree_vm_ref_release(&ref);
    }
    deferred_releases_.clear();
  }

  // NOTE: Ex* APIs are experimental and likely to be removed soon. Modules
  // using these APIs are not forward compatible.
  StatusOr<vm::ref<iree_hal_device_t>> ExSharedDevice();
  StatusOr<int32_t> ExMatchSupportedExecutableFormat(
      vm::ref<iree_hal_device_t>& device,
      absl::Span<const ExecutableFormat> available_formats);
  StatusOr<vm::ref<iree_hal_executable_t>> ExCacheExecutable(
      vm::ref<iree_hal_device_t>& device, ExecutableFormat executable_format,
      vm::ref<iree_vm_ro_byte_buffer_t>& executable_data);
  Status ExPushBinding(vm::ref<iree_hal_command_buffer_t>& command_buffer,
                       int32_t ordinal, vm::ref<iree_hal_buffer_t>& buffer,
                       absl::Span<const int32_t> shape,
                       iree_hal_element_type_t element_type);
  Status ExDeferRelease(vm::opaque_ref& operand);
  Status ExSubmitAndWait(vm::ref<iree_hal_device_t>& device,
                         vm::ref<iree_hal_command_buffer_t>& command_buffer);

  StatusOr<int32_t> AllocatorComputeSize(
      vm::ref<iree_hal_allocator_t>& allocator, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type);
  StatusOr<int32_t> AllocatorComputeOffset(
      vm::ref<iree_hal_allocator_t>& allocator, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type, absl::Span<const int32_t> indices);
  StatusOr<std::tuple<int32_t, int32_t>> AllocatorComputeRange(
      vm::ref<iree_hal_allocator_t>& allocator, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type,
      absl::Span<const int32_t> start_indices,
      absl::Span<const int32_t> lengths);
  StatusOr<vm::ref<iree_hal_buffer_t>> AllocatorAllocate(
      vm::ref<iree_hal_allocator_t>& allocator,
      iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
      int32_t allocation_size);
  StatusOr<vm::ref<iree_hal_buffer_t>> AllocatorAllocateConst(
      vm::ref<iree_hal_allocator_t>& allocator,
      iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
      absl::Span<const int32_t> shape, iree_hal_element_type_t element_type,
      vm::ref<iree_vm_ro_byte_buffer_t>& value);

  StatusOr<vm::ref<iree_hal_allocator_t>> BufferAllocator(
      vm::ref<iree_hal_buffer_t>& buffer);
  StatusOr<vm::ref<iree_hal_buffer_t>> BufferSubspan(
      vm::ref<iree_hal_buffer_t>& source_buffer, int32_t source_offset,
      int32_t length);
  Status BufferFill(vm::ref<iree_hal_buffer_t>& target_buffer,
                    int32_t target_offset, int32_t length, int32_t pattern);
  Status BufferReadData(vm::ref<iree_hal_buffer_t>& source_buffer,
                        int32_t source_offset,
                        vm::ref<iree_vm_rw_byte_buffer_t>& target_buffer,
                        int32_t target_offset, int32_t length);
  Status BufferWriteData(vm::ref<iree_hal_buffer_t>& target_buffer,
                         int32_t target_offset,
                         vm::ref<iree_vm_ro_byte_buffer_t>& source_buffer,
                         int32_t source_offset, int32_t length);
  Status BufferCopyData(vm::ref<iree_hal_buffer_t>& source_buffer,
                        int32_t source_offset,
                        vm::ref<iree_hal_buffer_t>& target_buffer,
                        int32_t target_offset, int32_t length);
  StatusOr<int32_t> BufferLoad(vm::ref<iree_hal_buffer_t>& source_buffer,
                               int32_t source_offset, int32_t length);
  Status BufferStore(int32_t value, vm::ref<iree_hal_buffer_t>& target_buffer,
                     int32_t target_offset, int32_t length);

  StatusOr<vm::ref<iree_hal_buffer_view_t>> BufferViewCreate(
      vm::ref<iree_hal_buffer_t>& buffer, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type);
  StatusOr<vm::ref<iree_hal_buffer_view_t>> BufferViewSubview(
      vm::ref<iree_hal_buffer_view_t>& buffer_view,
      absl::Span<const int32_t> indices, absl::Span<const int32_t> lengths);
  StatusOr<vm::ref<iree_hal_buffer_t>> BufferViewBuffer(
      vm::ref<iree_hal_buffer_view_t>& buffer_view);
  StatusOr<int32_t> BufferViewByteLength(
      vm::ref<iree_hal_buffer_view_t>& buffer_view);
  StatusOr<int32_t> BufferViewComputeOffset(
      vm::ref<iree_hal_buffer_view_t>& buffer_view,
      absl::Span<const int32_t> indices);
  StatusOr<std::tuple<int32_t, int32_t>> BufferViewComputeRange(
      vm::ref<iree_hal_buffer_view_t>& buffer_view,
      absl::Span<const int32_t> start_indices,
      absl::Span<const int32_t> lengths);

  StatusOr<vm::ref<iree_hal_command_buffer_t>> CommandBufferCreate(
      vm::ref<iree_hal_device_t>& device, iree_hal_command_buffer_mode_t modes,
      iree_hal_command_category_t command_categories);
  Status CommandBufferBegin(vm::ref<iree_hal_command_buffer_t>& command_buffer);
  Status CommandBufferEnd(vm::ref<iree_hal_command_buffer_t>& command_buffer);
  Status CommandBufferExecutionBarrier(
      vm::ref<iree_hal_command_buffer_t>& command_buffer,
      iree_hal_execution_stage_t source_stage_mask,
      iree_hal_execution_stage_t target_stage_mask,
      absl::Span<const int32_t> memory_barriers,
      absl::Span<const int32_t> buffer_barriers);
  Status CommandBufferFillBuffer(
      vm::ref<iree_hal_command_buffer_t>& command_buffer,
      vm::ref<iree_hal_buffer_t>& target_buffer, int32_t target_offset,
      int32_t length, uint32_t pattern);
  Status CommandBufferCopyBuffer(
      vm::ref<iree_hal_command_buffer_t>& command_buffer,
      vm::ref<iree_hal_buffer_t>& source_buffer, int32_t source_offset,
      vm::ref<iree_hal_buffer_t>& target_buffer, int32_t target_offset,
      int32_t length);
  Status CommandBufferBindDescriptorSet(
      vm::ref<iree_hal_command_buffer_t>& command_buffer,
      vm::ref<iree_hal_executable_layout_t>& executable_layout, int32_t set,
      vm::ref<iree_hal_descriptor_set_t>& descriptor_set,
      absl::Span<const int32_t> dynamic_offsets);
  Status CommandBufferDispatch(
      vm::ref<iree_hal_command_buffer_t>& command_buffer,
      vm::ref<iree_hal_executable_t>& executable, int32_t entry_point,
      int32_t workgroup_x, int32_t workgroup_y, int32_t workgroup_z);
  Status CommandBufferDispatchIndirect(
      vm::ref<iree_hal_command_buffer_t>& command_buffer,
      vm::ref<iree_hal_executable_t>& executable, int32_t entry_point,
      vm::ref<iree_hal_buffer_t>& workgroups_buffer, int32_t workgroups_offset);

  StatusOr<vm::ref<iree_hal_descriptor_set_t>> DescriptorSetCreate(
      vm::ref<iree_hal_device_t>& device,
      vm::ref<iree_hal_descriptor_set_layout_t>& set_layout,
      absl::Span<
          std::tuple<int32_t, vm::ref<iree_hal_buffer_t>, int32_t, int32_t>>
          bindings);

  StatusOr<vm::ref<iree_hal_descriptor_set_layout_t>> DescriptorSetLayoutCreate(
      vm::ref<iree_hal_device_t>& device,
      absl::Span<std::tuple<int32_t, iree_hal_descriptor_type_t,
                            iree_hal_memory_access_t>>
          bindings);

  StatusOr<vm::ref<iree_hal_allocator_t>> DeviceAllocator(
      vm::ref<iree_hal_device_t>& device);

  StatusOr<vm::ref<iree_hal_executable_layout_t>> ExecutableLayoutCreate(
      vm::ref<iree_hal_device_t>& device,
      absl::Span<vm::ref<iree_hal_descriptor_set_layout_t>> set_layouts,
      size_t push_constants);

 private:
  iree_allocator_t allocator_;
  ref_ptr<Device> shared_device_;
  ref_ptr<ExecutableCache> executable_cache_;

  std::vector<iree_vm_ref_t> deferred_releases_;

  std::vector<BufferBinding> bindings_;
};

//===----------------------------------------------------------------------===//
// Experimental APIs
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<iree_hal_device_t>> HALModuleState::ExSharedDevice() {
  return vm::retain_ref(
      reinterpret_cast<iree_hal_device_t*>(shared_device_.get()));
}

StatusOr<int32_t> HALModuleState::ExMatchSupportedExecutableFormat(
    vm::ref<iree_hal_device_t>& device,
    absl::Span<const ExecutableFormat> available_formats) {
  IREE_RETURN_IF_NULL(device);
  ExecutableFormat matched_format = 0;
  for (ExecutableFormat format : available_formats) {
    if (executable_cache_->CanPrepareFormat(format)) {
      matched_format = format;
      break;
    }
  }
  return matched_format;
}

StatusOr<vm::ref<iree_hal_executable_t>> HALModuleState::ExCacheExecutable(
    vm::ref<iree_hal_device_t>& device, ExecutableFormat executable_format,
    vm::ref<iree_vm_ro_byte_buffer_t>& executable_data) {
  IREE_TRACE_SCOPE0("HALModuleState::ExCacheExecutable");

  IREE_RETURN_IF_NULL(device);
  IREE_RETURN_IF_NULL(executable_data);

  ExecutableSpec spec;
  spec.format = executable_format;
  spec.executable_data = {executable_data->data.data,
                          executable_data->data.data_length};
  ASSIGN_OR_RETURN(auto executable, executable_cache_->PrepareExecutable(
                                        ExecutableCachingMode::kDefault, spec));

  return vm::assign_ref(
      reinterpret_cast<iree_hal_executable_t*>(executable.release()));
}

Status HALModuleState::ExPushBinding(
    vm::ref<iree_hal_command_buffer_t>& command_buffer, int32_t ordinal,
    vm::ref<iree_hal_buffer_t>& buffer, absl::Span<const int32_t> shape,
    iree_hal_element_type_t element_type) {
  IREE_RETURN_IF_NULL(command_buffer);
  IREE_RETURN_IF_NULL(buffer);
  if (ordinal >= bindings_.size()) {
    bindings_.resize(ordinal + 1);
  }
  auto& binding = bindings_[ordinal];
  binding.access = MemoryAccess::kAll;
  binding.buffer = reinterpret_cast<Buffer*>(buffer.get());
  binding.shape = Shape{shape};
  binding.element_size = iree_hal_element_byte_count(element_type);
  return OkStatus();
}

Status HALModuleState::ExDeferRelease(vm::opaque_ref& operand) {
  if (!iree_vm_ref_is_null(&operand.value)) {
    deferred_releases_.push_back({0});
    iree_vm_ref_move(&operand.value, &deferred_releases_.back());
  }
  return OkStatus();
}

Status HALModuleState::ExSubmitAndWait(
    vm::ref<iree_hal_device_t>& device,
    vm::ref<iree_hal_command_buffer_t>& command_buffer) {
  IREE_TRACE_SCOPE0("HALModuleState::ExSubmitAndWait");
  IREE_RETURN_IF_NULL(device);
  IREE_RETURN_IF_NULL(command_buffer);

  auto* device_ptr = reinterpret_cast<Device*>(device.get());
  auto* queue = device_ptr->dispatch_queues().front();
  ASSIGN_OR_RETURN(auto fence, device_ptr->CreateFence(0u));
  SubmissionBatch batch;
  CommandBuffer* command_buffers[1] = {
      reinterpret_cast<CommandBuffer*>(command_buffer.get())};
  batch.command_buffers = absl::MakeConstSpan(command_buffers);
  RETURN_IF_ERROR(queue->Submit(batch, {fence.get(), 1u}));
  RETURN_IF_ERROR(queue->WaitIdle());

  for (auto& ref : deferred_releases_) {
    iree_vm_ref_release(&ref);
  }
  deferred_releases_.clear();
  bindings_.clear();

  return OkStatus();
}

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

StatusOr<int32_t> HALModuleState::AllocatorComputeSize(
    vm::ref<iree_hal_allocator_t>& allocator, absl::Span<const int32_t> shape,
    iree_hal_element_type_t element_type) {
  IREE_RETURN_IF_NULL(allocator);
  iree_device_size_t allocation_size = 0;
  RETURN_IF_ERROR(FromApiStatus(iree_hal_allocator_compute_size(
                                    allocator.get(), shape.data(), shape.size(),
                                    element_type, &allocation_size),
                                IREE_LOC));
  return static_cast<int32_t>(allocation_size);
}

StatusOr<int32_t> HALModuleState::AllocatorComputeOffset(
    vm::ref<iree_hal_allocator_t>& allocator, absl::Span<const int32_t> shape,
    iree_hal_element_type_t element_type, absl::Span<const int32_t> indices) {
  IREE_RETURN_IF_NULL(allocator);
  iree_device_size_t offset = 0;
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_allocator_compute_offset(
                        allocator.get(), shape.data(), shape.size(),
                        element_type, indices.data(), indices.size(), &offset),
                    IREE_LOC));
  return static_cast<int32_t>(offset);
}

StatusOr<std::tuple<int32_t, int32_t>> HALModuleState::AllocatorComputeRange(
    vm::ref<iree_hal_allocator_t>& allocator, absl::Span<const int32_t> shape,
    iree_hal_element_type_t element_type,
    absl::Span<const int32_t> start_indices,
    absl::Span<const int32_t> lengths) {
  IREE_RETURN_IF_NULL(allocator);
  iree_device_size_t offset = 0;
  iree_device_size_t length = 0;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_allocator_compute_range(
          allocator.get(), shape.data(), shape.size(), element_type,
          start_indices.data(), start_indices.size(), lengths.data(),
          lengths.size(), &offset, &length),
      IREE_LOC));
  return std::make_tuple(static_cast<int32_t>(offset),
                         static_cast<int32_t>(length));
}

StatusOr<vm::ref<iree_hal_buffer_t>> HALModuleState::AllocatorAllocate(
    vm::ref<iree_hal_allocator_t>& allocator,
    iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
    int32_t allocation_size) {
  IREE_TRACE_SCOPE0("HALModuleState::AllocatorAllocate");
  IREE_RETURN_IF_NULL(allocator);
  vm::ref<iree_hal_buffer_t> buffer;
  RETURN_IF_ERROR(FromApiStatus(iree_hal_allocator_allocate_buffer(
                                    allocator.get(), memory_types, buffer_usage,
                                    allocation_size, &buffer),
                                IREE_LOC));
  return std::move(buffer);
}

StatusOr<vm::ref<iree_hal_buffer_t>> HALModuleState::AllocatorAllocateConst(
    vm::ref<iree_hal_allocator_t>& allocator,
    iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
    absl::Span<const int32_t> shape, iree_hal_element_type_t element_type,
    vm::ref<iree_vm_ro_byte_buffer_t>& value) {
  IREE_TRACE_SCOPE0("HALModuleState::AllocatorAllocateConst");
  IREE_RETURN_IF_NULL(allocator);
  IREE_RETURN_IF_NULL(value);

  iree_device_size_t allocation_size = 0;
  RETURN_IF_ERROR(FromApiStatus(iree_hal_allocator_compute_size(
                                    allocator.get(), shape.data(), shape.size(),
                                    element_type, &allocation_size),
                                IREE_LOC));
  if (allocation_size < value->data.data_length) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Constant data is too larger for the minimum allocation size";
  }

  vm::ref<iree_hal_buffer_t> buffer;
  RETURN_IF_ERROR(FromApiStatus(iree_hal_allocator_allocate_buffer(
                                    allocator.get(), memory_types, buffer_usage,
                                    allocation_size, &buffer),
                                IREE_LOC))
      << "Failed to allocate buffer";

  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_buffer_write_data(buffer.get(), 0, value->data.data,
                                 value->data.data_length),
      IREE_LOC))
      << "Writing constant data";

  return buffer;
}

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<iree_hal_allocator_t>> HALModuleState::BufferAllocator(
    vm::ref<iree_hal_buffer_t>& buffer) {
  IREE_RETURN_IF_NULL(buffer);
  return vm::retain_ref(iree_hal_buffer_allocator(buffer.get()));
}

StatusOr<vm::ref<iree_hal_buffer_t>> HALModuleState::BufferSubspan(
    vm::ref<iree_hal_buffer_t>& source_buffer, int32_t source_offset,
    int32_t length) {
  IREE_TRACE_SCOPE0("HALModuleState::BufferSubspan");
  IREE_RETURN_IF_NULL(source_buffer);
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferSubspan";
}

Status HALModuleState::BufferFill(vm::ref<iree_hal_buffer_t>& target_buffer,
                                  int32_t target_offset, int32_t length,
                                  int32_t pattern) {
  IREE_TRACE_SCOPE0("HALModuleState::BufferFill");
  IREE_RETURN_IF_NULL(target_buffer);
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferFill";
}

Status HALModuleState::BufferReadData(
    vm::ref<iree_hal_buffer_t>& source_buffer, int32_t source_offset,
    vm::ref<iree_vm_rw_byte_buffer_t>& target_buffer, int32_t target_offset,
    int32_t length) {
  IREE_TRACE_SCOPE0("HALModuleState::BufferReadData");
  IREE_RETURN_IF_NULL(source_buffer);
  IREE_RETURN_IF_NULL(target_buffer);
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferReadData";
}

Status HALModuleState::BufferWriteData(
    vm::ref<iree_hal_buffer_t>& target_buffer, int32_t target_offset,
    vm::ref<iree_vm_ro_byte_buffer_t>& source_buffer, int32_t source_offset,
    int32_t length) {
  IREE_TRACE_SCOPE0("HALModuleState::BufferWriteData");
  IREE_RETURN_IF_NULL(target_buffer);
  IREE_RETURN_IF_NULL(source_buffer);
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferWriteData";
}

Status HALModuleState::BufferCopyData(vm::ref<iree_hal_buffer_t>& source_buffer,
                                      int32_t source_offset,
                                      vm::ref<iree_hal_buffer_t>& target_buffer,
                                      int32_t target_offset, int32_t length) {
  IREE_TRACE_SCOPE0("HALModuleState::BufferCopyData");
  IREE_RETURN_IF_NULL(source_buffer);
  IREE_RETURN_IF_NULL(target_buffer);
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferCopyData";
}

StatusOr<int32_t> HALModuleState::BufferLoad(
    vm::ref<iree_hal_buffer_t>& source_buffer, int32_t source_offset,
    int32_t length) {
  IREE_TRACE_SCOPE0("HALModuleState::BufferLoad");
  IREE_RETURN_IF_NULL(source_buffer);

  uint32_t target_buffer = 0;
  if (length > sizeof(target_buffer)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Length " << length << " exceeds max";
  }

  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_buffer_read_data(source_buffer.get(), source_offset,
                                &target_buffer, length),
      IREE_LOC))
      << "Read failed";
  return target_buffer;
}

Status HALModuleState::BufferStore(int32_t value,
                                   vm::ref<iree_hal_buffer_t>& target_buffer,
                                   int32_t target_offset, int32_t length) {
  IREE_TRACE_SCOPE0("HALModuleState::BufferStore");
  IREE_RETURN_IF_NULL(target_buffer);

  if (target_offset + length >
      iree_hal_buffer_byte_length(target_buffer.get())) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Out of bounds store";
  } else if (length > sizeof(value)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Length " << length << " exceeds max";
  }

  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_buffer_write_data(target_buffer.get(),
                                               target_offset, &value, length),
                    IREE_LOC))
      << "Write failed";
  return OkStatus();
}

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<iree_hal_buffer_view_t>> HALModuleState::BufferViewCreate(
    vm::ref<iree_hal_buffer_t>& buffer, absl::Span<const int32_t> shape,
    iree_hal_element_type_t element_type) {
  IREE_RETURN_IF_NULL(buffer);
  vm::ref<iree_hal_buffer_view_t> buffer_view;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_buffer_view_create(buffer.get(), shape.data(), shape.size(),
                                  element_type, allocator_, &buffer_view),
      IREE_LOC))
      << "Failed to create buffer view";
  return std::move(buffer_view);
}

StatusOr<vm::ref<iree_hal_buffer_view_t>> HALModuleState::BufferViewSubview(
    vm::ref<iree_hal_buffer_view_t>& buffer_view,
    absl::Span<const int32_t> indices, absl::Span<const int32_t> lengths) {
  IREE_RETURN_IF_NULL(buffer_view);
  vm::ref<iree_hal_buffer_view_t> new_buffer_view;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_buffer_view_subview(
          buffer_view.get(), indices.data(), indices.size(), lengths.data(),
          lengths.size(), allocator_, &new_buffer_view),
      IREE_LOC))
      << "Failed to create subview";
  return std::move(new_buffer_view);
}

StatusOr<vm::ref<iree_hal_buffer_t>> HALModuleState::BufferViewBuffer(
    vm::ref<iree_hal_buffer_view_t>& buffer_view) {
  IREE_RETURN_IF_NULL(buffer_view);
  return vm::retain_ref(iree_hal_buffer_view_buffer(buffer_view.get()));
}

StatusOr<int32_t> HALModuleState::BufferViewByteLength(
    vm::ref<iree_hal_buffer_view_t>& buffer_view) {
  IREE_RETURN_IF_NULL(buffer_view);
  return iree_hal_buffer_view_byte_length(buffer_view.get());
}

StatusOr<int32_t> HALModuleState::BufferViewComputeOffset(
    vm::ref<iree_hal_buffer_view_t>& buffer_view,
    absl::Span<const int32_t> indices) {
  IREE_RETURN_IF_NULL(buffer_view);
  iree_device_size_t offset = 0;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_buffer_view_compute_offset(buffer_view.get(), indices.data(),
                                          indices.size(), &offset),
      IREE_LOC));
  return offset;
}

StatusOr<std::tuple<int32_t, int32_t>> HALModuleState::BufferViewComputeRange(
    vm::ref<iree_hal_buffer_view_t>& buffer_view,
    absl::Span<const int32_t> start_indices,
    absl::Span<const int32_t> lengths) {
  IREE_RETURN_IF_NULL(buffer_view);
  iree_device_size_t start_offset = 0;
  iree_device_size_t subspan_length = 0;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_buffer_view_compute_range(
          buffer_view.get(), start_indices.data(), start_indices.size(),
          lengths.data(), lengths.size(), &start_offset, &subspan_length),
      IREE_LOC));
  return std::make_tuple<int32_t, int32_t>(start_offset, subspan_length);
}

//===----------------------------------------------------------------------===//
// iree::hal::CommandBuffer
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<iree_hal_command_buffer_t>>
HALModuleState::CommandBufferCreate(
    vm::ref<iree_hal_device_t>& device, iree_hal_command_buffer_mode_t modes,
    iree_hal_command_category_t command_categories) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferCreate");
  IREE_RETURN_IF_NULL(device);

  vm::ref<iree_hal_command_buffer_t> command_buffer;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_command_buffer_create(device.get(), modes, command_categories,
                                     IREE_ALLOCATOR_SYSTEM, &command_buffer),
      IREE_LOC))
      << "Failed to create command buffer";
  return command_buffer;
}

Status HALModuleState::CommandBufferBegin(
    vm::ref<iree_hal_command_buffer_t>& command_buffer) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferBegin");
  IREE_RETURN_IF_NULL(command_buffer);
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_command_buffer_begin(command_buffer.get()), IREE_LOC))
      << "Failed to begin command buffer recording";
  return OkStatus();
}

Status HALModuleState::CommandBufferEnd(
    vm::ref<iree_hal_command_buffer_t>& command_buffer) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferEnd");
  IREE_RETURN_IF_NULL(command_buffer);
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_command_buffer_end(command_buffer.get()), IREE_LOC))
      << "Failed to end command buffer recording";
  return OkStatus();
}

Status HALModuleState::CommandBufferExecutionBarrier(
    vm::ref<iree_hal_command_buffer_t>& command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    absl::Span<const int32_t> memory_barriers,
    absl::Span<const int32_t> buffer_barriers) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferExecutionBarrier");
  IREE_RETURN_IF_NULL(command_buffer);

  // TODO(benvanik): decode barriers.
  iree_hal_memory_barrier_t global_barrier;
  global_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
  global_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_command_buffer_execution_barrier(
                        command_buffer.get(), source_stage_mask,
                        target_stage_mask, 1, &global_barrier, 0, nullptr),
                    IREE_LOC));
  return OkStatus();
}

Status HALModuleState::CommandBufferFillBuffer(
    vm::ref<iree_hal_command_buffer_t>& command_buffer,
    vm::ref<iree_hal_buffer_t>& target_buffer, int32_t target_offset,
    int32_t length, uint32_t pattern) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferFillBuffer");
  IREE_RETURN_IF_NULL(command_buffer);
  IREE_RETURN_IF_NULL(target_buffer);
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_command_buffer_fill_buffer(command_buffer.get(),
                                          target_buffer.get(), target_offset,
                                          length, &pattern, sizeof(pattern)),
      IREE_LOC));
  return OkStatus();
}

Status HALModuleState::CommandBufferCopyBuffer(
    vm::ref<iree_hal_command_buffer_t>& command_buffer,
    vm::ref<iree_hal_buffer_t>& source_buffer, int32_t source_offset,
    vm::ref<iree_hal_buffer_t>& target_buffer, int32_t target_offset,
    int32_t length) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferCopyBuffer");
  IREE_RETURN_IF_NULL(command_buffer);
  IREE_RETURN_IF_NULL(source_buffer);
  IREE_RETURN_IF_NULL(target_buffer);
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_command_buffer_copy_buffer(
          command_buffer.get(), source_buffer.get(), source_offset,
          target_buffer.get(), target_offset, length),
      IREE_LOC));
  return OkStatus();
}

Status HALModuleState::CommandBufferBindDescriptorSet(
    vm::ref<iree_hal_command_buffer_t>& command_buffer,
    vm::ref<iree_hal_executable_layout_t>& executable_layout, int32_t set,
    vm::ref<iree_hal_descriptor_set_t>& descriptor_set,
    absl::Span<const int32_t> dynamic_offsets) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferBindDescriptorSet");
  IREE_RETURN_IF_NULL(command_buffer);
  IREE_RETURN_IF_NULL(executable_layout);
  IREE_RETURN_IF_NULL(descriptor_set);
  absl::InlinedVector<iree_device_size_t, 4> dynamic_offset_values(
      dynamic_offsets.size());
  for (int i = 0; i < dynamic_offsets.size(); ++i) {
    dynamic_offset_values[i] =
        static_cast<iree_device_size_t>(dynamic_offsets[i]);
  }
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_command_buffer_bind_descriptor_set(
                        command_buffer.get(), executable_layout.get(), set,
                        descriptor_set.get(), dynamic_offset_values.size(),
                        dynamic_offset_values.data()),
                    IREE_LOC));
  return OkStatus();
}

Status HALModuleState::CommandBufferDispatch(
    vm::ref<iree_hal_command_buffer_t>& command_buffer,
    vm::ref<iree_hal_executable_t>& executable, int32_t entry_point,
    int32_t workgroup_x, int32_t workgroup_y, int32_t workgroup_z) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferDispatch");
  IREE_RETURN_IF_NULL(command_buffer);
  IREE_RETURN_IF_NULL(executable);

  DispatchRequest dispatch_request;
  dispatch_request.executable = reinterpret_cast<Executable*>(executable.get());
  dispatch_request.entry_point = entry_point;
  dispatch_request.workload = {workgroup_x, workgroup_y, workgroup_z};
  dispatch_request.bindings = bindings_;
  RETURN_IF_ERROR(reinterpret_cast<CommandBuffer*>(command_buffer.get())
                      ->Dispatch(dispatch_request));

  bindings_.clear();

  return OkStatus();
}

Status HALModuleState::CommandBufferDispatchIndirect(
    vm::ref<iree_hal_command_buffer_t>& command_buffer,
    vm::ref<iree_hal_executable_t>& executable, int32_t entry_point,
    vm::ref<iree_hal_buffer_t>& workgroups_buffer, int32_t workgroups_offset) {
  IREE_TRACE_SCOPE0("HALModuleState::CommandBufferDispatchIndirect");
  IREE_RETURN_IF_NULL(command_buffer);
  IREE_RETURN_IF_NULL(executable);
  IREE_RETURN_IF_NULL(workgroups_buffer);
  return UnimplementedErrorBuilder(IREE_LOC) << "CommandBufferDispatchIndirect";
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSet
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<iree_hal_descriptor_set_t>>
HALModuleState::DescriptorSetCreate(
    vm::ref<iree_hal_device_t>& device,
    vm::ref<iree_hal_descriptor_set_layout_t>& set_layout,
    absl::Span<
        std::tuple<int32_t, vm::ref<iree_hal_buffer_t>, int32_t, int32_t>>
        bindings) {
  IREE_TRACE_SCOPE0("HALModuleState::DescriptorSetCreate");
  IREE_RETURN_IF_NULL(device);
  IREE_RETURN_IF_NULL(set_layout);
  // TODO(benvanik): custom packing logic for the structs so that we marshal
  // the VM args directly into them instead of needing this temporary array.
  absl::InlinedVector<iree_hal_descriptor_set_binding_t, 4> binding_structs(
      bindings.size());
  for (int i = 0; i < bindings.size(); ++i) {
    binding_structs[i] = {
        std::get<0>(bindings[i]), std::get<1>(bindings[i]).get(),
        static_cast<iree_device_size_t>(std::get<2>(bindings[i])),
        static_cast<iree_device_size_t>(std::get<3>(bindings[i]))};
  }
  vm::ref<iree_hal_descriptor_set_t> descriptor_set;
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_descriptor_set_create(
                        device.get(), set_layout.get(), binding_structs.size(),
                        binding_structs.data(), allocator_, &descriptor_set),
                    IREE_LOC));
  return std::move(descriptor_set);
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSetLayout
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<iree_hal_descriptor_set_layout_t>>
HALModuleState::DescriptorSetLayoutCreate(
    vm::ref<iree_hal_device_t>& device,
    absl::Span<std::tuple<int32_t, iree_hal_descriptor_type_t,
                          iree_hal_memory_access_t>>
        bindings) {
  IREE_TRACE_SCOPE0("HALModuleState::DescriptorSetLayoutCreate");
  IREE_RETURN_IF_NULL(device);
  // TODO(benvanik): custom marshaling for the structs.
  absl::InlinedVector<iree_hal_descriptor_set_layout_binding_t, 4>
      binding_structs(bindings.size());
  for (int i = 0; i < bindings.size(); ++i) {
    binding_structs[i] = {std::get<0>(bindings[i]), std::get<1>(bindings[i]),
                          std::get<2>(bindings[i])};
  }
  vm::ref<iree_hal_descriptor_set_layout_t> descriptor_set_layout;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_descriptor_set_layout_create(
          device.get(), binding_structs.size(), binding_structs.data(),
          allocator_, &descriptor_set_layout),
      IREE_LOC));
  return std::move(descriptor_set_layout);
}

//===----------------------------------------------------------------------===//
// iree::hal::Device
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<iree_hal_allocator_t>> HALModuleState::DeviceAllocator(
    vm::ref<iree_hal_device_t>& device) {
  IREE_RETURN_IF_NULL(device);
  return vm::retain_ref(iree_hal_device_allocator(device.get()));
}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableLayout
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<iree_hal_executable_layout_t>>
HALModuleState::ExecutableLayoutCreate(
    vm::ref<iree_hal_device_t>& device,
    absl::Span<vm::ref<iree_hal_descriptor_set_layout_t>> set_layouts,
    size_t push_constants) {
  IREE_TRACE_SCOPE0("HALModuleState::ExecutableLayoutCreate");
  IREE_RETURN_IF_NULL(device);
  vm::ref<iree_hal_executable_layout_t> executable_layout;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_executable_layout_create(
          device.get(), set_layouts.size(),
          reinterpret_cast<const iree_hal_descriptor_set_layout_t**>(
              set_layouts.data()),
          push_constants, allocator_, &executable_layout),
      IREE_LOC));
  return std::move(executable_layout);
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

static const vm::NativeFunction<HALModuleState> kHALModuleFunctions[] = {
    vm::MakeNativeFunction("ex.shared_device", &HALModuleState::ExSharedDevice),
    vm::MakeNativeFunction("ex.match_supported_executable_format",
                           &HALModuleState::ExMatchSupportedExecutableFormat),
    vm::MakeNativeFunction("ex.cache_executable",
                           &HALModuleState::ExCacheExecutable),
    vm::MakeNativeFunction("ex.push_binding", &HALModuleState::ExPushBinding),
    vm::MakeNativeFunction("ex.defer_release", &HALModuleState::ExDeferRelease),
    vm::MakeNativeFunction("ex.submit_and_wait",
                           &HALModuleState::ExSubmitAndWait),

    vm::MakeNativeFunction("allocator.compute_size",
                           &HALModuleState::AllocatorComputeSize),
    vm::MakeNativeFunction("allocator.compute_offset",
                           &HALModuleState::AllocatorComputeOffset),
    vm::MakeNativeFunction("allocator.compute_range",
                           &HALModuleState::AllocatorComputeRange),
    vm::MakeNativeFunction("allocator.allocate",
                           &HALModuleState::AllocatorAllocate),
    vm::MakeNativeFunction("allocator.allocate.const",
                           &HALModuleState::AllocatorAllocateConst),

    vm::MakeNativeFunction("buffer.allocator",
                           &HALModuleState::BufferAllocator),
    vm::MakeNativeFunction("buffer.subspan", &HALModuleState::BufferSubspan),
    vm::MakeNativeFunction("buffer.fill", &HALModuleState::BufferFill),
    vm::MakeNativeFunction("buffer.read_data", &HALModuleState::BufferReadData),
    vm::MakeNativeFunction("buffer.write_data",
                           &HALModuleState::BufferWriteData),
    vm::MakeNativeFunction("buffer.copy_data", &HALModuleState::BufferCopyData),
    vm::MakeNativeFunction("buffer.load", &HALModuleState::BufferLoad),
    vm::MakeNativeFunction("buffer.store", &HALModuleState::BufferStore),

    vm::MakeNativeFunction("buffer_view.create",
                           &HALModuleState::BufferViewCreate),
    vm::MakeNativeFunction("buffer_view.subview",
                           &HALModuleState::BufferViewSubview),
    vm::MakeNativeFunction("buffer_view.buffer",
                           &HALModuleState::BufferViewBuffer),
    vm::MakeNativeFunction("buffer_view.byte_length",
                           &HALModuleState::BufferViewByteLength),
    vm::MakeNativeFunction("buffer_view.compute_offset",
                           &HALModuleState::BufferViewComputeOffset),
    vm::MakeNativeFunction("buffer_view.compute_range",
                           &HALModuleState::BufferViewComputeRange),

    vm::MakeNativeFunction("command_buffer.create",
                           &HALModuleState::CommandBufferCreate),
    vm::MakeNativeFunction("command_buffer.begin",
                           &HALModuleState::CommandBufferBegin),
    vm::MakeNativeFunction("command_buffer.end",
                           &HALModuleState::CommandBufferEnd),
    vm::MakeNativeFunction("command_buffer.execution_barrier",
                           &HALModuleState::CommandBufferExecutionBarrier),
    vm::MakeNativeFunction("command_buffer.fill_buffer",
                           &HALModuleState::CommandBufferFillBuffer),
    vm::MakeNativeFunction("command_buffer.copy_buffer",
                           &HALModuleState::CommandBufferCopyBuffer),
    vm::MakeNativeFunction("command_buffer.bind_descriptor_set",
                           &HALModuleState::CommandBufferBindDescriptorSet),
    vm::MakeNativeFunction("command_buffer.dispatch",
                           &HALModuleState::CommandBufferDispatch),
    vm::MakeNativeFunction("command_buffer.dispatch.indirect",
                           &HALModuleState::CommandBufferDispatchIndirect),
    vm::MakeNativeFunction("descriptor_set.create",
                           &HALModuleState::DescriptorSetCreate),
    vm::MakeNativeFunction("descriptor_set_layout.create",
                           &HALModuleState::DescriptorSetLayoutCreate),
    vm::MakeNativeFunction("device.allocator",
                           &HALModuleState::DeviceAllocator),
    vm::MakeNativeFunction("executable_layout.create",
                           &HALModuleState::ExecutableLayoutCreate),
};

class HALModule final : public vm::NativeModule<HALModuleState> {
 public:
  HALModule(iree_allocator_t allocator, ref_ptr<Device> shared_device)
      : vm::NativeModule<HALModuleState>(
            "hal", allocator, absl::MakeConstSpan(kHALModuleFunctions)),
        shared_device_(std::move(shared_device)) {}
  ~HALModule() = default;

  Status Initialize() {
    IREE_TRACE_SCOPE0("HALModule::Initialize");

    executable_cache_ = shared_device_->CreateExecutableCache();

    return OkStatus();
  }

  StatusOr<std::unique_ptr<HALModuleState>> CreateState(
      iree_allocator_t allocator) override {
    IREE_TRACE_SCOPE0("HALModule::CreateState");
    auto state = std::make_unique<HALModuleState>(
        allocator, add_ref(shared_device_), add_ref(executable_cache_));
    // TODO(benvanik): allocate context-specific variables (allocator pool,
    // etc).
    return state;
  }

 private:
  ref_ptr<Device> shared_device_;
  ref_ptr<ExecutableCache> executable_cache_;
};

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_module_create(iree_hal_device_t* device, iree_allocator_t allocator,
                       iree_vm_module_t** out_module) {
  if (!out_module) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module = nullptr;
  auto module = std::make_unique<HALModule>(
      allocator, add_ref(reinterpret_cast<Device*>(device)));
  IREE_API_RETURN_IF_ERROR(module->Initialize());
  *out_module = module.release()->interface();
  return IREE_STATUS_OK;
}

}  // namespace
}  // namespace hal
}  // namespace iree
