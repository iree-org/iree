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
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/api_detail.h"
#include "iree/hal/device.h"
#include "iree/vm/module_abi_cc.h"

namespace iree {
namespace hal {
namespace {

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
static iree_vm_ref_type_descriptor_t iree_hal_executable_cache_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_layout_descriptor = {
    0};
static iree_vm_ref_type_descriptor_t iree_hal_semaphore_descriptor = {0};

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_module_register_types() {
  static bool has_registered = false;
  if (has_registered) return iree_ok_status();

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
  IREE_VM_REGISTER_CC_TYPE(ExecutableCache, "hal.executable_cache",
                           iree_hal_executable_cache_descriptor);
  IREE_VM_REGISTER_CC_TYPE(ExecutableLayout, "hal.executable_layout",
                           iree_hal_executable_layout_descriptor);
  IREE_VM_REGISTER_CC_TYPE(Semaphore, "hal.semaphore",
                           iree_hal_semaphore_descriptor);

  has_registered = true;
  return iree_ok_status();
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
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable_cache,
                             iree_hal_executable_cache_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable_layout,
                             iree_hal_executable_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_semaphore, iree_hal_semaphore_t);

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

class HALModuleState final {
 public:
  HALModuleState(iree_allocator_t allocator, ref_ptr<Device> shared_device,
                 ref_ptr<ExecutableCache> executable_cache)
      : allocator_(allocator), shared_device_(std::move(shared_device)) {}

  ~HALModuleState() {
    for (auto& ref : deferred_releases_) {
      iree_vm_ref_release(&ref);
    }
    deferred_releases_.clear();
  }

  //===--------------------------------------------------------------------===//
  // Experimental APIs
  //===--------------------------------------------------------------------===//
  // NOTE: Ex* APIs are experimental and likely to be removed soon. Modules
  // using these APIs are not forward compatible.

  StatusOr<vm::ref<iree_hal_device_t>> ExSharedDevice() {
    return vm::retain_ref(
        reinterpret_cast<iree_hal_device_t*>(shared_device_.get()));
  }

  Status ExDeferRelease(absl::optional<vm::opaque_ref> operand) {
    if (operand.has_value()) {
      deferred_releases_.push_back({0});
      iree_vm_ref_move(&operand.value(), &deferred_releases_.back());
    }
    return OkStatus();
  }

  Status ExSubmitAndWait(vm::ref<iree_hal_device_t> device,
                         vm::ref<iree_hal_command_buffer_t> command_buffer) {
    IREE_TRACE_SCOPE0("HALModuleState::ExSubmitAndWait");

    vm::ref<iree_hal_semaphore_t> semaphore;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        device.get(), 0ull, iree_allocator_system(), &semaphore));

    iree_hal_submission_batch_t batch;
    memset(&batch, 0, sizeof(batch));
    batch.command_buffer_count = 1;
    iree_hal_command_buffer_t* command_buffer_ptrs[] = {command_buffer.get()};
    batch.command_buffers = command_buffer_ptrs;
    batch.signal_semaphores.count = 1;
    iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore.get()};
    batch.signal_semaphores.semaphores = semaphore_ptrs;
    uint64_t signal_value = 1ull;
    batch.signal_semaphores.payload_values = &signal_value;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_submit(
        device.get(), IREE_HAL_COMMAND_CATEGORY_ANY, 0, 1, &batch));

    IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait_with_deadline(
        semaphore.get(), 1ull, IREE_TIME_INFINITE_FUTURE));

    for (auto& ref : deferred_releases_) {
      iree_vm_ref_release(&ref);
    }
    deferred_releases_.clear();

    return OkStatus();
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::Allocator
  //===--------------------------------------------------------------------===//

  StatusOr<int32_t> AllocatorComputeSize(
      vm::ref<iree_hal_allocator_t> allocator, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type) {
    iree_device_size_t allocation_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_size(
        allocator.get(), shape.data(), shape.size(), element_type,
        &allocation_size));
    return static_cast<int32_t>(allocation_size);
  }

  StatusOr<int32_t> AllocatorComputeOffset(
      vm::ref<iree_hal_allocator_t> allocator, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type, absl::Span<const int32_t> indices) {
    iree_device_size_t offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_offset(
        allocator.get(), shape.data(), shape.size(), element_type,
        indices.data(), indices.size(), &offset));
    return static_cast<int32_t>(offset);
  }

  StatusOr<std::tuple<int32_t, int32_t>> AllocatorComputeRange(
      vm::ref<iree_hal_allocator_t> allocator, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type,
      absl::Span<const int32_t> start_indices,
      absl::Span<const int32_t> lengths) {
    iree_device_size_t offset = 0;
    iree_device_size_t length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_range(
        allocator.get(), shape.data(), shape.size(), element_type,
        start_indices.data(), start_indices.size(), lengths.data(),
        lengths.size(), &offset, &length));
    return std::make_tuple(static_cast<int32_t>(offset),
                           static_cast<int32_t>(length));
  }

  StatusOr<vm::ref<iree_hal_buffer_t>> AllocatorAllocate(
      vm::ref<iree_hal_allocator_t> allocator,
      iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
      int32_t allocation_size) {
    IREE_TRACE_SCOPE0("HALModuleState::AllocatorAllocate");
    vm::ref<iree_hal_buffer_t> buffer;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        allocator.get(), memory_types, buffer_usage, allocation_size, &buffer));
    return std::move(buffer);
  }

  StatusOr<vm::ref<iree_hal_buffer_t>> AllocatorAllocateConst(
      vm::ref<iree_hal_allocator_t> allocator,
      iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
      absl::Span<const int32_t> shape, iree_hal_element_type_t element_type,
      vm::ref<iree_vm_ro_byte_buffer_t> value) {
    IREE_TRACE_SCOPE0("HALModuleState::AllocatorAllocateConst");

    iree_device_size_t allocation_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_size(
        allocator.get(), shape.data(), shape.size(), element_type,
        &allocation_size));
    if (allocation_size < value->data.data_length) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Constant data is too large for the minimum allocation size";
    }

    vm::ref<iree_hal_buffer_t> buffer;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        allocator.get(), memory_types, buffer_usage, allocation_size, &buffer))
        << "Failed to allocate buffer";

    IREE_RETURN_IF_ERROR(iree_hal_buffer_write_data(
        buffer.get(), 0, value->data.data, value->data.data_length))
        << "Writing constant data";

    return buffer;
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::Buffer
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_allocator_t>> BufferAllocator(
      vm::ref<iree_hal_buffer_t> buffer) {
    return vm::retain_ref(iree_hal_buffer_allocator(buffer.get()));
  }

  StatusOr<vm::ref<iree_hal_buffer_t>> BufferSubspan(
      vm::ref<iree_hal_buffer_t> source_buffer, int32_t source_offset,
      int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferSubspan");
    return UnimplementedErrorBuilder(IREE_LOC) << "BufferSubspan";
  }

  Status BufferFill(vm::ref<iree_hal_buffer_t> target_buffer,
                    int32_t target_offset, int32_t length, int32_t pattern) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferFill");
    return UnimplementedErrorBuilder(IREE_LOC) << "BufferFill";
  }

  Status BufferReadData(vm::ref<iree_hal_buffer_t> source_buffer,
                        int32_t source_offset,
                        vm::ref<iree_vm_rw_byte_buffer_t> target_buffer,
                        int32_t target_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferReadData");
    return UnimplementedErrorBuilder(IREE_LOC) << "BufferReadData";
  }

  Status BufferWriteData(vm::ref<iree_hal_buffer_t> target_buffer,
                         int32_t target_offset,
                         vm::ref<iree_vm_ro_byte_buffer_t> source_buffer,
                         int32_t source_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferWriteData");
    return UnimplementedErrorBuilder(IREE_LOC) << "BufferWriteData";
  }

  Status BufferCopyData(vm::ref<iree_hal_buffer_t> source_buffer,
                        int32_t source_offset,
                        vm::ref<iree_hal_buffer_t> target_buffer,
                        int32_t target_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferCopyData");
    return UnimplementedErrorBuilder(IREE_LOC) << "BufferCopyData";
  }

  StatusOr<int32_t> BufferLoad(vm::ref<iree_hal_buffer_t> source_buffer,
                               int32_t source_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferLoad");

    uint32_t target_buffer = 0;
    if (length > sizeof(target_buffer)) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Length " << length << " exceeds max";
    }

    IREE_RETURN_IF_ERROR(iree_hal_buffer_read_data(
        source_buffer.get(), source_offset, &target_buffer, length))
        << "Read failed";
    return target_buffer;
  }

  Status BufferStore(int32_t value, vm::ref<iree_hal_buffer_t> target_buffer,
                     int32_t target_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferStore");

    if (target_offset + length >
        iree_hal_buffer_byte_length(target_buffer.get())) {
      return OutOfRangeErrorBuilder(IREE_LOC) << "Out of bounds store";
    } else if (length > sizeof(value)) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Length " << length << " exceeds max";
    }

    IREE_RETURN_IF_ERROR(iree_hal_buffer_write_data(
        target_buffer.get(), target_offset, &value, length))
        << "Write failed";
    return OkStatus();
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::BufferView
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_buffer_view_t>> BufferViewCreate(
      vm::ref<iree_hal_buffer_t> buffer, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type) {
    vm::ref<iree_hal_buffer_view_t> buffer_view;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(buffer.get(), shape.data(),
                                                     shape.size(), element_type,
                                                     allocator_, &buffer_view))
        << "Failed to create buffer view";
    return std::move(buffer_view);
  }

  StatusOr<vm::ref<iree_hal_buffer_view_t>> BufferViewSubview(
      vm::ref<iree_hal_buffer_view_t> buffer_view,
      absl::Span<const int32_t> indices, absl::Span<const int32_t> lengths) {
    vm::ref<iree_hal_buffer_view_t> new_buffer_view;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_subview(
        buffer_view.get(), indices.data(), indices.size(), lengths.data(),
        lengths.size(), allocator_, &new_buffer_view))
        << "Failed to create subview";
    return std::move(new_buffer_view);
  }

  StatusOr<vm::ref<iree_hal_buffer_t>> BufferViewBuffer(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    return vm::retain_ref(iree_hal_buffer_view_buffer(buffer_view.get()));
  }

  StatusOr<int32_t> BufferViewByteLength(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    return iree_hal_buffer_view_byte_length(buffer_view.get());
  }

  StatusOr<int32_t> BufferViewComputeOffset(
      vm::ref<iree_hal_buffer_view_t> buffer_view,
      absl::Span<const int32_t> indices) {
    iree_device_size_t offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_offset(
        buffer_view.get(), indices.data(), indices.size(), &offset));
    return offset;
  }

  StatusOr<std::tuple<int32_t, int32_t>> BufferViewComputeRange(
      vm::ref<iree_hal_buffer_view_t> buffer_view,
      absl::Span<const int32_t> start_indices,
      absl::Span<const int32_t> lengths) {
    iree_device_size_t start_offset = 0;
    iree_device_size_t subspan_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_range(
        buffer_view.get(), start_indices.data(), start_indices.size(),
        lengths.data(), lengths.size(), &start_offset, &subspan_length));
    return std::make_tuple<int32_t, int32_t>(
        static_cast<int32_t>(start_offset),
        static_cast<int32_t>(subspan_length));
  }

  StatusOr<int32_t> BufferViewRank(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    return static_cast<int32_t>(
        iree_hal_buffer_view_shape_rank(buffer_view.get()));
  }

  StatusOr<int32_t> BufferViewDim(vm::ref<iree_hal_buffer_view_t> buffer_view,
                                  int32_t index) {
    return static_cast<int32_t>(
        iree_hal_buffer_view_shape_dim(buffer_view.get(), index));
  }

  template <size_t N>
  StatusOr<std::array<int32_t, N>> BufferViewDimsN(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    std::array<int32_t, N> value;
    iree_host_size_t rank = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(buffer_view.get(), N, value.data(), &rank));
    return value;
  }

  StatusOr<std::array<int32_t, 1>> BufferViewDims1(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    return BufferViewDimsN<1>(std::move(buffer_view));
  }

  StatusOr<std::array<int32_t, 2>> BufferViewDims2(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    return BufferViewDimsN<2>(std::move(buffer_view));
  }

  StatusOr<std::array<int32_t, 3>> BufferViewDims3(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    return BufferViewDimsN<3>(std::move(buffer_view));
  }

  StatusOr<std::array<int32_t, 4>> BufferViewDims4(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    return BufferViewDimsN<4>(std::move(buffer_view));
  }

  Status BufferViewTrace(
      absl::Span<const vm::ref<iree_hal_buffer_view_t>> buffer_views) {
    // TODO(hanchung): Have better information for each dump, eg, having StrAttr
    // for each trace event so we can map the dump to dispatch functions easier.
    fprintf(stderr, "=== DEBUG DUMP ===\n");
    for (auto& view : buffer_views) {
      std::string result_str(4096, '\0');
      iree_status_t status;
      do {
        iree_host_size_t actual_length = 0;
        status = iree_hal_buffer_view_format(
            view.get(), /*max_element_count=*/1024, result_str.size() + 1,
            &result_str[0], &actual_length);
        result_str.resize(actual_length);
      } while (iree_status_is_out_of_range(status));
      IREE_RETURN_IF_ERROR(std::move(status));
      fprintf(stderr, "%s\n", result_str.c_str());
    }
    fprintf(stderr, "\n");
    return OkStatus();
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::CommandBuffer
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_command_buffer_t>> CommandBufferCreate(
      vm::ref<iree_hal_device_t> device, iree_hal_command_buffer_mode_t modes,
      iree_hal_command_category_t command_categories) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferCreate");

    vm::ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        device.get(), modes, command_categories, iree_allocator_system(),
        &command_buffer))
        << "Failed to create command buffer";
    return command_buffer;
  }

  Status CommandBufferBegin(vm::ref<iree_hal_command_buffer_t> command_buffer) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferBegin");
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer.get()))
        << "Failed to begin command buffer recording";
    return OkStatus();
  }

  Status CommandBufferEnd(vm::ref<iree_hal_command_buffer_t> command_buffer) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferEnd");
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer.get()))
        << "Failed to end command buffer recording";
    return OkStatus();
  }

  Status CommandBufferExecutionBarrier(
      vm::ref<iree_hal_command_buffer_t> command_buffer,
      iree_hal_execution_stage_t source_stage_mask,
      iree_hal_execution_stage_t target_stage_mask,
      absl::Span<const int32_t> memory_barriers,
      absl::Span<const int32_t> buffer_barriers) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferExecutionBarrier");

    // TODO(benvanik): decode barriers.
    iree_hal_memory_barrier_t global_barrier;
    global_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
    global_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_execution_barrier(
        command_buffer.get(), source_stage_mask, target_stage_mask, 1,
        &global_barrier, 0, nullptr));
    return OkStatus();
  }

  Status CommandBufferFillBuffer(
      vm::ref<iree_hal_command_buffer_t> command_buffer,
      vm::ref<iree_hal_buffer_t> target_buffer, int32_t target_offset,
      int32_t length, uint32_t pattern) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferFillBuffer");
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_fill_buffer(
        command_buffer.get(), target_buffer.get(), target_offset, length,
        &pattern, sizeof(pattern)));
    return OkStatus();
  }

  Status CommandBufferCopyBuffer(
      vm::ref<iree_hal_command_buffer_t> command_buffer,
      vm::ref<iree_hal_buffer_t> source_buffer, int32_t source_offset,
      vm::ref<iree_hal_buffer_t> target_buffer, int32_t target_offset,
      int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferCopyBuffer");
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_copy_buffer(
        command_buffer.get(), source_buffer.get(), source_offset,
        target_buffer.get(), target_offset, length));
    return OkStatus();
  }

  Status CommandBufferPushConstants(
      vm::ref<iree_hal_command_buffer_t> command_buffer,
      vm::ref<iree_hal_executable_layout_t> executable_layout, uint32_t offset,
      absl::Span<const uint32_t> values) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferPushConstants");
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_push_constants(
        command_buffer.get(), executable_layout.get(), offset, values.data(),
        values.size() * sizeof(uint32_t)));
    return OkStatus();
  }

  Status CommandBufferPushDescriptorSet(
      vm::ref<iree_hal_command_buffer_t> command_buffer,
      vm::ref<iree_hal_executable_layout_t> executable_layout, int32_t set,
      absl::Span<const int32_t> binding_ordinals,
      absl::Span<const vm::ref<iree_hal_buffer_t>> binding_buffers,
      absl::Span<const int32_t> binding_offsets,
      absl::Span<const int32_t> binding_lengths) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferPushDescriptorSet");
    absl::InlinedVector<iree_hal_descriptor_set_binding_t, 16> binding_structs(
        binding_ordinals.size());
    for (int i = 0; i < binding_ordinals.size(); ++i) {
      binding_structs[i] = {
          binding_ordinals[i], binding_buffers[i].get(),
          static_cast<iree_device_size_t>(binding_offsets[i]),
          static_cast<iree_device_size_t>(binding_lengths[i])};
      deferred_releases_.push_back(
          iree_hal_buffer_retain_ref(binding_buffers[i].get()));
    }
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_push_descriptor_set(
        command_buffer.get(), executable_layout.get(), set,
        binding_structs.size(), binding_structs.data()));
    return OkStatus();
  }

  Status CommandBufferBindDescriptorSet(
      vm::ref<iree_hal_command_buffer_t> command_buffer,
      vm::ref<iree_hal_executable_layout_t> executable_layout, int32_t set,
      vm::ref<iree_hal_descriptor_set_t> descriptor_set,
      absl::Span<const int32_t> dynamic_offsets) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferBindDescriptorSet");
    absl::InlinedVector<iree_device_size_t, 4> dynamic_offset_values(
        dynamic_offsets.size());
    for (int i = 0; i < dynamic_offsets.size(); ++i) {
      dynamic_offset_values[i] =
          static_cast<iree_device_size_t>(dynamic_offsets[i]);
    }
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_bind_descriptor_set(
        command_buffer.get(), executable_layout.get(), set,
        descriptor_set.get(), dynamic_offset_values.size(),
        dynamic_offset_values.data()));
    return OkStatus();
  }

  Status CommandBufferDispatch(
      vm::ref<iree_hal_command_buffer_t> command_buffer,
      vm::ref<iree_hal_executable_t> executable, int32_t entry_point,
      uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferDispatch");
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
        command_buffer.get(), executable.get(), entry_point, workgroup_x,
        workgroup_y, workgroup_z));
    return OkStatus();
  }

  Status CommandBufferDispatchIndirect(
      vm::ref<iree_hal_command_buffer_t> command_buffer,
      vm::ref<iree_hal_executable_t> executable, int32_t entry_point,
      vm::ref<iree_hal_buffer_t> workgroups_buffer, int32_t workgroups_offset) {
    IREE_TRACE_SCOPE0("HALModuleState::CommandBufferDispatchIndirect");
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch_indirect(
        command_buffer.get(), executable.get(), entry_point,
        workgroups_buffer.get(), workgroups_offset));
    return OkStatus();
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::DescriptorSet
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_descriptor_set_t>> DescriptorSetCreate(
      vm::ref<iree_hal_device_t> device,
      vm::ref<iree_hal_descriptor_set_layout_t> set_layout,
      absl::Span<const int32_t> binding_ordinals,
      absl::Span<const vm::ref<iree_hal_buffer_t>> binding_buffers,
      absl::Span<const int32_t> binding_offsets,
      absl::Span<const int32_t> binding_lengths) {
    IREE_TRACE_SCOPE0("HALModuleState::DescriptorSetCreate");
    absl::InlinedVector<iree_hal_descriptor_set_binding_t, 4> binding_structs(
        binding_ordinals.size());
    for (int i = 0; i < binding_ordinals.size(); ++i) {
      binding_structs[i] = {
          binding_ordinals[i],                                   // binding
          binding_buffers[i].get(),                              // buffer
          static_cast<iree_device_size_t>(binding_offsets[i]),   // offset
          static_cast<iree_device_size_t>(binding_lengths[i])};  // length
    }
    vm::ref<iree_hal_descriptor_set_t> descriptor_set;
    IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_create(
        device.get(), set_layout.get(), binding_structs.size(),
        binding_structs.data(), allocator_, &descriptor_set));
    return std::move(descriptor_set);
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::DescriptorSetLayout
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_descriptor_set_layout_t>> DescriptorSetLayoutCreate(
      vm::ref<iree_hal_device_t> device,
      iree_hal_descriptor_set_layout_usage_type_t usage_type,
      absl::Span<const std::tuple<int32_t, iree_hal_descriptor_type_t,
                                  iree_hal_memory_access_t>>
          bindings) {
    IREE_TRACE_SCOPE0("HALModuleState::DescriptorSetLayoutCreate");
    // TODO(benvanik): custom marshaling for the structs.
    absl::InlinedVector<iree_hal_descriptor_set_layout_binding_t, 4>
        binding_structs(bindings.size());
    for (int i = 0; i < bindings.size(); ++i) {
      binding_structs[i] = {std::get<0>(bindings[i]), std::get<1>(bindings[i]),
                            std::get<2>(bindings[i])};
    }
    vm::ref<iree_hal_descriptor_set_layout_t> descriptor_set_layout;
    IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_layout_create(
        device.get(), usage_type, binding_structs.size(),
        binding_structs.data(), allocator_, &descriptor_set_layout));
    return std::move(descriptor_set_layout);
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::Device
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_allocator_t>> DeviceAllocator(
      vm::ref<iree_hal_device_t> device) {
    return vm::retain_ref(iree_hal_device_allocator(device.get()));
  }

  StatusOr<int32_t> DeviceMatchID(vm::ref<iree_hal_device_t> device,
                                  absl::string_view pattern) {
    iree_string_view_t device_id = iree_hal_device_id(device.get());
    return iree_string_view_match_pattern(
               device_id, iree_string_view_t{pattern.data(), pattern.size()})
               ? 1
               : 0;
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::ExecutableCache
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_executable_cache_t>> ExecutableCacheCreate(
      vm::ref<iree_hal_device_t> device, absl::string_view identifier) {
    IREE_TRACE_SCOPE0("HALModuleState::ExecutableCacheCreate");
    vm::ref<iree_hal_executable_cache_t> executable_cache;
    IREE_RETURN_IF_ERROR(iree_hal_executable_cache_create(
        device.get(), iree_string_view_t{identifier.data(), identifier.size()},
        allocator_, &executable_cache));
    return std::move(executable_cache);
  }

  StatusOr<int32_t> ExecutableCacheSelectFormat(
      vm::ref<iree_hal_executable_cache_t> executable_cache,
      absl::Span<const iree_hal_executable_format_t> available_formats) {
    IREE_TRACE_SCOPE0("HALModuleState::ExecutableCacheSelectFormat");
    for (int i = 0; i < available_formats.size(); ++i) {
      if (iree_hal_executable_cache_can_prepare_format(executable_cache.get(),
                                                       available_formats[i])) {
        return i;
      }
    }
    return -1;
  }

  StatusOr<vm::ref<iree_hal_executable_t>> ExecutableCachePrepare(
      vm::ref<iree_hal_executable_cache_t> executable_cache,
      vm::ref<iree_hal_executable_layout_t> executable_layout,
      iree_hal_executable_caching_mode_t caching_mode,
      vm::ref<iree_vm_ro_byte_buffer_t> executable_data) {
    IREE_TRACE_SCOPE0("HALModuleState::ExecutableCachePrepare");
    vm::ref<iree_hal_executable_t> executable;
    IREE_RETURN_IF_ERROR(iree_hal_executable_cache_prepare_executable(
        executable_cache.get(), executable_layout.get(), caching_mode,
        executable_data->data, allocator_, &executable));
    return std::move(executable);
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::ExecutableLayout
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_executable_layout_t>> ExecutableLayoutCreate(
      vm::ref<iree_hal_device_t> device,
      absl::Span<const vm::ref<iree_hal_descriptor_set_layout_t>> set_layouts,
      int32_t push_constants) {
    IREE_TRACE_SCOPE0("HALModuleState::ExecutableLayoutCreate");
    vm::ref<iree_hal_executable_layout_t> executable_layout;
    IREE_RETURN_IF_ERROR(iree_hal_executable_layout_create(
        device.get(), set_layouts.size(),
        reinterpret_cast<iree_hal_descriptor_set_layout_t**>(
            const_cast<vm::ref<iree_hal_descriptor_set_layout_t>*>(
                set_layouts.data())),
        push_constants, allocator_, &executable_layout));
    return std::move(executable_layout);
  }

  //===--------------------------------------------------------------------===//
  // iree::hal::Semaphore
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_semaphore_t>> SemaphoreCreate(
      vm::ref<iree_hal_device_t> device, uint32_t initial_value) {
    IREE_TRACE_SCOPE0("HALModuleState::SemaphoreCreate");
    vm::ref<iree_hal_semaphore_t> semaphore;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(device.get(), initial_value,
                                                   allocator_, &semaphore));
    return std::move(semaphore);
  }

  StatusOr<std::tuple<int32_t, uint32_t>> SemaphoreQuery(
      vm::ref<iree_hal_semaphore_t> semaphore) {
    uint64_t value = 0;
    iree_status_t query_status =
        iree_hal_semaphore_query(semaphore.get(), &value);
    return std::make_tuple<int32_t, uint32_t>(iree_status_code(query_status),
                                              static_cast<uint32_t>(value));
  }

  Status SemaphoreSignal(vm::ref<iree_hal_semaphore_t> semaphore,
                         uint32_t new_value) {
    IREE_TRACE_SCOPE0("HALModuleState::SemaphoreSignal");
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_signal(semaphore.get(), new_value));
    return OkStatus();
  }

  Status SemaphoreFail(vm::ref<iree_hal_semaphore_t> semaphore,
                       int32_t status_code) {
    IREE_TRACE_SCOPE0("HALModuleState::SemaphoreFail");
    iree_status_t status = iree_make_status(
        static_cast<iree_status_code_t>(status_code & IREE_STATUS_CODE_MASK));
    iree_hal_semaphore_fail(semaphore.get(), status);
    return OkStatus();
  }

  StatusOr<int32_t> SemaphoreAwait(vm::ref<iree_hal_semaphore_t> semaphore,
                                   uint32_t new_value) {
    IREE_TRACE_SCOPE0("HALModuleState::SemaphoreAwait");
    // TODO(benvanik): allow for deadline exceeded returns? We don't allow
    // setting deadlines now (and when we do in the future it'll be the fiber
    // manager that handles them), so any failure indicates total failure.
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait_with_deadline(
        semaphore.get(), new_value, IREE_TIME_INFINITE_FUTURE));
    return OkStatus();
  }

 private:
  iree_allocator_t allocator_;
  ref_ptr<Device> shared_device_;

  std::vector<iree_vm_ref_t> deferred_releases_;
};

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

static const vm::NativeFunction<HALModuleState> kHALModuleFunctions[] = {
    vm::MakeNativeFunction("ex.shared_device", &HALModuleState::ExSharedDevice),
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
    vm::MakeNativeFunction("buffer_view.rank", &HALModuleState::BufferViewRank),
    vm::MakeNativeFunction("buffer_view.dim", &HALModuleState::BufferViewDim),
    vm::MakeNativeFunction("buffer_view.dims.1",
                           &HALModuleState::BufferViewDims1),
    vm::MakeNativeFunction("buffer_view.dims.2",
                           &HALModuleState::BufferViewDims2),
    vm::MakeNativeFunction("buffer_view.dims.3",
                           &HALModuleState::BufferViewDims3),
    vm::MakeNativeFunction("buffer_view.dims.4",
                           &HALModuleState::BufferViewDims4),
    vm::MakeNativeFunction("buffer_view.trace",
                           &HALModuleState::BufferViewTrace),

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
    vm::MakeNativeFunction("command_buffer.push_constants",
                           &HALModuleState::CommandBufferPushConstants),
    vm::MakeNativeFunction("command_buffer.push_descriptor_set",
                           &HALModuleState::CommandBufferPushDescriptorSet),
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
    vm::MakeNativeFunction("device.match.id", &HALModuleState::DeviceMatchID),

    vm::MakeNativeFunction("executable_cache.create",
                           &HALModuleState::ExecutableCacheCreate),
    vm::MakeNativeFunction("executable_cache.select_format",
                           &HALModuleState::ExecutableCacheSelectFormat),
    vm::MakeNativeFunction("executable_cache.prepare",
                           &HALModuleState::ExecutableCachePrepare),

    vm::MakeNativeFunction("executable_layout.create",
                           &HALModuleState::ExecutableLayoutCreate),

    vm::MakeNativeFunction("semaphore.create",
                           &HALModuleState::SemaphoreCreate),
    vm::MakeNativeFunction("semaphore.query", &HALModuleState::SemaphoreQuery),
    vm::MakeNativeFunction("semaphore.signal",
                           &HALModuleState::SemaphoreSignal),
    vm::MakeNativeFunction("semaphore.fail", &HALModuleState::SemaphoreFail),
    vm::MakeNativeFunction("semaphore.await", &HALModuleState::SemaphoreAwait),
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
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = nullptr;
  auto module = std::make_unique<HALModule>(
      allocator, add_ref(reinterpret_cast<Device*>(device)));
  IREE_RETURN_IF_ERROR(module->Initialize());
  *out_module = module.release()->interface();
  return iree_ok_status();
}

}  // namespace
}  // namespace hal
}  // namespace iree
