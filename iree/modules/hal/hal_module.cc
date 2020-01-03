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
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/device.h"

namespace iree {
namespace hal {
namespace {

// TODO(benvanik): remove when we have proper ABI wrapping.
static void ResetStackFrame(iree_vm_stack_frame_t* frame) {
  frame->return_registers = nullptr;
  for (int i = 0; i < frame->registers.ref_register_count; ++i) {
    iree_vm_ref_release(&frame->registers.ref[i]);
  }
}

// Pretty prints an array, e.g. [1, 2, 3, 4]
static std::string PrettyPrint(absl::Span<const int32_t> arr) {
  return "[" + absl::StrJoin(arr, ",") + "]";
}

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

static iree_vm_ref_type_descriptor_t iree_hal_allocator_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_command_buffer_descriptor = {0};
// TODO(benvanik): descriptor sets.
// static iree_vm_ref_type_descriptor_t
// iree_hal_descriptor_set_layout_descriptor =
//     {0};
static iree_vm_ref_type_descriptor_t iree_hal_device_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_descriptor = {0};

#define IREE_HAL_REGISTER_CC_TYPE(type, name, descriptor) \
  descriptor.type_name = iree_make_cstring_view(name);    \
  descriptor.offsetof_counter = type::offsetof_counter(); \
  descriptor.destroy = type::DirectDestroy;               \
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&descriptor));

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_module_register_types() {
  static bool has_registered = false;
  if (has_registered) return IREE_STATUS_OK;

  IREE_HAL_REGISTER_CC_TYPE(Allocator, "hal.allocator",
                            iree_hal_allocator_descriptor);
  IREE_HAL_REGISTER_CC_TYPE(Buffer, "hal.buffer", iree_hal_buffer_descriptor);
  IREE_HAL_REGISTER_CC_TYPE(CommandBuffer, "hal.command_buffer",
                            iree_hal_command_buffer_descriptor);
  // TODO(benvanik): descriptor sets.
  // IREE_HAL_REGISTER_CC_TYPE(DescriptorSetLayout, "hal.descriptor_set_layout",
  //                           iree_hal_descriptor_set_layout_descriptor);
  IREE_HAL_REGISTER_CC_TYPE(Device, "hal.device", iree_hal_device_descriptor);
  IREE_HAL_REGISTER_CC_TYPE(Executable, "hal.executable",
                            iree_hal_executable_descriptor);

  has_registered = true;
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// Type wrappers
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_allocator, iree_hal_allocator_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_command_buffer,
                             iree_hal_command_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_device, iree_hal_device_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

class HALModule final {
 public:
  static StatusOr<std::unique_ptr<HALModule>> Create(iree_allocator_t allocator,
                                                     ref_ptr<Device> device) {
    auto module = absl::WrapUnique(new HALModule(allocator, std::move(device)));
    RETURN_IF_ERROR(module->Initialize());
    return module;
  }

  static HALModule* FromPointer(void* ptr) {
    return reinterpret_cast<HALModule*>(ptr);
  }

  ~HALModule() = default;

  iree_vm_module_t* interface() { return &interface_; }
  const ref_ptr<Device>& shared_device() const { return shared_device_; }
  const ref_ptr<ExecutableCache>& executable_cache() const {
    return executable_cache_;
  }

 private:
  HALModule(iree_allocator_t allocator, ref_ptr<Device> shared_device)
      : allocator_(allocator), shared_device_(std::move(shared_device)) {
    iree_vm_module_init(&interface_, this);
  }

  Status Initialize();

  iree_vm_module_t interface_;
  iree_allocator_t allocator_;
  ref_ptr<Device> shared_device_;
  ref_ptr<ExecutableCache> executable_cache_;
};

class HALModuleState final {
 public:
  static HALModuleState* FromPointer(void* ptr) {
    return reinterpret_cast<HALModuleState*>(ptr);
  }

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
  Status ExSharedDevice(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status ExMatchSupportedExecutableFormat(iree_vm_stack_t* stack,
                                          iree_vm_stack_frame_t* frame);
  Status ExCacheExecutable(iree_vm_stack_t* stack,
                           iree_vm_stack_frame_t* frame);
  Status ExPushBinding(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status ExExecutableDescriptorSetLayout(iree_vm_stack_t* stack,
                                         iree_vm_stack_frame_t* frame);
  Status ExDeferRelease(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status ExSubmitAndWait(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);

  Status AllocatorComputeSize(iree_vm_stack_t* stack,
                              iree_vm_stack_frame_t* frame);
  Status AllocatorAllocate(iree_vm_stack_t* stack,
                           iree_vm_stack_frame_t* frame);
  Status AllocatorAllocateConst(iree_vm_stack_t* stack,
                                iree_vm_stack_frame_t* frame);
  Status AllocatorAllocateShaped(iree_vm_stack_t* stack,
                                 iree_vm_stack_frame_t* frame);

  Status BufferSubspan(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status BufferFill(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status BufferReadData(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status BufferWriteData(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status BufferCopyData(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status BufferLoad(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status BufferStore(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);

  Status BufferViewComputeOffset(iree_vm_stack_t* stack,
                                 iree_vm_stack_frame_t* frame);
  Status BufferViewComputeLength(iree_vm_stack_t* stack,
                                 iree_vm_stack_frame_t* frame);
  Status BufferViewComputeRange(iree_vm_stack_t* stack,
                                iree_vm_stack_frame_t* frame);
  Status BufferViewSlice(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);

  Status CommandBufferCreate(iree_vm_stack_t* stack,
                             iree_vm_stack_frame_t* frame);
  Status CommandBufferBegin(iree_vm_stack_t* stack,
                            iree_vm_stack_frame_t* frame);
  Status CommandBufferEnd(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);
  Status CommandBufferExecutionBarrier(iree_vm_stack_t* stack,
                                       iree_vm_stack_frame_t* frame);
  Status CommandBufferFillBuffer(iree_vm_stack_t* stack,
                                 iree_vm_stack_frame_t* frame);
  Status CommandBufferCopyBuffer(iree_vm_stack_t* stack,
                                 iree_vm_stack_frame_t* frame);
  Status CommandBufferBindDescriptorSet(iree_vm_stack_t* stack,
                                        iree_vm_stack_frame_t* frame);
  Status CommandBufferDispatch(iree_vm_stack_t* stack,
                               iree_vm_stack_frame_t* frame);
  Status CommandBufferDispatchIndirect(iree_vm_stack_t* stack,
                                       iree_vm_stack_frame_t* frame);

  Status DescriptorSetAllocate(iree_vm_stack_t* stack,
                               iree_vm_stack_frame_t* frame);
  Status DescriptorSetUpdate(iree_vm_stack_t* stack,
                             iree_vm_stack_frame_t* frame);

  Status DeviceAllocator(iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);

 private:
  iree_device_size_t CalculateBufferSize(absl::Span<const int32_t> shape,
                                         uint8_t element_size) {
    iree_device_size_t allocation_size = element_size;
    for (int i = 0; i < shape.size(); ++i) {
      allocation_size *= shape[i];
    }
    return allocation_size;
  }

  iree_device_size_t CalculateBufferOffset(absl::Span<const int32_t> shape,
                                           absl::Span<const int32_t> indices,
                                           uint8_t element_size) {
    iree_device_size_t offset = 0;
    for (int i = 0; i < indices.size(); ++i) {
      iree_device_size_t axis_offset = indices[i];
      for (int j = i + 1; j < shape.size(); ++j) {
        axis_offset *= shape[j];
      }
      offset += axis_offset;
    }
    offset *= element_size;
    return offset;
  }

  iree_allocator_t allocator_;
  ref_ptr<Device> shared_device_;
  ref_ptr<ExecutableCache> executable_cache_;

  std::vector<iree_vm_ref_t> deferred_releases_;

  std::vector<BufferBinding> bindings_;
};

//===----------------------------------------------------------------------===//
// Shared module code (across one or more contexts)
//===----------------------------------------------------------------------===//

Status HALModule::Initialize() {
  IREE_TRACE_SCOPE0("HALModule::Initialize");

  executable_cache_ = shared_device_->CreateExecutableCache();

  return OkStatus();
}

//===----------------------------------------------------------------------===//
// Method thunks
//===----------------------------------------------------------------------===//
// Ideally we would autogenerate these from the imports file or have some
// compile-time magic. For now this is all still experimental and we do it by
// hand.

// TODO(benvanik): replace with macro? helper for none/i32/etc
static const union {
  uint8_t reserved[2];
  iree_vm_register_list_t list;
} kReturnRef = {
    {1, 0 | IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT}};
static const union {
  uint8_t reserved[2];
  iree_vm_register_list_t list;
} kReturnI32 = {{1, 0}};
static const union {
  uint8_t reserved[3];
  iree_vm_register_list_t list;
} kReturn2xI32 = {{2, 0, 1}};

//===----------------------------------------------------------------------===//
// Experimental APIs
//===----------------------------------------------------------------------===//

Status HALModuleState::ExSharedDevice(iree_vm_stack_t* stack,
                                      iree_vm_stack_frame_t* frame) {
  frame->return_registers = &kReturnRef.list;
  frame->registers.ref_register_count = 1;
  frame->registers.ref[0] = iree_hal_device_retain_ref(
      reinterpret_cast<iree_hal_device_t*>(shared_device_.get()));
  return OkStatus();
}

Status HALModuleState::ExMatchSupportedExecutableFormat(
    iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame) {
  auto* device = iree_hal_device_deref(&frame->registers.ref[0]);
  if (!device) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'device' invalid";
  }
  auto available_formats = absl::MakeConstSpan(
      &frame->registers.i32[0], frame->return_registers->registers[1]);

  int32_t matched_format = 0;
  for (int32_t format : available_formats) {
    if (executable_cache_->CanPrepareFormat(
            static_cast<ExecutableFormat>(format))) {
      matched_format = format;
      break;
    }
  }

  ResetStackFrame(frame);
  frame->return_registers = &kReturnI32.list;
  frame->registers.i32[0] = matched_format;
  return OkStatus();
}

Status HALModuleState::ExCacheExecutable(iree_vm_stack_t* stack,
                                         iree_vm_stack_frame_t* frame) {
  auto* device = iree_hal_device_deref(&frame->registers.ref[0]);
  if (!device) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'device' invalid";
  }
  int32_t format = frame->registers.i32[0];
  auto* executable_data =
      iree_vm_ro_byte_buffer_deref(&frame->registers.ref[1]);
  if (!executable_data) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'executable_data' invalid";
  }

  ExecutableSpec spec;
  spec.format = static_cast<ExecutableFormat>(format);
  spec.executable_data = {executable_data->data.data,
                          executable_data->data.data_length};
  ASSIGN_OR_RETURN(auto executable, executable_cache_->PrepareExecutable(
                                        ExecutableCachingMode::kDefault, spec));

  ResetStackFrame(frame);
  frame->return_registers = &kReturnRef.list;
  frame->registers.ref_register_count = 1;
  frame->registers.ref[0] = iree_hal_executable_move_ref(
      reinterpret_cast<iree_hal_executable_t*>(executable.release()));
  return OkStatus();
}

Status HALModuleState::ExPushBinding(iree_vm_stack_t* stack,
                                     iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  int ri32 = 0;
  int32_t ordinal = frame->registers.i32[ri32++];
  auto* buffer = iree_hal_buffer_deref(&frame->registers.ref[1]);
  if (!buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'buffer' invalid";
  }
  int shape_rank = frame->return_registers->registers[3];
  auto shape = absl::MakeConstSpan(&frame->registers.i32[ri32], shape_rank);
  ri32 += shape_rank;
  uint8_t element_size = static_cast<uint8_t>(frame->registers.i32[ri32++]);

  if (ordinal >= bindings_.size()) {
    bindings_.resize(ordinal + 1);
  }
  auto& binding = bindings_[ordinal];
  binding.access = MemoryAccess::kAll;
  binding.buffer = reinterpret_cast<Buffer*>(buffer);
  binding.shape = Shape{shape};
  binding.element_size = element_size;

  ResetStackFrame(frame);
  return OkStatus();
}

Status HALModuleState::ExExecutableDescriptorSetLayout(
    iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC)
         << "ExExecutableDescriptorSetLayout";
}

Status HALModuleState::ExDeferRelease(iree_vm_stack_t* stack,
                                      iree_vm_stack_frame_t* frame) {
  if (!iree_vm_ref_is_null(&frame->registers.ref[0])) {
    deferred_releases_.push_back({0});
    iree_vm_ref_move(&frame->registers.ref[0], &deferred_releases_.back());
  }
  ResetStackFrame(frame);
  return OkStatus();
}

Status HALModuleState::ExSubmitAndWait(iree_vm_stack_t* stack,
                                       iree_vm_stack_frame_t* frame) {
  auto* device = iree_hal_device_deref(&frame->registers.ref[0]);
  if (!device) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'device' invalid";
  }
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[1]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }

  auto* queue = reinterpret_cast<Device*>(device)->dispatch_queues().front();
  ASSIGN_OR_RETURN(auto fence,
                   reinterpret_cast<Device*>(device)->CreateFence(0u));
  SubmissionBatch batch;
  batch.command_buffers = absl::MakeConstSpan(
      reinterpret_cast<CommandBuffer**>(&command_buffer), 1);
  RETURN_IF_ERROR(queue->Submit(batch, {fence.get(), 1u}));
  RETURN_IF_ERROR(queue->WaitIdle());

  for (auto& ref : deferred_releases_) {
    iree_vm_ref_release(&ref);
  }
  deferred_releases_.clear();
  bindings_.clear();

  ResetStackFrame(frame);
  return OkStatus();
}

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

Status HALModuleState::AllocatorComputeSize(iree_vm_stack_t* stack,
                                            iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "AllocatorComputeSize";
}

Status HALModuleState::AllocatorAllocate(iree_vm_stack_t* stack,
                                         iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "AllocatorAllocate";
}

Status HALModuleState::AllocatorAllocateConst(iree_vm_stack_t* stack,
                                              iree_vm_stack_frame_t* frame) {
  auto* allocator = iree_hal_allocator_deref(&frame->registers.ref[0]);
  if (!allocator) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'allocator' invalid";
  }
  auto* value_data = iree_vm_ro_byte_buffer_deref(&frame->registers.ref[1]);
  if (!value_data) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'value' invalid";
  }
  int ri32 = 0;
  iree_hal_memory_type_t memory_types =
      static_cast<iree_hal_memory_type_t>(frame->registers.i32[ri32++]);
  iree_hal_buffer_usage_t buffer_usage =
      static_cast<iree_hal_buffer_usage_t>(frame->registers.i32[ri32++]);
  int shape_rank = frame->return_registers->registers[3];
  auto shape = absl::MakeConstSpan(&frame->registers.i32[ri32], shape_rank);
  ri32 += shape_rank;
  uint8_t element_size = static_cast<uint8_t>(frame->registers.i32[ri32++]);

  // TODO(benvanik): generic compute size.
  iree_device_size_t allocation_size = CalculateBufferSize(shape, element_size);
  if (allocation_size < value_data->data.data_length) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Constant data is too larger for the minimum allocation size";
  }

  iree_hal_buffer_t* buffer = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_allocator_allocate_buffer(allocator, memory_types, buffer_usage,
                                         allocation_size, &buffer),
      IREE_LOC))
      << "Failed to allocate buffer";

  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_buffer_write_data(buffer, 0, value_data->data.data,
                                               value_data->data.data_length),
                    IREE_LOC))
      << "Writing constant data";

  ResetStackFrame(frame);
  frame->return_registers = &kReturnRef.list;
  frame->registers.ref_register_count = 1;
  frame->registers.ref[0] = iree_hal_buffer_move_ref(buffer);
  return OkStatus();
}

Status HALModuleState::AllocatorAllocateShaped(iree_vm_stack_t* stack,
                                               iree_vm_stack_frame_t* frame) {
  auto* allocator = iree_hal_allocator_deref(&frame->registers.ref[0]);
  if (!allocator) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'allocator' invalid";
  }
  int ri32 = 0;
  iree_hal_memory_type_t memory_types =
      static_cast<iree_hal_memory_type_t>(frame->registers.i32[ri32++]);
  iree_hal_buffer_usage_t buffer_usage =
      static_cast<iree_hal_buffer_usage_t>(frame->registers.i32[ri32++]);
  int shape_rank = frame->return_registers->registers[3];
  auto shape = absl::MakeConstSpan(&frame->registers.i32[ri32], shape_rank);
  ri32 += shape_rank;
  uint8_t element_size = static_cast<uint8_t>(frame->registers.i32[ri32++]);

  // TODO(benvanik): generic compute size.
  iree_device_size_t allocation_size = CalculateBufferSize(shape, element_size);

  iree_hal_buffer_t* buffer = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_allocator_allocate_buffer(allocator, memory_types, buffer_usage,
                                         allocation_size, &buffer),
      IREE_LOC))
      << "Failed to allocate buffer";

  ResetStackFrame(frame);
  frame->return_registers = &kReturnRef.list;
  frame->registers.ref_register_count = 1;
  frame->registers.ref[0] = iree_hal_buffer_move_ref(buffer);
  return OkStatus();
}

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

Status HALModuleState::BufferSubspan(iree_vm_stack_t* stack,
                                     iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferSubspan";
}

Status HALModuleState::BufferFill(iree_vm_stack_t* stack,
                                  iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferFill";
}

Status HALModuleState::BufferReadData(iree_vm_stack_t* stack,
                                      iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferReadData";
}

Status HALModuleState::BufferWriteData(iree_vm_stack_t* stack,
                                       iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferWriteData";
}

Status HALModuleState::BufferCopyData(iree_vm_stack_t* stack,
                                      iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferCopyData";
}

Status HALModuleState::BufferLoad(iree_vm_stack_t* stack,
                                  iree_vm_stack_frame_t* frame) {
  auto* source_buffer = iree_hal_buffer_deref(&frame->registers.ref[0]);
  if (!source_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'source_buffer' invalid";
  }
  iree_device_size_t source_offset = frame->registers.i32[0];
  iree_device_size_t length = frame->registers.i32[1];

  uint32_t target_buffer = 0;
  if (length > sizeof(target_buffer)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Length " << length << " exceeds max";
  }
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_buffer_read_data(source_buffer, source_offset,
                                              &target_buffer, length),
                    IREE_LOC))
      << "Read failed";

  ResetStackFrame(frame);
  frame->return_registers = &kReturnI32.list;
  frame->registers.i32[0] = target_buffer;
  return OkStatus();
}

Status HALModuleState::BufferStore(iree_vm_stack_t* stack,
                                   iree_vm_stack_frame_t* frame) {
  auto* target_buffer = iree_hal_buffer_deref(&frame->registers.ref[0]);
  if (!target_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'target_buffer' invalid";
  }
  uint32_t value = frame->registers.i32[0];
  iree_device_size_t target_offset = frame->registers.i32[1];
  iree_device_size_t length = frame->registers.i32[2];

  if (target_offset + length > iree_hal_buffer_byte_length(target_buffer)) {
    return OutOfRangeErrorBuilder(IREE_LOC) << "Out of bounds store";
  }

  if (length > sizeof(value)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Length " << length << " exceeds max";
  }
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_buffer_write_data(target_buffer, target_offset, &value, length),
      IREE_LOC))
      << "Write failed";

  ResetStackFrame(frame);
  return OkStatus();
}

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

Status HALModuleState::BufferViewComputeOffset(iree_vm_stack_t* stack,
                                               iree_vm_stack_frame_t* frame) {
  auto* buffer = iree_hal_buffer_deref(&frame->registers.ref[0]);
  if (!buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'buffer' invalid";
  }
  int ri32 = 0;
  int shape_rank = frame->return_registers->registers[1];
  auto shape = absl::MakeConstSpan(&frame->registers.i32[ri32], shape_rank);
  ri32 += shape_rank;
  int indices_rank = frame->return_registers->registers[2];
  auto indices = absl::MakeConstSpan(&frame->registers.i32[ri32], indices_rank);
  ri32 += indices_rank;
  uint8_t element_size = static_cast<uint8_t>(frame->registers.i32[ri32++]);

  iree_device_size_t offset =
      CalculateBufferOffset(shape, indices, element_size);

  ResetStackFrame(frame);
  frame->return_registers = &kReturnI32.list;
  frame->registers.i32[0] = offset;
  return OkStatus();
}

Status HALModuleState::BufferViewComputeLength(iree_vm_stack_t* stack,
                                               iree_vm_stack_frame_t* frame) {
  auto* buffer = iree_hal_buffer_deref(&frame->registers.ref[0]);
  if (!buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'buffer' invalid";
  }
  int ri32 = 0;
  int shape_rank = frame->return_registers->registers[1];
  auto shape = absl::MakeConstSpan(&frame->registers.i32[ri32], shape_rank);
  ri32 += shape_rank;
  uint8_t element_size = static_cast<uint8_t>(frame->registers.i32[ri32++]);

  iree_device_size_t length = CalculateBufferSize(shape, element_size);

  ResetStackFrame(frame);
  frame->return_registers = &kReturnI32.list;
  frame->registers.i32[0] = length;
  return OkStatus();
}

Status HALModuleState::BufferViewComputeRange(iree_vm_stack_t* stack,
                                              iree_vm_stack_frame_t* frame) {
  auto* buffer = iree_hal_buffer_deref(&frame->registers.ref[0]);
  if (!buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'buffer' invalid";
  }
  int ri32 = 0;
  int shape_rank = frame->return_registers->registers[1];
  auto shape = absl::MakeConstSpan(&frame->registers.i32[ri32], shape_rank);
  ri32 += shape_rank;
  int start_indices_rank = frame->return_registers->registers[2];
  auto start_indices =
      absl::MakeConstSpan(&frame->registers.i32[ri32], start_indices_rank);
  ri32 += start_indices_rank;
  int lengths_rank = frame->return_registers->registers[3];
  auto lengths = absl::MakeConstSpan(&frame->registers.i32[ri32], lengths_rank);
  ri32 += lengths_rank;
  uint8_t element_size = static_cast<uint8_t>(frame->registers.i32[ri32++]);

  if (start_indices.size() != shape.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Slice start_indices " << PrettyPrint(start_indices)
           << " do not match rank of shape " << PrettyPrint(shape);
  }
  if (start_indices.size() != lengths.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Slice start_indices " << PrettyPrint(start_indices)
           << " and lengths " << PrettyPrint(lengths)
           << " are not the same size";
  }

  absl::InlinedVector<int32_t, 6> end_indices(shape_rank);
  iree_device_size_t subspan_length = element_size;
  for (int i = 0; i < lengths.size(); ++i) {
    subspan_length *= lengths[i];
    end_indices[i] = start_indices[i] + lengths[i] - 1;
  }

  iree_device_size_t start_byte_offset =
      CalculateBufferOffset(shape, start_indices, element_size);
  iree_device_size_t end_byte_offset =
      CalculateBufferOffset(shape, end_indices, element_size);

  auto offset_length = end_byte_offset - start_byte_offset + element_size;
  if (subspan_length != offset_length) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Slice for non-contiguous region of memory unimplemented. "
              "start_indices: "
           << PrettyPrint(start_indices) << " lengths: " << PrettyPrint(lengths)
           << " " << subspan_length << " " << offset_length << " "
           << PrettyPrint(end_indices);
  }

  ResetStackFrame(frame);
  frame->return_registers = &kReturn2xI32.list;
  frame->registers.i32[0] = start_byte_offset;
  frame->registers.i32[1] = subspan_length;
  return OkStatus();
}

Status HALModuleState::BufferViewSlice(iree_vm_stack_t* stack,
                                       iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "BufferViewSlice";
}

//===----------------------------------------------------------------------===//
// iree::hal::CommandBuffer
//===----------------------------------------------------------------------===//

Status HALModuleState::CommandBufferCreate(iree_vm_stack_t* stack,
                                           iree_vm_stack_frame_t* frame) {
  auto* device = iree_hal_device_deref(&frame->registers.ref[0]);
  if (!device) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'device' invalid";
  }
  int ri32 = 0;
  iree_hal_command_buffer_mode_t mode =
      static_cast<iree_hal_command_buffer_mode_t>(frame->registers.i32[ri32++]);
  iree_hal_command_category_t command_categories =
      static_cast<iree_hal_command_category_t>(frame->registers.i32[ri32++]);

  iree_hal_command_buffer_t* command_buffer = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_command_buffer_create(device, mode, command_categories,
                                     IREE_ALLOCATOR_SYSTEM, &command_buffer),
      IREE_LOC))
      << "Failed to create command buffer";

  ResetStackFrame(frame);
  frame->return_registers = &kReturnRef.list;
  frame->registers.ref_register_count = 1;
  frame->registers.ref[0] = iree_hal_command_buffer_move_ref(command_buffer);
  return OkStatus();
}

Status HALModuleState::CommandBufferBegin(iree_vm_stack_t* stack,
                                          iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_command_buffer_begin(command_buffer), IREE_LOC))
      << "Failed to begin command buffer recording";
  ResetStackFrame(frame);
  return OkStatus();
}

Status HALModuleState::CommandBufferEnd(iree_vm_stack_t* stack,
                                        iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_command_buffer_end(command_buffer), IREE_LOC))
      << "Failed to end command buffer recording";
  ResetStackFrame(frame);
  return OkStatus();
}

Status HALModuleState::CommandBufferExecutionBarrier(
    iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  int ri32 = 0;
  iree_hal_execution_stage_t source_stage_mask =
      static_cast<iree_hal_execution_stage_t>(frame->registers.i32[ri32++]);
  iree_hal_execution_stage_t target_stage_mask =
      static_cast<iree_hal_execution_stage_t>(frame->registers.i32[ri32++]);

  // TODO(benvanik): decode barriers.
  iree_hal_memory_barrier_t global_barrier;
  global_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
  global_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
  RETURN_IF_ERROR(
      FromApiStatus(iree_hal_command_buffer_execution_barrier(
                        command_buffer, source_stage_mask, target_stage_mask, 1,
                        &global_barrier, 0, nullptr),
                    IREE_LOC));

  ResetStackFrame(frame);
  return OkStatus();
}

Status HALModuleState::CommandBufferFillBuffer(iree_vm_stack_t* stack,
                                               iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  return UnimplementedErrorBuilder(IREE_LOC) << "CommandBufferFillBuffer";
}

Status HALModuleState::CommandBufferCopyBuffer(iree_vm_stack_t* stack,
                                               iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  auto* source_buffer = iree_hal_buffer_deref(&frame->registers.ref[1]);
  if (!source_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'source_buffer' invalid";
  }
  auto* target_buffer = iree_hal_buffer_deref(&frame->registers.ref[2]);
  if (!target_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'target_buffer' invalid";
  }
  iree_device_size_t source_offset = frame->registers.i32[0];
  iree_device_size_t target_offset = frame->registers.i32[1];
  iree_device_size_t length = frame->registers.i32[2];

  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_command_buffer_copy_buffer(command_buffer, source_buffer,
                                          source_offset, target_buffer,
                                          target_offset, length),
      IREE_LOC));

  ResetStackFrame(frame);
  return OkStatus();
}

Status HALModuleState::CommandBufferBindDescriptorSet(
    iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  return UnimplementedErrorBuilder(IREE_LOC)
         << "CommandBufferBindDescriptorSet";
}

Status HALModuleState::CommandBufferDispatch(iree_vm_stack_t* stack,
                                             iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  auto* executable = iree_hal_executable_deref(&frame->registers.ref[1]);
  if (!executable) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'executable' invalid";
  }
  int32_t entry_point = frame->registers.i32[0];
  int32_t workgroup_x = frame->registers.i32[1];
  int32_t workgroup_y = frame->registers.i32[2];
  int32_t workgroup_z = frame->registers.i32[3];

  DispatchRequest dispatch_request;
  dispatch_request.executable = reinterpret_cast<Executable*>(executable);
  dispatch_request.entry_point = entry_point;
  dispatch_request.workload = {workgroup_x, workgroup_y, workgroup_z};
  dispatch_request.bindings = bindings_;

  RETURN_IF_ERROR(reinterpret_cast<CommandBuffer*>(command_buffer)
                      ->Dispatch(dispatch_request));

  bindings_.clear();

  ResetStackFrame(frame);
  return OkStatus();
}

Status HALModuleState::CommandBufferDispatchIndirect(
    iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame) {
  auto* command_buffer =
      iree_hal_command_buffer_deref(&frame->registers.ref[0]);
  if (!command_buffer) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'command_buffer' invalid";
  }
  return UnimplementedErrorBuilder(IREE_LOC) << "CommandBufferDispatchIndirect";
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSet
//===----------------------------------------------------------------------===//

Status HALModuleState::DescriptorSetAllocate(iree_vm_stack_t* stack,
                                             iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "DescriptorSetAllocate";
}

Status HALModuleState::DescriptorSetUpdate(iree_vm_stack_t* stack,
                                           iree_vm_stack_frame_t* frame) {
  return UnimplementedErrorBuilder(IREE_LOC) << "DescriptorSetUpdate";
}

//===----------------------------------------------------------------------===//
// iree::hal::Device
//===----------------------------------------------------------------------===//

Status HALModuleState::DeviceAllocator(iree_vm_stack_t* stack,
                                       iree_vm_stack_frame_t* frame) {
  auto* device = reinterpret_cast<Device*>(
      iree_hal_device_deref(&frame->registers.ref[0]));
  if (!device) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "'device' invalid";
  }

  ResetStackFrame(frame);
  frame->return_registers = &kReturnRef.list;
  frame->registers.ref_register_count = 1;
  frame->registers.ref[0] = iree_hal_allocator_retain_ref(
      reinterpret_cast<iree_hal_allocator_t*>(device->allocator()));
  return OkStatus();
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

using ExportFunctionPtr = Status (HALModuleState::*)(
    iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame);

struct ExportFunctionInfo {
  ExportFunctionPtr ptr;
  const char* name;
};

static const ExportFunctionInfo kHALExportFunctionInfos[] = {
    {&HALModuleState::ExSharedDevice, "ex.shared_device"},
    {&HALModuleState::ExMatchSupportedExecutableFormat,
     "ex.match_supported_executable_format"},
    {&HALModuleState::ExCacheExecutable, "ex.cache_executable"},
    {&HALModuleState::ExPushBinding, "ex.push_binding"},
    {&HALModuleState::ExExecutableDescriptorSetLayout,
     "ex.executable_descriptor_set_layout"},
    {&HALModuleState::ExDeferRelease, "ex.defer_release"},
    {&HALModuleState::ExSubmitAndWait, "ex.submit_and_wait"},
    {&HALModuleState::AllocatorComputeSize, "allocator.compute_size"},
    {&HALModuleState::AllocatorAllocate, "allocator.allocate"},
    {&HALModuleState::AllocatorAllocateConst, "allocator.allocate.const"},
    {&HALModuleState::AllocatorAllocateShaped, "allocator.allocate.shaped"},
    {&HALModuleState::BufferSubspan, "buffer.subspan"},
    {&HALModuleState::BufferFill, "buffer.fill"},
    {&HALModuleState::BufferReadData, "buffer.read_data"},
    {&HALModuleState::BufferWriteData, "buffer.write_data"},
    {&HALModuleState::BufferCopyData, "buffer.copy_data"},
    {&HALModuleState::BufferLoad, "buffer.load"},
    {&HALModuleState::BufferStore, "buffer.store"},
    {&HALModuleState::BufferViewComputeOffset, "buffer_view.compute_offset"},
    {&HALModuleState::BufferViewComputeLength, "buffer_view.compute_length"},
    {&HALModuleState::BufferViewComputeRange, "buffer_view.compute_range"},
    {&HALModuleState::BufferViewSlice, "buffer_view.slice"},
    {&HALModuleState::CommandBufferCreate, "command_buffer.create"},
    {&HALModuleState::CommandBufferBegin, "command_buffer.begin"},
    {&HALModuleState::CommandBufferEnd, "command_buffer.end"},
    {&HALModuleState::CommandBufferExecutionBarrier,
     "command_buffer.execution_barrier"},
    {&HALModuleState::CommandBufferFillBuffer, "command_buffer.fill_buffer"},
    {&HALModuleState::CommandBufferCopyBuffer, "command_buffer.copy_buffer"},
    {&HALModuleState::CommandBufferBindDescriptorSet,
     "command_buffer.bind_descriptor_set"},
    {&HALModuleState::CommandBufferDispatch, "command_buffer.dispatch"},
    {&HALModuleState::CommandBufferDispatchIndirect,
     "command_buffer.dispatch.indirect"},
    {&HALModuleState::DescriptorSetAllocate, "descriptor_set.allocate"},
    {&HALModuleState::DescriptorSetUpdate, "descriptor_set.update"},
    {&HALModuleState::DeviceAllocator, "device.allocator"},
};

static iree_status_t iree_hal_module_destroy(void* self) {
  delete HALModule::FromPointer(self);
  return IREE_STATUS_OK;
}

static iree_string_view_t iree_hal_module_name(void* self) {
  return iree_make_cstring_view("hal");
}

static iree_vm_module_signature_t iree_hal_module_signature(void* self) {
  iree_vm_module_signature_t signature = {0};
  signature.import_function_count = 0;
  signature.export_function_count = ABSL_ARRAYSIZE(kHALExportFunctionInfos);
  signature.internal_function_count = 0;
  return signature;
}

static iree_status_t iree_hal_module_get_function(
    void* self, iree_vm_function_linkage_t linkage, int32_t ordinal,
    iree_vm_function_t* out_function, iree_string_view_t* out_name,
    iree_vm_function_signature_t* out_signature) {
  if (out_function) {
    std::memset(out_function, 0, sizeof(*out_function));
  }
  if (out_name) {
    out_name->data = nullptr;
    out_name->size = 0;
  }
  if (out_signature) {
    std::memset(out_signature, 0, sizeof(*out_signature));
  }
  if (ordinal < 0 || ordinal > ABSL_ARRAYSIZE(kHALExportFunctionInfos)) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  const auto& info = kHALExportFunctionInfos[ordinal];

  if (out_function) {
    auto* module = HALModule::FromPointer(self);
    out_function->module = module->interface();
    out_function->linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT;
    out_function->ordinal = ordinal;
  }
  if (out_name) {
    *out_name = iree_make_cstring_view(info.name);
  }

  return IREE_STATUS_OK;
}

static iree_status_t iree_hal_module_lookup_function(
    void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
    iree_vm_function_t* out_function) {
  if (!out_function) return IREE_STATUS_INVALID_ARGUMENT;
  std::memset(out_function, 0, sizeof(*out_function));
  if (!name.data || !name.size) return IREE_STATUS_INVALID_ARGUMENT;

  auto* module = HALModule::FromPointer(self);
  out_function->module = module->interface();
  out_function->linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT;
  for (int i = 0; i < ABSL_ARRAYSIZE(kHALExportFunctionInfos); ++i) {
    if (iree_string_view_compare(
            name, iree_make_cstring_view(kHALExportFunctionInfos[i].name)) ==
        0) {
      out_function->ordinal = i;
      return IREE_STATUS_OK;
    }
  }
  return IREE_STATUS_NOT_FOUND;
}

static iree_status_t iree_hal_module_alloc_state(
    void* self, iree_allocator_t allocator,
    iree_vm_module_state_t** out_module_state) {
  if (!out_module_state) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module_state = nullptr;

  auto* module = HALModule::FromPointer(self);
  auto* module_state =
      new HALModuleState(allocator, add_ref(module->shared_device()),
                         add_ref(module->executable_cache()));

  // TODO(benvanik): allocate context-specific variables (allocator pool, etc).

  *out_module_state = reinterpret_cast<iree_vm_module_state_t*>(module_state);
  return IREE_STATUS_OK;
}

static iree_status_t iree_hal_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  if (!module_state) return IREE_STATUS_INVALID_ARGUMENT;
  delete HALModuleState::FromPointer(module_state);
  return IREE_STATUS_OK;
}

static iree_status_t iree_hal_module_resolve_import(
    void* self, iree_vm_module_state_t* module_state, int32_t ordinal,
    iree_vm_function_t function) {
  // Module does not have imports.
  return IREE_STATUS_FAILED_PRECONDITION;
}

static iree_status_t iree_hal_module_execute(
    void* self, iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame,
    iree_vm_execution_result_t* out_result) {
  if (!out_result) return IREE_STATUS_INVALID_ARGUMENT;
  std::memset(out_result, 0, sizeof(*out_result));
  if (!stack || !frame) return IREE_STATUS_INVALID_ARGUMENT;
  int32_t ordinal = frame->function.ordinal;
  if (ordinal < 0 || ordinal > ABSL_ARRAYSIZE(kHALExportFunctionInfos)) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto* state = HALModuleState::FromPointer(frame->module_state);

  const auto& info = kHALExportFunctionInfos[ordinal];
  auto status = (state->*(info.ptr))(stack, frame);
  if (!status.ok()) {
    return ToApiStatus(status);
  }

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_module_create(iree_hal_device_t* device, iree_allocator_t allocator,
                       iree_vm_module_t** out_module) {
  if (!out_module) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module = nullptr;

  IREE_API_ASSIGN_OR_RETURN(
      auto module,
      HALModule::Create(allocator, add_ref(reinterpret_cast<Device*>(device))));

  auto* interface = module->interface();
  interface->destroy = iree_hal_module_destroy;
  interface->name = iree_hal_module_name;
  interface->signature = iree_hal_module_signature;
  interface->get_function = iree_hal_module_get_function;
  interface->lookup_function = iree_hal_module_lookup_function;
  interface->alloc_state = iree_hal_module_alloc_state;
  interface->free_state = iree_hal_module_free_state;
  interface->resolve_import = iree_hal_module_resolve_import;
  interface->execute = iree_hal_module_execute;

  module.release();
  *out_module = interface;
  return IREE_STATUS_OK;
}

}  // namespace
}  // namespace hal
}  // namespace iree
