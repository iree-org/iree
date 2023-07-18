// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "module.h"

#include <cstdio>
#include <thread>

#include "iree/modules/hal/types.h"
#include "iree/vm/native_module_cc.h"

// NOTE: this module is written in C++ using the native module wrapper and uses
// template magic to handle marshaling arguments. For a lot of uses this is a
// much friendlier way of exposing modules to the IREE VM and if performance and
// code size are not a concern is a fine route to take. Here we do it for
// brevity but all of the internal IREE modules are implemented in C.

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

namespace {

using namespace iree;

// Approximation of some external library call that populates a buffer.
// It's assumed that when this is called the |source_buffer| is available to
// read and the |target_buffer| is available to write (no other readers exist).
// This sample assumes that the buffers are mappable so we can do the work here
// but they will not always be. APIs like iree_hal_allocator_import_buffer and
// iree_hal_allocator_export_buffer can be used in some cases to avoid
// potentially expensive operations but real applications that care about
// performance would want to issue async transfer command buffers.
//
// Only use this as a reference for when synchronous behavior is absolutely
// required (old-style blocking file IO/etc).
static Status SyncSimulatedHostOpI32(iree_hal_buffer_t* source_buffer,
                                     iree_hal_buffer_t* target_buffer,
                                     iree_hal_dim_t count) {
  Status status = OkStatus();

  // Map the source and target buffers into host memory. Note that not all
  // devices allow this but in this sample we assume they do.
  iree_hal_buffer_mapping_t source_mapping = {{0}};
  if (status.ok()) {
    status = iree_hal_buffer_map_range(
        source_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_WHOLE_BUFFER, &source_mapping);
  }
  iree_hal_buffer_mapping_t target_mapping = {{0}};
  if (status.ok()) {
    status =
        iree_hal_buffer_map_range(target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                  IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0,
                                  IREE_WHOLE_BUFFER, &target_mapping);
  }

  // Sad slow host work. Whenever possible it's worth it to move these into the
  // program so the IREE compiler can fuse and accelerate these operations.
  if (status.ok()) {
    const int32_t* source_ptr =
        reinterpret_cast<const int32_t*>(source_mapping.contents.data);
    int32_t* target_ptr =
        reinterpret_cast<int32_t*>(target_mapping.contents.data);
    for (iree_host_size_t i = 0; i < count; ++i) {
      target_ptr[i] = source_ptr[i] * 2;
    }
  }

  // We must unmap the buffers before they will be usable.
  // Note that it's possible for these to fail in cases where the buffer
  // required emulated mapping but on basic host-local devices like CPU assumed
  // in this sample that should never happen.
  iree_status_ignore(iree_hal_buffer_unmap_range(&source_mapping));
  iree_status_ignore(iree_hal_buffer_unmap_range(&target_mapping));

  return status;
}

// Per-context module state.
class CustomModuleState final {
 public:
  explicit CustomModuleState(vm::ref<iree_hal_device_t> device,
                             iree_allocator_t host_allocator)
      : device_(std::move(device)), host_allocator_(host_allocator) {}
  ~CustomModuleState() = default;

  StatusOr<vm::ref<iree_hal_buffer_view_t>> CallSync(
      const vm::ref<iree_hal_buffer_view_t> arg_view) {
    // We can directly access the buffer here but only for reading.
    // In the future it'll be possible to pass in-place buffers.
    auto* arg_buffer = iree_hal_buffer_view_buffer(arg_view.get());

    // Synchronously allocate the memory from the device allocator. We could
    // use queue-ordered allocations but that's unsafe to use from arbitrary
    // threads and we want to show how to safely do that using the thread-safe
    // device allocator.
    //
    // NOTE: if cloning host memory the initial_data can be passed in to
    // efficiently upload the memory to the device. If wrapping host memory then
    // iree_hal_allocator_import_buffer can be used to import the memory without
    // a copy (if supported). This simple example is showing an in-place style
    // external call.
    iree_hal_allocator_t* device_allocator =
        iree_hal_device_allocator(device_.get());
    iree_hal_buffer_params_t buffer_params = {
        /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT |
            IREE_HAL_BUFFER_USAGE_MAPPING,
        /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
        /*.type=*/IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
            IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
        /*.min_alignment=*/64,
    };
    vm::ref<iree_hal_buffer_t> result_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        device_allocator, buffer_params,
        iree_hal_buffer_view_byte_length(arg_view.get()),
        iree_const_byte_span_empty(), &result_buffer));

    // Hacky example accessing the source contents and producing the result
    // contents. This emulates what an external library the user is calling that
    // expects host void* buffers does.
    IREE_RETURN_IF_ERROR(SyncSimulatedHostOpI32(
        arg_buffer, result_buffer.get(),
        iree_hal_buffer_view_element_count(arg_view.get())));

    // Wrap the buffer in a buffer view that provides the metadata for
    // runtime verification.
    vm::ref<iree_hal_buffer_view_t> result_view;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create_like(
        result_buffer.get(), arg_view.get(), host_allocator_, &result_view));

    // Note that the caller may immediately use the buffer contents without
    // waiting as by being synchronous we've indicated that we waited ourselves
    // (the thread join above).
    return result_view;
  }

 private:
  // HAL device used for scheduling work and allocations.
  vm::ref<iree_hal_device_t> device_;

  // Allocator that the caller requested we use for any allocations we need to
  // perform during operation.
  iree_allocator_t host_allocator_;
};

// Function table mapping imported function names to their implementation.
static const vm::NativeFunction<CustomModuleState> kCustomModuleFunctions[] = {
    vm::MakeNativeFunction("call.sync", &CustomModuleState::CallSync),
};

// The module instance that will be allocated and reused across contexts.
class CustomModule final : public vm::NativeModule<CustomModuleState> {
 public:
  using vm::NativeModule<CustomModuleState>::NativeModule;

  void SetDevice(vm::ref<iree_hal_device_t> device) {
    device_ = std::move(device);
  }

  // Creates per-context state when the module is added to a new context.
  // May be called from any thread.
  StatusOr<std::unique_ptr<CustomModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    auto state = std::make_unique<CustomModuleState>(vm::retain_ref(device_),
                                                     host_allocator);
    return state;
  }

 private:
  vm::ref<iree_hal_device_t> device_;
};

}  // namespace

// Note that while we are using C++ bindings internally we still expose the
// module as a C instance. This hides the details of our implementation.
extern "C" iree_status_t iree_custom_module_sync_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // NOTE: this isn't using the allocator here and that's bad as it leaves
  // untracked allocations and pulls in the system allocator that may differ
  // from the one requested by the user.
  // TODO(benvanik): std::allocator wrapper around iree_allocator_t so this can
  // use that instead.
  auto module = std::make_unique<CustomModule>(
      "custom", /*version=*/0, instance, host_allocator,
      iree::span<const vm::NativeFunction<CustomModuleState>>(
          kCustomModuleFunctions));
  module->SetDevice(vm::retain_ref(device));

  *out_module = module.release()->interface();
  return iree_ok_status();
}
