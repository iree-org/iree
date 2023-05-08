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

// Represents some kind of stateful async operation.
// Here we spin up a thread to wait on the wait_fence, do some expensive work,
// and then signal the signal_fence.
//
// **This is not actually how this should be done** - spinning up a thread for
// each operation is extremely wasteful and doing so will contend with the
// threads IREE uses for scheduling its compute workloads. This is pretty much
// the worst way to run asynchronous work (but at least it's async!). Instead
// think of this as an example of calling off to some service/system layer where
// the ownership of the work scheduling is not in control of the application
// (like networking or RPC).
//
// Each AsyncOp instance is used for a single operation and deletes itself when
// the operation is complete. In order to prevent hangs it's critical that the
// signal_fence is signaled or marked as failing.
//
// TODO(benvanik): demonstrate getting the iree_task_executor_t for direct use.
class AsyncOp {
 public:
  static void Launch(vm::ref<iree_hal_buffer_view_t> source_view,
                     vm::ref<iree_hal_buffer_view_t> target_view,
                     vm::ref<iree_hal_fence_t> wait_fence,
                     vm::ref<iree_hal_fence_t> signal_fence) {
    new AsyncOp(std::move(source_view), std::move(target_view),
                std::move(wait_fence), std::move(signal_fence));
  }

 private:
  AsyncOp(vm::ref<iree_hal_buffer_view_t> source_view,
          vm::ref<iree_hal_buffer_view_t> target_view,
          vm::ref<iree_hal_fence_t> wait_fence,
          vm::ref<iree_hal_fence_t> signal_fence)
      : source_view_(std::move(source_view)),
        target_view_(std::move(target_view)),
        wait_fence_(std::move(wait_fence)),
        signal_fence_(std::move(signal_fence)),
        thread_([this]() {
          thread_.detach();
          ThreadEntry();
          delete this;  // self cleanup
        }) {}

  void ThreadEntry() {
    IREE_TRACE_SET_THREAD_NAME("std-thread-worker");
    IREE_TRACE_SCOPE();

    fprintf(stdout, "ASYNC: BEFORE WAIT\n");
    fflush(stdout);

    // Give a pause to simulate doing something expensive.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Wait until the tensor is ready for use. A real application could
    // export the fence to a native wait handle they could use with syscalls
    // or add the fence to a multi-wait operation. Here we just block the
    // thread until ready. Due to the nature of ordering it's possible the
    // fence has already been signaled by the time we get here.
    Status status =
        iree_hal_fence_wait(wait_fence_.get(), iree_infinite_timeout());

    fprintf(stdout, "ASYNC: AFTER WAIT\n");
    fflush(stdout);

    // Perform the expensive work while the input tensor is known good and
    // the output is ready to accept it.
    if (status.ok()) {
      // Hacky example accessing the source contents and producing the result
      // contents. This emulates what an external library the user is calling
      // that expects host void* buffers does.
      status = SyncSimulatedHostOpI32(
          iree_hal_buffer_view_buffer(source_view_.get()),
          iree_hal_buffer_view_buffer(target_view_.get()),
          iree_hal_buffer_view_element_count(source_view_.get()));
    }

    fprintf(stdout, "ASYNC: BEFORE SIGNAL\n");
    fflush(stdout);

    // Try to signal completion so that downstream consumers of the result
    // can get scheduled.
    if (status.ok()) {
      status = iree_hal_fence_signal(signal_fence_.get());
    }

    // If we failed then we propagate the failure status. This is likely to
    // result in complete failure of the invocation though when the user is
    // able to observe the failure is hard to determine as they may be
    // pipelined N invocations deep by the time this runs.
    if (!status.ok()) {
      iree_hal_fence_fail(signal_fence_.get(), status.release());
    }

    fprintf(stdout, "ASYNC: AFTER SIGNAL\n");
    fflush(stdout);
  }

  vm::ref<iree_hal_buffer_view_t> source_view_;
  vm::ref<iree_hal_buffer_view_t> target_view_;
  vm::ref<iree_hal_fence_t> wait_fence_;
  vm::ref<iree_hal_fence_t> signal_fence_;
  std::thread thread_;
};

// Per-context module state.
// This can contain "globals" and other arbitrary state.
//
// Thread-compatible; the runtime will not issue multiple calls at the same
// time using the same state. If the implementation uses external threads then
// it must synchronize itself.
class CustomModuleState final {
 public:
  explicit CustomModuleState(vm::ref<iree_hal_device_t> device,
                             iree_allocator_t host_allocator)
      : device_(std::move(device)), host_allocator_(host_allocator) {}
  ~CustomModuleState() = default;

  StatusOr<vm::ref<iree_hal_buffer_view_t>> CallAsync(
      const vm::ref<iree_hal_buffer_view_t> arg_view,
      const vm::ref<iree_hal_fence_t> wait_fence,
      const vm::ref<iree_hal_fence_t> signal_fence) {
    // TODO(benvanik): better fence helpers when timelines are not needed.
    vm::ref<iree_hal_semaphore_t> semaphore;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_create(device_.get(), 0ull, &semaphore));
    vm::ref<iree_hal_fence_t> alloca_fence;
    IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(
        semaphore.get(), 1ull, host_allocator_, &alloca_fence));

    // Asynchronously allocate the output memory for the call result.
    // This chains the allocation such that the wait_fence must be signaled
    // before the memory is allocated and our alloca_fence will be used to
    // sequence our work with the allocation:
    //
    // [wait_fence] -> alloca -> [alloca_fence] -> work -> [signal_fence]
    //
    // TODO(benvanik): extend to allowing result storage to be passed in (when
    // possible to compute sizes). For now all results need to be allocated.
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
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_alloca(
        device_.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_fence_semaphore_list(wait_fence.get()),
        iree_hal_fence_semaphore_list(alloca_fence.get()),
        IREE_HAL_ALLOCATOR_POOL_DEFAULT, buffer_params,
        iree_hal_buffer_view_byte_length(arg_view.get()), &result_buffer));

    // Wrap the buffer in a buffer view that provides the metadata for
    // runtime verification.
    vm::ref<iree_hal_buffer_view_t> result_view;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create_like(
        result_buffer.get(), arg_view.get(), host_allocator_, &result_view));

    // Launch the stateful async operation.
    // See the notes above - note that this is _not_ a good way of doing this!
    // Note that we should be using host_allocator_ here to create these objects
    // so that memory is properly tracked as originating from this call.
    AsyncOp::Launch(vm::retain_ref(arg_view), vm::retain_ref(result_view),
                    std::move(alloca_fence), std::move(signal_fence));

    // Note that the caller needs the buffer view back but is not allowed to
    // access its contents until we signal the signal_fence.
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
    vm::MakeNativeFunction("call.async", &CustomModuleState::CallAsync),
};

// The module instance that will be allocated and reused across contexts.
// Any context-specific state must be stored in a state structure such as
// CustomModuleState.
//
// Assumed thread-safe (by construction here, as it's immutable), though if any
// mutable state is stored here it will need to be synchronized by the
// implementation.
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
extern "C" iree_status_t iree_custom_module_async_create(
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
