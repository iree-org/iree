// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_COMMON_API_IMPL_H_
#define IREE_PJRT_PLUGIN_PJRT_COMMON_API_IMPL_H_

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/integrations/pjrt/common/compiler.h"
#include "iree/integrations/pjrt/common/platform.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/shape_util.h"

namespace iree::pjrt {

class ClientInstance;
class DeviceInstance;
class ErrorInstance;
class EventInstance;

//===----------------------------------------------------------------------===//
// PJRT_Error wrapper
// PJRT Errors are simple wrappers around an iree_status_t. They are
// infrequently created, so we make some ergonomic concessions (caching
// messages, etc).
//===----------------------------------------------------------------------===//

class ErrorInstance {
 public:
  ErrorInstance(iree_status_t status) : status_(status) {}
  ~ErrorInstance() { iree_status_ignore(status_); }
  static void BindApi(PJRT_Api* api);

  static const ErrorInstance* FromError(const PJRT_Error* error) {
    return reinterpret_cast<const ErrorInstance*>(error);
  }

  iree_status_t status() const { return status_; }
  const std::string& message() const;

 private:
  iree_status_t status_;
  mutable std::string cached_message_;
};

inline PJRT_Error* MakeError(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return nullptr;
  }
  auto alloced_error = std::make_unique<ErrorInstance>(status);
  return reinterpret_cast<PJRT_Error*>(alloced_error.release());
}

//===----------------------------------------------------------------------===//
// BufferInstance
//===----------------------------------------------------------------------===//

class BufferInstance {
 public:
  BufferInstance(DeviceInstance& device,
                 iree::vm::ref<iree_hal_buffer_view_t> buffer_view);
  ~BufferInstance();
  operator PJRT_Buffer*() { return reinterpret_cast<PJRT_Buffer*>(this); }
  static BufferInstance* Unwrap(PJRT_Buffer* buffer) {
    return reinterpret_cast<BufferInstance*>(buffer);
  }
  static void BindApi(PJRT_Api* api);

  iree_hal_buffer_view_t* buffer_view() { return buffer_view_.get(); }
  DeviceInstance& device() { return device_; }
  iree_status_t AsyncDeallocate();
  bool is_deleted() { return false; }
  bool is_on_cpu() {
    // TODO: Plumb through an indication if running on CPU and then implement
    // the hook to get an unsafe pointer (avoids a copy).
    return false;
  }
  iree_status_t GetXlaShape(xla::Shape** out_shape);

  // Gets the required host size in bytes to copy to host.
  iree_status_t GetHostSizeInBytes(iree_host_size_t* host_size);
  iree_status_t CopyToHost(void* dst, iree_host_size_t dst_size,
                           EventInstance** done_event);

  // Advance the ready and done fences.
  iree_status_t AdvanceReadyFence(iree_hal_semaphore_t* semaphore,
                                  uint64_t timepoint);
  iree_status_t AdvanceDoneFence(iree_hal_semaphore_t* semaphore,
                                 uint64_t timepoint);

  iree_hal_fence_t* ready_fence() { return ready_fence_.get(); }
  iree_hal_fence_t* done_fence() { return done_fence_.get(); }

 private:
  DeviceInstance& device_;
  iree::vm::ref<iree_hal_buffer_view_t> buffer_view_;
  // Various things require XLA's idea of shapes, layouts, etc.
  // We keep one around for such cases.
  std::optional<xla::Shape> cached_shape_;

  // Fences.
  // ready_fence_: Signalled when the buffer is ready to be consumed. Consumers
  //   should wait on this fence.
  // done_fence_: Signalled when all scheduled work on the buffer is done.
  //   Consumers should advance this fence when using it.
  iree::vm::ref<iree_hal_fence_t> ready_fence_;
  iree::vm::ref<iree_hal_fence_t> done_fence_;
};

//===----------------------------------------------------------------------===//
// DeviceInstance
//===----------------------------------------------------------------------===//

class DeviceInstance {
 public:
  DeviceInstance(int client_id, ClientInstance& client,
                 iree_hal_driver_t* driver, iree_hal_device_info_t* info)
      : client_id_(client_id), client_(client), driver_(driver), info_(info) {}
  ~DeviceInstance();
  operator PJRT_Device*() { return reinterpret_cast<PJRT_Device*>(this); }
  static void BindApi(PJRT_Api* api);
  static DeviceInstance* Unwrap(PJRT_Device* device) {
    return reinterpret_cast<DeviceInstance*>(device);
  }

  // Since the PJRT device id is a simple int and the IREE device_id is
  // a pointer-sized value, we just assign a synthetic id. Currently, this
  // is the offset into the devices() array on the client. Will need to be
  // revisited if ever supporting re-scanning (but many things would seem to
  // need updates then).
  int client_id() { return client_id_; }
  iree_hal_device_info_t* info() { return info_; }

  // Not yet implemented but plumbed through.
  bool is_addressable() { return true; }
  int process_index() { return 0; }
  int local_hardware_id() { return -1; }

  // Various debug descriptions of the device. Backing string data owned by
  // the device.
  std::string_view kind_string() { return kind_string_; }
  std::string_view debug_string() { return debug_string_; }
  std::string_view user_string() { return user_string_; }

  // Copies a host buffer to the device.
  // See PJRT_Client_BufferFromHostBuffer
  iree_status_t HostBufferToDevice(
      const void* data, PJRT_Buffer_Type type, const int64_t* dims,
      size_t num_dims, const int64_t* byte_strides, size_t num_byte_strides,
      PJRT_HostBufferSemantics host_buffer_semantics,
      EventInstance** out_done_with_host_buffer_event,
      BufferInstance** out_buffer);

  // TODO(laurenzo): Eagerly set up device to allow simple access.
  iree_status_t GetHalDevice(iree_hal_device_t** out_device);

  // Only valid once device opened.
  iree_hal_semaphore_t* main_timeline() { return main_timeline_.get(); }

  iree_hal_device_t* device() { return device_.get(); }
  iree_hal_allocator_t* device_allocator() {
    return iree_hal_device_allocator(device_.get());
  }
  // Creates a fence sized to the maximum number of semaphores in use by the
  // device.
  iree_status_t CreateFence(iree_hal_fence_t** out_fence);

 private:
  iree_status_t OpenDevice();
  iree_status_t AcquireHostStagingBuffer(
      iree_const_byte_span_t initial_contents,
      bool snapshot_initial_contents_now, bool* initial_contents_snapshotted,
      iree_hal_buffer_t** out_buffer);

  int client_id_;
  ClientInstance& client_;
  iree_hal_driver_t* driver_;  // Owned by client.
  iree::vm::ref<iree_hal_device_t> device_;
  iree::vm::ref<iree_hal_semaphore_t> main_timeline_;
  iree::vm::ref<iree_hal_semaphore_t> transfer_timeline_;
  // A fence that is initialized to the start of the transfer timeline,
  // effectively being signalled immediately.
  iree::vm::ref<iree_hal_fence_t> transfer_now_fence_;
  // The timepoint of the last transfer.
  uint64_t last_transfer_timepoint_ = 0;
  iree_hal_device_info_t* info_;

  // Debug strings (owned by device).
  std::string kind_string_;
  std::string debug_string_;
  std::string user_string_;
};

//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

class EventInstance {
 public:
  enum class Type {
    // An event that is always signalled.
    SIGNALLED,

    // An EXTERNAL event will have an outside caller invoke
    // ExternalSignalReady() when it is ready.
    // Any further OnReady callback will happen within the context of that
    // call (which can be on any thread).
    EXTERNAL,
  };

  // Default construction is always signalled.
  EventInstance(Type type);
  ~EventInstance();
  operator PJRT_Event*() { return reinterpret_cast<PJRT_Event*>(this); }
  static void BindApi(PJRT_Api* api);
  static EventInstance* Unwrap(PJRT_Event* exe) {
    return reinterpret_cast<EventInstance*>(exe);
  }

  iree_status_t OnReady(PJRT_Event_OnReadyCallback callback, void* user_arg);
  ErrorInstance* error();
  bool is_ready();

  // For EXTERNAL events: Signals that the event is ready.
  void ExternalSignalReady(iree_status_t status);

 private:
  std::mutex lock_;
  Type type_;
  iree_status_t status_ = iree_ok_status();
  bool is_ready_;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void*>> pending_callbacks_;
};

//===----------------------------------------------------------------------===//
// LoadedExecutableInstance
// PJRT models a LoadedExecutable, which can produce an Executable that can
// be used to serialize and get metadata.
// We have one additional layer, because our executables are just flat binary
// data until loaded onto a device context. We call this a ResidentExecutable
// to avoid name collisions.
//
// Correspondance:
//   PJRT_Executable -> ExecutableImage
//   PJRT_LoadedExecutable -> LoadedExecutableInstance
//   <None> -> ResidentExecutable
//
// Since we need to query metadata from a ResidentExecutable, we populate the
// metadata on an ExecutableImage lazily before returning it. The
// ExecutableImage is managed with ref counted semantics and owns the backing
// binary data (as well as doubling as a user-level PJRT_Executable).
//===----------------------------------------------------------------------===//

struct ExecutableImage {
  ExecutableImage(std::unique_ptr<CompilerOutput> binary)
      : ref_count(1), binary(std::move(binary)) {}
  operator PJRT_Executable*() {
    return reinterpret_cast<PJRT_Executable*>(this);
  }
  static ExecutableImage* Unwrap(PJRT_Executable* exe) {
    return reinterpret_cast<ExecutableImage*>(exe);
  }
  static void BindApi(PJRT_Api* api);

  void AddRef() { ref_count.fetch_add(1); }
  void DecRef() {
    if (ref_count.fetch_sub(1) == 0) {
      delete this;
    }
  }

 private:
  // The reference count. Must be disposed when reaching zero.
  std::atomic<int> ref_count;

 public:
  // Raw compiler output.
  std::unique_ptr<CompilerOutput> binary;

  // Meta-data about the executable is lazily set when an Executable is obtained
  // from a LoadedExecutable.
  iree_host_size_t arg_count;
  iree_host_size_t result_count;
  bool metadata_initialized = false;
};

// An executable loaded on all available devices.
struct ResidentExecutable {
  DeviceInstance* device_instance;
  iree::vm::ref<iree_vm_context_t> vm_context;
  iree::vm::ref<iree_vm_module_t> main_module;
  iree_vm_function_t main_function;
  iree_host_size_t arg_count;
  iree_host_size_t result_count;
};

class LoadedExecutableInstance {
 public:
  LoadedExecutableInstance(
      ClientInstance& client, ExecutableImage* image,
      const std::vector<DeviceInstance*>& addressable_devices)
      : client_(client),
        image_(image),
        addressable_devices_(addressable_devices) {}
  ~LoadedExecutableInstance() { image_->DecRef(); }

  operator PJRT_LoadedExecutable*() {
    return reinterpret_cast<PJRT_LoadedExecutable*>(this);
  }
  static void BindApi(PJRT_Api* api);
  static LoadedExecutableInstance* Unwrap(PJRT_LoadedExecutable* exe) {
    return reinterpret_cast<LoadedExecutableInstance*>(exe);
  }

  const std::vector<DeviceInstance*>& addressable_devices() {
    return addressable_devices_;
  }

  // Loads all executables to addressable devices.
  iree_status_t LoadAll();

  // Gets one loaded executable that can be used for querying metadata
  // and such.
  iree_status_t GetDefaultResidentExecutable(ResidentExecutable** out_loaded);

  // Gets the number of outputs.
  iree_status_t GetArgResultCount(iree_host_size_t* out_arg_count,
                                  iree_host_size_t* out_result_count);

  // Executes on a batch of devices. Since this is a complicated call,
  // we just give it the raw C argument struct vs breaking it down.
  iree_status_t BatchExecute(PJRT_LoadedExecutable_Execute_Args* args);

 private:
  ClientInstance& client_;
  ExecutableImage* image_;  // Ref-counted semantics.
  std::vector<DeviceInstance*> addressable_devices_;
  std::vector<ResidentExecutable> resident_executables_;
};

//===----------------------------------------------------------------------===//
// ClientInstance
// The root of the runtime hierarchy, these map to an IREE driver and are
// created against an API.
//===----------------------------------------------------------------------===//

struct ClientInstance {
 public:
  ClientInstance(std::unique_ptr<Platform> platform);
  virtual ~ClientInstance();

  // Binds monomorphic entry-points for the client.
  static void BindApi(PJRT_Api* api);

  static ClientInstance* Unwrap(PJRT_Client* client) {
    return reinterpret_cast<ClientInstance*>(client);
  }

  // Before the client is usable, it must be initialized.
  PJRT_Error* Initialize();

  Platform& platform() { return *platform_; }
  Logger& logger() { return platform_->logger(); }
  iree_allocator_t host_allocator() { return host_allocator_; }
  const std::vector<DeviceInstance*>& devices() { return devices_; }
  const std::vector<DeviceInstance*>& addressable_devices() {
    return addressable_devices_;
  }
  const std::string& cached_platform_name() { return cached_platform_name_; }
  const std::string& cached_platform_version() {
    return cached_platform_version_;
  }

  iree_vm_instance_t* vm_instance() { return vm_instance_.get(); }

  // Compiles.
  // See TODOs in PJRT_Client_Compile.
  PJRT_Error* Compile(PJRT_Program* program,
                      LoadedExecutableInstance** executable);

  // ---------------------------------------------------------------------------
  // Subclass hooks.
  // ---------------------------------------------------------------------------
  // Must be defined by concrete subclasses.
  virtual iree_status_t CreateDriver(iree_hal_driver_t** out_driver) = 0;

  // Populates the list of modules to load into a context for an executable
  // on a device. This can be customized by subclasses. The default
  // implementation constructs a hal module and appends:
  //   {hal_module, main_module}.
  virtual iree_status_t PopulateVMModules(
      std::vector<iree::vm::ref<iree_vm_module_t>>& modules,
      iree_hal_device_t* hal_device,
      iree::vm::ref<iree_vm_module_t>& main_module);

  // Sets default compiler flags for the client which apply to all executables
  // and devices.
  // Returns false on failure (and sets error information on the compiler_job).
  virtual bool SetDefaultCompilerFlags(CompilerJob* compiler_job) = 0;

  // Advances the timeline, returning (current, next) time point values.
  std::tuple<uint64_t, uint64_t> AdvanceTimeline();

 protected:
  iree_allocator_t host_allocator_;
  std::string cached_platform_name_;
  std::string cached_platform_version_;

 private:
  iree_status_t InitializeCompiler();
  iree_status_t InitializeVM();
  iree_status_t PopulateDevices();

  std::unique_ptr<Platform> platform_;

  // HAL.
  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_info_t* device_infos_ = nullptr;
  iree_host_size_t device_info_count_ = 0;
  std::vector<DeviceInstance*> devices_;
  std::vector<DeviceInstance*> addressable_devices_;

  // VM.
  iree::vm::ref<iree_vm_instance_t> vm_instance_;

  // Synchronization.
  // We keep one global execution timeline across all devices. The management
  // of this is currently somewhat primitive: we increment it by one for each
  // invocation. Batch invocations (i.e. across multiple devices), only
  // increment by one. In the future, additional parallelism could be plumbed
  // up to the framework to allow different kinds of timeline management.
  // Waiting on the current value of |execution_timeline_| will drain all
  // scheduled work to date.
  uint64_t execution_timeline_ = 0ull;
};

//===----------------------------------------------------------------------===//
// API binding
//===----------------------------------------------------------------------===//

// Binds all monomorphic API members and top-level API struct setup.
void BindMonomorphicApi(PJRT_Api* api);

// Fully binds the PJRT_Api struct for all types. Polymorphic types must be
// specified by template parameters.
template <typename PlatformTy, typename ClientInstanceTy>
static void BindApi(PJRT_Api* api) {
  BindMonomorphicApi(api);

  // Bind polymorphic entry-points.
  api->PJRT_Client_Create = +[](PJRT_Client_Create_Args* args) -> PJRT_Error* {
    auto platform = std::make_unique<PlatformTy>();

    // TODO: Once a client can be created with config, use it to populate
    // platform->config_vars().
    auto status = platform->Initialize();
    if (!iree_status_is_ok(status)) {
      return MakeError(status);
    }

    auto client = std::make_unique<ClientInstanceTy>(std::move(platform));
    auto* error = client->Initialize();
    if (error) return error;

    // Successful return.
    args->client = reinterpret_cast<PJRT_Client*>(client.release());
    return nullptr;
  };
}

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_COMMON_API_IMPL_H_
