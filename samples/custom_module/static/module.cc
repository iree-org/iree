// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree/vm/native_module_cc.h"

// NOTE: this module is written in C++ using the native module wrapper and uses
// template magic to handle marshaling arguments. For a lot of uses this is a
// much friendlier way of exposing modules to the IREE VM and if performance and
// code size are not a concern is a fine route to take. Here we do it for
// brevity but all of the internal IREE modules are implemented in C.

//===----------------------------------------------------------------------===//
// !custom.string type
//===----------------------------------------------------------------------===//

// The "string" type we use to store and retain string data.
// This could be arbitrarily complex or simply wrap another user-defined type.
// The descriptor that is registered at startup defines how to manage the
// lifetime of the type (such as which destruction function is called, if any).
// See ref.h for more information and additional utilities.
typedef struct iree_sample_string_t {
  // Must be the first field; used to track the reference count of the object.
  iree_vm_ref_object_t ref_object;
  // Allocator the string data was allocated from.
  // Ideally pools and nested allocators would be used to avoid needing to store
  // the allocator with every object.
  iree_allocator_t allocator;
  // Non-NUL-terminated string value.
  iree_string_view_t value;
} iree_sample_string_t;

// Runtime type descriptor for the !custom.string describing how to manage it
// and destroy it. The type ID is allocated at runtime and does not need to
// match the compiler ID.
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_sample_string, iree_sample_string_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_sample_string, iree_sample_string_t);

// Creates a new !custom.string object with a copy of the given |value|.
// Applications could use this and any other methods we wanted to expose to
// interop with the loaded VM modules - such as passing in/out the objects.
// We don't need this for the demo but creating the sample object, appending it
// to the invocation input list, and then consuming it in the compiled module
// is straightforward.
static iree_status_t iree_sample_string_create(
    iree_string_view_t value, iree_allocator_t allocator,
    iree_sample_string_t** out_string) {
  IREE_ASSERT_ARGUMENT(out_string);
  // Note that we allocate the string and the string value together.
  iree_sample_string_t* string = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(*string) + value.size, (void**)&string));
  string->ref_object.counter = IREE_ATOMIC_VAR_INIT(1);
  string->allocator = allocator;
  string->value.data = ((const char*)string) + sizeof(iree_sample_string_t);
  string->value.size = value.size;
  memcpy((void*)string->value.data, value.data, string->value.size);
  *out_string = string;
  return iree_ok_status();
}

static void iree_sample_string_destroy(void* ptr) {
  iree_sample_string_t* string = (iree_sample_string_t*)ptr;
  iree_allocator_free(string->allocator, ptr);
}

static iree_vm_ref_type_descriptor_t iree_sample_string_descriptor_storage = {
    0};

// Registers types provided by the sample module.
// We must call this before any of our types can be resolved.
static iree_status_t iree_sample_module_basic_register_types(
    iree_vm_instance_t* instance) {
  iree_sample_string_descriptor_storage.destroy = iree_sample_string_destroy;
  iree_sample_string_descriptor_storage.type_name = IREE_SV("custom.string");
  iree_sample_string_descriptor_storage.offsetof_counter =
      offsetof(iree_sample_string_t, ref_object.counter) /
      IREE_VM_REF_COUNTER_ALIGNMENT;
  return iree_vm_instance_register_type(instance,
                                        &iree_sample_string_descriptor_storage,
                                        &iree_sample_string_registration);
}

// Unregisters types previously registered.
// In dynamic modules it's critical that types are unregistered before the
// library is unloaded.
static void iree_sample_module_basic_unregister_types(
    iree_vm_instance_t* instance) {
  iree_vm_instance_unregister_type(instance,
                                   &iree_sample_string_descriptor_storage);
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

namespace {

using namespace iree;

// Per-context module state.
// This can contain "globals" and other arbitrary state.
//
// Thread-compatible; the runtime will not issue multiple calls at the same
// time using the same state. If the implementation uses external threads then
// it must synchronize itself.
class CustomModuleState final {
 public:
  explicit CustomModuleState(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}
  ~CustomModuleState() = default;

  // Creates a new string with a copy of the given string data.
  // No NUL terminator is required.
  StatusOr<vm::ref<iree_sample_string_t>> StringFromTensor(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    char string_buffer[512];
    iree_host_size_t string_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_format(
        buffer_view.get(), 128, IREE_ARRAYSIZE(string_buffer), string_buffer,
        &string_length));

    vm::ref<iree_sample_string_t> string;
    IREE_RETURN_IF_ERROR(iree_sample_string_create(
        iree_make_string_view(string_buffer, string_length), host_allocator_,
        &string));
    fprintf(stdout, "CREATE %.*s\n", static_cast<int>(string->value.size),
            string->value.data);
    fflush(stdout);
    return std::move(string);
  }

  // Prints the contents of the string to stdout.
  Status StringPrint(const vm::ref<iree_sample_string_t> string) {
    if (!string) return OkStatus();  // no-op
    fprintf(stdout, "PRINT %.*s\n", static_cast<int>(string->value.size),
            string->value.data);
    fflush(stdout);
    return OkStatus();
  }

 private:
  // Allocator that the caller requested we use for any allocations we need to
  // perform during operation.
  iree_allocator_t host_allocator_;
};

// Function table mapping imported function names to their implementation.
static const vm::NativeFunction<CustomModuleState> kCustomModuleFunctions[] = {
    vm::MakeNativeFunction("string.from_tensor",
                           &CustomModuleState::StringFromTensor),
    vm::MakeNativeFunction("string.print", &CustomModuleState::StringPrint),
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

  ~CustomModule() override {
    iree_sample_module_basic_unregister_types(instance());
  }

  // Creates per-context state when the module is added to a new context.
  // May be called from any thread.
  StatusOr<std::unique_ptr<CustomModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    auto state = std::make_unique<CustomModuleState>(host_allocator);
    return state;
  }
};

}  // namespace

// Registers types exported by the module on |instance|.
// This is only required when statically linking modules into tooling as dynamic
// dependency resolution is performed out of order from creation. Normally if
// there's 'base' and 'derived' modules the 'base' module is expected to have
// been created and thus registered its types before the 'derived' module but
// the tooling will try to create 'derived' first. This can be changed in the
// future by making modules lookup types when instantiated in a context instead
// of when created but that's TBD.
extern "C" iree_status_t register_sample_module_types(
    iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(iree_sample_module_basic_register_types(instance));
  return iree_ok_status();
}

// Creates a native sample module that can be reused in multiple contexts.
// The module itself may hold state that can be shared by all instantiated
// copies but it will require the module to provide synchronization; usually
// it's safer to just treat the module as immutable and keep state within the
// instantiated module states instead.
//
// Note that while we are using C++ bindings internally we still expose the
// module as a C instance. This hides the details of our implementation and
// is required for working across the dynamic library boundary.
extern "C" iree_status_t create_sample_module(iree_vm_instance_t* instance,
                                              iree_allocator_t host_allocator,
                                              iree_vm_module_t** out_module) {
  // Register sample types used by the module against the instance.
  // Note that this function must be safe to call multiple times as the module
  // may be loaded multiple times.
  IREE_RETURN_IF_ERROR(iree_sample_module_basic_register_types(instance));

  // Create the sample module and return it to the runtime.
  // NOTE: this isn't using the allocator here and that's bad as it leaves
  // untracked allocations and pulls in the system allocator that may differ
  // from the one requested by the user.
  // TODO(benvanik): std::allocator wrapper around iree_allocator_t so this can
  // use that instead.
  auto module = std::make_unique<CustomModule>(
      "custom", /*version=*/0, instance, host_allocator,
      iree::span<const vm::NativeFunction<CustomModuleState>>(
          kCustomModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}
