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

#include "iree/samples/custom_modules/native_module.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "iree/vm2/stack.h"

// These functions match the imports defined in custom.imports.mlir, though the
// ordinals are only used internally (string lookups are used at runtime). We
// could have a fancier compile-time generated structure for this, possibly
// derived from the imports file, however this is fine for now.
#define kTotalExportCount 3
#define kExportOrdinalPrint 0
#define kExportOrdinalReverse 1
#define kExportOrdinalGetUniqueMessage 2

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

// The module instance that will be allocated and reused across contexts.
// Any context-specific state must be stored in a state structure such as
// iree_custom_native_module_state_t below.
//
// Assumed thread-safe (by construction here, as it's immutable), though if more
// state is stored here it will need to be synchronized by the implementation.
typedef struct {
  // Must be first in the struct as we dereference the interface to find our
  // members below.
  iree_vm_module_t interface;

  // Allocator this module was allocated with and must be freed with.
  iree_allocator_t allocator;

  // The next ID to allocate for states that will be used to form the
  // per-context unique message. This shows state at the module level. Note that
  // this must be thread-safe.
  atomic_int next_unique_id;
} iree_custom_native_module_t;

// Per-context module state.
// This can contain "globals" and other arbitrary state.
//
// Thread-compatible; the runtime will not issue multiple calls at the same
// time using the same state. If the implementation uses external threads then
// it must synchronize itself.
typedef struct {
  // Allocator this state was allocated with and must be freed with.
  iree_allocator_t allocator;

  // A unique message owned by the state and returned to the VM.
  // This demonstrates any arbitrary per-context state one may want to store.
  iree_vm_ref_t unique_message;
} iree_custom_native_module_state_t;

//===----------------------------------------------------------------------===//
// !custom.message type
//===----------------------------------------------------------------------===//

// Runtime type descriptor for the !custom.message describing how to manage it
// and destroy it. The type ID is allocated at runtime and does not need to
// match the compiler ID.
static iree_vm_ref_type_descriptor_t iree_custom_message_descriptor = {0};
#define IREE_CUSTOM_MESSAGE_TYPE_ID iree_custom_message_descriptor.type

// The "message" type we use to store string messages to print.
// This could be arbitrarily complex or simply wrap another user-defined type.
// The descriptor that is registered at startup defines how to manage the
// lifetime of the type (such as which destruction function is called, if any).
// See ref.h for more information and additional utilities.
typedef struct iree_custom_message {
  // Ideally first; used to track the reference count of the object.
  iree_vm_ref_object_t ref_object;
  // Allocator the message was created from.
  // Ideally pools and nested allocators would be used to avoid needing to store
  // the allocator with every object.
  iree_allocator_t allocator;
  // String message value.
  iree_string_view_t value;
} iree_custom_message_t;

iree_status_t iree_custom_message_create(iree_string_view_t value,
                                         iree_allocator_t allocator,
                                         iree_vm_ref_t* out_message_ref) {
  // Note that we allocate the message and the string value together.
  iree_custom_message_t* message = NULL;
  IREE_API_RETURN_IF_API_ERROR(allocator.alloc(
      allocator.self, sizeof(iree_custom_message_t) + value.size,
      (void**)&message));
  memset(message, 0, sizeof(iree_custom_message_t));
  message->ref_object.counter = 1;
  message->allocator = allocator;
  message->value.data = (uint8_t*)message + sizeof(iree_custom_message_t);
  message->value.size = value.size;
  memcpy((void*)message->value.data, value.data, message->value.size);
  return iree_vm_ref_wrap(message, IREE_CUSTOM_MESSAGE_TYPE_ID,
                          out_message_ref);
}

iree_status_t iree_custom_message_wrap(iree_string_view_t value,
                                       iree_allocator_t allocator,
                                       iree_vm_ref_t* out_message_ref) {
  iree_custom_message_t* message = NULL;
  IREE_API_RETURN_IF_API_ERROR(allocator.alloc(
      allocator.self, sizeof(iree_custom_message_t), (void**)&message));
  message->ref_object.counter = 1;
  message->allocator = allocator;
  message->value = value;  // Unowned.
  return iree_vm_ref_wrap(message, IREE_CUSTOM_MESSAGE_TYPE_ID,
                          out_message_ref);
}

void iree_custom_message_destroy(void* ptr) {
  iree_custom_message_t* message = (iree_custom_message_t*)ptr;
  message->allocator.free(message->allocator.self, ptr);
}

iree_status_t iree_custom_message_read_value(iree_vm_ref_t* message_ref,
                                             char* buffer,
                                             size_t buffer_capacity) {
  IREE_VM_DEREF_OR_RETURN(iree_custom_message_t, message, message_ref,
                          IREE_CUSTOM_MESSAGE_TYPE_ID);
  if (buffer_capacity < message->value.size + 1) {
    return IREE_STATUS_OUT_OF_RANGE;
  }
  memcpy(buffer, message->value.data, message->value.size);
  buffer[message->value.size] = 0;
  return IREE_STATUS_OK;
}

iree_status_t iree_custom_native_module_register_types() {
  if (iree_custom_message_descriptor.type) {
    return IREE_STATUS_OK;  // Already registered.
  }
  iree_custom_message_descriptor.type_name = iree_make_cstring_view("message");
  iree_custom_message_descriptor.offsetof_counter =
      offsetof(iree_custom_message_t, ref_object.counter);
  iree_custom_message_descriptor.destroy = iree_custom_message_destroy;
  return iree_vm_ref_register_user_defined_type(
      &iree_custom_message_descriptor);
}

//===----------------------------------------------------------------------===//
// Method thunks
//===----------------------------------------------------------------------===//
// Ideally we would autogenerate these from the imports file or have some
// compile-time magic. For now this is all still experimental and we do it by
// hand.

static iree_status_t iree_custom_native_print_thunk(
    iree_custom_native_module_t* module,
    iree_custom_native_module_state_t* state, iree_vm_stack_t* stack,
    iree_vm_stack_frame_t* frame, iree_vm_execution_result_t* out_result) {
  frame->registers.ref_register_count = 1;
  IREE_VM_DEREF_OR_RETURN(iree_custom_message_t, message,
                          &frame->registers.ref[0],
                          IREE_CUSTOM_MESSAGE_TYPE_ID);
  int count = frame->registers.i32[0];
  for (int i = 0; i < count; ++i) {
    fwrite(message->value.data, 1, message->value.size, stdout);
    fputc('\n', stdout);
  }
  fflush(stdout);
  return IREE_STATUS_OK;
}

static iree_status_t iree_custom_native_reverse_thunk(
    iree_custom_native_module_t* module,
    iree_custom_native_module_state_t* state, iree_vm_stack_t* stack,
    iree_vm_stack_frame_t* frame, iree_vm_execution_result_t* out_result) {
  frame->registers.ref_register_count = 1;
  IREE_VM_DEREF_OR_RETURN(iree_custom_message_t, src_message,
                          &frame->registers.ref[0],
                          IREE_CUSTOM_MESSAGE_TYPE_ID);
  IREE_API_RETURN_IF_API_ERROR(iree_custom_message_create(
      src_message->value, src_message->allocator, &frame->registers.ref[0]));
  IREE_VM_DEREF_OR_RETURN(iree_custom_message_t, dst_message,
                          &frame->registers.ref[0],
                          IREE_CUSTOM_MESSAGE_TYPE_ID);
  char* str_ptr = (char*)dst_message->value.data;
  for (int low = 0, high = dst_message->value.size - 1; low < high;
       ++low, --high) {
    char temp = str_ptr[low];
    str_ptr[low] = str_ptr[high];
    str_ptr[high] = temp;
  }
  // TODO(benvanik): replace with macro? helper for none/i32/etc
  static const union {
    uint8_t reserved[2];
    iree_vm_register_list_t list;
  } return_registers = {
      {1, 0 | IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT}};
  frame->return_registers = &return_registers.list;
  return IREE_STATUS_OK;
}

static iree_status_t iree_custom_native_get_unique_message_thunk(
    iree_custom_native_module_t* module,
    iree_custom_native_module_state_t* state, iree_vm_stack_t* stack,
    iree_vm_stack_frame_t* frame, iree_vm_execution_result_t* out_result) {
  // TODO(benvanik): replace with macro? helper for none/i32/etc
  static const union {
    uint8_t reserved[2];
    iree_vm_register_list_t list;
  } return_registers = {
      {1, 0 | IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT}};
  frame->return_registers = &return_registers.list;
  frame->registers.ref_register_count = 1;
  memset(&frame->registers.ref[0], 0, sizeof(iree_vm_ref_t));
  iree_vm_ref_retain(&state->unique_message, &frame->registers.ref[0]);
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_custom_native_module_destroy(void* self) {
  iree_custom_native_module_t* module = (iree_custom_native_module_t*)self;
  return module->allocator.free(module->allocator.self, module);
}

static iree_string_view_t iree_custom_native_module_name(void* self) {
  return iree_make_cstring_view("custom");
}

static iree_vm_module_signature_t iree_custom_native_module_signature(
    void* self) {
  iree_vm_module_signature_t signature;
  memset(&signature, 0, sizeof(signature));
  signature.import_function_count = 0;
  signature.export_function_count = kTotalExportCount;
  signature.internal_function_count = 0;
  return signature;
}

static iree_status_t iree_custom_native_module_get_function(
    void* self, iree_vm_function_linkage_t linkage, int32_t ordinal,
    iree_vm_function_t* out_function, iree_string_view_t* out_name,
    iree_vm_function_signature_t* out_signature) {
  if (out_function) {
    memset(out_function, 0, sizeof(iree_vm_function_t));
  }
  if (out_name) {
    out_name->data = NULL;
    out_name->size = 0;
  }
  if (out_signature) {
    memset(out_signature, 0, sizeof(iree_vm_function_signature_t));
  }

  const char* name = NULL;
  switch (ordinal) {
    case kExportOrdinalPrint:
      name = "print";
      if (out_signature) {
        out_signature->argument_count = 2;
        out_signature->result_count = 0;
      }
      break;
    case kExportOrdinalReverse:
      name = "reverse";
      if (out_signature) {
        out_signature->argument_count = 1;
        out_signature->result_count = 1;
      }
      break;
    case kExportOrdinalGetUniqueMessage:
      name = "get_unique_message";
      if (out_signature) {
        out_signature->argument_count = 0;
        out_signature->result_count = 1;
      }
      break;
    default:
      // Invalid function ordinal.
      return IREE_STATUS_INVALID_ARGUMENT;
  }
  if (out_function) {
    iree_custom_native_module_t* module = (iree_custom_native_module_t*)self;
    out_function->module = &module->interface;
    out_function->linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT;
    out_function->ordinal = ordinal;
  }
  if (out_name && name) {
    *out_name = iree_make_cstring_view(name);
  }

  return IREE_STATUS_OK;
}

static iree_status_t iree_custom_native_module_lookup_function(
    void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
    iree_vm_function_t* out_function) {
  if (!out_function) return IREE_STATUS_INVALID_ARGUMENT;
  memset(out_function, 0, sizeof(iree_vm_function_t));

  if (!name.data || !name.size) return IREE_STATUS_INVALID_ARGUMENT;

  iree_custom_native_module_t* module = (iree_custom_native_module_t*)self;
  out_function->module = &module->interface;
  out_function->linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT;
  if (iree_string_view_compare(name, iree_make_cstring_view("print")) == 0) {
    out_function->ordinal = kExportOrdinalPrint;
  } else if (iree_string_view_compare(name,
                                      iree_make_cstring_view("reverse")) == 0) {
    out_function->ordinal = kExportOrdinalReverse;
  } else if (iree_string_view_compare(
                 name, iree_make_cstring_view("get_unique_message")) == 0) {
    out_function->ordinal = kExportOrdinalGetUniqueMessage;
  } else {
    return IREE_STATUS_NOT_FOUND;
  }
  return IREE_STATUS_OK;
}

static iree_status_t iree_custom_native_module_alloc_state(
    void* self, iree_allocator_t allocator,
    iree_vm_module_state_t** out_module_state) {
  if (!out_module_state) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module_state = NULL;

  iree_custom_native_module_t* module = (iree_custom_native_module_t*)self;

  iree_custom_native_module_state_t* state = NULL;
  IREE_API_RETURN_IF_API_ERROR(
      allocator.alloc(allocator.self, sizeof(iree_custom_native_module_state_t),
                      (void**)&state));
  memset(state, 0, sizeof(iree_custom_native_module_state_t));
  state->allocator = allocator;

  // Allocate a unique ID to demonstrate per-context state.
  int unique_id = atomic_fetch_add(&module->next_unique_id, 1);
  char buffer[16];
  snprintf(buffer, 16, "ctx_%d", unique_id);
  IREE_API_RETURN_IF_API_ERROR(iree_custom_message_create(
      iree_make_cstring_view(buffer), allocator, &state->unique_message));

  *out_module_state = (iree_vm_module_state_t*)state;
  return IREE_STATUS_OK;
}

static iree_status_t iree_custom_native_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  iree_custom_native_module_state_t* state =
      (iree_custom_native_module_state_t*)module_state;
  if (!state) return IREE_STATUS_INVALID_ARGUMENT;

  // Release any state we have been holding on to.
  iree_vm_ref_release(&state->unique_message);

  return state->allocator.free(state->allocator.self, module_state);
}

static iree_status_t iree_custom_native_module_resolve_import(
    void* self, iree_vm_module_state_t* module_state, int32_t ordinal,
    iree_vm_function_t function) {
  // Module does not have imports.
  return IREE_STATUS_FAILED_PRECONDITION;
}

static iree_status_t iree_custom_native_module_execute(
    void* self, iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame,
    iree_vm_execution_result_t* out_result) {
  if (!out_result) return IREE_STATUS_INVALID_ARGUMENT;
  memset(out_result, 0, sizeof(iree_vm_execution_result_t));
  if (!stack || !frame) return IREE_STATUS_INVALID_ARGUMENT;
  if (frame->function.module != self) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  iree_custom_native_module_t* module = (iree_custom_native_module_t*)self;
  iree_custom_native_module_state_t* state =
      (iree_custom_native_module_state_t*)frame->module_state;
  switch (frame->function.ordinal) {
    case kExportOrdinalPrint:
      return iree_custom_native_print_thunk(module, state, stack, frame,
                                            out_result);
    case kExportOrdinalReverse:
      return iree_custom_native_reverse_thunk(module, state, stack, frame,
                                              out_result);
    case kExportOrdinalGetUniqueMessage:
      return iree_custom_native_get_unique_message_thunk(module, state, stack,
                                                         frame, out_result);
    default:
      // Invalid function ordinal.
      return IREE_STATUS_INVALID_ARGUMENT;
  }
}

iree_status_t iree_custom_native_module_create(iree_allocator_t allocator,
                                               iree_vm_module_t** out_module) {
  if (!out_module) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module = NULL;

  iree_custom_native_module_t* module = NULL;
  IREE_API_RETURN_IF_API_ERROR(allocator.alloc(
      allocator.self, sizeof(iree_custom_native_module_t), (void**)&module));
  module->allocator = allocator;
  module->next_unique_id = 0;

  module->interface.self = module;
  module->interface.destroy = iree_custom_native_module_destroy;
  module->interface.name = iree_custom_native_module_name;
  module->interface.signature = iree_custom_native_module_signature;
  module->interface.get_function = iree_custom_native_module_get_function;
  module->interface.lookup_function = iree_custom_native_module_lookup_function;
  module->interface.alloc_state = iree_custom_native_module_alloc_state;
  module->interface.free_state = iree_custom_native_module_free_state;
  module->interface.resolve_import = iree_custom_native_module_resolve_import;
  module->interface.execute = iree_custom_native_module_execute;

  *out_module = &module->interface;
  return IREE_STATUS_OK;
}
