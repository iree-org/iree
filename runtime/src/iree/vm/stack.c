// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/stack.h"

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/vm/module.h"

//===----------------------------------------------------------------------===//
// Stack implementation
//===----------------------------------------------------------------------===//
//
// The stack is (currently) designed to contain enough information to allow us
// to build some nice debugging tools. This means that we try hard to preserve
// all information needed for complete and precise stack dumps as well as
// allowing inspection of both current and previous stack frame registers.
// In the future we may want to toggle these modes such that registers, for
// example, are hidden by the module implementations to allow for more
// optimization opportunity but as a whole we tradeoff minimal memory
// consumption for flexibility and debugging. Given that a single activation
// tensor will usually dwarf the entire size of the stack used for an invocation
// it's generally acceptable :)
//
// Stack frames and storage
// ------------------------
// Frames are stored as a linked list of iree_vm_stack_frame_header_t's
// containing the API-visible stack frame information (such as which function
// the frame is in and it's program counter) and the storage for registers used
// by the frame. As all operations including stack dumps only ever need to
// enumerate the frames in storage order there's no need to be able to randomly
// index into them and the linked list combined with dynamic stack growth gives
// us (practically) unlimited stack depth.
//
// [iree_vm_stack_t]
//   +- top -------> [frame 3 header] [registers] ---+
//                                                   |
//              +--- [frame 2 header] [registers] <--+
//              |
//              +--> [frame 1 header] [registers] ---+
//                                                   |
//         NULL <--- [frame 0 header] [registers] <--+
//
// To allow for static stack allocation and make allocating the VM stack on the
// host stack or within an existing data structure the entire stack, including
// all frame storage, can be placed into an existing allocation. This is similar
// to inlined vectors/etc where some storage is available directly in the object
// and only when exceeded will it switch to a dynamic allocation.
//
// Dynamic stack growth
// --------------------
// Though most of the stacks we deal with are rather shallow due to aggressive
// inlining in the compiler it's still possible to spill any reasonably-sized
// static storage allocation. This can be especially true in modules compiled
// with optimizations disabled; for example the debug register allocator may
// expand the required register count for a function from 30 to 3000.
//
// To support these cases the stack can optionally be provided an allocator to
// enable it to grow the stack when the initial storage is exhausted. As we
// store pointers to the stack storage within the storage itself (such as the
// iree_vm_registers_t pointers) this means we need to perform a fixup step
// during reallocation to ensure they are all updated. This also means that the
// pointers to the stack frames are possibly invalidated on every function
// entry and that users of the stack cannot rely on pointer stability during
// execution.
//
// Calling convention
// ------------------
// Callers provide an arguments buffer and results buffer sized appropriately
// for the call and with the arguments buffer populated. Callees will push
// their new stack frame, copy or move the arguments from the caller buffer into
// the callee frame, and then begin execution. Upon return the callee function
// will move return values to the result buffer and pop their stack frame.
//
// By making the actual stack frame setup and teardown callee-controlled we can
// have optimized implementations that treat register storage differently across
// various frames. For example, native modules that store their registers in
// host-machine specific registers can marshal the caller registers in/out of
// the host registers (or stack/etc) without exposing the actual implementation
// to the caller.
//
// Calling into the VM
// -------------------
// Calls from external code into the VM such as via iree_vm_invoke reuse the
// same calling convention as internal-to-internal calls: callees load arguments
// from the caller frame and store results into the caller frame.
//
// Marshaling arguments is easy given that the caller controls these and we can
// trivially map the ordered set of argument types into the VM calling
// convention buffers.
//
// A side-effect (beyond code reuse) is that ref types are retained by the VM
// for the entire lifetime they may be accessible by VM routines. This lets us
// get rich stack traces without needing to hook into external code and lets us
// timeshift via coroutines where we may otherwise not know when the external
// caller will resume a yielded call and actually read back the results.
//
// The overhead of this marshaling is minimal as external functions can always
// use move semantics on the ref objects. Since we are reusing the normal VM
// code paths which are likely still in instruction cache the bulk of the work
// amounts to some small memcpys.

// Multiplier on the capacity of the stack frame storage when growing.
// Since we never shrink stacks it's nice to keep this relative low. If we
// measure a lot of growth happening in normal models we should increase this
// but otherwise leave as small as we can to avoid overallocation.
#define IREE_VM_STACK_GROWTH_FACTOR 2

// A private stack frame header that allows us to walk the linked list of
// frames without exposing their exact structure through the API. This makes it
// easier for us to add/version additional information or hide implementation
// details.
typedef struct iree_vm_stack_frame_header_t {
  // Size, in bytes, of the frame header and frame payload including registers.
  // Adding this value to the base header pointer will yield the next available
  // memory location. Ensure that it does not exceed the total
  // frame_storage_capacity.
  iree_host_size_t frame_size;

  // Pointer to the parent stack frame, usually immediately preceding this one
  // in the frame storage. May be NULL.
  struct iree_vm_stack_frame_header_t* parent;

  // Size, in bytes, of the additional stack frame data that follows the frame.
  iree_host_size_t data_size;

  // Opened trace zone ID or 0 if none assigned.
  IREE_TRACE(iree_zone_id_t trace_zone;)

  // Function called when the stack frame is left.
  iree_vm_stack_frame_cleanup_fn_t frame_cleanup_fn;

  // Actual stack frame as visible through the API.
  // The registers within the frame will (likely) point to addresses immediately
  // following this header in memory.
  iree_vm_stack_frame_t frame;
} iree_vm_stack_frame_header_t;

// Core stack storage. This will be mapped either into dynamic memory allocated
// by the member allocator or static memory allocated externally. Static stacks
// cannot grow when storage runs out while dynamic ones will resize their stack.
struct iree_vm_stack_t {
  // NOTE: to get better cache hit rates we put the most frequently accessed
  // members first.

  // Pointer to the current top of the stack.
  // This can be used to walk the stack from top to bottom by following the
  // |parent| pointers. Note that these pointers are invalidated each time the
  // stack grows (if dynamic growth is enabled) and all of the frames will need
  // updating.
  iree_vm_stack_frame_header_t* top;

  // Base pointer to stack storage.
  // For statically-allocated stacks this will (likely) point to immediately
  // after the iree_vm_stack_t in memory. For dynamically-allocated stacks this
  // will (likely) point to heap memory.
  iree_host_size_t frame_storage_capacity;
  iree_host_size_t frame_storage_size;
  void* frame_storage;

  // Flags controlling the behavior of the invocation owning this stack.
  iree_vm_invocation_flags_t flags;

  // True if the stack owns the frame_storage and should free it when it is no
  // longer required. Host stack-allocated stacks don't own their storage but
  // may transition to owning it on dynamic growth.
  bool owns_frame_storage;

  // Resolves a module to a module state within a context.
  // This will be called on function entry whenever module transitions occur.
  iree_vm_state_resolver_t state_resolver;

  // Allocator used for dynamic stack allocations. May be the null allocator
  // if growth is prohibited.
  iree_allocator_t allocator;
};

//===----------------------------------------------------------------------===//
// Stack implementation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_vm_stack_initialize(
    iree_byte_span_t storage, iree_vm_invocation_flags_t flags,
    iree_vm_state_resolver_t state_resolver, iree_allocator_t allocator,
    iree_vm_stack_t** out_stack) {
  IREE_ASSERT_ARGUMENT(out_stack);
  *out_stack = NULL;
  if (storage.data_length < IREE_VM_STACK_MIN_SIZE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "stack storage under minimum required amount: %zu < %d",
        storage.data_length, IREE_VM_STACK_MIN_SIZE);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_stack_t* stack = (iree_vm_stack_t*)storage.data;
  memset(stack, 0, sizeof(iree_vm_stack_t));
  stack->owns_frame_storage = false;
  stack->flags = flags;
  stack->state_resolver = state_resolver;
  stack->allocator = allocator;

  iree_host_size_t storage_offset =
      iree_host_align(sizeof(iree_vm_stack_t), 16);
  stack->frame_storage_capacity = storage.data_length - storage_offset;
  stack->frame_storage_size = 0;
  stack->frame_storage = storage.data + storage_offset;

  stack->top = NULL;

  *out_stack = stack;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_stack_reset(iree_vm_stack_t* stack) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Pop each frame of the stack in reverse.
  while (stack->top) {
    iree_status_ignore(iree_vm_stack_function_leave(stack));
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_vm_stack_deinitialize(iree_vm_stack_t* stack) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release stack frame resources.
  iree_vm_stack_reset(stack);

  // Drop allocated frame storage.
  if (stack->owns_frame_storage) {
    iree_allocator_free(stack->allocator, stack->frame_storage);
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_vm_stack_allocate(
    iree_vm_invocation_flags_t flags, iree_vm_state_resolver_t state_resolver,
    iree_allocator_t allocator, iree_vm_stack_t** out_stack) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_stack = NULL;

  iree_host_size_t storage_size = IREE_VM_STACK_DEFAULT_SIZE;
  void* storage = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, storage_size, &storage);
  iree_vm_stack_t* stack = NULL;
  if (iree_status_is_ok(status)) {
    iree_byte_span_t storage_span = iree_make_byte_span(storage, storage_size);
    status = iree_vm_stack_initialize(storage_span, flags, state_resolver,
                                      allocator, &stack);
  }

  *out_stack = stack;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_vm_stack_free(iree_vm_stack_t* stack) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t allocator = stack->allocator;
  void* storage = (void*)stack;
  iree_vm_stack_deinitialize(stack);
  iree_allocator_free(allocator, storage);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_allocator_t
iree_vm_stack_allocator(const iree_vm_stack_t* stack) {
  return stack->allocator;
}

IREE_API_EXPORT iree_vm_invocation_flags_t
iree_vm_stack_invocation_flags(const iree_vm_stack_t* stack) {
  return stack->flags;
}

IREE_API_EXPORT iree_vm_stack_frame_t* iree_vm_stack_top(
    iree_vm_stack_t* stack) {
  if (!stack->top) {
    // Stack empty.
    return NULL;
  } else if (stack->top->frame.type != IREE_VM_STACK_FRAME_WAIT) {
    // Non-wait frame; return current.
    return &stack->top->frame;
  } else {
    // Wait frame; return parent frame.
    // Only one wait frame can be active so we know that the parent isn't one.
    // We technically also know that the parent can't be NULL, but check for it
    // to be safe.
    return stack->top->parent ? &stack->top->parent->frame : NULL;
  }
}

IREE_API_EXPORT iree_vm_stack_frame_t* iree_vm_stack_current_frame(
    iree_vm_stack_t* stack) {
  return stack->top ? &stack->top->frame : NULL;
}

IREE_API_EXPORT iree_vm_stack_frame_t* iree_vm_stack_parent_frame(
    iree_vm_stack_t* stack) {
  if (!stack->top) return NULL;
  iree_vm_stack_frame_header_t* parent_header = stack->top->parent;
  return parent_header ? &parent_header->frame : NULL;
}

IREE_API_EXPORT iree_status_t iree_vm_stack_query_module_state(
    iree_vm_stack_t* stack, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state) {
  return stack->state_resolver.query_module_state(stack->state_resolver.self,
                                                  module, out_module_state);
}

// Attempts to grow the stack store to hold at least |minimum_capacity|.
// Pointers to existing stack frames will be invalidated and any pointers
// embedded in the stack frame data structures will be updated.
// Fails if dynamic stack growth is disabled or the allocator is OOM.
static iree_status_t iree_vm_stack_grow(iree_vm_stack_t* stack,
                                        iree_host_size_t minimum_capacity) {
  if (IREE_UNLIKELY(stack->allocator.ctl == NULL)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "stack initialized on the host stack and cannot grow");
  }

  // Ensure we grow at least as much as required.
  iree_host_size_t new_capacity = stack->frame_storage_capacity;
  do {
    new_capacity *= IREE_VM_STACK_GROWTH_FACTOR;
  } while (new_capacity < minimum_capacity);
  if (new_capacity > IREE_VM_STACK_MAX_SIZE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "new stack size would exceed maximum size: %" PRIhsz
                            " > %d",
                            new_capacity, IREE_VM_STACK_MAX_SIZE);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Reallocate the frame storage. 99.9999% chance the new storage pointer will
  // differ and we'll need to fix up pointers so we just always do that.
  void* old_storage = stack->frame_storage;
  void* new_storage = stack->frame_storage;
  iree_status_t status;
  if (stack->owns_frame_storage) {
    // We own the storage already likely from a previous growth operation.
    status =
        iree_allocator_realloc(stack->allocator, new_capacity, &new_storage);
  } else {
    // We don't own the original storage so we are going to switch to our own
    // newly-allocated storage instead. We need to make sure we copy over the
    // existing stack contents.
    status =
        iree_allocator_malloc(stack->allocator, new_capacity, &new_storage);
    if (iree_status_is_ok(status)) {
      memcpy(new_storage, old_storage, stack->frame_storage_capacity);
    }
  }
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  stack->frame_storage = new_storage;
  stack->frame_storage_capacity = new_capacity;
  stack->owns_frame_storage = true;

#define REBASE_POINTER(type, ptr, old_base, new_base)           \
  if (ptr) {                                                    \
    (ptr) = (type)(((uintptr_t)(ptr) - (uintptr_t)(old_base)) + \
                   (uintptr_t)(new_base));                      \
  }

  // Fixup embedded stack frame pointers.
  REBASE_POINTER(iree_vm_stack_frame_header_t*, stack->top, old_storage,
                 new_storage);
  iree_vm_stack_frame_header_t* frame_header = stack->top;
  while (frame_header != NULL) {
    REBASE_POINTER(iree_vm_stack_frame_header_t*, frame_header->parent,
                   old_storage, new_storage);
    frame_header = frame_header->parent;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
static iree_zone_id_t iree_vm_stack_trace_wait_zone_begin(
    iree_vm_wait_type_t wait_type, iree_host_size_t wait_count) {
  switch (wait_type) {
    default:
    case IREE_VM_WAIT_UNTIL: {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_vm_stack_wait_until");
      return z0;
    }
    case IREE_VM_WAIT_ANY: {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_vm_stack_wait_any");
      IREE_TRACE_ZONE_APPEND_VALUE(z0, wait_count);
      return z0;
    }
    case IREE_VM_WAIT_ALL: {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_vm_stack_wait_all");
      IREE_TRACE_ZONE_APPEND_VALUE(z0, wait_count);
      return z0;
    }
  }
}
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

IREE_API_EXPORT iree_status_t iree_vm_stack_wait_enter(
    iree_vm_stack_t* stack, iree_vm_wait_type_t wait_type,
    iree_host_size_t wait_count, iree_timeout_t timeout,
    iree_zone_id_t trace_zone, iree_vm_wait_frame_t** out_wait_frame) {
  IREE_ASSERT_ARGUMENT(out_wait_frame);
  *out_wait_frame = NULL;

  iree_host_size_t frame_size = iree_host_align(
      sizeof(iree_vm_wait_frame_t) + wait_count * sizeof(iree_wait_source_t),
      16);

  // Allocate stack space and grow stack, if required.
  iree_host_size_t header_size = sizeof(iree_vm_stack_frame_header_t);
  iree_host_size_t new_top =
      stack->frame_storage_size + header_size + frame_size;
  if (IREE_UNLIKELY(new_top > stack->frame_storage_capacity)) {
    IREE_RETURN_IF_ERROR(iree_vm_stack_grow(stack, new_top));
  }

  // Try to reuse the same module state if the caller and callee are from the
  // same module. Otherwise, query the state from the registered handler.
  iree_vm_stack_frame_header_t* caller_frame_header = stack->top;
  iree_vm_stack_frame_t* caller_frame =
      caller_frame_header ? &caller_frame_header->frame : NULL;

  // Bump pointer and get real stack pointer offsets.
  iree_vm_stack_frame_header_t* frame_header =
      (iree_vm_stack_frame_header_t*)((uintptr_t)stack->frame_storage +
                                      stack->frame_storage_size);
  memset(frame_header, 0, header_size + frame_size);

  frame_header->frame_size = header_size + frame_size;
  frame_header->parent = stack->top;
  frame_header->data_size = frame_size;
  // TODO(benvanik): allow a custom cleanup function so callers can be notified
  // of aborted waits on the stack? Today normal stack cleanup will take care of
  // things but we may want to support cancellation.
  frame_header->frame_cleanup_fn = NULL;

  iree_vm_stack_frame_t* callee_frame = &frame_header->frame;
  callee_frame->type = IREE_VM_STACK_FRAME_WAIT;
  callee_frame->depth = caller_frame ? caller_frame->depth + 1 : 0;

  stack->frame_storage_size = new_top;
  stack->top = frame_header;

  IREE_TRACE({
    frame_header->trace_zone =
        iree_vm_stack_trace_wait_zone_begin(wait_type, wait_count);
  });

  iree_vm_wait_frame_t* wait_frame =
      (iree_vm_wait_frame_t*)iree_vm_stack_frame_storage(callee_frame);
  wait_frame->wait_type = wait_type;
  wait_frame->deadline_ns = iree_timeout_as_deadline_ns(timeout);
  IREE_TRACE(wait_frame->trace_zone = trace_zone);
  wait_frame->count = wait_count;
  *out_wait_frame = wait_frame;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_stack_wait_leave(
    iree_vm_stack_t* stack, iree_vm_wait_result_t* out_wait_result) {
  IREE_ASSERT_ARGUMENT(out_wait_result);
  memset(out_wait_result, 0, sizeof(*out_wait_result));

  if (IREE_UNLIKELY(!stack->top)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unbalanced stack leave");
  } else if (stack->top->frame.type != IREE_VM_STACK_FRAME_WAIT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unbalanced wait leave");
  }

  // Fetch wait status from the wait storage.
  iree_vm_wait_frame_t* wait_frame =
      iree_vm_stack_frame_storage(&stack->top->frame);
  out_wait_result->status = wait_frame->wait_status;

  // Call (optional) frame storage cleanup function.
  if (stack->top->frame_cleanup_fn) {
    stack->top->frame_cleanup_fn(&stack->top->frame);
  }

  IREE_TRACE({
    if (stack->top->trace_zone) {
      IREE_TRACE_ZONE_END(stack->top->trace_zone);
    }
    out_wait_result->trace_zone = wait_frame->trace_zone;
  });

  // Restore the frame pointer to the caller.
  stack->frame_storage_size -= stack->top->frame_size;
  stack->top = stack->top->parent;

  return iree_ok_status();
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
static iree_zone_id_t iree_vm_stack_trace_function_zone_begin(
    iree_vm_stack_frame_type_t frame_type, const iree_vm_function_t* function) {
  if (frame_type != IREE_VM_STACK_FRAME_NATIVE) {
    // TODO(benvanik): cache source location and query from module.
    iree_string_view_t function_name = iree_vm_function_name(function);
    IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, function_name.data,
                                        function_name.size);
    return z0;
  } else {
    return 0;
  }
}
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

IREE_API_EXPORT iree_status_t iree_vm_stack_function_enter(
    iree_vm_stack_t* stack, const iree_vm_function_t* function,
    iree_vm_stack_frame_type_t frame_type, iree_host_size_t frame_size,
    iree_vm_stack_frame_cleanup_fn_t frame_cleanup_fn,
    iree_vm_stack_frame_t* IREE_RESTRICT* out_callee_frame) {
  if (out_callee_frame) *out_callee_frame = NULL;

  // Allocate stack space and grow stack, if required.
  iree_host_size_t header_size = sizeof(iree_vm_stack_frame_header_t);
  iree_host_size_t new_top =
      stack->frame_storage_size + header_size + frame_size;
  if (IREE_UNLIKELY(new_top > stack->frame_storage_capacity)) {
    IREE_RETURN_IF_ERROR(iree_vm_stack_grow(stack, new_top));
  }

  // Try to reuse the same module state if the caller and callee are from the
  // same module. Otherwise, query the state from the registered handler.
  iree_vm_stack_frame_header_t* caller_frame_header = stack->top;
  iree_vm_stack_frame_t* caller_frame =
      caller_frame_header ? &caller_frame_header->frame : NULL;
  iree_vm_module_state_t* module_state = NULL;
  if (caller_frame && caller_frame->function.module == function->module) {
    module_state = caller_frame->module_state;
  } else if (function->module != NULL) {
    IREE_RETURN_IF_ERROR(stack->state_resolver.query_module_state(
        stack->state_resolver.self, function->module, &module_state));
  }

  // Bump pointer and get real stack pointer offsets.
  iree_vm_stack_frame_header_t* frame_header =
      (iree_vm_stack_frame_header_t*)((uintptr_t)stack->frame_storage +
                                      stack->frame_storage_size);
  memset(frame_header, 0, header_size + frame_size);

  frame_header->frame_size = header_size + frame_size;
  frame_header->parent = stack->top;
  frame_header->data_size = frame_size;
  frame_header->frame_cleanup_fn = frame_cleanup_fn;

  iree_vm_stack_frame_t* callee_frame = &frame_header->frame;
  callee_frame->type = frame_type;
  callee_frame->function = *function;
  callee_frame->module_state = module_state;
  callee_frame->pc = 0;
  callee_frame->depth = caller_frame ? caller_frame->depth + 1 : 0;

  stack->frame_storage_size = new_top;
  stack->top = frame_header;

  IREE_TRACE({
    frame_header->trace_zone =
        iree_vm_stack_trace_function_zone_begin(frame_type, function);
    if (frame_header->trace_zone) {
      IREE_TRACE_ZONE_APPEND_VALUE(frame_header->trace_zone, (uint64_t)stack);
    }
  });

  if (out_callee_frame) *out_callee_frame = callee_frame;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_vm_stack_function_leave(iree_vm_stack_t* stack) {
  if (IREE_UNLIKELY(!stack->top)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unbalanced stack leave");
  }

  // Call (optional) frame storage cleanup function.
  if (stack->top->frame_cleanup_fn) {
    stack->top->frame_cleanup_fn(&stack->top->frame);
  }

  IREE_TRACE({
    if (stack->top->trace_zone) {
      IREE_TRACE_ZONE_END(stack->top->trace_zone);
    }
  });

  // Restore the frame pointer to the caller.
  stack->frame_storage_size -= stack->top->frame_size;
  stack->top = stack->top->parent;

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_stack_format_backtrace(
    iree_vm_stack_t* stack, iree_string_builder_t* builder) {
  for (iree_vm_stack_frame_header_t* frame = stack->top; frame != NULL;
       frame = frame->parent) {
    // Stack frame prefix.
    const char* type_str;
    switch (frame->frame.type) {
      default:
        type_str = "??";
        break;
      case IREE_VM_STACK_FRAME_EXTERNAL:
        type_str = "external";
        break;
      case IREE_VM_STACK_FRAME_NATIVE:
        type_str = "native";
        break;
      case IREE_VM_STACK_FRAME_BYTECODE:
        type_str = "bytecode";
        break;
      case IREE_VM_STACK_FRAME_WAIT:
        type_str = "wait";
        break;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "\n[%*" PRId32 "] %*s ", 2, frame->frame.depth, 8, type_str));

    iree_vm_module_t* module = frame->frame.function.module;
    if (!module) continue;

    // Common module/function name and PC.
    iree_string_view_t module_name = iree_vm_module_name(module);
    iree_string_view_t function_name =
        iree_vm_function_name(&frame->frame.function);
    if (iree_string_view_is_empty(function_name)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%.*s@%d", (int)module_name.size, module_name.data,
          (int)frame->frame.function.ordinal));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%.*s.%.*s", (int)module_name.size, module_name.data,
          (int)function_name.size, function_name.data));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ":%" PRIu64 " ", (uint64_t)frame->frame.pc));

    iree_vm_source_location_t source_location;
    iree_status_t status = iree_vm_module_resolve_source_location(
        module, &frame->frame, &source_location);
    if (iree_status_is_ok(status)) {
      status = iree_vm_source_location_format(
          &source_location, IREE_VM_SOURCE_LOCATION_FORMAT_FLAG_NONE, builder);
    }
    if (iree_status_is_unavailable(status)) {
      // TODO(benvanik): if this is an import/export we can get that name.
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "-"));
    } else if (!iree_status_is_ok(status)) {
      return status;
    }
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_stack_annotate_backtrace(
    iree_vm_stack_t* stack, iree_status_t base_status) {
  if (IREE_LIKELY(iree_status_is_ok(base_status))) return base_status;
  iree_string_builder_t builder;
  iree_string_builder_initialize(stack->allocator, &builder);
  iree_status_t status = iree_vm_stack_format_backtrace(stack, &builder);
  if (iree_status_is_ok(status)) {
    // TODO(benvanik): don't duplicate the buffer here - we should be attaching
    // a payload but that requires additional plumbing.
    status = iree_status_annotate_f(base_status, "%.*s",
                                    (int)iree_string_builder_size(&builder),
                                    iree_string_builder_buffer(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION) && \
    !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FIBERS)

// These helpers perform manual fiber-like stack suspend/resume when not
// natively supported by the tracing library. See
// iree_vm_stack_suspend_trace_zones for more information.

IREE_API_EXPORT void iree_vm_stack_suspend_trace_zones(iree_vm_stack_t* stack) {
  // Walk top->bottom to unwind the trace zones.
  iree_vm_stack_frame_header_t* frame_header = stack->top;
  while (frame_header) {
    if (frame_header->trace_zone) {
      IREE_TRACE_ZONE_END(frame_header->trace_zone);
      frame_header->trace_zone = 0;
    }
    if (frame_header->frame.type == IREE_VM_STACK_FRAME_WAIT) {
      iree_vm_wait_frame_t* wait_frame =
          (iree_vm_wait_frame_t*)iree_vm_stack_frame_storage(
              &frame_header->frame);
      if (wait_frame->trace_zone) {
        IREE_TRACE_ZONE_END(wait_frame->trace_zone);
        wait_frame->trace_zone = 0;
      }
    }
    frame_header = frame_header->parent;
  }
}

static void iree_vm_stack_resume_trace_zones_recursive(
    iree_vm_stack_t* stack, iree_vm_stack_frame_header_t* frame_header) {
  if (frame_header->parent) {
    // To get bottom->top ordering we recurse into parent frames first.
    iree_vm_stack_resume_trace_zones_recursive(stack, frame_header->parent);
  }

  IREE_ASSERT_EQ(frame_header->trace_zone, 0);
  if (frame_header->frame.type == IREE_VM_STACK_FRAME_WAIT) {
    iree_vm_wait_frame_t* wait_frame =
        (iree_vm_wait_frame_t*)iree_vm_stack_frame_storage(
            &frame_header->frame);
    // TODO(benvanik): find a good way to recover the wait zone; for now we just
    // mark it as "?".
    IREE_ASSERT_EQ(wait_frame->trace_zone, 0);
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_vm_stack_wait_recover_?");
    wait_frame->trace_zone = z0;
    frame_header->trace_zone = iree_vm_stack_trace_wait_zone_begin(
        wait_frame->wait_type, wait_frame->count);
  } else {
    frame_header->trace_zone = iree_vm_stack_trace_function_zone_begin(
        frame_header->frame.type, &frame_header->frame.function);
    if (frame_header->trace_zone) {
      IREE_TRACE_ZONE_APPEND_VALUE(frame_header->trace_zone, (uint64_t)stack);
    }
  }
}
IREE_API_EXPORT void iree_vm_stack_resume_trace_zones(iree_vm_stack_t* stack) {
  // Walking the stack bottom->top only happens in this case and it's not worth
  // storing additional metadata in order to make it efficient.
  if (stack->top) {
    iree_vm_stack_resume_trace_zones_recursive(stack, stack->top);
  }
}

#else

IREE_API_EXPORT void iree_vm_stack_resume_trace_zones(iree_vm_stack_t* stack) {}
IREE_API_EXPORT void iree_vm_stack_suspend_trace_zones(iree_vm_stack_t* stack) {
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
