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

#include "iree/vm/stack.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/vm/module.h"

#ifndef NDEBUG
#define VMCHECK(expr) assert(expr)
#else
#define VMCHECK(expr)
#endif  // NDEBUG

#define VMMAX(a, b) (((a) > (b)) ? (a) : (b))
#define VMMIN(a, b) (((a) < (b)) ? (a) : (b))

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
// Stack frames are managed by callees. A caller will provide a list of argument
// and result registers in the caller frame that should be passed into and out
// of the callee. The callee will first push its stack frame and reserve the
// register storage it requires, copy or move the argument values from the
// caller frame into the callee frame, and then begin execution. Upon return the
// callee function will move return registers from its frame out to the caller
// frame and pop its stack frame.
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
// from the caller frame and store results into the caller frame. To enable this
// a set of iree_vm_stack_external_enter/iree_vm_stack_external_leave functions
// are provided to setup the special `[external]` stack frame that allows the
// external code to marshal in the arguments and marshal out the results.
//
// Marshaling arguments is easy given that the caller controls these and we can
// trivially map the ordered set of argument types into the ABI registers.
//
// TODO(#1979): know result types ahead of time.
// Results are more difficult as today we do not know the order or type of the
// function results upon entry. As such when the callee goes to leave its frame
// and store the results back to the caller we cannot place directly into
// registers that then the iree_vm_stack_external_leave can access. To work
// around this until we can more cleanly provide this information we check for
// external callers when leaving frames and duplicate the register mapping. This
// can result in use-after-free if the result register mapping was not in rdata
// however today both our C++ and bytecode implementations are.
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
//
// Alternative register widths
// ---------------------------
// Registers in the VM are just a blob of memory and not physical device
// registers. They have a natural width of 32-bits as that covers a majority of
// our usage for i32/f32 but can be accessed at larger widths such as 64-bits or
// more for vector operations. The base of each frame's register memory is
// 16-byte aligned and accessing any individual register as a 32-bit value is
// always 4-byte aligned.
//
// Supporting other register widths is "free" in that the registers for all
// widths alias the same register storage memory. This is similar to how
// physical registers work in x86 where each register can be accessed at
// different sizes (like EAX/RAX alias and the SIMD registers alias as XMM1 is
// 128-bit, YMM1 is 256-bit, and ZMM1 is 512-bit but all the same storage).
//
// The requirements for doing this is that the base alignment for any register
// must be a multiple of 4 (due to the native 32-bit storage) AND aligned to the
// natural size of the register (so 8 bytes for i64, 16 bytes for v128, etc).
// This alignment can easily be done by masking off the low bits such that we
// know for any valid `reg` ordinal aligned to 4 bytes `reg/N` will still be
// within register storage. For example, i64 registers are accessed as `reg&~1`
// to align to 8 bytes starting at byte 0 of the register storage.
//
// Transferring between register types can be done with vm.ext.* and vm.trunc.*
// ops. For example, vm.trunc.i64.i32 will read an 8 byte register and write a
// two 4 byte registers (effectively) with hi=0 and lo=the lower 32-bits of the
// value.

// Multiplier on the capacity of the stack frame storage when growing.
// Since we never shrink stacks it's nice to keep this relative low. If we
// measure a lot of growth happening in normal models we should increase this
// but otherwise leave as small as we can to avoid overallocation.
#define IREE_VM_STACK_GROWTH_FACTOR 2

enum {
  // Represents an `[external]` frame that needs to marshal args/results.
  // These frames have no source location and are tracked so that we know when
  // transitions occur into/out-of external code.
  IREE_VM_STACK_FRAME_EXTERNAL = 0,
  // Normal VM stack frame using the internal register storage.
  IREE_VM_STACK_FRAME_INTERNAL = 1,
};
typedef uint8_t iree_vm_stack_frame_type_t;

// A private stack frame header that allows us to walk the linked list of
// frames without exposing their exact structure through the API. This makes it
// easier for us to add/version additional information or hide implementation
// details.
typedef struct iree_vm_stack_frame_header {
  // Size, in bytes, of the frame header and frame payload including registers.
  // Adding this value to the base header pointer will yield the next available
  // memory location. Ensure that it does not exceed the total
  // frame_storage_capacity.
  iree_host_size_t frame_size;

  // Pointer to the parent stack frame, usually immediately preceding this one
  // in the frame storage. May be NULL.
  struct iree_vm_stack_frame_header* parent;

  // Stack frame type used to determine which fields are valid.
  iree_vm_stack_frame_type_t type;

  // Pointer to a register list within the stack frame where return registers
  // will be stored by callees upon return.
  //
  // TODO(#1979): external frames will have a direct reference to the callee
  // frame return registers.
  const iree_vm_register_list_t* return_registers;

  // Actual stack frame as visible through the API.
  // The registers within the frame will (likely) point to addresses immediately
  // following this header in memory.
  iree_vm_stack_frame_t frame;
} iree_vm_stack_frame_header_t;

// Core stack storage. This will be mapped either into dynamic memory allocated
// by the member allocator or static memory allocated externally. Static stacks
// cannot grow when storage runs out while dynamic ones will resize their stack.
struct iree_vm_stack {
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
// Math utilities, kept here to limit dependencies
//===----------------------------------------------------------------------===//

// Haswell or later, gcc compile time option: -mlzcnt
#if defined(__LZCNT__)
#include <x86intrin.h>
#endif  // __LZCNT__

// Clang on Windows has __builtin_clz; otherwise we need to use the
// Windows intrinsic functions.
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#endif  // _MSC_VER

// https://en.wikipedia.org/wiki/Find_first_set
static inline int iree_math_count_leading_zeros_u32(uint32_t n) {
#if defined(_MSC_VER)
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse(&result, n)) {
    return 31 - result;
  }
  return 32;
#elif defined(__GNUC__)
  // Use __builtin_clz, which uses the following instructions:
  //  x86: bsr
  //  ARM64: clz
  //  PPC: cntlzd
  static_assert(sizeof(int) == sizeof(n),
                "__builtin_clz does not take 32-bit arg");

#if defined(__LCZNT__)
  // NOTE: LZCNT is a risky instruction; it is not supported on architectures
  // before Haswell, yet it is encoded as 'rep bsr', which typically ignores
  // invalid rep prefixes, and interprets it as the 'bsr' instruction, which
  // returns the index of the value rather than the count, resulting in
  // incorrect code.
  return __lzcnt32(n);
#endif  // defined(__LCZNT__)

  // Handle 0 as a special case because __builtin_clz(0) is undefined.
  return n ? __builtin_clz(n) : 32;
#else
#error No clz for this arch.
#endif
}

// Rounds up the value to the nearest power of 2 (if not already a power of 2).
static inline uint32_t iree_math_round_up_to_pow2_u32(uint32_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

//===----------------------------------------------------------------------===//
// Stack implementation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_initialize(
    iree_byte_span_t storage, iree_vm_state_resolver_t state_resolver,
    iree_allocator_t allocator, iree_vm_stack_t** out_stack) {
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
  stack->state_resolver = state_resolver;
  stack->allocator = allocator;

  iree_host_size_t storage_offset =
      iree_math_align(sizeof(iree_vm_stack_t), 16);
  stack->frame_storage_capacity = storage.data_length - storage_offset;
  stack->frame_storage_size = 0;
  stack->frame_storage = storage.data + storage_offset;

  stack->top = NULL;

  *out_stack = stack;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_stack_deinitialize(iree_vm_stack_t* stack) {
  IREE_TRACE_ZONE_BEGIN(z0);

  while (stack->top) {
    iree_vm_stack_function_leave(stack, NULL, NULL);
  }

  if (stack->owns_frame_storage) {
    iree_allocator_free(stack->allocator, stack->frame_storage);
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_allocate(
    iree_vm_state_resolver_t state_resolver, iree_allocator_t allocator,
    iree_vm_stack_t** out_stack) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_stack = NULL;

  iree_host_size_t storage_size = IREE_VM_STACK_DEFAULT_SIZE;
  void* storage = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, storage_size, &storage);
  iree_vm_stack_t* stack = NULL;
  if (iree_status_is_ok(status)) {
    iree_byte_span_t storage_span = iree_make_byte_span(storage, storage_size);
    status = iree_vm_stack_initialize(storage_span, state_resolver, allocator,
                                      &stack);
  }

  *out_stack = stack;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void IREE_API_CALL iree_vm_stack_free(iree_vm_stack_t* stack) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t allocator = stack->allocator;
  void* storage = (void*)stack;
  iree_vm_stack_deinitialize(stack);
  iree_allocator_free(allocator, storage);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_vm_stack_frame_t* IREE_API_CALL
iree_vm_stack_current_frame(iree_vm_stack_t* stack) {
  return stack->top ? &stack->top->frame : NULL;
}

IREE_API_EXPORT iree_vm_stack_frame_t* IREE_API_CALL
iree_vm_stack_parent_frame(iree_vm_stack_t* stack) {
  if (!stack->top) return NULL;
  iree_vm_stack_frame_header_t* parent_header = stack->top->parent;
  return parent_header ? &parent_header->frame : NULL;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_query_module_state(
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
  if (stack->allocator.alloc == NULL) {
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
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "new stack size would exceed maximum size: %zu > %d", new_capacity,
        IREE_VM_STACK_MAX_SIZE);
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
    iree_vm_registers_t* registers = &frame_header->frame.registers;
    REBASE_POINTER(int32_t*, registers->i32, old_storage, new_storage);
    REBASE_POINTER(iree_vm_ref_t*, registers->ref, old_storage, new_storage);
    REBASE_POINTER(iree_vm_stack_frame_header_t*, frame_header->parent,
                   old_storage, new_storage);
    frame_header = frame_header->parent;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Remaps argument/result registers from a source list in the caller/callee
// frame to the 0-N ABI registers in the callee/caller frame. This assumes that
// the destination stack frame registers are unused and ok to overwrite
// directly.
static void iree_vm_stack_frame_remap_abi_registers(
    const iree_vm_registers_t src_regs,
    const iree_vm_register_list_t* src_reg_list,
    const iree_vm_registers_t dst_regs) {
  // Each bank begins left-aligned at 0 and increments per arg of its type.
  int i32_reg_offset = 0;
  int ref_reg_offset = 0;
  for (int i = 0; i < src_reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = src_reg_list->registers[i];
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      uint16_t dst_reg = ref_reg_offset++;
      memset(&dst_regs.ref[dst_reg & dst_regs.ref_mask], 0,
             sizeof(iree_vm_ref_t));
      iree_vm_ref_retain_or_move(src_reg & IREE_REF_REGISTER_MOVE_BIT,
                                 &src_regs.ref[src_reg & src_regs.ref_mask],
                                 &dst_regs.ref[dst_reg & dst_regs.ref_mask]);
    } else {
      uint16_t dst_reg = i32_reg_offset++;
      dst_regs.i32[dst_reg & dst_regs.i32_mask] =
          src_regs.i32[src_reg & src_regs.i32_mask];
    }
  }
}

// Remaps registers from source to destination, possibly across frames.
// Registers from the |src_regs| will be copied/moved to |dst_regs| with the
// mappings provided by |src_reg_list| and |dst_reg_list|. It's assumed that the
// mappings are matching by type and - in the case that they aren't - things
// will get weird (but not crash).
static void iree_vm_stack_frame_remap_registers(
    const iree_vm_registers_t src_regs,
    const iree_vm_register_list_t* src_reg_list,
    const iree_vm_registers_t dst_regs,
    const iree_vm_register_list_t* dst_reg_list) {
  VMCHECK(src_reg_list->size <= dst_reg_list->size);
  if (src_reg_list->size > dst_reg_list->size) return;
  for (int i = 0; i < src_reg_list->size; ++i) {
    // TODO(benvanik): change encoding to avoid this branching.
    // Could write two arrays: one for prims and one for refs.
    uint16_t src_reg = src_reg_list->registers[i];
    uint16_t dst_reg = dst_reg_list->registers[i];
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      iree_vm_ref_retain_or_move(src_reg & IREE_REF_REGISTER_MOVE_BIT,
                                 &src_regs.ref[src_reg & src_regs.ref_mask],
                                 &dst_regs.ref[dst_reg & dst_regs.ref_mask]);
    } else {
      dst_regs.i32[dst_reg & dst_regs.i32_mask] =
          src_regs.i32[src_reg & src_regs.i32_mask];
    }
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_function_enter(
    iree_vm_stack_t* stack, iree_vm_function_t function,
    const iree_vm_register_list_t* argument_registers,
    const iree_vm_register_list_t* result_registers,
    iree_vm_stack_frame_t** out_callee_frame) {
  if (out_callee_frame) *out_callee_frame = NULL;

  // Try to reuse the same module state if the caller and callee are from the
  // same module. Otherwise, query the state from the registered handler.
  iree_vm_stack_frame_header_t* caller_frame_header = stack->top;
  iree_vm_stack_frame_t* caller_frame =
      caller_frame_header ? &caller_frame_header->frame : NULL;
  iree_vm_module_state_t* module_state = NULL;
  if (caller_frame && caller_frame->function.module == function.module) {
    module_state = caller_frame->module_state;
  } else if (function.module != NULL) {
    IREE_RETURN_IF_ERROR(stack->state_resolver.query_module_state(
        stack->state_resolver.self, function.module, &module_state));
  }

  // We first compute the frame size of the callee and the masks we'll use to
  // bounds check register access. This lets us allocate the entire frame
  // (header, frame, and register storage) as a single pointer bump below.
  iree_vm_registers_t registers;
  memset(&registers, 0, sizeof(registers));

  // Round up register counts to the nearest power of 2 (if not already).
  // This let's us use bit masks on register accesses to do bounds checking
  // instead of more complex logic. The cost of these extra registers is only at
  // worst 2x the required cost: so not large when thinking about the normal
  // size of data used in an IREE app for tensors.
  //
  // Note that to allow the masking to work as a guard we need to ensure we at
  // least allocate 1 register; this way an i32[reg & mask] will always point at
  // valid memory even if mask == 0.
  uint32_t i32_register_count =
      iree_math_round_up_to_pow2_u32(VMMAX(1, function.i32_register_count));
  uint32_t ref_register_count =
      iree_math_round_up_to_pow2_u32(VMMAX(1, function.ref_register_count));
  if (i32_register_count > IREE_I32_REGISTER_MASK ||
      ref_register_count > IREE_REF_REGISTER_MASK) {
    // Register count overflow. A valid compiler should never produce files that
    // hit this.
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "register count overflow");
  }
  // NOTE: >> by the bit width is undefined so we use a 64bit mask here to
  // ensure we are ok.
  registers.i32_mask =
      (uint16_t)(0xFFFFFFFFull >>
                 (iree_math_count_leading_zeros_u32(i32_register_count) + 1)) &
      IREE_I32_REGISTER_MASK;
  registers.ref_mask =
      (uint16_t)(0xFFFFFFFFull >>
                 (iree_math_count_leading_zeros_u32(ref_register_count) + 1)) &
      IREE_REF_REGISTER_MASK;

  // We need to align the ref register start to the natural machine alignment
  // in case the compiler is expecting that (it makes it easier to debug too).
  iree_host_size_t i32_register_size =
      iree_math_align(i32_register_count * sizeof(int32_t), 16);
  iree_host_size_t ref_register_size =
      iree_math_align(ref_register_count * sizeof(iree_vm_ref_t), 16);
  iree_host_size_t header_size =
      iree_math_align(sizeof(iree_vm_stack_frame_header_t), 16);
  iree_host_size_t frame_size =
      header_size + i32_register_size + ref_register_size;

  // Grow stack, if required.
  iree_host_size_t new_top = stack->frame_storage_size + frame_size;
  if (new_top > stack->frame_storage_capacity) {
    IREE_RETURN_IF_ERROR(iree_vm_stack_grow(stack, new_top));

    // NOTE: the caller_frame pointer may have changed if the stack grew.
    caller_frame = iree_vm_stack_current_frame(stack);
  }

  // Bump pointer and get real stack pointer offsets.
  iree_vm_stack_frame_header_t* frame_header =
      (iree_vm_stack_frame_header_t*)((uintptr_t)stack->frame_storage +
                                      stack->frame_storage_size);
  memset(frame_header, 0, frame_size);
  registers.i32 = (int32_t*)((uintptr_t)frame_header + header_size);
  registers.ref =
      (iree_vm_ref_t*)((uintptr_t)registers.i32 + i32_register_size);

  frame_header->frame_size = frame_size;
  frame_header->parent = stack->top;
  frame_header->type = IREE_VM_STACK_FRAME_INTERNAL;
  frame_header->return_registers = result_registers;

  iree_vm_stack_frame_t* callee_frame = &frame_header->frame;
  callee_frame->pc = 0;
  callee_frame->registers = registers;
  callee_frame->function = function;
  callee_frame->module_state = module_state;
  callee_frame->depth = caller_frame ? caller_frame->depth + 1 : 0;

  stack->frame_storage_size = new_top;
  stack->top = frame_header;

  // Remap arguments from the caller stack frame into the callee stack frame.
  if (caller_frame && argument_registers) {
    iree_vm_stack_frame_remap_abi_registers(
        caller_frame->registers, argument_registers, callee_frame->registers);
  }

  if (out_callee_frame) *out_callee_frame = callee_frame;
  return iree_ok_status();
}

// The external caller doesn't know the register types that it's going to be
// getting back today and as such cannot provide them the way normal calls can.
// Here we take the callee register list and construct a left-aligned register
// list (each register type starting from 0-N in each bank).
static void iree_vm_stack_populate_external_result_list(
    const iree_vm_register_list_t* callee_registers,
    const iree_vm_register_list_t* external_registers) {
  VMCHECK(external_registers->size >= callee_registers->size);
  uint16_t i32_reg_ordinal = 0;
  uint16_t ref_reg_ordinal = 0;
  ((iree_vm_register_list_t*)external_registers)->size = callee_registers->size;
  uint16_t* dst_reg_list = (uint16_t*)external_registers->registers;
  for (int i = 0; i < callee_registers->size; ++i) {
    uint16_t src_reg = callee_registers->registers[i];
    uint16_t abi_reg;
    if (src_reg & IREE_REF_REGISTER_TYPE_BIT) {
      abi_reg = IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT |
                (ref_reg_ordinal++);
    } else {
      abi_reg = i32_reg_ordinal++;
    }
    dst_reg_list[i] = abi_reg;
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_function_leave(
    iree_vm_stack_t* stack, const iree_vm_register_list_t* result_registers,
    iree_vm_stack_frame_t** out_caller_frame) {
  if (!stack->top) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unbalanced stack leave");
  }

  iree_vm_stack_frame_header_t* frame_header = stack->top;
  iree_vm_stack_frame_t* callee_frame = &frame_header->frame;
  iree_vm_stack_frame_t* caller_frame =
      frame_header->parent ? &frame_header->parent->frame : NULL;

  // TODO(#1979): avoid this hack to propagate results to external frames.
  if (frame_header->parent &&
      frame_header->parent->type == IREE_VM_STACK_FRAME_EXTERNAL &&
      result_registers) {
    iree_vm_stack_populate_external_result_list(result_registers,
                                                frame_header->return_registers);
    frame_header->parent->return_registers = frame_header->return_registers;
  }

  // Remap result registers from the callee frame to the caller frame.
  if (caller_frame && result_registers) {
    iree_vm_stack_frame_remap_registers(
        callee_frame->registers, result_registers, caller_frame->registers,
        frame_header->parent->return_registers);
  }

  // Release the reserved register storage to restore the frame pointer.
  // TODO(benvanik): allow the VM to elide this when it's known that there are
  // no more live registers.
  iree_vm_registers_t* registers = &callee_frame->registers;
  for (int i = 0; i <= registers->ref_mask; ++i) {
    iree_vm_ref_release(&registers->ref[i]);
  }

  // Restore the frame pointer to the caller.
  stack->top = stack->top->parent;
  stack->frame_storage_size -= frame_header->frame_size;

  if (out_caller_frame) *out_caller_frame = caller_frame;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_stack_external_enter(
    iree_vm_stack_t* stack, iree_string_view_t name,
    iree_host_size_t register_count, iree_vm_stack_frame_t** out_callee_frame) {
  iree_vm_function_t external_function;
  memset(&external_function, 0, sizeof(external_function));
  external_function.ref_register_count = (uint16_t)register_count;
  external_function.i32_register_count = (uint16_t)register_count;

  IREE_RETURN_IF_ERROR(iree_vm_stack_function_enter(
      stack, external_function, NULL, NULL, out_callee_frame));

  stack->top->type = IREE_VM_STACK_FRAME_EXTERNAL;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_stack_external_leave(iree_vm_stack_t* stack) {
  if (!stack->top) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unbalanced stack leave");
  } else if (stack->top->type != IREE_VM_STACK_FRAME_EXTERNAL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unbalanced stack leave (not external)");
  }
  return iree_vm_stack_function_leave(stack, NULL, NULL);
}
