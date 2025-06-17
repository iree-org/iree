// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NOTE: these structs/enums are taken from the HSA spec, the hsa.h and
// hsa_ext_amd.h headers, and the LLVM AMDGPU device library headers.
// We define them locally as the HSA headers cannot be directly used in
// bare-metal C and the device libraries are only available in a fork of LLM.
//
// Sources:
// https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
// https://github.com/ROCm/ROCR-Runtime
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_QUEUE_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

//===----------------------------------------------------------------------===//
// HSA/AMDGPU AQL Queue
//===----------------------------------------------------------------------===//

typedef enum {
  // Queue supports multiple producers.
  IREE_HSA_QUEUE_TYPE_MULTI = 0,
  // Queue only supports a single producer.
  IREE_HSA_QUEUE_TYPE_SINGLE = 1,
} iree_hsa_queue_type_t;

// NOTE: this is not our struct and we cannot change it.
typedef struct iree_hsa_queue_t {
  // Queue type.
  iree_hsa_queue_type_t type;

  // Queue features mask. This is a bit-field of iree_hsa_queue_feature_t
  // values. Applications should ignore any unknown set bits.
  uint32_t features;

  // Packet storage. Must be accessible on any agents that may operate on it and
  // aligned to at least 64 (the size of an AQL packet).
  void* base_address;

  // Signal object used by the application to indicate the ID of a packet that
  // is ready to be processed. The HSA runtime or hardware packet processor
  // manages the doorbell signal. If the application tries to replace or destroy
  // this signal the behavior is undefined.
  //
  // If type is HSA_QUEUE_TYPE_SINGLE the doorbell signal value must be
  // updated in a monotonically increasing fashion. If type is
  // HSA_QUEUE_TYPE_MULTI the doorbell signal value can be updated with any
  // value and the act of writing a differing value is enough to wake the
  // processor. On AMD GPUs today it is reportedly not any more efficient to
  // use SINGLE queues as the packet processor handles both the same way.
  iree_hsa_signal_t doorbell_signal;

  // Maximum number of packets the queue can hold. Must be a power of 2.
  uint32_t size;

  uint32_t reserved1;  // must be 0

  // Queue identifier, which is unique over the lifetime of the application even
  // if the queue is reallocated.
  uint64_t id;
} iree_hsa_queue_t;

#define IREE_AMD_HSA_BITS_CREATE_ENUM_ENTRIES(name, shift, width) \
  name##_SHIFT = (shift), name##_WIDTH = (width),                 \
  name = (((1 << (width)) - 1) << (shift))
#define IREE_AMD_HSA_BITS_SET(dst, mask, val) \
  dst &= (~(1 << mask##_SHIFT) & ~mask);      \
  dst |= (((val) << mask##_SHIFT) & mask)

enum iree_amd_queue_properties_t {
  IREE_AMD_HSA_BITS_CREATE_ENUM_ENTRIES(
      IREE_AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER, 0, 1),
  // All devices we care about are 64-bit.
  IREE_AMD_HSA_BITS_CREATE_ENUM_ENTRIES(IREE_AMD_QUEUE_PROPERTIES_IS_PTR64, 1,
                                        1),
  IREE_AMD_HSA_BITS_CREATE_ENUM_ENTRIES(
      IREE_AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS, 2, 1),
  // Timestamps will be stored on signals (start_ts/end_ts).
  IREE_AMD_HSA_BITS_CREATE_ENUM_ENTRIES(
      IREE_AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, 3, 1),
  IREE_AMD_HSA_BITS_CREATE_ENUM_ENTRIES(
      IREE_AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE, 4, 1),
  IREE_AMD_HSA_BITS_CREATE_ENUM_ENTRIES(IREE_AMD_QUEUE_PROPERTIES_RESERVED1, 5,
                                        27)
};
typedef uint32_t iree_amd_queue_properties32_t;

// An AQL packet queue.
// We generally treat these as opaque except for if we need to read queue
// properties to check modes - otherwise we just treat any queue handle as
// an iree_hsa_queue_t.
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_amd_queue_t {
  iree_hsa_queue_t hsa_queue;
  uint32_t caps;
  uint32_t reserved1[3];
  volatile uint64_t write_dispatch_id;
  uint32_t group_segment_aperture_base_hi;
  uint32_t private_segment_aperture_base_hi;
  uint32_t max_cu_id;
  uint32_t max_wave_id;
  volatile uint64_t max_legacy_doorbell_dispatch_id_plus_1;
  volatile uint32_t legacy_doorbell_lock;
  uint32_t reserved2[9];
  volatile uint64_t read_dispatch_id;
  uint32_t read_dispatch_id_field_base_byte_offset;
  uint32_t compute_tmpring_size;
  uint32_t scratch_resource_descriptor[4];
  uint64_t scratch_backing_memory_location;
  uint64_t scratch_backing_memory_byte_size;
  uint32_t scratch_workitem_byte_size;
  iree_amd_queue_properties32_t queue_properties;
  volatile uint64_t scratch_last_used_index; /* async-reclaim */
  iree_hsa_signal_t queue_inactive_signal;
  uint32_t reserved4[2];
  volatile uint64_t alt_scratch_last_used_index; /* async-reclaim */
  uint64_t alt_scratch_backing_memory_location;  /* async-reclaim */
  uint64_t alt_scratch_backing_memory_byte_size; /* async-reclaim */
  uint32_t alt_scratch_dispatch_limit_x;         /* async-reclaim */
  uint32_t alt_scratch_dispatch_limit_y;         /* async-reclaim */
  uint32_t alt_scratch_dispatch_limit_z;         /* async-reclaim */
  uint32_t alt_scratch_wave64_lane_byte_size;    /* async-reclaim */
  uint32_t alt_compute_tmpring_size;             /* async-reclaim */
  uint32_t reserved5;
} iree_amd_queue_t;

//===----------------------------------------------------------------------===//
// Cached HSA/AMD Queue
//===----------------------------------------------------------------------===//

// A variant of iree_hsa_queue_t/iree_amd_queue_t with all of the hot fields
// hoisted into the struct. Queue metadata is immutable so it is always safe to
// cache the fields and doing so allows us to avoid one indirection on every
// access. Since queue metadata may live in either host or device memory it also
// prevents the device from possibly needing to dereference the host-side
// metadata in order to get the ringbuffer pointer or indices.
//
// NOTE: we try to keep this to the smallest possible set so that it can fit in
// a single cache line. We want one fetch from local memory to get all the info
// needed to start pushing writes.
typedef IREE_AMDGPU_ALIGNAS(
    iree_amdgpu_destructive_interference_size) struct iree_amd_cached_queue_t {
  // Packet storage. Must be accessible on any agents that may operate on it and
  // aligned to at least 64 (the size of an AQL packet).
  void* base_address;

  // Maximum number of packets the queue can hold. Must be a power of 2.
  uint32_t size;

  uint32_t reserved;

  // Signal object used by the application to indicate the ID of a packet that
  // is ready to be processed. The HSA runtime or hardware packet processor
  // manages the doorbell signal. If the application tries to replace or destroy
  // this signal the behavior is undefined.
  iree_hsa_signal_t doorbell_signal;

  // Pointer to the ringbuffer write dispatch ID atomic.
  iree_amdgpu_scoped_atomic_uint64_t* write_dispatch_id;

  // Pointer to the ringbuffer read dispatch ID atomic.
  iree_amdgpu_scoped_atomic_uint64_t* read_dispatch_id;

  // Backing queue that may live in host or device memory.
  iree_amd_queue_t* queue;
} iree_amd_cached_queue_t;

// Returns an HSA queue reference with the important
static inline iree_amd_cached_queue_t iree_amd_make_cached_queue(
    iree_hsa_queue_t* queue) {
  iree_amd_cached_queue_t result = {
      /*.base_address=*/queue->base_address,
      /*.size=*/queue->size,
      /*.reserved=*/0u,
      /*.doorbell_signal=*/{(uint64_t)&queue->doorbell_signal},
      /*.write_dispatch_id=*/
      (iree_amdgpu_scoped_atomic_uint64_t*)&((iree_amd_queue_t*)queue)
          ->write_dispatch_id,
      /*.read_dispatch_id=*/
      (iree_amdgpu_scoped_atomic_uint64_t*)&((iree_amd_queue_t*)queue)
          ->read_dispatch_id,
      /*.queue=*/(iree_amd_queue_t*)queue,
  };
  return result;
}

//===----------------------------------------------------------------------===//
// HSA/AMDGPU AQL Packets
//===----------------------------------------------------------------------===//

typedef enum {
  // Handled entirely by the packet processor and will vary agent to agent.
  IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  // Invalid packet (not yet populated) that will stall the packet processor.
  IREE_HSA_PACKET_TYPE_INVALID = 1,
  // iree_hsa_kernel_dispatch_packet_t
  IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH = 2,
  // iree_hsa_barrier_and_packet_t
  IREE_HSA_PACKET_TYPE_BARRIER_AND = 3,
  // iree_hsa_agent_dispatch_packet_t
  IREE_HSA_PACKET_TYPE_AGENT_DISPATCH = 4,
  // iree_hsa_barrier_or_packet_t
  IREE_HSA_PACKET_TYPE_BARRIER_OR = 5,
} iree_hsa_packet_type_t;

// Bit offsets within the header word of various values.
// We have to perform the bit manipulation ourselves because OpenCL has no
// bitfields. Crazy.
//
// If we did have bitfields the struct would look like:
// typedef struct {
//   uint16_t type : 8;
//   uint16_t barrier : 1;
//   uint16_t scacquire_fence_scope : 2;
//   uint16_t screlease_fence_scope : 2;
//   uint16_t reserved : 3;  // must be 0
// } iree_hsa_packet_header_t;
//
// Since the smallest atomic width is 32-bits and this header is 16-bits any
// operations updating the header must include the subsequent 16-bits of the
// packet (e.g. setup for kernel dispatches).
//
// See spec 2.9.1 and child entries for the full details.
typedef enum {
  // Determines the packet type as processed by the packet processor.
  // The header is the same for all packets but all other following contents may
  // change.
  IREE_HSA_PACKET_HEADER_TYPE = 0,
  // If set then processing of the packet will only begin when all preceding
  // packets are complete. There is no implicit fence defined as part of the
  // barrier and an acquire fence scope must still be specified if any is
  // required.
  IREE_HSA_PACKET_HEADER_BARRIER = 8,
  // A packet memory acquire fence ensures any subsequent global segment or
  // image loads by any unit of execution that belongs to a dispatch that has
  // not yet entered the active phase on any queue of the same agent, sees any
  // data previously released at the scopes specified by the packet acquire
  // fence.
  //
  // Behavior:
  //   IREE_HSA_FENCE_SCOPE_NONE:
  //     No fence is applied and the packet relies on an earlier acquire fence
  //     performed on the agent or acquire fences within the operation (e.g. by
  //     the kernel).
  //   IREE_HSA_FENCE_SCOPE_AGENT:
  //     The acquire fence is applied with agent scope for the global segment.
  //   IREE_HSA_FENCE_SCOPE_SYSTEM:
  //     The acquire fence is applied across both agent and system scope for the
  //     global segment.
  IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9,
  // A packet memory release fence makes any global segment or image data that
  // was stored by any unit of execution that belonged to a dispatch that has
  // completed the active phase on any queue of the same agent visible in all
  // the scopes specified by the packet release fence.
  //
  // Behavior:
  //   IREE_HSA_FENCE_SCOPE_NONE:
  //     No fence is applied and the packet relies on a later release fence
  //     performed on the agent or release fences within the operation (e.g. by
  //     the kernel).
  //   IREE_HSA_FENCE_SCOPE_AGENT:
  //     The release fence is applied with agent scope for the global segment.
  //   IREE_HSA_FENCE_SCOPE_SYSTEM:
  //     The release fence is applied across both agent and system scope for the
  //     global segment.
  IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11,
} iree_hsa_packet_header_t;

// Width in bits of the sub-fields in iree_hsa_packet_header_t.
typedef enum {
  IREE_HSA_PACKET_HEADER_WIDTH_TYPE = 8,
  IREE_HSA_PACKET_HEADER_WIDTH_BARRIER = 1,
  IREE_HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE = 2,
  IREE_HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE = 2,
} iree_hsa_packet_header_width_t;

// Forms a packet 16-bit AQL packet header.
#define iree_hsa_make_packet_header(type, is_barrier, scacquire_fence_scope,   \
                                    screlease_fence_scope)                     \
  (((type) << IREE_HSA_PACKET_HEADER_TYPE) |                                   \
   ((is_barrier ? 1 : 0) << IREE_HSA_PACKET_HEADER_BARRIER) |                  \
   ((scacquire_fence_scope) << IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) | \
   ((screlease_fence_scope) << IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE))

typedef enum {
  // No cache management occurs.
  IREE_HSA_FENCE_SCOPE_NONE = 0,
  // Invalidates I, K and L1 caches. Changes will be available to any queue on
  // the same agent but may not be available on any other agent.
  IREE_HSA_FENCE_SCOPE_AGENT = 1,
  // Invalidates L1, L2 and flushes L2 caches. Changes will be available on all
  // agents in the system after the fence completes.
  IREE_HSA_FENCE_SCOPE_SYSTEM = 2,
} iree_hsa_fence_scope_t;

// Kernel dispatch (2.9.6 in the spec).
//
// Pseudo-code:
//   for (uint32_t z = 0; z < grid_size[2] / workgroup_size[2]; ++z) {
//     for (uint32_t y = 0; y < grid_size[1] / workgroup_size[1]; ++y) {
//       for (uint32_t x = 0; x < grid_size[0] / workgroup_size[0]; ++x) {
//         kernel_object(*kernarg_address);
//       }
//     }
//   }
//   iree_hsa_signal_subtract(completion_signal, 1);
//
// The acquire fence is applied at the end of the launch phase just before the
// packet enters the active phase. The release fence is applied at the start of
// the completion phase of the packet.
typedef struct iree_hsa_kernel_dispatch_packet_t {
  // AQL packet header. See iree_hsa_packet_header_t for details.
  uint16_t header;
  // Number of grid dimensions (1, 2, or 3 - we always use 3).
  uint16_t setup;
  // Work-group size in work-items.
  uint16_t workgroup_size[3];
  uint16_t reserved0;  // must be 0
  // Grid size in work-items.
  uint32_t grid_size[3];
  // Total size in bytes of the per-work-item memory.
  uint32_t private_segment_size;
  // Total size in bytes of the per-work-group memory.
  uint32_t group_segment_size;
  // Kernel object (function) handle as returned from a query on the symbol
  // of HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT.
  uint64_t kernel_object;
  // Kernel arguments as required by the function.
  // Must be 16-byte aligned and live until the dispatch has completed.
  void* kernarg_address;
  uint64_t reserved2;  // must be 0
  iree_hsa_signal_t completion_signal;
  // Optional signal indicating completion of all work-groups.
} iree_hsa_kernel_dispatch_packet_t;

// Agent dispatch (2.9.7 in the spec).
//
// Pseudo-code:
//   *return_address = fns[type](arg[0], arg[1], arg[2], arg[3]);
//   iree_hsa_signal_subtract(completion_signal, 1);
//
// The acquire fence is applied at the end of the launch phase just before the
// packet enters the active phase. The release fence is applied at the start of
// the completion phase of the packet.
typedef struct iree_hsa_agent_dispatch_packet_t {
  // AQL packet header. See iree_hsa_packet_header_t for details.
  uint16_t header;
  // Agent-defined type (discriminator).
  uint16_t type;
  uint32_t reserved0;  // must be 0
  // Pointer to store the return value(s) in with the contents and layout
  // defined by the type.
  void* return_address;
  // Arguments to the dispatch as defined by the type.
  uint64_t arg[4];
  uint64_t reserved2;  // must be 0
  // Optional signal indicating completion of the dispatch.
  iree_hsa_signal_t completion_signal;
} iree_hsa_agent_dispatch_packet_t;

// Barrier-AND (2.9.8 in the spec).
// Waits until all dep_signals reach the value 0 at the same time and then
// decrements the completion_signal. Ignores any 0 (null) signals.
//
// Pseudo-code:
//   do {
//     bool any_unsatisfied = false;
//     for (int i = 0; i < 5; ++i) {
//       if (iree_hsa_signal_load(dep_signal[i]) != 0) any_unsatisfied = true;
//     }
//     if (!any_unsatisfied) break;
//     iree_amdgpu_yield();
//   } while(true);
//   iree_hsa_signal_subtract(completion_signal, 1);
//
// The acquire fence is processed first in the completion phase of the packet
// after the barrier condition has been met. The release fence is processed
// after the acquire fence in the completion phase.
typedef struct iree_hsa_barrier_and_packet_t {
  // AQL packet header. See iree_hsa_packet_header_t for details.
  uint16_t header;
  uint16_t reserved0;  // must be 0
  uint32_t reserved1;  // must be 0
  // Handles for dependent signaling objects to be evaluated by the packet
  // processor. Any 0 (null) handles are ignored.
  iree_hsa_signal_t dep_signal[5];
  uint64_t reserved2;
  // Signal to decrement when all dep_signals are satisfied.
  iree_hsa_signal_t completion_signal;
} iree_hsa_barrier_and_packet_t;

// Barrier-OR (2.9.9 in the spec).
// Waits until any one dep_signal reaches the value 0 and then decrements the
// completion_signal. Ignores any 0 (null) signals.
//
// Pseudo-code:
//   do {
//     for (int i = 0; i < 5; ++i) {
//       if (iree_hsa_signal_load(dep_signal[i]) == 0) break;
//     }
//     iree_amdgpu_yield();
//   } while(true);
//   iree_hsa_signal_subtract(completion_signal, 1);
//
// The acquire fence is processed first in the completion phase of the packet
// after the barrier condition has been met. The release fence is processed
// after the acquire fence in the completion phase.
typedef struct iree_hsa_barrier_or_packet_t {
  // AQL packet header. See iree_hsa_packet_header_t for details.
  uint16_t header;
  uint16_t reserved0;  // must be 0
  uint32_t reserved1;  // must be 0
  // Handles for dependent signaling objects to be evaluated by the packet
  // processor. Any 0 (null) handles are ignored.
  iree_hsa_signal_t dep_signal[5];
  uint64_t reserved2;  // must be 0
  // Signal to decrement when any dep_signal is satisfied.
  iree_hsa_signal_t completion_signal;
} iree_hsa_barrier_or_packet_t;

typedef enum {
  // iree_hsa_amd_barrier_value_packet_t
  IREE_HSA_AMD_PACKET_TYPE_BARRIER_VALUE = 2,
} iree_hsa_amd_packet_type_t;
typedef uint8_t iree_hsa_amd_packet_type8_t;

// Prefix of AMD-specific vendor packets.
typedef struct iree_hsa_amd_packet_header_t {
  // AQL packet header. See iree_hsa_packet_header_t for details.
  uint16_t header;
  // Secondary type indicating which AMD-specific packet this is.
  iree_hsa_amd_packet_type8_t AmdFormat;
  uint8_t reserved;  // must be 0
} iree_hsa_amd_vendor_packet_header_t;

// Barrier value extension.
// Halts packet processing and waits for `(signal_value & mask) cond value` to
// be satisfied before decrementing the completion_signal.
//
// Pseudo-code:
//   do {
//     if (iree_hsa_evaluate_signal_condition(
//         /*condition=*/cond,
//         /*current_value=*/(iree_hsa_signal_load(signal) & mask),
//         /*desired_value=*/value)) {
//       break;
//     }
//     iree_amdgpu_yield();
//   } while(true);
//   iree_hsa_signal_subtract(completion_signal, 1);
//
// The acquire fence is processed first in the completion phase of the packet
// after the barrier condition has been met. The release fence is processed
// after the acquire fence in the completion phase.
typedef struct iree_hsa_amd_barrier_value_packet_t {
  // AMD vendor-specific packet header.
  iree_hsa_amd_vendor_packet_header_t header;
  uint32_t reserved0;  // must be 0
  // Dependent signal object. A 0 (null) signal will be treated as satisfied.
  iree_hsa_signal_t signal;
  // Value to compare the signal against (no mask applied).
  iree_hsa_signal_value_t value;
  // Bitmask applied to the current signal value.
  iree_hsa_signal_value_t mask;
  // Comparison operation.
  iree_hsa_signal_condition32_t cond;
  uint32_t reserved1;  // must be 0
  uint64_t reserved2;  // must be 0
  uint64_t reserved3;  // must be 0
  // Signal to decrement when any dep_signal is satisfied.
  iree_hsa_signal_t completion_signal;
} iree_hsa_amd_barrier_value_packet_t;

//===----------------------------------------------------------------------===//
// iree_amdgpu_kernel_implicit_args_t
//===----------------------------------------------------------------------===//

// Implicit kernel arguments passed to OpenCL/HIP kernels that use them.
// Not all kernels require this and the metadata needs to be checked to detect
// its use (or if the total kernargs size is > what we think it should be).
// Layout-wise explicit args always start at offset 0 and implicit args follow
// those with 8-byte alignment.
//
// The metadata will contain exact fields and offsets and most driver code will
// carefully walk to detect, align, pad, and write each field:
// OpenCL/HIP: (`amd::KernelParameterDescriptor`...)
// https://github.com/ROCm/clr/blob/5da72f9d524420c43fe3eee44b11ac875d884e0f/rocclr/device/rocm/rocvirtual.cpp#L3197
//
// This complex construction was required once upon a time. The LLVM code
// producing the kernargs layout and metadata handles these cases much more
// simply by only ever truncating the implicit args at the last used field:
// https://github.com/llvm/llvm-project/blob/7f1b465c6ae476e59dc90652d58fc648932d23b1/llvm/lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp#L389
//
// Then at some point in time someone was like "meh, who cares about optimizing"
// and decided to include all of them always 🤦:
// https://github.com/llvm/llvm-project/blob/7f1b465c6ae476e59dc90652d58fc648932d23b1/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp#L299
//
// What this means in practice is that if any implicit arg is used then all will
// be included and declared in the metadata even if only one is actually read by
// the kernel -- there's no way for us to know. In the ideal case none of them
// are read and the kernel function gets the `amdgpu-no-implicitarg-ptr` attr
// so that all of them can be skipped. Otherwise we reserve the 256 bytes and
// just splat them all in. This at least keeps our code simple relative to all
// the implementations that enumerate the metadata and write args one at a time.
// We really should try to force `amdgpu-no-implicitarg-ptr` when we generate
// code, though.
//
// For our bare-metal C runtime device code we have total freedom and don't use
// any OpenCL/HIP-related things that would emit the implicit args.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_amdgpu_kernel_implicit_args_t {
  // Grid dispatch workgroup count.
  // Some languages, such as OpenCL, support a last workgroup in each
  // dimension being partial. This count only includes the non-partial
  // workgroup count. This is not the same as the value in the AQL dispatch
  // packet, which has the grid size in workitems.
  //
  // Represented in metadata as:
  //   hidden_block_count_x
  //   hidden_block_count_y
  //   hidden_block_count_z
  uint32_t block_count[3];  // + 0/4/8

  // Grid dispatch workgroup size.
  // This size only applies to the non-partial workgroups. This is the same
  // value as the AQL dispatch packet workgroup size.
  //
  // Represented in metadata as:
  //   hidden_group_size_x
  //   hidden_group_size_y
  //   hidden_group_size_z
  uint16_t group_size[3];  // + 12/14/16

  // Grid dispatch work group size of the partial work group, if it exists.
  // Any dimension that does not exist must be 0.
  //
  // Represented in metadata as:
  //   hidden_remainder_x
  //   hidden_remainder_y
  //   hidden_remainder_z
  uint16_t remainder[3];  // + 18/20/22

  uint64_t reserved0;  // + 24 hidden_tool_correlation_id
  uint64_t reserved1;  // + 32

  // OpenCL grid dispatch global offset.
  //
  // Represented in metadata as:
  //   hidden_global_offset_x
  //   hidden_global_offset_y
  //   hidden_global_offset_z
  uint64_t global_offset[3];  // + 40/48/56

  // Grid dispatch dimensionality. This is the same value as the AQL
  // dispatch packet dimensionality. Must be a value between 1 and 3.
  //
  // Represented in metadata as:
  //   hidden_grid_dims
  uint16_t grid_dims;  // + 64

  // Represented in metadata as:
  //   hidden_printf_buffer
  void* printf_buffer;

  // Represented in metadata as:
  //   hidden_hostcall_buffer
  void* hostcall_buffer;

  // Represented in metadata as:
  //   hidden_multigrid_sync_arg
  uint64_t multigrid_sync_arg;

  // Represented in metadata as:
  //   hidden_heap_v1
  uint64_t heap_v1;

  // Represented in metadata as:
  //   hidden_default_queue
  uint64_t default_queue;

  // Represented in metadata as:
  //   hidden_completion_action
  uint64_t completion_action;

  // Represented in metadata as:
  //   hidden_dynamic_lds_size
  uint32_t dynamic_lds_size;

  uint8_t reserved[68];

  // Represented in metadata as:
  //   hidden_private_base
  uint32_t deprecated_private_base;

  // Represented in metadata as:
  //   hidden_shared_base
  uint32_t deprecated_shared_base;

  // Represented in metadata as:
  //   hidden_queue_ptr;
  iree_hsa_queue_t* queue_ptr;
} iree_amdgpu_kernel_implicit_args_t;

//===----------------------------------------------------------------------===//
// OpenCL/HIP Dispatch ABI
//===----------------------------------------------------------------------===//
// These come from llvm-project/amd/device-libs/ockl/src/workitem.cl (the ockl
// functions) and llvm-project/clang/lib/CodeGen/CGBuiltin.cpp (e.g.
// EmitAMDGPUWorkGroupSize). Using either runs a chance of pulling in the
// entire iree_amdgpu_kernel_implicit_args_t struct and we don't want to set
// that. We also don't need it: we aren't requiring OpenCL compatibility and
// have no need for the extra features provided by the implicit args (like
// workgroup offset and device-side enqueue - that's our job).

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Returns the pointer to the iree_hsa_kernel_dispatch_packet_t being executed.
#define iree_amdgcn_dispatch_ptr()                                 \
  ((const iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT) \
       __builtin_amdgcn_dispatch_ptr())

// __ockl_get_global_id(0) / get_global_id_x using OLD_ABI
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_x(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_x();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_x();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  return (group_id * group_size + local_id);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_y(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_y();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_y();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  return (group_id * group_size + local_id);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_z(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_z();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_z();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  return (group_id * group_size + local_id);
}

// __ockl_get_group_id(0)
#define iree_hal_amdgpu_device_group_id_x() __builtin_amdgcn_workgroup_id_x()
#define iree_hal_amdgpu_device_group_id_y() __builtin_amdgcn_workgroup_id_y()
#define iree_hal_amdgpu_device_group_id_z() __builtin_amdgcn_workgroup_id_z()

// __ockl_get_num_groups(0)
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_x(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[0];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  const uint32_t q = grid_size / group_size;
  return q + (grid_size > q * group_size);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_y(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[1];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  const uint32_t q = grid_size / group_size;
  return q + (grid_size > q * group_size);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_z(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[2];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  const uint32_t q = grid_size / group_size;
  return q + (grid_size > q * group_size);
}

// __ockl_get_local_id(0)
#define iree_hal_amdgpu_device_local_id_x() __builtin_amdgcn_workitem_id_x()
#define iree_hal_amdgpu_device_local_id_y() __builtin_amdgcn_workitem_id_y()
#define iree_hal_amdgpu_device_local_id_z() __builtin_amdgcn_workitem_id_z()

// __ockl_get_local_size(0) / get_local_size_x
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_x(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_x();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[0];
  const uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_y(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_y();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[1];
  const uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_z(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_z();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[2];
  const uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_1d(void) {
  return iree_hal_amdgpu_device_group_id_x() *
             iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
         iree_hal_amdgpu_device_local_id_x();
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_2d(void) {
  const size_t id_x = iree_hal_amdgpu_device_group_id_x() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
                      iree_hal_amdgpu_device_local_id_x();
  const size_t id_y = iree_hal_amdgpu_device_group_id_y() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[1] +
                      iree_hal_amdgpu_device_local_id_y();
  return id_y * iree_amdgcn_dispatch_ptr()->grid_size[0] + id_x;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_3d(void) {
  const size_t id_x = iree_hal_amdgpu_device_group_id_x() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
                      iree_hal_amdgpu_device_local_id_x();
  const size_t id_y = iree_hal_amdgpu_device_group_id_y() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[1] +
                      iree_hal_amdgpu_device_local_id_y();
  const size_t id_z = iree_hal_amdgpu_device_group_id_z() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[2] +
                      iree_hal_amdgpu_device_local_id_z();
  return (id_z * iree_amdgcn_dispatch_ptr()->grid_size[1] + id_y) *
             iree_amdgcn_dispatch_ptr()->grid_size[0] +
         id_x;
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Device Library Functions
//===----------------------------------------------------------------------===//
// These are cloned from llvm-project/amd/device-libs/ockl/src/hsaqs.cl so that
// we don't need to rely on ockl.bc (and don't have to try to find them every
// time we want to see what they do).

#if defined(IREE_AMDGPU_TARGET_DEVICE)

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hsa_queue_load_read_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue,
    iree_amdgpu_memory_order_t memory_order) {
  return iree_amdgpu_scoped_atomic_load(queue->read_dispatch_id, memory_order,
                                        iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void
iree_hsa_queue_store_read_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue, uint64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amdgpu_scoped_atomic_store(queue->read_dispatch_id, value, memory_order,
                                  iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hsa_queue_load_write_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue,
    iree_amdgpu_memory_order_t memory_order) {
  return iree_amdgpu_scoped_atomic_load(queue->write_dispatch_id, memory_order,
                                        iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hsa_queue_add_write_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue, uint64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  return iree_amdgpu_scoped_atomic_fetch_add(queue->write_dispatch_id, value,
                                             memory_order,
                                             iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hsa_queue_cas_write_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue,
    uint64_t expected, uint64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  uint64_t e = expected;
  iree_amdgpu_scoped_atomic_compare_exchange_strong(
      queue->write_dispatch_id, &e, value, memory_order,
      iree_amdgpu_memory_order_relaxed, iree_amdgpu_memory_scope_system);
  return e;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void
iree_hsa_queue_store_write_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue, uint64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amdgpu_scoped_atomic_store(queue->write_dispatch_id, value, memory_order,
                                  iree_amdgpu_memory_scope_system);
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_QUEUE_H_
