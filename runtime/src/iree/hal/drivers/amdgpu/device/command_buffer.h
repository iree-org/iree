// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_COMMAND_BUFFER_H_

#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/device/buffer.h"
#include "iree/hal/drivers/amdgpu/device/kernels.h"
#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/queue.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"

typedef struct iree_hal_amdgpu_device_queue_scheduler_t
    iree_hal_amdgpu_device_queue_scheduler_t;

//===----------------------------------------------------------------------===//
// Device-side Command Buffer
//===----------------------------------------------------------------------===//
//
// Command buffers are represented by a host-side wrapper that implements the
// IREE HAL API and a device-side data structure holding the recorded contents.
// All information required to execute a command buffer lives on the device and
// a command buffer can be submitted from the device without host involvement.
// Command buffer data structures are immutable once constructed and can be
// executed concurrently and repeatedly based on the same recording because
// mutable execution state is stored separately as part of the issuing queue
// operation. Though the device-side recorded command buffer closely follows the
// HAL command buffer API it does not need to match 1:1. The initial
// implementation bakes out a lot of information as part of recording but leaves
// AQL packet construction to issue-time; future iterations could blend the two
// approaches by preconstructing packets to copy when profitable.
//
// The recorded command buffer is partitioned into one or more command blocks.
// Each block represents a yieldable point in the execution where the command
// buffer scheduler is allowed to suspend processing. Segmenting allows for
// basic control flow to be implemented within a command buffer by skipping,
// branching, or looping over blocks and also enables execution when hardware
// queues may not have capacity for the entire command buffer. Conceptually
// command buffers are like coroutines/fibers in that any number may be
// simultaneously executing the same program on the same hardware resources with
// independent states.
//
// +----------------------------------+
// | iree_hal_amdgpu_command_buffer_t | (host)
// +-----------------------v----------+
//                         |   +-----------------------------------------+
//                (device) +---> iree_hal_amdgpu_device_command_buffer_t |
//                             +------------------v----------------------+
//                                                |
//      +------------+------------+------------+--+------+--+------------+
//      |            |            |            |            |            |
// +----v----+  +----v----+  +----v----+  +----v----+  +----v----+  +----v----+
// |  block  |..|  block  |..|  block  |..|  block  |..|  block  |..|  block  |
// +----v----+  +---------+  +---------+  +---------+  +---------+  +---------+
//      |
//      |    +------------------------------+
//      +----> command entry[]              | fixed-length struct array
//      |    +------------------------------+
//      +----> embedded command data...     | variable length packed buffer
//           +------------------------------+
//
// Each block contains one or more commands encoded in fixed-length entries.
// This allows commands to be indexed by ordinal within the block such that
// command processing can be parallelized. An extra buffer is used to embed any
// variable-length data the commands require in read-only memory such as update
// source buffers, dispatch constants, and dispatch binding references.
// Execution-invariant information is stored in the command and any
// execution-dependent information is stored as either deltas/relative values or
// bits that can be used to derive the information when the command is issued.
//
// Execution starts with the first block and progresses through blocks based
// on control commands. Blocks are translated to AQL packets in parallel via
// the command buffer block issue kernel. Each command may translate to one or
// more AQL packets and space is reserved for the maximum potential AQL packets
// that are required when the block is launched. Execution uses a state
// structure that resides on device and is valid for the full duration of the
// command buffer execution. Every concurrently executing instance of a command
// buffer has its own state referencing its own kernel arguments buffer. Note
// that any optional AQL packet not used must still be set to a valid no-op
// packet in order for the command processor to progress through the AQL queue.
//
// Processing behavior:
//   1. Initialize iree_hal_amdgpu_device_execution_state_t:
//     a. Allocate execution state from the queue ringbuffer
//     b. Assign the target hardware AQL queue to receive packets
//     c. Reserve kernel arguments buffer with max size used by any block
//     d. Copy binding table into the state (if present)
//     e. Assign the first command buffer block as the entry block
//   2. Enqueue iree_hal_amdgpu_device_cmd_block_issue:
//     a. Reserve queue space for all command AQL packets
//     b. Enqueue command processor kernel for the next block with barrier bit
//   3. Command processor, parallelized over each command in a block:
//     a. Assign/copy kernel arguments to scratch buffer (if needed)
//     b. Construct AQL packet(s) for the command
//     c. Change from type INVALID to the real type
//   4. Repeat 2 and 3 until all blocks completed
//   5. Enqueue top-level queue scheduler upon completion
//   6. Deinitialize execution state (release resources)
//
//=----------------------------------------------------------------------------=
//
// Command buffer scheduling is always performed on the scheduler queue.
// Execution of the commands is allowed to target another queue dedicated to
// execution. When using multiple queues it's possible for the hardware to begin
// executing the initial commands while the rest of the commands are still being
// issued. This also allows the thread-compatible tracing logic to operate in
// single-threaded mode with the scheduler queue being the only one producing
// the synchronous "CPU" trace events while the execution queue produces the
// asynchronous "GPU" trace events.
//
//              +=========+  +-------------+                      +--------+
// scheduler q: | execute |->| issue block |...                ...| retire |
//              +=========+  +|-|-|-|-|-|-|+                      +--------+
//                            \ \ \ \ \ \ \                       ^
//                             v v v v v v v                     /
//                             +-----+-----+-----+-----+-----+--|--+
// execution q:                | cmd | cmd | cmd | cmd | cmd | ret |
//                             +-----+-----+-----+-----+-----+-----+
//
// The additional scheduler/execution queue hops between the command buffer
// execution request, each block, and the retire are insignificant compared to
// the actual execution time _when command buffers are large_ and it allows us
// to use queue priorities to ensure that scheduling runs ASAP even if the
// execution queue is heavily utilized. It also allows us to have one scheduler
// target multiple execution queues for concurrent command buffer processing or
// multiple schedulers target a single execution queue to ensure it is always
// utilized. It is not ideal for many small command buffers: there's currently
// some naive attempts at mitigating the latency (such as serially issuing
// small blocks) but more work would be required to optimize for that case.
// CUDA stream-like APIs are not the target here (as they aren't good) and we
// align more with the graph-based approaches of modern CUDA (though that too
// has issues). Latency on tiny command buffers (1-8 commands) is really the
// major flaw of this implementation. I'll probably end up having to solve that
// in the future with more special-casing or scheduler-level carve-outs. Hooray.
//
//=----------------------------------------------------------------------------=
//
// Command buffers are recorded with a forward progress guarantee ensuring that
// once issued they will complete even if no other work can be executed on the
// same queue. Events used within the command buffer have a signal-before-wait
// requirement when used on the same queue.
//
// Dispatches have their kernel arguments packed while their packets are
// constructed and enqueued. Some arguments are fixed (constants, directly
// referenced buffers) and copied directly from the command data buffer while
// others may be substituted with per-invocation state (indirectly referenced
// buffers from a binding table).
//
// Though most AQL packets are written once during their initial enqueuing some
// commands such as indirect dispatches require updating the packets after they
// have been placed in the target queue. Indirect dispatch parameters may either
// be declared static and captured at the start of command buffer processing
// or dynamic until immediately prior to when the particular dispatch is
// executed. Static parameters are preferred as the command scheduler can
// enqueue the dispatch packet by dereferencing the workgroups buffer while
// constructing the AQL packet. Dynamic parameters require dispatching a special
// fixup kernel immediately prior to the actual dispatch that does the
// indirection and updates the following packet in the queue. The AQL queue
// processing model is exploited by having the actual dispatch packet encoded as
// INVALID and thus halting the hardware command processor and the fixup
// dispatch is what switches it to a valid KERNEL_DISPATCH type.
//
// Disclaimer: that's how it's _supposed_ to work. I've discovered that all is
// not what it declares to be in AQL-land. Static indirect dispatch is always
// possible but the dynamic indirect dispatch with packet fixup may need some
// big hammers to workaround: in the worst case we put them in their own blocks
// and treat indirect dispatches as branches that route back through the
// scheduler to ensure each block has all indirect parameters available when the
// block is issued... 🤞
//
//=----------------------------------------------------------------------------=
//
// AQL agents launch packets in order but may complete them in any order.
// The two mechanisms of controlling the launch timeline are the barrier bit and
// barrier packets. When set on a packet the barrier bit indicates that all
// prior work on the queue must complete before the packet can be launched and
// that behavior matches our HAL execution barrier. Barrier packets can be used
// to set up dependencies via HSA signals roughly matching our HAL events.
//
// When a command buffer is recorded we use the execution barrier commands to
// set the barrier bit on recorded packets and in many cases end up with no
// additional barrier packets:
//  +------------+
//  | barrier    |      (no aql packet needed)
//  +------------+
//  | dispatch A |  --> AQL dispatch w/ barrier = true (await all prior)
//  +------------+
//  | barrier    |      (no aql packet needed)
//  +------------+
//  | dispatch B |  --> AQL dispatch w/ barrier = true (await dispatch A)
//  +------------+
//
// In cases of concurrency a nop packet is needed to allow multiple dispatches
// to launch without blocking. The complication is that at the time we get the
// execution barrier command we don't know how many commands will follow before
// the next barrier. To support single-pass recording we do some tricks with
// moving packets in order to insert barrier packets as required:
//  +------------+
//  | dispatch A |  --> dispatch w/ barrier = true (await all prior)
//  +------------+
//  | dispatch B |  --> dispatch w/ barrier = false (execute concurrently)
//  +------------+
//
// Fence acquire/release behavior is supported on nop barrier packets
// allowing for commands on either side to potentially avoid setting the
// behavior themselves. For example in serialized cases without the barrier
// packets the dispatches would need to acquire/release:
//  +------------+
//  | dispatch A |  --> acquire (as needed), release AGENT
//  +------------+
//  | dispatch B |  --> acquire AGENT, release (as needed)
//  +------------+
// While the barrier packet can allow this to be avoided:
//  +------------+
//  | barrier    |  --> acquire (as needed), release AGENT
//  +------------+
//  | dispatch A |  --> acquire/release NONE
//  +------------+
//  | dispatch B |  --> acquire/release NONE
//  +------------+
//  | barrier    |  --> release (as needed)
//  +------------+
// The recording logic is more complex than desired but by figuring it out at
// record-time the command buffer logic running here on device is kept much
// more straightforward.
//
// TODO(benvanik): define how events map to packets.

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_cmd_t
//===----------------------------------------------------------------------===//

// Defines the recorded command type.
// Note that commands may expand to zero or more AQL packets in the target
// execution queue as they may be routed to other queues or require multiple
// packets to complete.
typedef uint8_t iree_hal_amdgpu_device_cmd_type_t;
enum iree_hal_amdgpu_device_cmd_type_e {
  // iree_hal_amdgpu_device_cmd_debug_group_begin_t
  IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN = 0u,
  // iree_hal_amdgpu_device_cmd_debug_group_end_t
  IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END,
  // iree_hal_amdgpu_device_cmd_barrier_t
  IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER,
  // iree_hal_amdgpu_device_cmd_signal_event_t
  IREE_HAL_AMDGPU_DEVICE_CMD_SIGNAL_EVENT,
  // iree_hal_amdgpu_device_cmd_reset_event_t
  IREE_HAL_AMDGPU_DEVICE_CMD_RESET_EVENT,
  // iree_hal_amdgpu_device_cmd_wait_events_t
  IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS,
  // iree_hal_amdgpu_device_cmd_fill_buffer_t
  IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER,
  // iree_hal_amdgpu_device_cmd_copy_buffer_t
  IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER,
  // iree_hal_amdgpu_device_cmd_dispatch_t
  IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH,
  // iree_hal_amdgpu_device_cmd_dispatch_t
  IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC,
  // iree_hal_amdgpu_device_cmd_branch_t
  IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH,
  // iree_hal_amdgpu_device_cmd_cond_branch_t
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_BRANCH,
  // iree_hal_amdgpu_device_cmd_return_t
  IREE_HAL_AMDGPU_DEVICE_CMD_RETURN,
  // TODO(benvanik): trace flush block for intra-block query/sampling resets.
  // Today we assume command blocks under the query pool size.
  IREE_HAL_AMDGPU_DEVICE_CMD_MAX = IREE_HAL_AMDGPU_DEVICE_CMD_RETURN,
};

enum {
  IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_ACQUIRE_BIT = 1,  // bit 1 & 2
  IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_RELEASE_BIT = 3,  // bit 3 & 4
};

// Flags controlling command processing behavior.
typedef uint8_t iree_hal_amdgpu_device_cmd_flags_t;
enum iree_hal_amdgpu_device_cmd_flag_bits_t {
  IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_NONE = 0u,

  // Sets the barrier bit in the first AQL packet of the command in order to
  // force a wait on all prior packets to complete before processing the command
  // packets. This is much lighter weight than barriers and signals for the
  // common case of straight-line execution.
  IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER = 1u << 0,

  // Sets HSA_FENCE_SCOPE_AGENT on the AQL packet acquire scope.
  // This invalidates the I/K/L1 caches.
  IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_ACQUIRE_AGENT =
      IREE_HSA_FENCE_SCOPE_AGENT
      << IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_ACQUIRE_BIT,
  // Sets HSA_FENCE_SCOPE_SYSTEM on the AQL packet acquire scope.
  // This invalidates the L1/L2 caches and flushes L2.
  IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_ACQUIRE_SYSTEM =
      IREE_HSA_FENCE_SCOPE_SYSTEM
      << IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_ACQUIRE_BIT,

  // Sets HSA_FENCE_SCOPE_AGENT on the AQL packet release scope.
  // This flushes the I/K/L1 caches.
  IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_RELEASE_AGENT =
      IREE_HSA_FENCE_SCOPE_AGENT << 3,
  // Sets HSA_FENCE_SCOPE_SYTEM on the AQL packet release scope.
  // This flushes the L1/L2 caches.
  IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_RELEASE_SYSTEM =
      IREE_HSA_FENCE_SCOPE_SYSTEM << 3,
};

// Commands are fixed-size to allow for indexing into an array of commands.
// Additional variable-length data is stored out-of-band of the command struct.
#define IREE_HAL_AMDGPU_DEVICE_CMD_SIZE 64

// Header at the start of every command used to control command processing.
typedef struct iree_hal_amdgpu_device_cmd_header_t {
  // Command type indicating the parent structure.
  iree_hal_amdgpu_device_cmd_type_t type;
  // Flags controlling command processing behavior.
  iree_hal_amdgpu_device_cmd_flags_t flags;
  // Offset into the queue where AQL packets for the command should be placed.
  // If more than one packet is required they are stored contiguously from the
  // base offset.
  uint16_t packet_offset;
} iree_hal_amdgpu_device_cmd_header_t;
static_assert(sizeof(iree_hal_amdgpu_device_cmd_header_t) == 4,
              "header should be small as it's embedded in every command");

// Pushes a new debug group to the stack.
// All trace zones emitted between this and the corresponding
// iree_hal_amdgpu_device_cmd_debug_group_end_t command will be nested within.
//
// NOTE: the pointers used in the command are in the host address space. This is
// wonky, but the host trace buffer translation checks first to see if the
// address is in the expected range of device pointers and otherwise passes it
// right through.
//
// Recorded by:
//  iree_hal_command_buffer_begin_debug_group
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_cmd_debug_group_begin_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  // Source location pointer, if available. May be in the host address space.
  iree_hal_amdgpu_trace_src_loc_ptr_t src_loc;
  // Label for the group.
  // Value must be a pointer to a process-lifetime string literal. The host-side
  // command buffer recorder should perform interning if required.
  uint64_t label_literal;
  // Length of the label_literal in characters.
  uint32_t label_literal_length;
  // Color of the group. 0 indicates unspecified/default.
  iree_hal_amdgpu_trace_color_t color;
} iree_hal_amdgpu_device_cmd_debug_group_begin_t;
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)
#define IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN_AQL_PACKET_COUNT 1
#else
#define IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN_AQL_PACKET_COUNT 0
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

// Pops the current debug group from the stack.
//
// Recorded by:
//  iree_hal_command_buffer_end_debug_group
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_cmd_debug_group_end_t {
  iree_hal_amdgpu_device_cmd_header_t header;
} iree_hal_amdgpu_device_cmd_debug_group_end_t;
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)
#define IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END_AQL_PACKET_COUNT 1
#else
#define IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END_AQL_PACKET_COUNT 0
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

// Performs a full queue barrier causing subsequent commands to block until all
// prior commands have completed. This is effectively a no-op packet that just
// has the IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER bit set but could
// be used to perform coarse synchronization (acquire/release agent/system,
// etc).
//
// Recorded by:
//  iree_hal_command_buffer_execution_barrier (sometimes)
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_device_cmd_barrier_t {
  iree_hal_amdgpu_device_cmd_header_t header;
} iree_hal_amdgpu_device_cmd_barrier_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER_AQL_PACKET_COUNT 1

// TODO(benvanik): rework events so that they can be reused. We really should
// have an events table-like thing or something that allows capture at time of
// issue (if we even want to allow events to be used across command buffers).
// Today events are similar to Vulkan ones which don't support concurrent issue
// and that limits us here.
//
// Storing an ordinal to the event table would let us bulk allocate them as part
// of the execution state. Recording would need to track the unique set of
// events used in order to determine the capacity. We could make it be declared
// similar to the binding table capacity and swap to recording with ordinals but
// that makes it more difficult for users to compose. Recording could also only
// support events created from the command buffer during recording
// (iree_hal_command_buffer_acquire_event, etc) and that could also be used to
// verify lifetime and invalid cross-command-buffer usage. The event handle
// could just be an integer all the way into the compiler.
//
// For now the event-based code below uses an opaque value that we can
// substitute with whatever we come up with.
typedef uint32_t iree_hal_amdgpu_device_event_ordinal_t;

// Signals event after prior commands complete.
// The AQL signal will be decremented from a value of 1 to 0 to allow AQL
// dependencies to be satisfied directly.
//
// Recorded by:
//  iree_hal_command_buffer_signal_event
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_cmd_signal_event_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  iree_hal_amdgpu_device_event_ordinal_t event;
} iree_hal_amdgpu_device_cmd_signal_event_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_SIGNAL_EVENT_AQL_PACKET_COUNT 1

// Resets event to unsignaled after prior commands complete.
// The AQL signal will be set to a value of 1.
//
// Recorded by:
//  iree_hal_command_buffer_reset_event
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_cmd_reset_event_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  iree_hal_amdgpu_device_event_ordinal_t event;
} iree_hal_amdgpu_device_cmd_reset_event_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_RESET_EVENT_AQL_PACKET_COUNT 1

// Number of events that can be stored inline in a
// iree_hal_amdgpu_device_cmd_wait_events_t command. This is the same as the AQL
// barrier-and packet and allows us to avoid additional storage/indirections in
// the common case of waits one or two events.
#define IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENT_INLINE_CAPACITY 5

// Waits for the given events to be signaled before proceeding.
// All events much reach a value of 0. May be decomposed into multiple barrier
// packets if the event count exceeds the capacity of the barrier-and packet.
//
// Recorded by:
//  iree_hal_command_buffer_wait_events
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_cmd_wait_events_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  // Number of events being waited upon.
  uint32_t event_count;
  union {
    // Inlined events if event_count is less than
    // IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENT_INLINE_CAPACITY.
    iree_hal_amdgpu_device_event_ordinal_t
        events[IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENT_INLINE_CAPACITY];
    // Externally stored events if event_count is greater than
    // IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENT_INLINE_CAPACITY.
    iree_hal_amdgpu_device_event_ordinal_t* events_ptr;
  };
} iree_hal_amdgpu_device_cmd_wait_events_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS_PER_AQL_PACKET 5
#define IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS_AQL_PACKET_COUNT(event_count) \
  IREE_AMDGPU_CEIL_DIV((event_count),                                        \
                       IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS_PER_AQL_PACKET)

// Fills a buffer with a repeating pattern.
// Performed via a blit kernel.
//
// Recorded by:
//  iree_hal_command_buffer_fill_buffer
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_cmd_fill_buffer_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  // Block-relative kernel arguments address.
  uint32_t kernarg_offset;
  // Target buffer to fill.
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
  // 1 to 8 pattern bytes, little endian.
  uint64_t pattern;
  // Length in bytes of the pattern.
  uint8_t pattern_length;
} iree_hal_amdgpu_device_cmd_fill_buffer_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER_AQL_PACKET_COUNT 1

// Copies between buffers.
// Performed via a blit kernel. May be implementable with SDMA but it is
// currently unverified.
//
// Recorded by:
//  iree_hal_command_buffer_copy_buffer
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_cmd_copy_buffer_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  // Block-relative kernel arguments address.
  uint32_t kernarg_offset;
  // Copy source.
  iree_hal_amdgpu_device_buffer_ref_t source_ref;
  // Copy target.
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
} iree_hal_amdgpu_device_cmd_copy_buffer_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER_AQL_PACKET_COUNT 1

// Bitfield specifying flags controlling a dispatch operation.
typedef uint16_t iree_hal_amdgpu_device_dispatch_flags_t;
enum iree_hal_amdgpu_device_dispatch_flag_bits_t {
  IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_NONE = 0,
  // Dispatch requires the iree_amdgpu_kernel_implicit_args_t kernargs to be
  // appended (with 8-byte alignment) to the kernargs.
  IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_IMPLICIT_ARGS = 1u << 0,
  // Dispatch uses an indirect workgroup count that is constant and available
  // prior to command buffer execution. The command processor will read the
  // workgroup count and embed it directly in the AQL kernel dispatch packet.
  IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_STATIC = 1u << 1,
  // Dispatch uses an indirect workgroup count that is dynamic and may change up
  // to the exact moment the dispatch is issue. The command processor will
  // enqueue a kernel that performs the indirection and updates the kernel
  // dispatch packet with the value before allowing the hardware queue to
  // continue.
  IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC = 1u << 2,
};

// Either directly embedded workgroup count XYZ dimensions or a thin buffer ref
// pointing to a buffer containing the `uint32_t dims[3]` count.
typedef union iree_hal_amdgpu_device_workgroup_count_t {
  // XYZ dimensions of the grid, in workgroups. Must be greater than 0.
  // If the grid has fewer than 3 dimensions the unused ones must be 1.
  // Unused if the dispatch is indirect and instead the workgroups buffer
  // reference in the parameters is used.
  uint32_t dims[3];
  // Optional buffer containing the workgroup count.
  // Processing is controlled by the
  // IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_* flags.
  iree_hal_amdgpu_device_workgroup_count_buffer_ref_t ref;
} iree_hal_amdgpu_device_workgroup_count_t;

// AQL/HAL dispatch parameters as recorded.
// Some parameters may be overwritten as the packet is enqueued or during
// execution (such as for indirect dispatches).
typedef struct iree_hal_amdgpu_device_cmd_dispatch_config_t {
  // Dispatch control flags.
  iree_hal_amdgpu_device_dispatch_flags_t flags;
  uint16_t reserved0;
  // sharedMemBytes from the original dispatch. Added to the group_segment_size
  // during packet production.
  uint32_t dynamic_lds_size;
  // Kernel arguments used to dispatch the kernel.
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args;
  // Direct or indirect workgroup count based on the flags.
  iree_hal_amdgpu_device_workgroup_count_t workgroup_count;
} iree_hal_amdgpu_device_cmd_dispatch_config_t;
static_assert(
    sizeof(iree_hal_amdgpu_device_cmd_dispatch_config_t) == 32,
    "dispatch packet template is inlined into cmd structs and must be small");
#define IREE_HAL_AMDGPU_DEVICE_WORKGROUP_COUNT_UPDATE_KERNARG_SIZE \
  (3 * sizeof(uint64_t))

// Dispatches (directly or indirectly) a kernel.
// All information required to build the AQL packet is stored within the command
// such that it can be enqueued without additional indirection.
//
// Bindings and constants used by the dispatch are stored in an external data
// segment and may not be adjacent to each other. It's possible for multiple
// dispatches to share the same bindings and constants. Translation into
// kernargs happens when the command is issued on device.
//
// Recorded by:
//  iree_hal_command_buffer_dispatch
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_device_cmd_dispatch_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  // Block-relative kernel arguments address.
  // This will be added to the per-execution base kernel arguments address
  // during packet production.
  //
  // If the IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC bit is set
  // then this will include an additional
  // IREE_HAL_AMDGPU_DEVICE_WORKGROUP_COUNT_UPDATE_KERNARG_SIZE prefix that is
  // used for dispatching the
  // `iree_hal_amdgpu_device_command_buffer_workgroup_count_update` builtin
  // kernel.
  uint32_t kernarg_offset;
  // AQL packet template and dispatch parameters.
  iree_hal_amdgpu_device_cmd_dispatch_config_t config;
  // References describing how binding pointers are passed to the kernel.
  // References may include direct device pointers, allocation or slots in the
  // binding table included as part of the execution request.
  const iree_hal_amdgpu_device_buffer_ref_t* bindings /*[binding_count]*/;
  // Dispatch constants passed to the kernel.
  const uint32_t* constants /*[constant_count]*/;
} iree_hal_amdgpu_device_cmd_dispatch_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_DIRECT_AQL_PACKET_COUNT 1
#define IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_STATIC_AQL_PACKET_COUNT 1
#define IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC_AQL_PACKET_COUNT 2
#define IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_AQL_PACKET_COUNT(dispatch_flags)         \
  (((dispatch_flags) &                                                               \
    IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_STATIC) != 0)                      \
      ? IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_STATIC_AQL_PACKET_COUNT         \
      : ((((dispatch_flags) &                                                        \
           IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC) != 0)              \
             ? IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC_AQL_PACKET_COUNT \
             : IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_DIRECT_AQL_PACKET_COUNT)

// TODO(benvanik): better specify control flow; maybe conditional support.
// The current implementation is a placeholder for more sophisticated control
// flow both within a command buffer (branching) and across command buffers
// (calls). Calls will require nesting execution state and we may need to
// preallocate that (a primary command buffer keeping track of the max nesting
// depth).

// Unconditionally branches from the current block to a new block within the
// same command buffer.
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_device_cmd_branch_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  // Block-relative kernel arguments address.
  uint32_t kernarg_offset;
  // Block ordinal within the parent command buffer where execution will
  // continue. The block pointer can be retrieved from the command buffer
  // blocks list.
  uint32_t target_block;
} iree_hal_amdgpu_device_cmd_branch_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH_AQL_PACKET_COUNT 1
#define IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH_KERNARG_SIZE (2 * sizeof(uint64_t))
#define IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH_KERNARG_ALIGNMENT 8

// Specifies a `*ref <cond> value` operation.
typedef uint8_t iree_hal_amdgpu_device_cmd_cond_t;
enum iree_hal_amdgpu_device_cmd_cond_e {
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_EQ = 0u,  // *ref == value
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_NE,       // *ref != value
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_SLT,      // *ref < value (signed)
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_SLE,      // *ref <= value (signed)
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_SGT,      // *ref > value (signed)
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_SGE,      // *ref >= value (signed)
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_ULT,      // *ref < value (unsigned)
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_ULE,      // *ref <= value (unsigned)
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_UGT,      // *ref > value (unsigned)
  IREE_HAL_AMDGPU_DEVICE_CMD_COND_UGE,      // *ref >= value (unsigned)
};

// Conditionally branches from the current block to a new block within the
// same command buffer based on the specified condition evaluated at the time
// the command is executed.
//
// It is assumed that during recording any true_block==false_block conditions
// have been turned into unconditional iree_hal_amdgpu_device_cmd_branch_t
// commands.
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_cmd_cond_branch_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  // Block-relative kernel arguments address.
  uint32_t kernarg_offset;
  // Buffer containing the uint64_t value used for the condition comparison.
  iree_hal_amdgpu_device_uint64_buffer_ref_t ref;
  // Conditional operation (`*ref cond value`).
  iree_hal_amdgpu_device_cmd_cond_t cond;
  uint8_t reserved[7];
  // Value compared against using cond. Represented as signless bits with the
  // interpretation of the both `ref` and this value being based on the `cond`.
  uint64_t value;
  // Block ordinal within the parent command buffer where execution will
  // continue when the condition evaluates to true. The block pointer can be
  // retrieved from the command buffer blocks list.
  uint32_t true_block;
  // Block ordinal within the parent command buffer where execution will
  // continue when the condition evaluates to false. The block pointer can be
  // retrieved from the command buffer blocks list.
  uint32_t false_block;
} iree_hal_amdgpu_device_cmd_cond_branch_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_COND_BRANCH_AQL_PACKET_COUNT 1
#define IREE_HAL_AMDGPU_DEVICE_CMD_COND_BRANCH_KERNARG_SIZE \
  (6 * sizeof(uint64_t))
#define IREE_HAL_AMDGPU_DEVICE_CMD_COND_BRANCH_KERNARG_ALIGNMENT 8

// Returns from processing a command buffer by launching the scheduler.
//
// TODO(benvanik): differentiate return to scheduler from return to caller
// command buffer. Today this always assumes the scheduler is going to be the
// target.
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_device_cmd_return_t {
  iree_hal_amdgpu_device_cmd_header_t header;
  // Block-relative kernel arguments address.
  uint32_t kernarg_offset;
} iree_hal_amdgpu_device_cmd_return_t;
#define IREE_HAL_AMDGPU_DEVICE_CMD_RETURN_AQL_PACKET_COUNT 1
#define IREE_HAL_AMDGPU_DEVICE_CMD_RETURN_KERNARG_SIZE (1 * sizeof(uint64_t))
#define IREE_HAL_AMDGPU_DEVICE_CMD_RETURN_KERNARG_ALIGNMENT 8

static_assert(sizeof(iree_hal_amdgpu_device_cmd_debug_group_begin_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_debug_group_end_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_barrier_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_signal_event_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_reset_event_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_wait_events_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_fill_buffer_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_copy_buffer_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_dispatch_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_branch_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_cond_branch_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");
static_assert(sizeof(iree_hal_amdgpu_device_cmd_return_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");

// A command describing an operation that may translate to zero or more AQL
// packets.
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_device_cmd_t {
  union {
    iree_hal_amdgpu_device_cmd_header_t header;
    iree_hal_amdgpu_device_cmd_debug_group_begin_t debug_group_begin;
    iree_hal_amdgpu_device_cmd_debug_group_end_t debug_group_end;
    iree_hal_amdgpu_device_cmd_barrier_t barrier;
    iree_hal_amdgpu_device_cmd_signal_event_t signal_event;
    iree_hal_amdgpu_device_cmd_reset_event_t reset_event;
    iree_hal_amdgpu_device_cmd_wait_events_t wait_events;
    iree_hal_amdgpu_device_cmd_fill_buffer_t fill_buffer;
    iree_hal_amdgpu_device_cmd_copy_buffer_t copy_buffer;
    iree_hal_amdgpu_device_cmd_dispatch_t dispatch;
    iree_hal_amdgpu_device_cmd_branch_t branch;
    iree_hal_amdgpu_device_cmd_cond_branch_t cond_branch;
    iree_hal_amdgpu_device_cmd_return_t ret /*urn*/;
  };
} iree_hal_amdgpu_device_cmd_t;
static_assert(sizeof(iree_hal_amdgpu_device_cmd_t) <=
                  IREE_HAL_AMDGPU_DEVICE_CMD_SIZE,
              "commands must fit within the fixed command size");

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_command_buffer_t
//===----------------------------------------------------------------------===//

// Tracing query IDs used by a single command depending on tracing mode.
// These IDs are relative to the command block they are referenced from and
// added to whatever query ringbuffer base ID is used.
//
// Query IDs of 0xFFFF (IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID)
// indicate that a particular command does not use a query ID.
typedef struct iree_hal_amdgpu_device_command_query_id_t {
  // Query ID used for the command when the control flag is set:
  // IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL
  iree_hal_amdgpu_trace_execution_query_id_t control_id;
  // Query ID used for the command when the control+dispatch flag is set:
  // IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH
  iree_hal_amdgpu_trace_execution_query_id_t dispatch_id;
} iree_hal_amdgpu_device_command_query_id_t;
static_assert(sizeof(iree_hal_amdgpu_device_command_query_id_t) == 4,
              "query IDs interleaved/packed");

// Information required to allocate and map commands to query IDs used with
// tracing/profiling. The counts control how many unique query signals are
// allocated from the query ringbuffer when issuing the block. The embedded ID
// map is from each command to a relative query ID based on the ringbuffer's
// returned base ID. Query IDs must not be reused within the same command block
// as they are only captured during a flush.
typedef struct iree_hal_amdgpu_device_command_query_map_t {
  // Maximum number of queries used when in control mode:
  // IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL
  uint16_t max_control_query_count;
  // Maximum number of queries used with in control+dispatch mode:
  // IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH
  uint16_t max_dispatch_query_count;
  uint32_t reserved;  // may be uninitialized
  // One query ID entry per command when profiling/tracing is enabled.
  // Each entry contains the query ID to use in control-only mode and the one to
  // use in control+dispatch mode.
  iree_hal_amdgpu_device_command_query_id_t query_ids[/*command_count*/];
} iree_hal_amdgpu_device_command_query_map_t;

// A block of commands within a command buffer.
// Each block represents one or more commands that should be issued to target
// AQL queues as part of a single parallelized issue in a single contiguous
// span.
//
// Blocks are immutable once recorded and a block may be executed multiple
// times concurrently or serially with pipelining. Blocks are replicated per
// device such that any embedded device-local pointers are always valid for any
// queue the block is issued on. Any pointers that reference per-execution
// state (such as kernel argument buffers) are encoded as relative offsets to be
// added to whatever base pointer is reserved for the execution.
//
// Blocks are stored in a read-only memory region.
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_device_command_block_t {
  // Maximum number of AQL packets that the block will enqueue during a single
  // execution. Fewer packets may be used but they will still be populated with
  // valid no-op AQL packets to ensure forward progress by the packet processor.
  uint32_t max_packet_count;
  // Total number of commands in the block.
  uint32_t command_count;
  // Aligned storage for fixed-length command structures.
  const iree_hal_amdgpu_device_cmd_t* commands;
  // Tracing/profiling query map for commands in the block.
  iree_hal_amdgpu_device_command_query_map_t query_map;  // tail array
} iree_hal_amdgpu_device_command_block_t;

// A program consisting of one or more blocks of commands and control flow
// between them. Command buffers are immutable once recorded and retained in
// device local memory. A command buffer may be enqueued multiple times
// concurrently or in sequence as any state needed is stored separately in
// iree_hal_amdgpu_device_execution_state_t.
//
// Execution of a command buffer starts at block[0] and continues based on
// control flow commands at the tail of each block. Blocks may direct execution
// within the same command buffer or transfer control to other command buffers
// by nesting. Upon completion a return command at the tail of a block will
// return back to the caller.
//
// Command buffers are stored in a read-only memory region.
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_device_command_buffer_t {
  // Minimum required kernel argument buffer capacity to execute all blocks.
  // Only one block executes at a time and the storage will be reused.
  uint32_t max_kernarg_capacity;
  // Total number of blocks in the command buffer.
  uint32_t block_count;
  // A list of all blocks with block[0] being the entry point.
  // Commands reference blocks by ordinal in this list.
  const iree_hal_amdgpu_device_command_block_t* blocks[];  // tail array
} iree_hal_amdgpu_device_command_buffer_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_execution_state_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_DEVICE_EXECUTION_ISSUE_BLOCK_KERNARG_SIZE \
  (3 * sizeof(uint64_t))
#define IREE_HAL_AMDGPU_DEVICE_EXECUTION_CONTROL_KERNARG_SIZE \
  IREE_AMDGPU_MAX(8 * sizeof(uint64_t),                       \
                  IREE_HAL_AMDGPU_DEVICE_EXECUTION_ISSUE_BLOCK_KERNARG_SIZE)

// Controls command buffer execution behavior.
typedef uint8_t iree_hal_amdgpu_device_execution_flags_t;
enum iree_hal_amdgpu_device_execution_flag_bits_e {
  IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_NONE = 0u,
  // Issues work on the execution queue serially from the control queue.
  // This reduces execution latency but decreases throughput as one single
  // thread is populating all packets instead of it being done in parallel.
  IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_ISSUE_SERIALLY = (1u << 0),
  // Forces every command executed to have the AQL barrier bit set. This
  // serializes execution such that only one command can execute at a time. When
  // debugging dispatch exceptions or data corruption this can be used to ensure
  // only one dispatch at a time is executing on the device.
  IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_SERIALIZE = (1u << 1),
  // Forces cache invalidations/flushes between every command. This can be used
  // when stepping to ensure the host and device can see changes made on either
  // immediately.
  IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_UNCACHED = (1u << 2),
  // Enables tracing of command buffer control logic and instrumentation.
  // Implicit zones such as the total command buffer execution time, each
  // scheduling stage, and other events will be produced. Explicit zones created
  // via HAL command buffer debug APIs will be included.
  //
  // TODO(benvanik): find a way to avoid serializing here.
  IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL =
      (1u << 6) | IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_SERIALIZE,
  // Enables tracing of every dispatch (or DMA) command.
  // Timings are captured by the hardware and stored on a per-command query
  // signal. Forces all commands to be executed serially so that trace zones
  // remain perfectly nested and timing does not have any interference from
  // other concurrently executing commands. Note that total latency is expected
  // to increase due to the lack of concurrency.
  IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH =
      (1u << 7) | IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_SERIALIZE |
      IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL,
};

// Transient state used during the execution of a command buffer.
// Command buffers are executed like coroutines by having the command processor
// issue a sequence of commands before tail-enqueuing further processing or a
// return back to the top-level scheduler.
//
// Execution state is stored in mutable global memory so that the scheduler can
// manipulate it. Though immutable command buffer storage can be shared across
// all agents the execution state is initialized for the agent is executing on.
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_execution_state_t {
  // Flags controlling execution behavior.
  iree_hal_amdgpu_device_execution_flags_t flags;

  uint8_t reserved0[7];

  // Command buffer being executed.
  const iree_hal_amdgpu_device_command_buffer_t* command_buffer;

  // Block that should be executed when the state is scheduled.
  // Updated by control flow operations and set to NULL when returning.
  const iree_hal_amdgpu_device_command_block_t* block;

  // Scheduler that is managing the execution state lifetime.
  // When the command buffer completes it will be scheduled to handle cleanup
  // and resuming queue processing.
  iree_hal_amdgpu_device_queue_scheduler_t* scheduler;

  // Parent scheduler queue entry that this execution state is associated with.
  // Used to notify the scheduler when execution completes.
  uint64_t scheduler_queue_entry;

  // Queue used for control operations. This _may_ be the execution queue but
  // is usually independent to allow for command issue to overlap with
  // execution.
  iree_amd_cached_queue_t* control_queue;

  union {
    // State used for transfer operations done as part of the command buffer.
    iree_hal_amdgpu_device_buffer_transfer_context_t transfer_context;
    // NOTE: must match iree_hal_amdgpu_device_buffer_transfer_context_t.
    // This lets us toll-free share the transfer_state with the command buffer
    // itself.
    struct {
      // Queue used for command execution.
      // This may differ from the top-level scheduling queue.
      iree_amd_cached_queue_t execution_queue;
      // Handles to opaque kernel objects used to dispatch builtin kernels.
      const iree_hal_amdgpu_device_kernels_t* kernels;
      // Optional trace buffer used when tracing infrastructure is available.
      iree_hal_amdgpu_device_trace_buffer_t* trace_buffer;
    };
  };

  // Storage with space for control kernel arguments.
  // Initialized to contain the required set of args on initialization and
  // reused without change as we store control changes in the state struct.
  // Must be at least IREE_HAL_AMDGPU_DEVICE_EXECUTION_CONTROL_KERNARG_SIZE
  // bytes.
  IREE_AMDGPU_ALIGNAS(8) uint8_t* control_kernarg_storage;

  // Reserved storage for kernel arguments of at least the size specified by the
  // command buffer max_kernarg_capacity. Only one block can be executed
  // at a time and storage is reused. Note that storage is uninitialized and
  // must be fully specified by the command processor.
  IREE_AMDGPU_ALIGNAS(8) uint8_t* execution_kernarg_storage;

  // Last acquired base query ringbuffer index.
  // Used for all commands in the current block and reset after each block.
  uint64_t trace_block_query_base_id;
  // Total number of queries allocated from the ringbuffer for the last block.
  uint16_t trace_block_query_count;

  // TODO(benvanik): stack for remembering resume blocks when returning from
  // nested command buffers. For now we don't have calls so it's not needed.
  // uint32_t block_stack[...];

  // Binding table used to resolve indirect binding references.
  // Contains enough elements to satisfy all slots referenced by
  // iree_hal_amdgpu_device_buffer_ref_t in the command buffer.
  //
  // The enqueuing agent populates this and must ensure that all bindings stay
  // live until the command buffer completes executing by attaching a resource
  // set.
  //
  // Note that bindings here will not reference slots (though maybe we could
  // support that in the future for silly aliasing tricks).
  IREE_AMDGPU_ALIGNAS(64)
  iree_hal_amdgpu_device_buffer_ref_t bindings[];  // tail array
} iree_hal_amdgpu_device_execution_state_t;

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Launches a command buffer with the given initialized execution state.
// The command buffer will begin execution at the block specified in the state
// and continue (possibly rescheduling itself) until a return command is
// reached.
//
// Forward progress is only guaranteed so long as the hardware scheduling queue
// is not blocked (such as by waiting on the completion signal). Upon completion
// the command buffer return command will enqueue the scheduler so that it can
// clean up the execution state and resume processing the queue.
void iree_hal_amdgpu_device_command_buffer_enqueue_next_block(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_COMMAND_BUFFER_H_
