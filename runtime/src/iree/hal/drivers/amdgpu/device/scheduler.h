// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SCHEDULER_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SCHEDULER_H_

#include "iree/hal/drivers/amdgpu/device/allocator.h"
#include "iree/hal/drivers/amdgpu/device/command_buffer.h"
#include "iree/hal/drivers/amdgpu/device/host.h"
#include "iree/hal/drivers/amdgpu/device/semaphore.h"
#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/queue.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"
#include "iree/hal/drivers/amdgpu/device/support/signal_pool.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_entry_header_t / entries
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_device_queue_entry_header_t
    iree_hal_amdgpu_device_queue_entry_header_t;

// Queue entry type indicating the type and size of the arguments.
typedef uint8_t iree_hal_amdgpu_device_queue_entry_type_t;
enum iree_hal_amdgpu_device_queue_entry_type_e {
  IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_ALLOCA = 0,
  IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_DEALLOCA,
  IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_FILL,
  IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_UPDATE,
  IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_COPY,
  IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_EXECUTE,
  IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_BARRIER,
};

// Flags indicating how queue entries are to be processed.
typedef uint16_t iree_hal_amdgpu_device_queue_entry_flags_t;
enum iree_hal_amdgpu_device_queue_entry_flag_bits_e {
  IREE_HAL_AMDGPU_DEVICE_DEVICE_QUEUE_ENTRY_FLAG_NONE = 0u,
};

// Header for all scheduler queue entries.
// Each entry contains a set of fixed information used when initially enqueuing
// it in the incoming scheduler mailbox and dynamic information maintainined by
// the scheduler as the entry moves through its state machine.
typedef struct iree_hal_amdgpu_device_queue_entry_header_t {
  // Type of the queue entry used to issue the operation.
  iree_hal_amdgpu_device_queue_entry_type_t type;

  // Index into the active set the entry has been assigned while active.
  // This may change over the lifetime of the entry if it is made active
  // multiple times (such as after yielding).
  uint8_t active_bit_index;

  // Flags controlling queue entry behavior.
  iree_hal_amdgpu_device_queue_entry_flags_t flags;

  // Monotonically increasing value with lower values indicating entries that
  // were enqueued first. This is used to ensure FIFO execution ordering when
  // inserting into the run list. Assigned by the scheduler when accepting the
  // entry as there may be multiple producers and the epoch should be
  // scheduler-specific.
  uint32_t epoch;

  // Maximum number of bytes of the execution kernarg ringbuffer are required.
  // The entry will stall before issuing until capacity is available.
  // Must be aligned to IREE_HAL_AMDGPU_DEVICE_KERNARG_ALIGNMENT.
  uint32_t max_kernarg_capacity;

  // Whether the queue entry was allocated from the small (0) or large (1) block
  // pool.
  // TODO(benvanik): find another way to derive this bit. If we had size buckets
  // instead of just small/large we'd need more bits for that. The alternative
  // is storing a 64-bit pointer back to the pool and that feels excessive.
  uint32_t allocation_pool;

  // Host-side iree_hal_amdgpu_block_token_t from the queue entry allocation.
  // TODO(benvanik): move the allocator to be shared with the device-side
  // library so we can free entries from the device.
  uint64_t allocation_token;

  // Allocated absolute kernarg ringbuffer offset of max_kernarg_capacity bytes.
  // May be suballocated by the entry. Use
  // iree_hal_amdgpu_device_kernarg_ringbuffer_resolve to get the pointer.
  // UINT64_MAX indicates no kernargs are used (as would
  // max_kernarg_capacity=0).
  uint64_t kernarg_offset;

  // Semaphores that must be signaled before the queue entry is issued.
  // Semaphores will be removed from the list as they complete.
  // The semaphores must remain valid for the lifetime of the queue entry.
  iree_hal_amdgpu_device_semaphore_list_t* wait_list;

  // Semaphores to be signaled when the queue entry completes.
  // Semaphores that can be signaled on the device will be removed from the list
  // while any host-only semaphores (ones external to the HAL implementation or
  // that need a host callback) will remain for the host-side entry retirement
  // to handle.
  // The semaphores must remain valid for the lifetime of the queue entry.
  iree_hal_amdgpu_device_semaphore_list_t* signal_list;

  // Host-side iree_hal_resource_set_t tracking all resources (including wait
  // and signal semaphores) used by this entry. All will be kept live until the
  // entry is retired on the host.
  uint64_t resource_set;

  // Intrusive pointer used when the entry is in a linked list (wait list, run
  // list, etc).
  iree_hal_amdgpu_device_queue_entry_header_t* list_next;
} iree_hal_amdgpu_device_queue_entry_header_t;
static_assert(sizeof(iree_hal_amdgpu_device_queue_entry_header_t) <= 64,
              "queue entries should be kept as small as possible; avoid adding "
              "to the fixed header struct that increases the size of all "
              "entries unless it is something used by all entry types");

#define IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE \
  (2 * sizeof(uint64_t))

typedef struct iree_hal_amdgpu_device_queue_alloca_entry_t {
  // Type: IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_ALLOCA
  iree_hal_amdgpu_device_queue_entry_header_t header;
  iree_hal_amdgpu_device_allocation_pool_id_t pool_id;
  uint32_t min_alignment;
  uint64_t allocation_size;
  iree_hal_amdgpu_device_allocation_handle_t* handle;
} iree_hal_amdgpu_device_queue_alloca_entry_t;
#define IREE_HAL_AMDGPU_DEVICE_QUEUE_ALLOCA_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_device_queue_dealloca_entry_t {
  // Type: IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_DEALLOCA
  iree_hal_amdgpu_device_queue_entry_header_t header;
  iree_hal_amdgpu_device_allocation_handle_t* handle;
} iree_hal_amdgpu_device_queue_dealloca_entry_t;
#define IREE_HAL_AMDGPU_DEVICE_QUEUE_DEALLOCA_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_device_queue_fill_entry_t {
  // Type: IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_FILL
  iree_hal_amdgpu_device_queue_entry_header_t header;
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
  uint64_t pattern;
  uint8_t pattern_length;
} iree_hal_amdgpu_device_queue_fill_entry_t;
#define IREE_HAL_AMDGPU_DEVICE_QUEUE_FILL_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE +           \
   IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_device_queue_update_entry_t {
  // Type: IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_UPDATE
  iree_hal_amdgpu_device_queue_entry_header_t header;
  const IREE_AMDGPU_DEVICE_PTR void* source_ptr;
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
} iree_hal_amdgpu_device_queue_update_entry_t;
#define IREE_HAL_AMDGPU_DEVICE_QUEUE_UPDATE_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE +             \
   IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_device_queue_copy_entry_t {
  // Type: IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_COPY
  iree_hal_amdgpu_device_queue_entry_header_t header;
  iree_hal_amdgpu_device_buffer_ref_t source_ref;
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
} iree_hal_amdgpu_device_queue_copy_entry_t;
#define IREE_HAL_AMDGPU_DEVICE_QUEUE_COPY_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE +           \
   IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_device_queue_execute_entry_t {
  // Type: IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_EXECUTE
  iree_hal_amdgpu_device_queue_entry_header_t header;
  // Execution control flags.
  iree_hal_amdgpu_device_execution_flags_t flags;
  // Command buffer being executed.
  const iree_hal_amdgpu_device_command_buffer_t* command_buffer;
  // State used during command buffer execution. Mutated in-place.
  // Enqueuers only need to populate bindings.
  //
  // TODO(benvanik): move this to ensure device-local memory instead of sharing
  // with the queue storage. We'd need to have the bindings here and then
  // replicate them. It would allow us to remove the allocation_handle
  // dereferences from the command issue path by doing it once at entry issue.
  iree_hal_amdgpu_device_execution_state_t state;
} iree_hal_amdgpu_device_queue_execute_entry_t;
#define IREE_HAL_AMDGPU_DEVICE_QUEUE_EXECUTE_MAX_KERNARG_CAPACITY( \
    command_buffer_max_kernarg_capacity)                           \
  (IREE_HAL_AMDGPU_DEVICE_EXECUTION_CONTROL_KERNARG_SIZE +         \
   (command_buffer_max_kernarg_capacity))

typedef struct iree_hal_amdgpu_device_queue_barrier_entry_t {
  // Type: IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_BARRIER
  iree_hal_amdgpu_device_queue_entry_header_t header;
} iree_hal_amdgpu_device_queue_barrier_entry_t;
#define IREE_HAL_AMDGPU_DEVICE_QUEUE_BARRIER_MAX_KERNARG_CAPACITY 0

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernarg_ringbuffer_t
//===----------------------------------------------------------------------===//

// A multi-producer/single-consumer ringbuffer containing out-of-order or
// remotely freed kernarg allocations. It has a fixed size as the number of
// active queue entries that can have allocations outstanding is limited by the
// fixed size iree_hal_amdgpu_device_queue_active_set_t. Kernargs are always
// allocated forward in the ringbuffer but may be freed out-of-order as queue
// operations complete. If a queue operation is retired by the scheduler on
// device and the kernarg allocation is at the read index then the read index
// will be advanced and otherwise the pending free request will be added to the
// free list. When the scheduler ticks it will run down the free list and try to
// advance the read index as far as possible. The host or other agents that
// retire queue entries can also add their kernarg allocations to the free list.
// Note that the common host enqueue -> device execute -> host wait pattern will
// always produce the host-side retire and add entries to the free list. We only
// store the min/max range of the kernargs and don't directly reference the
// original queue entry that was retired as by the time the list is reclaimed
// the queue entry may have been freed or reused.
//
// Thread-safe for producers, thread-compatible for the consumer.
// This is intended to be written to from the host or other agents that may be
// retiring queue entries that have associated kernarg allocations. Only the
// owner of the kernarg ringbuffer is allowed to read as it runs down the free
// list and updates its local data structures.
typedef struct iree_hal_amdgpu_device_kernarg_reclaim_list_t {
  iree_amdgpu_scoped_atomic_uint64_t read_index;
  iree_amdgpu_scoped_atomic_uint64_t write_index;
  struct {
    uint64_t min;
    uint64_t max;
  } slots[64];
} iree_hal_amdgpu_device_kernarg_reclaim_list_t;

#define iree_hal_amdgpu_device_kernarg_reclaim_list_capacity(list) \
  IREE_AMDGPU_ARRAYSIZE((list)->slots)

// All kernargs start at this offset and any kernarg allocation is rounded up to
// this size.
#define IREE_HAL_AMDGPU_DEVICE_KERNARG_ALIGNMENT 64

// A lightweight single-producer/single-consumer ringbuffer used for kernargs.
//
// Writes must check for overflow by ensuring that there is sufficient capacity
// for their reservation. An example write sequence:
// if (write_offset + requested_size - read_offset >= ringbuffer_capacity) {
//    iree_amdgpu_yield();
// }
// memcpy(base_ptr + write_offset, contents, requested_size);
// write_offset += requested_size;
//
// This presents as a ringbuffer that does not need any special logic for
// wrapping from base offsets used when copying in memory. It follows the
// approach documented in of virtual memory mapping the buffer multiple times:
// https://github.com/google/wuffs/blob/main/script/mmap-ring-buffer.c
// We use SVM to allocate the physical memory of the ringbuffer and then stitch
// together 3 virtual memory ranges in one contiguous virtual allocation that
// alias the physical allocation. By treating the middle range as the base
// buffer pointer we are then able to freely dereference both before and after
// the base pointer by up to the ringbuffer size in length.
//   physical: <ringbuffer size> --+------+------+
//                                 v      v      v
//                        virtual: [prev] [base] [next]
//                                        ^
//                                        +-- base_ptr
//
// Because of the mapping trick we have a maximum outstanding ringbuffer size
// equal to the ringbuffer capacity (modulo alignment requirements).
// The host must allocate the ringbuffer storage from the device kernarg pool.
//
// Thread-compatible; only used by a single scheduler on the control queue.
typedef struct iree_hal_amdgpu_device_kernarg_ringbuffer_t {
  // Base ringbuffer pointer used for all relative addressing.
  // Pointers must always be within the range of
  // (base_ptr-capacity, base_ptr+capacity).
  uint8_t* IREE_AMDGPU_RESTRICT base_ptr;
  // Total size in bytes of the kernarg ringbuffer.
  // Note that this is the size of the underlying physical allocation and the
  // virtual range is 3x that. Must be a power of two.
  uint64_t capacity;
  // Current absolute read offset. The write offset must not advance past this.
  uint64_t read_offset;
  // Current absolute write offset. All data up to this point is valid.
  uint64_t write_offset;
  // A list of pending kernarg range reclaimations.
  // Any range placed into the list will be reclaimed on the next call to
  // iree_hal_amdgpu_device_kernarg_ringbuffer_reclaim.
  iree_hal_amdgpu_device_kernarg_reclaim_list_t reclaim_list;
} iree_hal_amdgpu_device_kernarg_ringbuffer_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_list_t
//===----------------------------------------------------------------------===//

// An singly-linked intrusive list of queue entries.
// This uses the `list_next` field of each entry and requires that an entry only
// be in one list at a time. Because we use these lists to manage wait and run
// lists and entries can only be in one at a time we don't run into collisions.
//
// List order is determined by how entries are inserted. Producers must ensure
// they are consistent about either inserting in FIFO list order or FIFO
// submission order (using queue entry epochs).
//
// Thread-compatible; expected to only be accessed locally.
// Zero initialization compatible.
typedef struct iree_hal_amdgpu_device_queue_list_t {
  iree_hal_amdgpu_device_queue_entry_header_t* head;
  iree_hal_amdgpu_device_queue_entry_header_t* tail;
} iree_hal_amdgpu_device_queue_list_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_active_set_t
//===----------------------------------------------------------------------===//

// A set of active queue entries that are scheduled on the execution queue.
// The scheduler adds entries to the active set when they are scheduled and then
// entries subsequently notify the scheduler of their desired resolution. If an
// entry needs follow-up scheduling work (such as continuing from a yield point)
// then the corresponding reschedule bit should be set before requesting a
// scheduler run. If an entry has completed and need to be moved to the retire
// list the corresponding retire bit should be set.
//
// Thread-compatible/thread-safe; management of the active entries is expected
// to happen locally on the scheduler queue but requests for rescheduling and
// retirement can happen on any agent.
typedef struct iree_hal_amdgpu_device_queue_active_set_t {
  // Bitmask of entries that are valid.
  // This is used to quickly find available slots, check capacity, and optimize
  // scans over the active set.
  uint64_t active_bits;
  // Bitmask of active entries that need to be rescheduled on the next tick.
  // Entries will be moved to the ready list and scheduled as normal.
  iree_amdgpu_scoped_atomic_uint64_t reschedule_bits;
  // Bitmask of active entries that need to be retired on the next tick.
  // Entries will be moved to the retire list and handled as soon as possible.
  iree_amdgpu_scoped_atomic_uint64_t retire_bits;
  // Pointers to queue entries. Only entries indicated via active_bits are
  // valid.
  iree_hal_amdgpu_device_queue_entry_header_t* IREE_AMDGPU_RESTRICT entries[64];
} iree_hal_amdgpu_device_queue_active_set_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_mailbox_t
//===----------------------------------------------------------------------===//

// Incoming mailbox capacity in entries. Must be a power-of-two.
#define IREE_HAL_AMDGPU_DEVICE_MAILBOX_CAPACITY (256)

// Sentinel value used to indicate an invalid mailbox entry.
#define IREE_HAL_AMDGPU_DEVICE_MAILBOX_ENTRY_INVALID UINT64_MAX

// Small multi-producer/single-consumer ringbuffer.
// Operates like a paired down HSA queue: producers acquire a write index by
// incrementing and then populate the entry with the returned index masked by
// the queue capacity with the new entry. The scheduler consumes entries by
// advancing the read index to the write index and swapping out entries with the
// IREE_HAL_AMDGPU_DEVICE_MAILBOX_ENTRY_INVALID sentinel. As with HSA queues
// producers must first reserve the index and spin until the value is
// IREE_HAL_AMDGPU_DEVICE_MAILBOX_ENTRY_INVALID indicating that any prior write
// has been read by the scheduler.
typedef struct iree_hal_amdgpu_device_mailbox_t {
  iree_amdgpu_scoped_atomic_uint64_t read_index;
  iree_amdgpu_scoped_atomic_uint64_t write_index;
  uint64_t entries[IREE_HAL_AMDGPU_DEVICE_MAILBOX_CAPACITY];
} iree_hal_amdgpu_device_mailbox_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_scheduler_t
//===----------------------------------------------------------------------===//

// Soft queue agent dispatch packet type.
enum iree_hal_amdgpu_device_queue_scheduler_call_e {
  // Enqueues a new queue entry. Execution will respect the wait semaphores on
  // the entry and signal semaphores will be notified when execution completes.
  //
  // Signature:
  //   arg0: iree_hal_amdgpu_device_queue_scheduler_t* scheduler
  //   arg1: iree_hal_amdgpu_device_queue_entry_header_t* entry
  //   arg2: unused
  //   arg3: unused
  //   return_address: unused
  //   completion_signal: unused
  IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULER_CALL_ENQUEUE = 0u,
};

// Controls scheduler behavior.
typedef uint8_t iree_hal_amdgpu_device_queue_scheduling_mode_t;
enum iree_hal_amdgpu_device_queue_scheduling_mode_bits_e {
  IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULING_MODE_DEFAULT = 0u,

  // Only one queue entry is allowed to be active at a time. Others will wait in
  // the ready list and execute in epoch order.
  IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULING_MODE_EXCLUSIVE = 1u << 0,

  // Attempt to schedule entries out-of-order to fill available resources.
  // This may reduce overall latency if small entries come in while large ones
  // are outstanding - or may make things worse as large entries may come in
  // and acquire resources just before a prior entry completes.
  // https://en.wikipedia.org/wiki/Work-conserving_scheduler
  IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULING_MODE_WORK_CONSERVING = 1u << 1,
};

// Indicates one or more actions that should happen on the next tick.
typedef uint64_t iree_hal_amdgpu_device_queue_tick_action_set_t;
enum iree_hal_amdgpu_device_queue_tick_action_bits_e {
  // No actions are required.
  IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_NONE = 0x0ul,
  // Work is scheduled in the mailbox. The mailbox should always be checked.
  IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_INCOMING = 1ul << 0,
  // A waiter is likely to have resolved (even if partially).
  IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_WAKE = 1ul << 1,
  // The previous queue tick had remaining work but yielded early.
  IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_CONTINUE = 1ul << 2,
  // One or more queue entries have been added to the queue reschedule set.
  IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_RESCHEDULE = 1ul << 3,
  // One or more queue entries have been added to the queue retire set.
  IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_RETIRE = 1ul << 4,
};

// Parameters used to initialize a queue scheduler.
// This is populated by the device and passed as part of the initialization
// args.
typedef struct iree_hal_amdgpu_device_queue_scheduler_params_t {
  iree_hal_amdgpu_device_queue_scheduling_mode_t mode;
  uint64_t queue;
  iree_hal_amdgpu_device_host_t host;
  iree_hal_amdgpu_device_allocator_t* allocator;
  void* control_kernargs;
  struct {
    uint8_t* IREE_AMDGPU_RESTRICT base_ptr;
    uint64_t capacity;
  } kernarg_ringbuffer;  // iree_hal_amdgpu_device_kernarg_ringbuffer_t
  iree_hal_amdgpu_device_kernels_t kernels;
  struct {
    iree_hal_amdgpu_device_signal_pool_t* IREE_AMDGPU_RESTRICT ptr;
    uint32_t count;
    iree_hsa_signal_t* IREE_AMDGPU_RESTRICT values;
  } signal_pool;  // iree_hal_amdgpu_device_signal_pool_t
  iree_hsa_queue_t* control_queue;
  uint32_t execution_queue_count;
  uint32_t reserved0;
  iree_hsa_queue_t* execution_queues[/*execution_queue_count*/];
} iree_hal_amdgpu_device_queue_scheduler_params_t;

// A logical HAL queue scheduler.
// Manages the HAL queue for a single queue affinity bit on a HAL device. There
// may be multiple HAL queues for a single HSA agent and each will manage at
// least one HSA execution queue. Multiple queue schedulers are allowed to share
// the same HSA queues for control operations (ticking, retiring entries, etc)
// and for execution.
//
// HSA queues do not support out-of-order execution while HAL queues do: this
// means a user may be enqueuing entries that wait on semaphores that have not
// yet had a signal enqueued _on the same queue_. To avoid both deadlocks and
// head-of-line blocking we manage our own wait list and move queue entries to
// an active set when all of their waits have resolved. Once active a queue
// entry gets issued on an HSA queue and is expected to complete in a finite
// amount of time. Once the queue entry completes it is retired from the active
// set and its resources are reclaimed.
//
// Practically all execution requires kernarg storage. Each queue entry declares
// how much storage it needs and the scheduler only issues entries when there is
// sufficient capacity in the queue-specific kernarg storage ringbuffer. If
// capacity is unavailable the entry will be retried after other entries retire
// and free up capacity.
//
// The host owns the scheduler and its resources and is responsible for
// allocating them on startup and deallocating them after the queue has fully
// shut down. Many of the resources utilized on device are wrapped in host
// objects or at least referenced by them to ensure they remain valid. When
// queue entries retire the host is notified of any resources it may be able to
// reclaim.
typedef struct iree_hal_amdgpu_device_queue_scheduler_t {
  // Controls scheduling behavior.
  iree_hal_amdgpu_device_queue_scheduling_mode_t mode;

  uint8_t reserved0[3];
  uint32_t reserved1;

  // Host-side queue owning this scheduler. Used to route back device->host to
  // the queue.
  uint64_t queue;

  // Host agent used to perform services at the request of the device runtime.
  // May be shared with multiple schedulers.
  iree_hal_amdgpu_device_host_t host;

  // Device-side allocator.
  // May be shared with multiple schedulers but always represents device-local
  // memory.
  iree_hal_amdgpu_device_allocator_t* allocator;

  // Mailbox used for accepting incoming queue entries.
  iree_hal_amdgpu_device_mailbox_t mailbox;

  // Queue used for launching the top-level scheduler after execution completes.
  iree_hsa_queue_t* control_queue;

  // Cached kernargs that is used for scheduler ticks.
  // Contains just the scheduler pointer. This allows remote schedulers and
  // execution queues to dispatch a tick without needing to dynamically allocate
  // kernarg space. Should be device local to avoid additional traffic.
  //
  // This is reused for initialization and contains:
  //   iree_hal_amdgpu_device_queue_scheduler_t* scheduler;
  //   iree_hal_amdgpu_device_queue_scheduler_params_t* params;
  //
  // Additional storage may be allocated for other components managed by the
  // scheduler such as the trace buffer.
  void* control_kernargs;

  // A ringbuffer used for kernargs during execution.
  // It is assumed that operations will complete and release their ringbuffer
  // space in bounded time. Must have at least as much capacity for the largest
  // allocation of kernargs required by any execution.
  iree_hal_amdgpu_device_kernarg_ringbuffer_t kernarg_ringbuffer;

  union {
    // State used for transfer operations done as part of the command buffer.
    iree_hal_amdgpu_device_buffer_transfer_state_t transfer_state;
    // NOTE: must match iree_hal_amdgpu_device_buffer_transfer_state_t.
    // This lets us toll-free share the transfer_state with the command buffer
    // itself.
    struct {
      // Queue used for command buffer execution.
      // This may differ from the top-level scheduling queue.
      //
      // TODO(benvanik): allow multiple queues. We could allow multiple command
      // buffers to issue/execute concurrently so long as their dependencies are
      // respected. Or allow a single command buffer to target multiple hardware
      // queues. We'd need to change trace buffer scoping in that case.
      iree_hsa_queue_t* execution_queue;

      // Handles to opaque kernel objects used to dispatch builtin kernels.
      const iree_hal_amdgpu_device_kernels_t* IREE_AMDGPU_RESTRICT kernels;

      // Trace buffer dedicated to this scheduler. Only this scheduler can write
      // to the buffer and only the host can read from the buffer.
      iree_hal_amdgpu_device_trace_buffer_t* trace_buffer;
    };
  };

  // Pool of HSA signals that can be used by device code.
  // The pool will be used by the scheduler as well as various subsystems to
  // get signals as they are opaque objects that must have been allocated on the
  // host. Note that when the pool is exhausted the scheduler will abort.
  iree_hal_amdgpu_device_signal_pool_t* signal_pool;

  // Epoch allocated to the next incoming queue entry. We assign epochs on
  // acceptance as we may have multiple producers (host or device) and don't
  // want them to have to synchronize. The mailbox is FIFO so original request
  // order is still observed.
  uint64_t next_epoch;

  // A bitmask of iree_hal_amdgpu_device_queue_tick_action_set_t values
  // indicating work that needs to be performed on the next tick. Those
  // requesting ticks must OR their required actions prior to enqueuing a tick.
  // The next tick that runs will process all actions.
  //
  // This _may_ be used to optimize ticks by only walking data structures if
  // there's likely to be something changed wtih. At minimum it is useful for
  // tracking what operations are pending with tracing annotations.
  IREE_AMDGPU_ALIGNAS(64)
  iree_amdgpu_scoped_atomic_uint64_t tick_action_set;
  uint8_t
      tick_action_set_padding[64 - sizeof(iree_amdgpu_scoped_atomic_uint64_t)];

  // Queue entries blocked waiting for their dependencies to resolve.
  // The scheduler will poll each entry and move it to the ready list when
  // possible.
  iree_hal_amdgpu_device_queue_list_t wait_list;

  // Queue entries that are ready to be issued.
  // The scheduler will issue the entries as soon as resources are available.
  iree_hal_amdgpu_device_queue_list_t ready_list;

  // Queue entries that are in the active/issued state.
  // These are asynchronously executing and will move themselves to the retire
  // list when they are complete.
  iree_hal_amdgpu_device_queue_active_set_t active_set;

  // Scheduler targets that should be woken after the current tick completes.
  // This is only used during the tick but large enough that we keep it in the
  // heap.
  iree_hal_amdgpu_device_wake_set_t wake_set;

  // A pool of semaphore-to-wake mappings. Fixed size and should be checked for
  // exhaustion.
  iree_hal_amdgpu_device_wake_pool_t wake_pool;

  // Kernels resolved for the agent this scheduler is on.
  iree_hal_amdgpu_device_kernels_t kernel_storage;
} iree_hal_amdgpu_device_queue_scheduler_t;

// Control kernargs used when launching the scheduler.
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_queue_scheduler_kernargs_t {
  // Expected control kernel arguments:
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_scheduler_t* scheduler;
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_scheduler_params_t*
      params;

  // Additional storage for payloads.
  // This is to avoid extra allocations.
  struct {
    IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_trace_buffer_kernargs_t trace_buffer;
    IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_device_queue_scheduler_params_t params;  // tail array
  } storage;
} iree_hal_amdgpu_device_queue_scheduler_kernargs_t;

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULER_KERNARG_SIZE (1 * sizeof(void*))

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Enqueues a scheduler tick from any scheduling control queue.
void iree_hal_amdgpu_device_queue_scheduler_enqueue_from_control_queue(
    iree_hal_amdgpu_device_queue_scheduler_t* IREE_AMDGPU_RESTRICT scheduler,
    iree_hal_amdgpu_device_queue_tick_action_set_t action_set);

// Enqueues a scheduler tick from any execution queue.
void iree_hal_amdgpu_device_queue_scheduler_enqueue_from_execution_queue(
    iree_hal_amdgpu_device_queue_scheduler_t* IREE_AMDGPU_RESTRICT scheduler,
    iree_hal_amdgpu_device_queue_tick_action_set_t action_set);

// Marks a queue entry as needing to be rescheduled and possibly enqueues a
// scheduler tick. The argument is the opaque queue entry pointer.
void iree_hal_amdgpu_device_queue_scheduler_reschedule_from_execution_queue(
    iree_hal_amdgpu_device_queue_scheduler_t* IREE_AMDGPU_RESTRICT scheduler,
    uint64_t scheduler_queue_entry);

// Marks a queue entry as retired and possibly enqueues a scheduler tick.
// The argument is the opaque queue entry pointer. The queue entry may be
// deallocated immediately and callers must not access any resources that may
// have been stored within it.
void iree_hal_amdgpu_device_queue_scheduler_retire_from_execution_queue(
    iree_hal_amdgpu_device_queue_scheduler_t* IREE_AMDGPU_RESTRICT scheduler,
    uint64_t scheduler_queue_entry);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SCHEDULER_H_
