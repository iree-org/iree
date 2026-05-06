// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/queue.h"

#include <stdio.h>
#include <string.h>

#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/command_buffer.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/local/transient_buffer.h"
#include "iree/hal/utils/memory_file.h"

#define IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_SLOT_ABSENT UINT32_MAX
#define IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_SLOT_RESERVED UINT64_MAX
#define IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX 2

typedef void(IREE_API_PTR* iree_hal_vulkan_queue_completion_action_fn_t)(
    void* user_data, iree_status_t completion_status);

typedef struct iree_hal_vulkan_queue_completion_action_t {
  // Completion callback invoked after the queue epoch retires.
  iree_hal_vulkan_queue_completion_action_fn_t fn;

  // User data passed to |fn|.
  void* user_data;

  // Optional resource retained until |fn| has been invoked or cancelled.
  iree_hal_resource_t* resource;
} iree_hal_vulkan_queue_completion_action_t;

static iree_hal_vulkan_queue_completion_action_t
iree_hal_vulkan_queue_completion_action_null(void) {
  return (iree_hal_vulkan_queue_completion_action_t){0};
}

struct iree_hal_vulkan_queue_descriptor_block_t {
  // Next block in the queue-owned descriptor cache.
  iree_hal_vulkan_queue_descriptor_block_t* next;

  // Descriptor pool backing |sets|.
  VkDescriptorPool pool;

  // Descriptor sets leased by built-in queue operations.
  VkDescriptorSet sets[IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY];

  // Owning queue epoch for each descriptor set, or 0 when free.
  uint64_t owner_epochs[IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY];

  // Next slot considered when acquiring from this block.
  uint32_t next_slot;

  // Number of descriptor sets whose |owner_epochs| entry is 0.
  uint32_t free_count;
};

typedef struct iree_hal_vulkan_queue_descriptor_lease_t {
  // Descriptor block containing |set|.
  iree_hal_vulkan_queue_descriptor_block_t* block;

  // Descriptor set slot within |block|.
  uint32_t slot;
} iree_hal_vulkan_queue_descriptor_lease_t;

typedef enum iree_hal_vulkan_queue_submission_kind_e {
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER = 0,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL = 1,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL = 2,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE = 3,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY = 4,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE = 5,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA = 6,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA = 7,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND = 8,
} iree_hal_vulkan_queue_submission_kind_t;

typedef enum iree_hal_vulkan_queue_deferred_state_e {
  // Linked on the deferred list and waiting for software dependencies.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING = 0,

  // A dependency callback owns promotion from deferred to ready.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PROMOTING = 1,

  // Linked on the ready list for native submission by the completion thread.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_READY = 2,

  // Cancellation owns the unsubmitted node after unlinking it from deferred.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_CANCELLING = 3,
} iree_hal_vulkan_queue_deferred_state_t;

typedef enum iree_hal_vulkan_queue_alloca_memory_wait_kind_e {
  // No active queue-owned memory readiness wait.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE = 0,

  // Staged backing is reserved but cannot be used until a pool death frontier
  // retires.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER = 1,

  // No reservation is currently available; retry after the pool notification
  // advances.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION = 2,
} iree_hal_vulkan_queue_alloca_memory_wait_kind_t;

typedef struct iree_hal_vulkan_queue_wait_entry_t {
  // Async semaphore timepoint registered for one software-resolved wait.
  iree_async_semaphore_timepoint_t timepoint;

  // Deferred submission owning this wait entry.
  iree_hal_vulkan_queue_pending_submission_t* submission;

  // Set to non-zero after the timepoint callback's final access completes.
  iree_atomic_int32_t callback_complete;
} iree_hal_vulkan_queue_wait_entry_t;

typedef struct iree_hal_vulkan_queue_wait_resolution_t {
  // Native Vulkan wait semaphores resolved from queue last-signal metadata.
  VkSemaphoreSubmitInfo* wait_infos;

  // Number of populated wait_infos entries.
  uint32_t wait_info_count;

  // Capacity of the wait_infos storage.
  uint32_t wait_info_capacity;

  // Whether unresolved waits require software timepoint deferral.
  bool needs_deferral;
} iree_hal_vulkan_queue_wait_resolution_t;

typedef struct iree_hal_vulkan_queue_submission_result_t {
  // Whether ownership moved to the native pending queue.
  bool submitted;

  // Submission parked on a queue-owned memory readiness wait.
  iree_hal_vulkan_queue_pending_submission_t* memory_wait_submission;

  // Sticky queue failure status recorded during native submission.
  iree_status_t queue_failure_status;
} iree_hal_vulkan_queue_submission_result_t;

struct iree_hal_vulkan_queue_pending_submission_t {
  // Next submission in the queue-owned list currently holding this object.
  iree_hal_vulkan_queue_pending_submission_t* next;

  // Queue owning this submission.
  iree_hal_vulkan_queue_t* queue;

  // Queue epoch signaled by the native Vulkan submit.
  uint64_t epoch;

  // Queue frontier published by this submission.
  iree_hal_vulkan_queue_frontier_t frontier;

  // Wait semaphores retained until the native submit retires.
  iree_hal_semaphore_list_t wait_semaphore_list;

  // Signal semaphores retained until completion side effects retire.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Submission kind controlling completion-side actions.
  iree_hal_vulkan_queue_submission_kind_t kind;

  // HAL-native profiling metadata captured for this submission.
  struct {
    // Recorder active when the submission was captured. Borrowed.
    iree_hal_local_profile_recorder_t* recorder;

    // Queue metadata scope active when the submission was captured.
    iree_hal_local_profile_queue_scope_t scope;

    // Queue event type corresponding to |kind|.
    iree_hal_profile_queue_event_type_t type;

    // Queue event flags accumulated during dependency resolution.
    iree_hal_profile_queue_event_flags_t flags;

    // Strategy used to satisfy wait dependencies.
    iree_hal_profile_queue_dependency_strategy_t dependency_strategy;

    // Host timestamp when the queue operation was captured.
    iree_time_t submit_host_time_ns;

    // Host timestamp when the operation was accepted for native/deferred work.
    iree_time_t ready_host_time_ns;

    // Profile submission id assigned to this queue operation.
    uint64_t submission_id;

    // Number of payload operations represented by this queue operation.
    uint32_t operation_count;

    // Type-specific payload length represented by this queue operation.
    uint64_t payload_length;

    // Set after the queue event has been appended to the recorder.
    bool queue_event_recorded;

    // Query pool receiving device timestamps, or VK_NULL_HANDLE.
    VkQueryPool query_pool;

    // Number of query slots allocated in query_pool.
    uint32_t query_count;

    // Query index written before native command payloads, or ABSENT.
    uint32_t queue_start_query;

    // Query index written after native command payloads, or ABSENT.
    uint32_t queue_end_query;

    // First query slot for per-dispatch timestamp pairs, or ABSENT.
    uint32_t dispatch_base_query;

    // Number of dispatch commands with per-dispatch timestamp pairs.
    uint32_t dispatch_query_count;

    // Host storage receiving query results after native completion.
    uint64_t* query_values;

    // Set after the queue device event has been appended to the recorder.
    bool queue_device_event_recorded;
  } profile;

  // Native Vulkan command buffer submitted for GPU-encoded work.
  VkCommandBuffer native_command_buffer;

  // Descriptor cache leases held by built-in queue operations.
  iree_hal_vulkan_queue_descriptor_lease_t native_descriptor_leases
      [IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX];

  // Number of entries populated in native_descriptor_leases.
  uint32_t native_descriptor_slot_count;

  // Descriptor pool backing native command-buffer descriptor sets.
  VkDescriptorPool native_descriptor_pool;

  // Optional action invoked after native queue completion.
  iree_hal_vulkan_queue_completion_action_t completion_action;

  // Queue-ordered sparse buffer memory binding payload.
  struct {
    // Sparse buffer receiving the memory binds.
    VkBuffer buffer;

    // Queue-owned sparse memory bind array.
    VkSparseMemoryBind* binds;

    // Number of populated entries in |binds|.
    uint32_t bind_count;
  } sparse_bind;

  // Wait entries registered for software-resolved wait-before-signal edges.
  iree_hal_vulkan_queue_wait_entry_t* wait_entries;

  // Number of entries in wait_entries.
  iree_host_size_t wait_entry_count;

  // Outstanding software wait callbacks before the submission becomes ready.
  iree_atomic_int32_t wait_count;

  // Lifecycle owner for software-deferred submission promotion/cancellation.
  iree_atomic_int32_t deferred_state;

  // First software wait failure status. Owned by this submission.
  iree_atomic_intptr_t wait_failure_status;

  // Wakes cancellation or promotion owners waiting for callbacks to quiesce.
  iree_notification_t callback_notification;

  // Queue-ordered host action payload.
  struct {
    // Host callback and user data captured from queue_host_call.
    iree_hal_host_call_t call;

    // User arguments copied from queue_host_call.
    uint64_t args[4];

    // HAL host-call flags captured from queue_host_call.
    iree_hal_host_call_flags_t flags;
  } host_call;

  // Native buffer fill payload.
  struct {
    // Target buffer retained until the fill retires.
    iree_hal_buffer_t* target_buffer;

    // Target byte offset captured from queue_fill.
    iree_device_size_t target_offset;

    // Number of bytes to fill in the target buffer.
    iree_device_size_t length;

    // Fill pattern bytes captured from queue_fill.
    uint8_t pattern[4];

    // Number of bytes in pattern.
    iree_host_size_t pattern_length;

    // HAL fill flags captured from queue_fill.
    iree_hal_fill_flags_t flags;
  } fill;

  // Native buffer update payload.
  struct {
    // Source bytes copied from the queue_update caller.
    void* source_data;

    // Target buffer retained until the update retires.
    iree_hal_buffer_t* target_buffer;

    // Target byte offset captured from queue_update.
    iree_device_size_t target_offset;

    // Number of bytes to copy into the target buffer.
    iree_device_size_t length;

    // HAL update flags captured from queue_update.
    iree_hal_update_flags_t flags;
  } update;

  // Host-mediated buffer copy payload.
  struct {
    // Source buffer retained until the copy retires.
    iree_hal_buffer_t* source_buffer;

    // Source byte offset captured from queue_copy.
    iree_device_size_t source_offset;

    // Target buffer retained until the copy retires.
    iree_hal_buffer_t* target_buffer;

    // Target byte offset captured from queue_copy.
    iree_device_size_t target_offset;

    // Number of bytes to copy into the target buffer.
    iree_device_size_t length;

    // HAL copy flags captured from queue_copy.
    iree_hal_copy_flags_t flags;
  } copy;

  // Queue-ordered allocation payload.
  struct {
    // Transient buffer retained until the alloca retires.
    iree_hal_buffer_t* buffer;

    // Backing strategy selected by the allocator.
    iree_hal_vulkan_queue_alloca_strategy_t strategy;

    // Borrowed allocator used by the sparse strategy.
    iree_hal_allocator_t* allocator;

    // Borrowed pool used by the pool strategy.
    iree_hal_pool_t* pool;

    // Buffer parameters captured from queue_alloca after normalization.
    iree_hal_buffer_params_t params;

    // Physical allocation size requested from pool.
    iree_device_size_t allocation_size;

    // HAL allocation flags captured from queue_alloca.
    iree_hal_alloca_flags_t flags;

    // Pool reservation flags used when probing pool.
    iree_hal_pool_reserve_flags_t reserve_flags;

    // Active memory-readiness wait kind.
    iree_hal_vulkan_queue_alloca_memory_wait_kind_t memory_wait_kind;

    // Set to non-zero after the memory wait callback's final access completes.
    iree_atomic_int32_t memory_wait_callback_complete;

    // Pool death frontier borrowed while buffer keeps the reservation armed.
    const iree_async_frontier_t* wait_frontier;

    // Tracker waiter storage for wait_frontier.
    iree_async_frontier_waiter_t frontier_waiter;

    // Pool notification borrowed while a notification wait is active.
    iree_async_notification_t* pool_notification;

    // Notification epoch observed before retrying pool reservation.
    uint32_t pool_notification_wait_token;

    // Whether an observe scope is held until the wait operation is submitted.
    bool pool_notification_observation_held;

    // Async wait operations rotated so callbacks can arm another wait.
    iree_async_notification_wait_operation_t pool_notification_wait_ops[2];

    // Active slot in pool_notification_wait_ops.
    uint8_t pool_notification_wait_slot;
  } alloca;

  // Queue-ordered deallocation payload.
  struct {
    // Transient buffer retained until the dealloca retires.
    iree_hal_buffer_t* buffer;

    // HAL dealloca flags captured from queue_dealloca.
    iree_hal_dealloca_flags_t flags;
  } dealloca;

  // Recorded command-buffer execution payload.
  struct {
    // Command buffer retained until execution retires.
    iree_hal_command_buffer_t* command_buffer;

    // Captured binding table entries used by indirect command references.
    iree_hal_buffer_binding_t* binding_table_bindings;

    // Number of entries populated in binding_table_bindings.
    iree_host_size_t binding_table_count;

    // HAL execute flags captured from queue_execute.
    iree_hal_execute_flags_t flags;
  } execute;
};

static iree_status_t iree_hal_vulkan_queue_create_timeline_semaphore(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    uint64_t initial_value, VkSemaphore* out_semaphore) {
  *out_semaphore = VK_NULL_HANDLE;
  VkSemaphoreTypeCreateInfo timeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = initial_value,
  };
  VkSemaphoreCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timeline_create_info,
  };
  return iree_vkCreateSemaphore(IREE_VULKAN_DEVICE(syms), logical_device,
                                &create_info, /*pAllocator=*/NULL,
                                out_semaphore);
}

static iree_status_t iree_hal_vulkan_queue_create_command_pool(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    uint32_t queue_family_index, VkCommandPool* out_command_pool) {
  *out_command_pool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = queue_family_index,
  };
  return iree_vkCreateCommandPool(IREE_VULKAN_DEVICE(syms), logical_device,
                                  &create_info, /*pAllocator=*/NULL,
                                  out_command_pool);
}

static iree_status_t iree_hal_vulkan_queue_descriptor_block_create(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_descriptor_block_t** out_block) {
  *out_block = NULL;
  iree_hal_vulkan_queue_descriptor_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*block), (void**)&block));
  memset(block, 0, sizeof(*block));
  block->free_count = IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY;

  VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY,
  };
  VkDescriptorPoolCreateInfo pool_create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
  };
  iree_status_t status = iree_vkCreateDescriptorPool(
      IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device,
      &pool_create_info, /*pAllocator=*/NULL, &block->pool);

  VkDescriptorSetLayout* set_layouts = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        queue->host_allocator, IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY,
        sizeof(*set_layouts), (void**)&set_layouts);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY;
         ++i) {
      set_layouts[i] = queue->builtins->storage_buffer_descriptor_set_layout;
    }
    VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = block->pool,
        .descriptorSetCount = IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY,
        .pSetLayouts = set_layouts,
    };
    status = iree_vkAllocateDescriptorSets(IREE_VULKAN_DEVICE(&queue->syms),
                                           queue->logical_device,
                                           &allocate_info, block->sets);
  }
  iree_allocator_free(queue->host_allocator, set_layouts);
  if (iree_status_is_ok(status)) {
    *out_block = block;
  } else {
    if (block->pool) {
      iree_vkDestroyDescriptorPool(IREE_VULKAN_DEVICE(&queue->syms),
                                   queue->logical_device, block->pool,
                                   /*pAllocator=*/NULL);
    }
    iree_allocator_free(queue->host_allocator, block);
  }
  return status;
}

static void iree_hal_vulkan_queue_descriptor_block_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_descriptor_block_t* block) {
  if (!block) return;
  if (block->pool) {
    iree_vkDestroyDescriptorPool(IREE_VULKAN_DEVICE(&queue->syms),
                                 queue->logical_device, block->pool,
                                 /*pAllocator=*/NULL);
  }
  iree_allocator_free(queue->host_allocator, block);
}

static void iree_hal_vulkan_queue_descriptor_cache_append_block(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_descriptor_block_t* block) {
  block->next = NULL;
  if (queue->descriptor_cache.tail) {
    queue->descriptor_cache.tail->next = block;
  } else {
    queue->descriptor_cache.head = block;
  }
  queue->descriptor_cache.tail = block;
  if (!queue->descriptor_cache.cursor) {
    queue->descriptor_cache.cursor = block;
  }
  queue->descriptor_cache.block_count = queue->descriptor_cache.block_count + 1;
}

static iree_status_t iree_hal_vulkan_queue_descriptor_cache_initialize(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_descriptor_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_queue_descriptor_block_create(queue, &block));
  iree_hal_vulkan_queue_descriptor_cache_append_block(queue, block);
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_descriptor_cache_deinitialize(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_descriptor_block_t* block =
      queue->descriptor_cache.head;
  while (block) {
    iree_hal_vulkan_queue_descriptor_block_t* next = block->next;
    iree_hal_vulkan_queue_descriptor_block_destroy(queue, block);
    block = next;
  }
  memset(&queue->descriptor_cache, 0, sizeof(queue->descriptor_cache));
}

typedef void(IREE_API_PTR* iree_hal_vulkan_queue_staging_waiter_fn_t)(
    void* user_data);

typedef struct iree_hal_vulkan_queue_staging_waiter_t
    iree_hal_vulkan_queue_staging_waiter_t;
struct iree_hal_vulkan_queue_staging_waiter_t {
  // Next waiter in the ring-local list.
  iree_hal_vulkan_queue_staging_waiter_t* next;

  // Callback invoked after staging capacity may be available.
  iree_hal_vulkan_queue_staging_waiter_fn_t fn;

  // User data passed to |fn|.
  void* user_data;

  // Resource retained while the waiter is linked.
  iree_hal_resource_t* resource;

  // Whether this waiter is currently linked on a staging ring.
  bool is_queued;
};

typedef struct iree_hal_vulkan_queue_staging_slot_t {
  // Slot ordinal in the queue-local staging ring.
  uint32_t ordinal;

  // Byte offset of this slot within the staging buffer.
  iree_device_size_t buffer_offset;

  // Host mapped byte span for this slot.
  iree_byte_span_t host_span;

  // Whether a transfer currently owns this slot.
  bool in_use;
} iree_hal_vulkan_queue_staging_slot_t;

struct iree_hal_vulkan_queue_staging_ring_t {
  // Mutex protecting slot ownership and waiters.
  iree_slim_mutex_t mutex;

  // Host allocator used for ring metadata.
  iree_allocator_t host_allocator;

  // Queue owning this staging ring. Borrowed.
  iree_hal_vulkan_queue_t* queue;

  // Backing buffer for all slots.
  iree_hal_buffer_t* buffer;

  // Persistent host mapping of |buffer|.
  iree_hal_buffer_mapping_t mapping;

  // Byte length of each staging slot.
  iree_device_size_t slot_size;

  // Number of slots in |slots|.
  uint32_t slot_count;

  // Head of transfers waiting for capacity.
  iree_hal_vulkan_queue_staging_waiter_t* waiter_head;

  // Tail link for transfers waiting for capacity.
  iree_hal_vulkan_queue_staging_waiter_t** waiter_tail;

  // Slot table.
  iree_hal_vulkan_queue_staging_slot_t slots[];
};

static iree_status_t iree_hal_vulkan_queue_staging_ring_create(
    iree_hal_vulkan_queue_t* queue, iree_device_size_t slot_size,
    uint32_t slot_count, iree_hal_vulkan_queue_staging_ring_t** out_ring) {
  *out_ring = NULL;
  if (slot_count == 0 || slot_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan staging ring requires non-zero capacity");
  }
  if (slot_size > IREE_DEVICE_SIZE_MAX / slot_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staging ring allocation size overflows");
  }

  iree_host_size_t slots_size = 0;
  if (!iree_host_size_checked_mul(slot_count,
                                  sizeof(iree_hal_vulkan_queue_staging_slot_t),
                                  &slots_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staging ring slot table overflows");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(sizeof(**out_ring), slots_size,
                                  &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staging ring metadata overflows");
  }

  iree_hal_vulkan_queue_staging_ring_t* ring = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(queue->host_allocator, total_size, (void**)&ring));
  memset(ring, 0, total_size);
  iree_slim_mutex_initialize(&ring->mutex);
  ring->host_allocator = queue->host_allocator;
  ring->queue = queue;
  ring->slot_size = slot_size;
  ring->slot_count = slot_count;
  ring->waiter_tail = &ring->waiter_head;

  const iree_device_size_t allocation_size = slot_size * slot_count;
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
               IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE,
      .queue_affinity = queue->queue_affinity,
  };
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      queue->device_allocator, params, allocation_size, &ring->buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        ring->buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        /*byte_offset=*/0, allocation_size, &ring->mapping);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < slot_count; ++i) {
      const iree_device_size_t slot_offset = slot_size * i;
      ring->slots[i].ordinal = i;
      ring->slots[i].buffer_offset = slot_offset;
      ring->slots[i].host_span =
          iree_make_byte_span(ring->mapping.contents.data + slot_offset,
                              (iree_host_size_t)slot_size);
    }
    *out_ring = ring;
  } else {
    if (ring->mapping.contents.data) {
      iree_hal_buffer_unmap_range(&ring->mapping);
    }
    iree_hal_buffer_release(ring->buffer);
    iree_slim_mutex_deinitialize(&ring->mutex);
    iree_allocator_free(queue->host_allocator, ring);
  }
  return status;
}

static void iree_hal_vulkan_queue_staging_ring_destroy(
    iree_hal_vulkan_queue_staging_ring_t* ring) {
  if (!ring) return;
  if (ring->mapping.contents.data) {
    iree_hal_buffer_unmap_range(&ring->mapping);
  }
  iree_hal_buffer_release(ring->buffer);
  iree_slim_mutex_deinitialize(&ring->mutex);
  iree_allocator_free(ring->host_allocator, ring);
}

static bool iree_hal_vulkan_queue_staging_ring_try_acquire(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_slot_t** out_slot) {
  *out_slot = NULL;
  iree_slim_mutex_lock(&ring->mutex);
  for (uint32_t i = 0; i < ring->slot_count; ++i) {
    if (!ring->slots[i].in_use) {
      ring->slots[i].in_use = true;
      *out_slot = &ring->slots[i];
      break;
    }
  }
  iree_slim_mutex_unlock(&ring->mutex);
  return *out_slot != NULL;
}

static bool iree_hal_vulkan_queue_staging_ring_queue_waiter(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_waiter_t* waiter,
    iree_hal_vulkan_queue_staging_waiter_fn_t fn, void* user_data,
    iree_hal_resource_t* resource) {
  iree_slim_mutex_lock(&ring->mutex);
  bool should_queue = true;
  for (uint32_t i = 0; i < ring->slot_count; ++i) {
    if (!ring->slots[i].in_use) {
      should_queue = false;
      break;
    }
  }
  if (should_queue && !waiter->is_queued) {
    waiter->next = NULL;
    waiter->fn = fn;
    waiter->user_data = user_data;
    waiter->resource = resource;
    waiter->is_queued = true;
    if (waiter->resource) {
      iree_hal_resource_retain(waiter->resource);
    }
    *ring->waiter_tail = waiter;
    ring->waiter_tail = &waiter->next;
  }
  iree_slim_mutex_unlock(&ring->mutex);
  return should_queue;
}

static bool iree_hal_vulkan_queue_staging_ring_cancel_waiter(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_waiter_t* waiter) {
  bool was_queued = false;
  iree_slim_mutex_lock(&ring->mutex);
  iree_hal_vulkan_queue_staging_waiter_t** link = &ring->waiter_head;
  while (*link) {
    if (*link == waiter) {
      *link = waiter->next;
      if (!waiter->next) ring->waiter_tail = link;
      waiter->next = NULL;
      waiter->is_queued = false;
      was_queued = true;
      break;
    }
    link = &(*link)->next;
  }
  iree_slim_mutex_unlock(&ring->mutex);
  if (was_queued && waiter->resource) {
    iree_hal_resource_release(waiter->resource);
    waiter->resource = NULL;
  }
  return was_queued;
}

static void iree_hal_vulkan_queue_staging_ring_release(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_slot_t* slot) {
  iree_hal_vulkan_queue_staging_waiter_t* waiter_head = NULL;
  iree_slim_mutex_lock(&ring->mutex);
  slot->in_use = false;
  waiter_head = ring->waiter_head;
  ring->waiter_head = NULL;
  ring->waiter_tail = &ring->waiter_head;
  for (iree_hal_vulkan_queue_staging_waiter_t* waiter = waiter_head; waiter;
       waiter = waiter->next) {
    waiter->is_queued = false;
  }
  iree_slim_mutex_unlock(&ring->mutex);

  while (waiter_head) {
    iree_hal_vulkan_queue_staging_waiter_t* waiter = waiter_head;
    waiter_head = waiter->next;
    waiter->next = NULL;
    iree_hal_resource_t* resource = waiter->resource;
    waiter->resource = NULL;
    waiter->fn(waiter->user_data);
    if (resource) {
      iree_hal_resource_release(resource);
    }
  }
}

static iree_status_t iree_hal_vulkan_queue_allocate_native_command_buffer(
    iree_hal_vulkan_queue_t* queue, VkCommandBuffer* out_command_buffer) {
  *out_command_buffer = VK_NULL_HANDLE;
  if (!queue->command_pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue has no command pool");
  }

  VkCommandBufferAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = queue->command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_status_t status = iree_vkAllocateCommandBuffers(
      IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device, &allocate_info,
      out_command_buffer);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_create_timestamp_query_pool(
    iree_hal_vulkan_queue_t* queue, uint32_t query_count,
    VkQueryPool* out_query_pool) {
  *out_query_pool = VK_NULL_HANDLE;
  VkQueryPoolCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = query_count,
  };
  return iree_vkCreateQueryPool(IREE_VULKAN_DEVICE(&queue->syms),
                                queue->logical_device, &create_info,
                                /*pAllocator=*/NULL, out_query_pool);
}

static bool iree_hal_vulkan_queue_try_acquire_descriptor_cache_sets_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    uint32_t descriptor_set_count, VkDescriptorSet* out_descriptor_sets) {
  iree_hal_vulkan_queue_descriptor_lease_t
      leases[IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX];
  uint32_t lease_count = 0;

  iree_hal_vulkan_queue_descriptor_block_t* first_block =
      queue->descriptor_cache.cursor ? queue->descriptor_cache.cursor
                                     : queue->descriptor_cache.head;
  iree_hal_vulkan_queue_descriptor_block_t* block = first_block;
  while (block && lease_count < descriptor_set_count) {
    if (block->free_count != 0) {
      for (uint32_t probe_ordinal = 0;
           probe_ordinal < IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY &&
           lease_count < descriptor_set_count;
           ++probe_ordinal) {
        const uint32_t slot = (block->next_slot + probe_ordinal) %
                              IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY;
        if (block->owner_epochs[slot] == 0) {
          leases[lease_count++] = (iree_hal_vulkan_queue_descriptor_lease_t){
              .block = block,
              .slot = slot,
          };
        }
      }
    }
    block = block->next;
    if (!block && first_block != queue->descriptor_cache.head) {
      block = queue->descriptor_cache.head;
      first_block = queue->descriptor_cache.head;
    }
  }

  if (lease_count < descriptor_set_count) return false;
  for (uint32_t i = 0; i < descriptor_set_count; ++i) {
    iree_hal_vulkan_queue_descriptor_lease_t lease = leases[i];
    lease.block->owner_epochs[lease.slot] =
        IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_SLOT_RESERVED;
    lease.block->next_slot =
        (lease.slot + 1) % IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY;
    lease.block->free_count = lease.block->free_count - 1;
    submission
        ->native_descriptor_leases[submission->native_descriptor_slot_count++] =
        lease;
    out_descriptor_sets[i] = lease.block->sets[lease.slot];
    queue->descriptor_cache.cursor = lease.block;
  }
  return true;
}

static iree_status_t iree_hal_vulkan_queue_acquire_descriptor_cache_sets(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    uint32_t descriptor_set_count, VkDescriptorSet* out_descriptor_sets) {
  if (descriptor_set_count == 0) return iree_ok_status();
  if (descriptor_set_count >
      IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan queue built-in descriptor set count %u exceeds limit %u",
        descriptor_set_count,
        IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX);
  }
  if (submission->native_descriptor_slot_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue submission already owns built-in descriptor sets");
  }

  for (;;) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    bool acquired =
        iree_hal_vulkan_queue_try_acquire_descriptor_cache_sets_under_lock(
            queue, submission, descriptor_set_count, out_descriptor_sets);
    iree_slim_mutex_unlock(&queue->submission_mutex);
    if (acquired) return iree_ok_status();

    const iree_host_size_t drained_count =
        iree_hal_vulkan_queue_drain_completions(queue);
    if (drained_count != 0) continue;

    iree_hal_vulkan_queue_descriptor_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_queue_descriptor_block_create(queue, &block));
    iree_slim_mutex_lock(&queue->submission_mutex);
    iree_hal_vulkan_queue_descriptor_cache_append_block(queue, block);
    iree_slim_mutex_unlock(&queue->submission_mutex);
  }
}

static void iree_hal_vulkan_queue_publish_descriptor_cache_sets_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  (void)queue;
  for (uint32_t i = 0; i < submission->native_descriptor_slot_count; ++i) {
    iree_hal_vulkan_queue_descriptor_lease_t lease =
        submission->native_descriptor_leases[i];
    lease.block->owner_epochs[lease.slot] = submission->epoch;
  }
}

static void iree_hal_vulkan_queue_release_descriptor_cache_sets(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->native_descriptor_slot_count == 0) return;
  iree_slim_mutex_lock(&queue->submission_mutex);
  for (uint32_t i = 0; i < submission->native_descriptor_slot_count; ++i) {
    iree_hal_vulkan_queue_descriptor_lease_t* lease =
        &submission->native_descriptor_leases[i];
    if (lease->slot != IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_SLOT_ABSENT) {
      lease->block->owner_epochs[lease->slot] = 0;
      lease->block->free_count = lease->block->free_count + 1;
      queue->descriptor_cache.cursor = lease->block;
      lease->block = NULL;
      lease->slot = IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_SLOT_ABSENT;
    }
  }
  submission->native_descriptor_slot_count = 0;
  iree_slim_mutex_unlock(&queue->submission_mutex);
}

static bool iree_hal_vulkan_queue_has_pending(iree_hal_vulkan_queue_t* queue) {
  bool has_pending = false;
  iree_slim_mutex_lock(&queue->submission_mutex);
  has_pending = queue->pending_head != NULL || queue->deferred_head != NULL ||
                queue->ready_head != NULL;
  iree_slim_mutex_unlock(&queue->submission_mutex);
  return has_pending;
}

static void iree_hal_vulkan_queue_append_deferred_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  submission->next = queue->deferred_head;
  queue->deferred_head = submission;
}

static bool iree_hal_vulkan_queue_unlink_deferred_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_pending_submission_t** link = &queue->deferred_head;
  while (*link) {
    if (*link == submission) {
      *link = submission->next;
      submission->next = NULL;
      return true;
    }
    link = &(*link)->next;
  }
  return false;
}

static void iree_hal_vulkan_queue_append_ready_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  submission->next = NULL;
  if (queue->ready_tail) {
    queue->ready_tail->next = submission;
  } else {
    queue->ready_head = submission;
  }
  queue->ready_tail = submission;
}

static bool iree_hal_vulkan_queue_unlink_ready_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_pending_submission_t* previous = NULL;
  iree_hal_vulkan_queue_pending_submission_t** link = &queue->ready_head;
  while (*link) {
    if (*link == submission) {
      *link = submission->next;
      if (queue->ready_tail == submission) queue->ready_tail = previous;
      submission->next = NULL;
      return true;
    }
    previous = *link;
    link = &(*link)->next;
  }
  return false;
}

static iree_hal_vulkan_queue_pending_submission_t*
iree_hal_vulkan_queue_pop_ready_submission(iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_pending_submission_t* submission = queue->ready_head;
  if (!submission) return NULL;
  queue->ready_head = submission->next;
  if (!queue->ready_head) queue->ready_tail = NULL;
  submission->next = NULL;
  return submission;
}

static iree_status_t iree_hal_vulkan_queue_store_error(
    iree_hal_vulkan_queue_t* queue, iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status),
              "queue failure status must be non-OK");
  intptr_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &queue->failure_status, &expected, (intptr_t)status,
          iree_memory_order_release, iree_memory_order_acquire)) {
    return iree_status_clone(status);
  }
  iree_status_free(status);
  return iree_status_clone((iree_status_t)expected);
}

static iree_status_t iree_hal_vulkan_queue_check_error(
    iree_hal_vulkan_queue_t* queue) {
  iree_status_t status = (iree_status_t)iree_atomic_load(
      &queue->failure_status, iree_memory_order_acquire);
  return iree_status_is_ok(status) ? iree_ok_status()
                                   : iree_status_clone(status);
}

static iree_status_t iree_hal_vulkan_queue_signal_wakeup(
    iree_hal_vulkan_queue_t* queue) {
  const uint64_t new_value =
      (uint64_t)iree_atomic_fetch_add(&queue->wakeup_value, 1,
                                      iree_memory_order_acq_rel) +
      1;
  VkSemaphoreSignalInfo signal_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = queue->wakeup_semaphore,
      .value = new_value,
  };
  return iree_vkSignalSemaphore(IREE_VULKAN_DEVICE(&queue->syms),
                                queue->logical_device, &signal_info);
}

static iree_status_t iree_hal_vulkan_queue_validate_host_call(
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  const iree_hal_host_call_flags_t known_flags =
      IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
      IREE_HAL_HOST_CALL_FLAG_WAIT_ACTIVE | IREE_HAL_HOST_CALL_FLAG_RELAXED;
  if (!call.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host_call callback must be non-null");
  }
  if (!args) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host_call args must be non-null");
  }
  if (iree_any_bit_set(flags, ~known_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported host_call flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_validate_semaphore_list(
    iree_hal_vulkan_queue_t* queue, iree_hal_semaphore_list_t semaphore_list,
    iree_string_view_t usage) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_semaphore_t* semaphore = semaphore_list.semaphores[i];
    if (!iree_hal_vulkan_semaphore_is_local(semaphore, queue->device)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Vulkan queue %.*s semaphore %" PRIhsz
                              " is not a local Vulkan semaphore",
                              (int)usage.size, usage.data, i);
    }
    if (semaphore_list.payload_values[i] > IREE_HAL_SEMAPHORE_MAX_VALUE) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan queue %.*s semaphore value %" PRIu64
          " exceeds the maximum HAL semaphore value %" PRIu64,
          (int)usage.size, usage.data, semaphore_list.payload_values[i],
          (uint64_t)IREE_HAL_SEMAPHORE_MAX_VALUE);
    }
  }
  return iree_ok_status();
}

static uint32_t iree_hal_vulkan_queue_profile_count(iree_host_size_t value) {
  return value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
}

static iree_hal_profile_queue_event_type_t iree_hal_vulkan_queue_profile_type(
    iree_hal_vulkan_queue_submission_kind_t kind) {
  switch (kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER:
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA;
  }
  return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
}

static uint32_t iree_hal_vulkan_queue_profile_operation_count(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER:
      return 0;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND:
      return submission->sparse_bind.bind_count;
    default:
      return 1;
  }
}

static uint64_t iree_hal_vulkan_queue_profile_payload_length(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL:
      return submission->fill.length;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE:
      return submission->update.length;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY:
      return submission->copy.length;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA:
      return submission->alloca.allocation_size;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND: {
      uint64_t total_length = 0;
      for (uint32_t i = 0; i < submission->sparse_bind.bind_count; ++i) {
        const VkDeviceSize bind_size = submission->sparse_bind.binds[i].size;
        if (total_length > UINT64_MAX - bind_size) return UINT64_MAX;
        total_length += bind_size;
      }
      return total_length;
    }
    default:
      return 0;
  }
}

static void iree_hal_vulkan_queue_profile_submission_initialize(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_local_profile_recorder_t* recorder = queue->profile_recorder;
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                        IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                        IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
                        IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS)) {
    return;
  }
  submission->profile.recorder = recorder;
  submission->profile.scope = queue->profile_scope;
  submission->profile.type =
      iree_hal_vulkan_queue_profile_type(submission->kind);
  submission->profile.submit_host_time_ns = iree_time_now();
  if (queue->profile_submission_counter) {
    submission->profile.submission_id = (uint64_t)iree_atomic_fetch_add(
        queue->profile_submission_counter, 1, iree_memory_order_relaxed);
  }
}

static void iree_hal_vulkan_queue_profile_set_dependency_resolution(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_vulkan_queue_wait_resolution_t* resolution) {
  if (!submission->profile.recorder) return;
  if (submission->profile.dependency_strategy ==
      IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER) {
    return;
  }
  if (submission->wait_semaphore_list.count == 0) {
    submission->profile.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE;
  } else if (resolution->needs_deferral) {
    submission->profile.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
    submission->profile.flags |=
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
  } else if (resolution->wait_info_count != 0) {
    submission->profile.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER;
  } else {
    submission->profile.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE;
  }
}

static void iree_hal_vulkan_queue_profile_force_software_defer(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.recorder) return;
  submission->profile.dependency_strategy =
      IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
  submission->profile.flags |=
      IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
}

static uint64_t iree_hal_vulkan_queue_profile_allocation_id(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA:
      return iree_hal_local_transient_buffer_profile_id(
          submission->alloca.buffer);
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA:
      return iree_hal_local_transient_buffer_profile_id(
          submission->dealloca.buffer);
    default:
      return 0;
  }
}

static uint64_t iree_hal_vulkan_queue_profile_command_buffer_id(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->kind != IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE ||
      !submission->execute.command_buffer ||
      submission->profile.type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH) {
    return 0;
  }
  return submission->execute.command_buffer->profile_id;
}

static void iree_hal_vulkan_queue_profile_populate_submission_metrics(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.recorder) return;
  if (submission->profile.operation_count == 0) {
    submission->profile.operation_count =
        iree_hal_vulkan_queue_profile_operation_count(submission);
  }
  if (submission->profile.payload_length == 0) {
    submission->profile.payload_length =
        iree_hal_vulkan_queue_profile_payload_length(submission);
  }
}

static void iree_hal_vulkan_queue_profile_record_submission(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.recorder ||
      submission->profile.queue_event_recorded) {
    return;
  }
  if (!iree_hal_local_profile_recorder_is_enabled(
          submission->profile.recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    return;
  }
  submission->profile.queue_event_recorded = true;
  submission->profile.ready_host_time_ns = iree_time_now();
  iree_hal_vulkan_queue_profile_populate_submission_metrics(submission);

  iree_hal_local_profile_queue_event_info_t event_info =
      iree_hal_local_profile_queue_event_info_default();
  event_info.type = submission->profile.type;
  event_info.flags = submission->profile.flags;
  event_info.dependency_strategy = submission->profile.dependency_strategy;
  event_info.scope = submission->profile.scope;
  event_info.host_time_ns = submission->profile.submit_host_time_ns;
  event_info.ready_host_time_ns = submission->profile.ready_host_time_ns;
  event_info.submission_id = submission->profile.submission_id;
  event_info.command_buffer_id =
      iree_hal_vulkan_queue_profile_command_buffer_id(submission);
  event_info.allocation_id =
      iree_hal_vulkan_queue_profile_allocation_id(submission);
  event_info.wait_count = iree_hal_vulkan_queue_profile_count(
      submission->wait_semaphore_list.count);
  event_info.signal_count = iree_hal_vulkan_queue_profile_count(
      submission->signal_semaphore_list.count);
  event_info.operation_count = submission->profile.operation_count;
  event_info.payload_length = submission->profile.payload_length;
  iree_hal_local_profile_recorder_append_queue_event(
      submission->profile.recorder, &event_info, /*out_event_id=*/NULL);
}

static bool iree_hal_vulkan_queue_profile_requests_queue_device_event(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  return iree_hal_local_profile_recorder_is_enabled(
      submission->profile.recorder,
      IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS);
}

static bool iree_hal_vulkan_queue_profile_requests_dispatch_events(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  return iree_hal_local_profile_recorder_is_enabled(
      submission->profile.recorder,
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS);
}

static bool iree_hal_vulkan_queue_profile_submission_requires_native_timestamp(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL:
      return submission->fill.length != 0;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE:
      return submission->update.length != 0;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY:
      return submission->copy.length != 0;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE:
      return true;
    default:
      return false;
  }
}

static iree_status_t iree_hal_vulkan_queue_profile_dispatch_count(
    const iree_hal_vulkan_queue_pending_submission_t* submission,
    uint32_t* out_dispatch_count) {
  *out_dispatch_count = 0;
  if (!iree_hal_vulkan_queue_profile_requests_dispatch_events(submission)) {
    return iree_ok_status();
  }
  if (submission->kind != IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE ||
      !submission->execute.command_buffer) {
    return iree_ok_status();
  }
  const iree_host_size_t dispatch_count =
      iree_hal_vulkan_command_buffer_dispatch_count(
          submission->execute.command_buffer);
  if (dispatch_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan dispatch profile dispatch count exceeds uint32_t");
  }
  *out_dispatch_count = (uint32_t)dispatch_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_profile_prepare_native_timestamps(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->native_command_buffer) {
    if (!iree_hal_vulkan_queue_profile_submission_requires_native_timestamp(
            submission)) {
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan device timestamp profiling requires native command recording "
        "for queue event type %u",
        (uint32_t)submission->profile.type);
  }
  if (submission->profile.query_pool) return iree_ok_status();

  const bool needs_queue_device_timestamps =
      iree_hal_vulkan_queue_profile_requests_queue_device_event(submission) &&
      iree_hal_vulkan_queue_profile_submission_requires_native_timestamp(
          submission);
  uint32_t dispatch_query_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_profile_dispatch_count(
      submission, &dispatch_query_count));
  if (!needs_queue_device_timestamps && dispatch_query_count == 0) {
    return iree_ok_status();
  }

  uint32_t query_count = 0;
  if (needs_queue_device_timestamps) {
    submission->profile.queue_start_query = query_count++;
    submission->profile.queue_end_query = query_count++;
  }
  if (dispatch_query_count != 0) {
    if (dispatch_query_count > (UINT32_MAX - query_count) / 2) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan dispatch profile query count exceeds uint32_t");
    }
    submission->profile.dispatch_base_query = query_count;
    submission->profile.dispatch_query_count = dispatch_query_count;
    query_count += dispatch_query_count * 2;
  }

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_create_timestamp_query_pool(
      queue, query_count, &submission->profile.query_pool));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      queue->host_allocator, query_count, sizeof(uint64_t),
      (void**)&submission->profile.query_values));
  submission->profile.query_count = query_count;
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_profile_write_timestamp_begin(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.query_pool ||
      submission->profile.queue_start_query ==
          IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    return;
  }
  iree_vkCmdResetQueryPool(IREE_VULKAN_DEVICE(&queue->syms),
                           submission->native_command_buffer,
                           submission->profile.query_pool, /*firstQuery=*/0,
                           submission->profile.query_count);
  iree_vkCmdWriteTimestamp2(
      IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, submission->profile.query_pool,
      submission->profile.queue_start_query);
}

static void iree_hal_vulkan_queue_profile_write_timestamp_end(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.query_pool ||
      submission->profile.queue_end_query ==
          IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    return;
  }
  iree_vkCmdWriteTimestamp2(
      IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, submission->profile.query_pool,
      submission->profile.queue_end_query);
}

static iree_status_t iree_hal_vulkan_queue_profile_read_native_timestamps(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.query_pool) {
    return iree_ok_status();
  }
  VkResult result = iree_vkGetQueryPoolResults_raw(
      &queue->syms, queue->logical_device, submission->profile.query_pool,
      /*firstQuery=*/0, submission->profile.query_count,
      submission->profile.query_count * sizeof(uint64_t),
      submission->profile.query_values, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT);
  if (result != VK_SUCCESS) {
    return iree_status_from_vk_result(__FILE__, __LINE__, result,
                                      "vkGetQueryPoolResults");
  }
  if (submission->profile.queue_start_query !=
      IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    const uint64_t start_tick =
        submission->profile.query_values[submission->profile.queue_start_query];
    const uint64_t end_tick =
        submission->profile.query_values[submission->profile.queue_end_query];
    if (end_tick < start_tick) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "Vulkan queue device profiling timestamp range is not monotonic");
    }
  }
  for (uint32_t i = 0; i < submission->profile.dispatch_query_count; ++i) {
    const uint32_t query_index =
        submission->profile.dispatch_base_query + i * 2;
    const uint64_t start_tick = submission->profile.query_values[query_index];
    const uint64_t end_tick = submission->profile.query_values[query_index + 1];
    if (end_tick < start_tick) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "Vulkan dispatch profiling timestamp range is not monotonic");
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_profile_record_dispatch_events(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!iree_hal_vulkan_queue_profile_requests_dispatch_events(submission) ||
      submission->profile.dispatch_query_count == 0) {
    return iree_ok_status();
  }
  if (submission->profile.dispatch_base_query ==
      IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan dispatch profiling has no dispatch timestamp query range");
  }
  return iree_hal_vulkan_command_buffer_append_dispatch_profile_events(
      submission->execute.command_buffer, submission->profile.recorder,
      submission->profile.scope, submission->profile.submission_id,
      iree_hal_vulkan_queue_profile_command_buffer_id(submission),
      &submission->profile
           .query_values[submission->profile.dispatch_base_query],
      submission->profile.dispatch_query_count);
}

static iree_status_t iree_hal_vulkan_queue_profile_record_queue_device_event(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!iree_hal_vulkan_queue_profile_requests_queue_device_event(submission) ||
      submission->profile.queue_device_event_recorded) {
    return iree_ok_status();
  }
  if (submission->profile.queue_start_query ==
      IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    return iree_ok_status();
  }
  submission->profile.queue_device_event_recorded = true;

  iree_hal_vulkan_queue_profile_populate_submission_metrics(submission);
  iree_hal_local_profile_queue_device_event_info_t event_info =
      iree_hal_local_profile_queue_device_event_info_default();
  event_info.type = submission->profile.type;
  event_info.flags = submission->profile.flags;
  event_info.scope = submission->profile.scope;
  event_info.submission_id = submission->profile.submission_id;
  event_info.command_buffer_id =
      iree_hal_vulkan_queue_profile_command_buffer_id(submission);
  event_info.allocation_id =
      iree_hal_vulkan_queue_profile_allocation_id(submission);
  event_info.operation_count = submission->profile.operation_count;
  event_info.payload_length = submission->profile.payload_length;
  event_info.start_tick =
      submission->profile.query_values[submission->profile.queue_start_query];
  event_info.end_tick =
      submission->profile.query_values[submission->profile.queue_end_query];
  return iree_hal_local_profile_recorder_append_queue_device_event(
      submission->profile.recorder, &event_info, /*out_event_id=*/NULL);
}

static void iree_hal_vulkan_queue_profile_populate_memory_event_pool_stats(
    iree_hal_pool_t* pool, iree_hal_profile_memory_event_t* event) {
  if (!pool) return;
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  event->flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS;
  event->pool_bytes_reserved = stats.bytes_reserved;
  event->pool_bytes_free = stats.bytes_free;
  event->pool_bytes_committed = stats.bytes_committed;
  event->pool_budget_limit = stats.budget_limit;
  event->pool_reservation_count = stats.reservation_count;
  event->pool_slab_count = stats.slab_count;
}

static void iree_hal_vulkan_queue_profile_record_memory_event(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_hal_profile_memory_event_type_t type,
    iree_hal_profile_memory_event_flags_t flags, uint32_t result,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation, uint64_t backing_id,
    iree_device_size_t length, uint64_t frontier_entry_count) {
  if (!iree_hal_local_profile_recorder_is_enabled(
          submission->profile.recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS)) {
    return;
  }

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = type;
  event.flags = flags;
  event.result = result;
  event.allocation_id = iree_hal_vulkan_queue_profile_allocation_id(submission);
  event.pool_id = (uint64_t)(uintptr_t)pool;
  event.backing_id = backing_id;
  event.submission_id = submission->profile.submission_id;
  event.physical_device_ordinal =
      submission->profile.scope.physical_device_ordinal;
  event.queue_ordinal = submission->profile.scope.queue_ordinal;
  event.frontier_entry_count =
      (uint32_t)iree_min(frontier_entry_count, (uint64_t)UINT32_MAX);
  event.memory_type = params.type;
  event.buffer_usage = params.usage;
  event.length = length;
  event.alignment = params.min_alignment ? params.min_alignment : 1;
  if (reservation) {
    event.flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION;
    event.backing_id = reservation->block_handle;
    event.offset = reservation->offset;
    if (type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA &&
        type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA) {
      event.length = reservation->length;
    }
  }
  iree_hal_vulkan_queue_profile_populate_memory_event_pool_stats(pool, &event);
  iree_hal_local_profile_recorder_append_memory_event(
      submission->profile.recorder, &event, /*out_event_id=*/NULL);
}

static iree_hal_profile_memory_event_flags_t
iree_hal_vulkan_queue_profile_pool_reserve_event_flags(
    iree_hal_pool_acquire_result_t acquire_result) {
  iree_hal_profile_memory_event_flags_t flags =
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
  if (acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
    flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER;
  } else if (acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
             acquire_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
    flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION;
  }
  return flags;
}

static void iree_hal_vulkan_queue_set_completion_action(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_hal_vulkan_queue_completion_action_t action) {
  submission->completion_action = action;
  if (submission->completion_action.resource) {
    iree_hal_resource_retain(submission->completion_action.resource);
  }
}

static void iree_hal_vulkan_queue_release_completion_action(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->completion_action.resource) {
    iree_hal_resource_release(submission->completion_action.resource);
  }
  submission->completion_action =
      iree_hal_vulkan_queue_completion_action_null();
}

static void iree_hal_vulkan_queue_consume_completion_action(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_vulkan_queue_completion_action_t action =
      submission->completion_action;
  submission->completion_action =
      iree_hal_vulkan_queue_completion_action_null();
  if (action.fn) {
    action.fn(action.user_data, completion_status);
  }
  if (action.resource) {
    iree_hal_resource_release(action.resource);
  }
}

static iree_status_t iree_hal_vulkan_queue_pending_submission_create(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_queue_submission_kind_t kind, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags,
    iree_hal_vulkan_queue_pending_submission_t** out_submission) {
  *out_submission = NULL;
  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, sizeof(*submission), (void**)&submission));
  memset(submission, 0, sizeof(*submission));
  submission->queue = queue;
  submission->kind = kind;
  for (uint32_t i = 0;
       i < IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX; ++i) {
    submission->native_descriptor_leases[i].slot =
        IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_SLOT_ABSENT;
  }
  submission->profile.queue_start_query = IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT;
  submission->profile.queue_end_query = IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT;
  submission->profile.dispatch_base_query =
      IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT;
  iree_notification_initialize(&submission->callback_notification);
  iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 1,
                    iree_memory_order_relaxed);
  if (kind == IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL) {
    submission->host_call.call = call;
    memcpy(submission->host_call.args, args,
           sizeof(submission->host_call.args));
    submission->host_call.flags = flags;
  }
  iree_hal_vulkan_queue_profile_submission_initialize(queue, submission);

  iree_status_t status =
      iree_hal_semaphore_list_clone(&wait_semaphore_list, queue->host_allocator,
                                    &submission->wait_semaphore_list);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(&signal_semaphore_list,
                                           queue->host_allocator,
                                           &submission->signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    if (submission->wait_semaphore_list.semaphores) {
      iree_hal_semaphore_list_free(submission->wait_semaphore_list,
                                   queue->host_allocator);
    }
    iree_notification_deinitialize(&submission->callback_notification);
    iree_allocator_free(queue->host_allocator, submission);
  }
  return status;
}

static void iree_hal_vulkan_queue_pending_submission_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->wait_semaphore_list.semaphores) {
    iree_hal_semaphore_list_free(submission->wait_semaphore_list,
                                 queue->host_allocator);
  }
  if (submission->signal_semaphore_list.semaphores) {
    iree_hal_semaphore_list_free(submission->signal_semaphore_list,
                                 queue->host_allocator);
  }
  iree_status_t wait_failure_status = (iree_status_t)iree_atomic_exchange(
      &submission->wait_failure_status, 0, iree_memory_order_acquire);
  iree_status_free(wait_failure_status);
  if (submission->wait_entries) {
    iree_allocator_free(queue->host_allocator, submission->wait_entries);
  }
  if (submission->sparse_bind.binds) {
    iree_allocator_free(queue->host_allocator, submission->sparse_bind.binds);
  }
  iree_hal_vulkan_queue_release_descriptor_cache_sets(queue, submission);
  if (submission->fill.target_buffer) {
    iree_hal_buffer_release(submission->fill.target_buffer);
  }
  if (submission->update.target_buffer) {
    iree_hal_buffer_release(submission->update.target_buffer);
  }
  if (submission->update.source_data) {
    iree_allocator_free(queue->host_allocator, submission->update.source_data);
  }
  if (submission->copy.source_buffer) {
    iree_hal_buffer_release(submission->copy.source_buffer);
  }
  if (submission->copy.target_buffer) {
    iree_hal_buffer_release(submission->copy.target_buffer);
  }
  if (submission->alloca.buffer) {
    if (submission->alloca.pool_notification_observation_held) {
      submission->alloca.pool_notification_observation_held = false;
      iree_async_notification_end_observe(submission->alloca.pool_notification);
    }
    iree_hal_buffer_release(submission->alloca.buffer);
  }
  if (submission->dealloca.buffer) {
    iree_hal_buffer_release(submission->dealloca.buffer);
  }
  if (submission->native_descriptor_pool) {
    iree_vkDestroyDescriptorPool(IREE_VULKAN_DEVICE(&queue->syms),
                                 queue->logical_device,
                                 submission->native_descriptor_pool,
                                 /*pAllocator=*/NULL);
  }
  iree_hal_vulkan_queue_release_completion_action(submission);
  if (submission->profile.query_pool) {
    iree_vkDestroyQueryPool(IREE_VULKAN_DEVICE(&queue->syms),
                            queue->logical_device,
                            submission->profile.query_pool,
                            /*pAllocator=*/NULL);
  }
  if (submission->profile.query_values) {
    iree_allocator_free(queue->host_allocator,
                        submission->profile.query_values);
  }
  if (submission->native_command_buffer) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    iree_vkFreeCommandBuffers(IREE_VULKAN_DEVICE(&queue->syms),
                              queue->logical_device, queue->command_pool,
                              /*commandBufferCount=*/1,
                              &submission->native_command_buffer);
    iree_slim_mutex_unlock(&queue->submission_mutex);
  }
  if (submission->execute.command_buffer) {
    iree_hal_command_buffer_release(submission->execute.command_buffer);
  }
  if (submission->execute.binding_table_bindings) {
    if (!iree_any_bit_set(
            submission->execute.flags,
            IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
      for (iree_host_size_t i = 0; i < submission->execute.binding_table_count;
           ++i) {
        iree_hal_buffer_release(
            submission->execute.binding_table_bindings[i].buffer);
      }
    }
    iree_allocator_free(queue->host_allocator,
                        submission->execute.binding_table_bindings);
  }
  iree_notification_deinitialize(&submission->callback_notification);
  iree_allocator_free(queue->host_allocator, submission);
}

static void iree_hal_vulkan_queue_append_pending_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  submission->next = NULL;
  if (queue->pending_tail) {
    queue->pending_tail->next = submission;
  } else {
    queue->pending_head = submission;
  }
  queue->pending_tail = submission;
}

static iree_hal_vulkan_queue_pending_submission_t*
iree_hal_vulkan_queue_pop_completed_submissions(iree_hal_vulkan_queue_t* queue,
                                                uint64_t completed_epoch) {
  iree_hal_vulkan_queue_pending_submission_t* completed_head =
      queue->pending_head;
  iree_hal_vulkan_queue_pending_submission_t* completed_tail = NULL;
  while (queue->pending_head && queue->pending_head->epoch <= completed_epoch) {
    completed_tail = queue->pending_head;
    queue->pending_head = queue->pending_head->next;
  }
  if (!completed_tail) return NULL;

  if (!queue->pending_head) queue->pending_tail = NULL;
  completed_tail->next = NULL;
  return completed_head;
}

static void iree_hal_vulkan_queue_fail_signal_list(
    iree_hal_semaphore_list_t signal_semaphore_list, iree_status_t status) {
  if (signal_semaphore_list.count == 0) {
    iree_status_free(status);
    return;
  }
  iree_hal_semaphore_list_fail(signal_semaphore_list, status);
}

static void iree_hal_vulkan_queue_signal_list_or_fail(
    iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* frontier) {
  iree_status_t status =
      iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, status);
  }
}

static void iree_hal_vulkan_queue_execute_host_call(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);

  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
    return;
  }

  const bool is_nonblocking = iree_any_bit_set(
      submission->host_call.flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (is_nonblocking) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  }

  iree_hal_host_call_context_t context = {
      .device = (iree_hal_device_t*)queue->device,
      .queue_affinity = queue->queue_affinity,
      .signal_semaphore_list = is_nonblocking ? iree_hal_semaphore_list_empty()
                                              : signal_semaphore_list,
  };
  iree_status_t call_status =
      submission->host_call.call.fn(submission->host_call.call.user_data,
                                    submission->host_call.args, &context);

  if (is_nonblocking || iree_status_is_deferred(call_status)) {
    iree_status_free(call_status);
  } else if (iree_status_is_ok(call_status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, call_status);
  }
}

static bool iree_hal_vulkan_queue_buffer_has_recordable_backing(
    iree_hal_buffer_t* buffer) {
  return !iree_hal_local_transient_buffer_isa(buffer) ||
         iree_hal_local_transient_buffer_backing_buffer(buffer) != NULL;
}

static iree_status_t iree_hal_vulkan_queue_validate_recordable_backing(
    iree_hal_buffer_t* buffer, iree_string_view_t usage) {
  if (!iree_hal_local_transient_buffer_isa(buffer)) return iree_ok_status();
  if (iree_hal_local_transient_buffer_backing_buffer(buffer)) {
    return iree_ok_status();
  }
  if (iree_hal_local_transient_buffer_is_dealloca_queued(buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue %.*s buffer has been queued for deallocation",
        (int)usage.size, usage.data);
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "Vulkan queue %.*s buffer has no recordable device backing",
      (int)usage.size, usage.data);
}

static iree_status_t iree_hal_vulkan_queue_buffer_is_native(
    iree_hal_buffer_t* buffer, bool* out_is_native) {
  *out_is_native = false;
  if (!iree_hal_vulkan_queue_buffer_has_recordable_backing(buffer)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(buffer, &backing_buffer));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(backing_buffer);
  *out_is_native = iree_hal_vulkan_buffer_isa(allocated_buffer) ||
                   iree_hal_vulkan_sparse_buffer_isa(allocated_buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_expand_fill_pattern(
    const uint8_t* pattern, iree_host_size_t pattern_length,
    uint32_t* out_expanded_pattern) {
  *out_expanded_pattern = 0;
  switch (pattern_length) {
    case sizeof(uint8_t):
      *out_expanded_pattern = pattern[0];
      *out_expanded_pattern |= *out_expanded_pattern << 8;
      *out_expanded_pattern |= *out_expanded_pattern << 16;
      break;
    case sizeof(uint16_t): {
      uint16_t pattern16 = 0;
      memcpy(&pattern16, pattern, sizeof(pattern16));
      *out_expanded_pattern = pattern16;
      *out_expanded_pattern |= *out_expanded_pattern << 16;
      break;
    }
    case sizeof(uint32_t):
      memcpy(out_expanded_pattern, pattern, sizeof(*out_expanded_pattern));
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue fill pattern length must be 1, 2, "
                              "or 4 bytes (got %" PRIhsz ")",
                              pattern_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_record_fill_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(submission->fill.target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  iree_hal_buffer_t* target_backing = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing(
      submission->fill.target_buffer, &target_backing));
  VkDeviceMemory target_memory = VK_NULL_HANDLE;
  VkBuffer target_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_handle(
      target_backing, &target_memory, &target_handle));
  (void)target_memory;

  iree_device_size_t target_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      submission->fill.target_buffer, target_backing,
      submission->fill.target_offset, &target_offset));

  uint32_t pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_expand_fill_pattern(
      submission->fill.pattern, submission->fill.pattern_length, &pattern));
  const bool needs_unaligned_fill =
      target_offset % sizeof(uint32_t) != 0 ||
      submission->fill.length % sizeof(uint32_t) != 0;
  if (needs_unaligned_fill && !queue->builtins) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan unaligned native queue fill requires built-in pipelines");
  }
  if (needs_unaligned_fill &&
      submission->fill.length >
          IREE_DEVICE_SIZE_MAX - (iree_device_size_t)target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan native queue fill range overflows");
  }
  VkDescriptorSet
      descriptor_sets[IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX] =
          {
              VK_NULL_HANDLE,
              VK_NULL_HANDLE,
          };
  uint32_t descriptor_set_count = 0;
  if (needs_unaligned_fill) {
    descriptor_set_count =
        iree_hal_vulkan_builtins_fill_unaligned_descriptor_set_count(
            (VkDeviceSize)target_offset, (VkDeviceSize)submission->fill.length);
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_acquire_descriptor_cache_sets(
        queue, submission, descriptor_set_count, descriptor_sets));
  }

  VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  IREE_RETURN_IF_ERROR(iree_vkBeginCommandBuffer(
      IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
      &begin_info));
  iree_hal_vulkan_queue_profile_write_timestamp_begin(queue, submission);
  VkDeviceSize fill_offset = (VkDeviceSize)target_offset;
  VkDeviceSize fill_length = (VkDeviceSize)submission->fill.length;
  if (needs_unaligned_fill) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_builtins_record_fill_unaligned_descriptor_sets(
            queue->builtins, submission->native_command_buffer, descriptor_sets,
            descriptor_set_count, target_handle, (VkDeviceSize)target_offset,
            (VkDeviceSize)submission->fill.length, submission->fill.pattern,
            submission->fill.pattern_length));
    const VkDeviceSize target_end =
        (VkDeviceSize)target_offset + (VkDeviceSize)submission->fill.length;
    const VkDeviceSize aligned_target_offset =
        iree_device_align(target_offset, sizeof(uint32_t));
    const VkDeviceSize aligned_target_end =
        target_end & ~(VkDeviceSize)(sizeof(uint32_t) - 1);
    if (aligned_target_offset >= aligned_target_end) {
      fill_length = 0;
    } else {
      fill_offset = aligned_target_offset;
      fill_length = aligned_target_end - aligned_target_offset;
    }
  }
  if (fill_length != 0) {
    iree_vkCmdFillBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                         submission->native_command_buffer, target_handle,
                         fill_offset, fill_length, pattern);
  }
  iree_hal_vulkan_queue_profile_write_timestamp_end(queue, submission);
  return iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                 submission->native_command_buffer);
}

static void iree_hal_vulkan_queue_execute_fill(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
  }
}

static iree_status_t iree_hal_vulkan_queue_record_update_chunks(
    iree_hal_vulkan_queue_t* queue, VkCommandBuffer native_command_buffer,
    VkBuffer target_handle, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* source_data, iree_host_size_t source_data_offset) {
  if (length == 0) return iree_ok_status();
  if (length > (VkDeviceSize)(IREE_HOST_SIZE_MAX - source_data_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan native queue update source offset "
                            "overflows");
  }

  const VkDeviceSize max_update_length = 65536;
  VkDeviceSize remaining_length = length;
  VkDeviceSize update_offset = target_offset;
  iree_host_size_t update_source_offset = source_data_offset;
  while (remaining_length != 0) {
    const VkDeviceSize chunk_length =
        iree_min(remaining_length, max_update_length);
    iree_vkCmdUpdateBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                           native_command_buffer, target_handle, update_offset,
                           chunk_length, source_data + update_source_offset);
    update_offset += chunk_length;
    update_source_offset += (iree_host_size_t)chunk_length;
    remaining_length -= chunk_length;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_record_update_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(submission->update.target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  iree_hal_buffer_t* target_backing = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing(
      submission->update.target_buffer, &target_backing));
  VkDeviceMemory target_memory = VK_NULL_HANDLE;
  VkBuffer target_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_handle(
      target_backing, &target_memory, &target_handle));
  (void)target_memory;

  iree_device_size_t target_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      submission->update.target_buffer, target_backing,
      submission->update.target_offset, &target_offset));

  const bool needs_unaligned_update =
      target_offset % sizeof(uint32_t) != 0 ||
      submission->update.length % sizeof(uint32_t) != 0;
  if (needs_unaligned_update && !queue->builtins) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan unaligned native queue update requires built-in pipelines");
  }
  if (needs_unaligned_update &&
      submission->update.length >
          IREE_DEVICE_SIZE_MAX - (iree_device_size_t)target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan native queue update range overflows");
  }
  VkDescriptorSet
      descriptor_sets[IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX] =
          {
              VK_NULL_HANDLE,
              VK_NULL_HANDLE,
          };
  uint32_t descriptor_set_count = 0;
  if (needs_unaligned_update) {
    descriptor_set_count =
        iree_hal_vulkan_builtins_update_unaligned_descriptor_set_count(
            (VkDeviceSize)target_offset,
            (VkDeviceSize)submission->update.length);
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_acquire_descriptor_cache_sets(
        queue, submission, descriptor_set_count, descriptor_sets));
  }

  VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  IREE_RETURN_IF_ERROR(iree_vkBeginCommandBuffer(
      IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
      &begin_info));
  iree_hal_vulkan_queue_profile_write_timestamp_begin(queue, submission);
  VkDeviceSize update_offset = (VkDeviceSize)target_offset;
  VkDeviceSize update_length = (VkDeviceSize)submission->update.length;
  iree_host_size_t source_data_offset = 0;
  if (needs_unaligned_update) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_builtins_record_update_unaligned_descriptor_sets(
            queue->builtins, submission->native_command_buffer, descriptor_sets,
            descriptor_set_count, target_handle, (VkDeviceSize)target_offset,
            (VkDeviceSize)submission->update.length,
            submission->update.source_data,
            (iree_host_size_t)submission->update.length));
    const VkDeviceSize target_end =
        (VkDeviceSize)target_offset + (VkDeviceSize)submission->update.length;
    const VkDeviceSize aligned_target_offset =
        iree_device_align(target_offset, sizeof(uint32_t));
    const VkDeviceSize aligned_target_end =
        target_end & ~(VkDeviceSize)(sizeof(uint32_t) - 1);
    if (aligned_target_offset >= aligned_target_end) {
      update_length = 0;
    } else {
      update_offset = aligned_target_offset;
      update_length = aligned_target_end - aligned_target_offset;
      source_data_offset =
          (iree_host_size_t)(aligned_target_offset - target_offset);
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_record_update_chunks(
      queue, submission->native_command_buffer, target_handle, update_offset,
      update_length, submission->update.source_data, source_data_offset));
  iree_hal_vulkan_queue_profile_write_timestamp_end(queue, submission);
  return iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                 submission->native_command_buffer);
}

static void iree_hal_vulkan_queue_execute_update(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
  }
}

static iree_status_t iree_hal_vulkan_queue_record_copy_native_buffers(
    iree_hal_vulkan_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    VkCommandBuffer command_buffer,
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  iree_hal_buffer_t* source_backing = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(source_buffer, &source_backing));
  iree_hal_buffer_t* target_backing = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(target_buffer, &target_backing));
  VkDeviceMemory source_memory = VK_NULL_HANDLE;
  VkBuffer source_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_handle(
      source_backing, &source_memory, &source_handle));
  (void)source_memory;
  VkDeviceMemory target_memory = VK_NULL_HANDLE;
  VkBuffer target_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_handle(
      target_backing, &target_memory, &target_handle));
  (void)target_memory;

  iree_device_size_t source_backing_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      source_buffer, source_backing, source_offset, &source_backing_offset));
  iree_device_size_t target_backing_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      target_buffer, target_backing, target_offset, &target_backing_offset));

  VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  IREE_RETURN_IF_ERROR(iree_vkBeginCommandBuffer(
      IREE_VULKAN_DEVICE(&queue->syms), command_buffer, &begin_info));
  if (submission) {
    iree_hal_vulkan_queue_profile_write_timestamp_begin(queue, submission);
  }
  VkBufferCopy copy_region = {
      .srcOffset = (VkDeviceSize)source_backing_offset,
      .dstOffset = (VkDeviceSize)target_backing_offset,
      .size = (VkDeviceSize)length,
  };
  iree_vkCmdCopyBuffer(IREE_VULKAN_DEVICE(&queue->syms), command_buffer,
                       source_handle, target_handle, /*regionCount=*/1,
                       &copy_region);
  if (submission) {
    iree_hal_vulkan_queue_profile_write_timestamp_end(queue, submission);
  }
  return iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                 command_buffer);
}

static iree_status_t iree_hal_vulkan_queue_record_copy_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  return iree_hal_vulkan_queue_record_copy_native_buffers(
      queue, submission->copy.source_buffer, submission->copy.source_offset,
      submission->copy.target_buffer, submission->copy.target_offset,
      submission->copy.length, submission->native_command_buffer, submission);
}

static void iree_hal_vulkan_queue_execute_copy(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
  }
}

static void iree_hal_vulkan_queue_execute_command_buffer(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
  }
}

static void iree_hal_vulkan_queue_complete_alloca(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (iree_status_is_ok(completion_status)) {
    iree_hal_local_transient_buffer_commit(submission->alloca.buffer);
    iree_hal_pool_t* pool = NULL;
    iree_hal_pool_reservation_t reservation;
    const bool has_reservation =
        iree_hal_local_transient_buffer_query_reservation(
            submission->alloca.buffer, &pool, &reservation);
    iree_hal_vulkan_queue_profile_record_memory_event(
        submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX, pool,
        submission->alloca.params, has_reservation ? &reservation : NULL,
        (uint64_t)(uintptr_t)submission->sparse_bind.buffer,
        submission->alloca.allocation_size, frontier->entry_count);
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_pool_t* pool = NULL;
    iree_hal_pool_reservation_t reservation;
    const bool has_reservation =
        iree_hal_local_transient_buffer_query_reservation(
            submission->alloca.buffer, &pool, &reservation);
    iree_hal_local_transient_buffer_decommit(submission->alloca.buffer);
    iree_hal_local_transient_buffer_release_reservation(
        submission->alloca.buffer, submission->alloca.wait_frontier);
    iree_hal_vulkan_queue_profile_record_memory_event(
        submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
        iree_status_code(completion_status), pool, submission->alloca.params,
        has_reservation ? &reservation : NULL,
        (uint64_t)(uintptr_t)submission->sparse_bind.buffer,
        submission->alloca.allocation_size,
        submission->alloca.wait_frontier
            ? submission->alloca.wait_frontier->entry_count
            : 0);
    if (has_reservation) {
      iree_hal_vulkan_queue_profile_record_memory_event(
          submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
          iree_status_code(completion_status), pool, submission->alloca.params,
          &reservation, /*backing_id=*/0, submission->alloca.allocation_size,
          submission->alloca.wait_frontier
              ? submission->alloca.wait_frontier->entry_count
              : 0);
    }
    submission->alloca.wait_frontier = NULL;
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
  }
}

static void iree_hal_vulkan_queue_complete_dealloca(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (iree_status_is_ok(completion_status)) {
    iree_hal_pool_t* pool = NULL;
    iree_hal_pool_reservation_t reservation;
    const bool has_reservation =
        iree_hal_local_transient_buffer_query_reservation(
            submission->dealloca.buffer, &pool, &reservation);
    const iree_hal_buffer_params_t params = {
        .type = iree_hal_buffer_memory_type(submission->dealloca.buffer),
        .access = iree_hal_buffer_allowed_access(submission->dealloca.buffer),
        .usage = iree_hal_buffer_allowed_usage(submission->dealloca.buffer),
    };
    const iree_device_size_t allocation_size =
        iree_hal_buffer_allocation_size(submission->dealloca.buffer);
    iree_hal_local_transient_buffer_decommit(submission->dealloca.buffer);
    iree_hal_local_transient_buffer_release_reservation(
        submission->dealloca.buffer, frontier);
    iree_hal_vulkan_queue_profile_record_memory_event(
        submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX, pool,
        params, has_reservation ? &reservation : NULL, /*backing_id=*/0,
        allocation_size, frontier->entry_count);
    if (has_reservation) {
      iree_hal_vulkan_queue_profile_record_memory_event(
          submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX, pool,
          params, &reservation, /*backing_id=*/0, allocation_size,
          frontier->entry_count);
    }
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    const iree_hal_buffer_params_t params = {
        .type = iree_hal_buffer_memory_type(submission->dealloca.buffer),
        .access = iree_hal_buffer_allowed_access(submission->dealloca.buffer),
        .usage = iree_hal_buffer_allowed_usage(submission->dealloca.buffer),
    };
    iree_hal_vulkan_queue_profile_record_memory_event(
        submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
        iree_status_code(completion_status), /*pool=*/NULL, params,
        /*reservation=*/NULL, /*backing_id=*/0,
        iree_hal_buffer_allocation_size(submission->dealloca.buffer),
        /*frontier_entry_count=*/0);
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
  }
}

static void iree_hal_vulkan_queue_complete_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_status_t terminal_status = iree_status_clone(completion_status);
  if (iree_status_is_ok(terminal_status)) {
    terminal_status =
        iree_hal_vulkan_queue_profile_read_native_timestamps(queue, submission);
    if (iree_status_is_ok(terminal_status)) {
      terminal_status =
          iree_hal_vulkan_queue_profile_record_dispatch_events(submission);
    }
    if (iree_status_is_ok(terminal_status)) {
      terminal_status =
          iree_hal_vulkan_queue_profile_record_queue_device_event(submission);
    }
  }

  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER:
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND:
      if (iree_status_is_ok(terminal_status)) {
        iree_hal_vulkan_queue_signal_list_or_fail(
            submission->signal_semaphore_list,
            iree_async_fixed_frontier_as_const_frontier(&submission->frontier));
      } else {
        iree_hal_vulkan_queue_fail_signal_list(
            submission->signal_semaphore_list,
            iree_status_clone(terminal_status));
      }
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL:
      iree_hal_vulkan_queue_execute_host_call(queue, submission,
                                              terminal_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL:
      iree_hal_vulkan_queue_execute_fill(submission, terminal_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE:
      iree_hal_vulkan_queue_execute_update(submission, terminal_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY:
      iree_hal_vulkan_queue_execute_copy(submission, terminal_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE:
      iree_hal_vulkan_queue_execute_command_buffer(submission, terminal_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA:
      iree_hal_vulkan_queue_complete_alloca(submission, terminal_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA:
      iree_hal_vulkan_queue_complete_dealloca(submission, terminal_status);
      break;
    default:
      iree_hal_vulkan_queue_fail_signal_list(
          submission->signal_semaphore_list,
          iree_make_status(IREE_STATUS_INTERNAL,
                           "unknown Vulkan queue submission kind %u",
                           (uint32_t)submission->kind));
      break;
  }

  iree_hal_vulkan_queue_consume_completion_action(submission, terminal_status);

  if (iree_status_is_ok(terminal_status) && queue->frontier_tracker) {
    iree_async_frontier_tracker_advance(queue->frontier_tracker, queue->axis,
                                        submission->epoch);
  }
  iree_atomic_store(&queue->last_drained_epoch, (int64_t)submission->epoch,
                    iree_memory_order_release);
  iree_status_free(terminal_status);
}

static void iree_hal_vulkan_queue_fail_unsubmitted_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status),
              "unsubmitted queue failure status must be non-OK");
  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA: {
      iree_hal_pool_t* alloca_pool = NULL;
      iree_hal_pool_reservation_t alloca_reservation;
      const bool alloca_has_reservation =
          iree_hal_local_transient_buffer_query_reservation(
              submission->alloca.buffer, &alloca_pool, &alloca_reservation);
      iree_hal_local_transient_buffer_decommit(submission->alloca.buffer);
      iree_hal_local_transient_buffer_release_reservation(
          submission->alloca.buffer, submission->alloca.wait_frontier);
      iree_hal_vulkan_queue_profile_record_memory_event(
          submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
          iree_status_code(status), alloca_pool, submission->alloca.params,
          alloca_has_reservation ? &alloca_reservation : NULL,
          (uint64_t)(uintptr_t)submission->sparse_bind.buffer,
          submission->alloca.allocation_size,
          submission->alloca.wait_frontier
              ? submission->alloca.wait_frontier->entry_count
              : 0);
      if (alloca_has_reservation) {
        iree_hal_vulkan_queue_profile_record_memory_event(
            submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
            IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
            iree_status_code(status), alloca_pool, submission->alloca.params,
            &alloca_reservation, /*backing_id=*/0,
            submission->alloca.allocation_size,
            submission->alloca.wait_frontier
                ? submission->alloca.wait_frontier->entry_count
                : 0);
      }
      submission->alloca.wait_frontier = NULL;
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             status);
      break;
    }
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA: {
      iree_hal_local_transient_buffer_abort_dealloca(
          submission->dealloca.buffer);
      const iree_hal_buffer_params_t dealloca_params = {
          .type = iree_hal_buffer_memory_type(submission->dealloca.buffer),
          .access = iree_hal_buffer_allowed_access(submission->dealloca.buffer),
          .usage = iree_hal_buffer_allowed_usage(submission->dealloca.buffer),
      };
      iree_hal_vulkan_queue_profile_record_memory_event(
          submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
          iree_status_code(status), /*pool=*/NULL, dealloca_params,
          /*reservation=*/NULL, /*backing_id=*/0,
          iree_hal_buffer_allocation_size(submission->dealloca.buffer),
          /*frontier_entry_count=*/0);
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             status);
      break;
    }
    default:
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             status);
      break;
  }
  iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
}

static iree_status_t iree_hal_vulkan_queue_resolve_waits(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_vulkan_queue_frontier_t* pending_frontier,
    bool allow_software_deferral,
    iree_hal_vulkan_queue_wait_resolution_t* resolution) {
  resolution->wait_info_count = 0;
  resolution->needs_deferral = false;
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    iree_hal_semaphore_t* semaphore = wait_semaphore_list.semaphores[i];
    const uint64_t minimum_value = wait_semaphore_list.payload_values[i];

    uint64_t current_value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(semaphore, &current_value));
    if (current_value >= minimum_value) continue;

    iree_hal_vulkan_last_signal_flags_t signal_flags = 0;
    iree_async_axis_t producer_axis = 0;
    uint64_t producer_epoch = 0;
    uint64_t producer_value = 0;
    const bool has_last_signal = iree_hal_vulkan_last_signal_load(
        iree_hal_vulkan_semaphore_last_signal(semaphore), &signal_flags,
        &producer_axis, &producer_epoch, &producer_value);
    (void)signal_flags;
    if (!has_last_signal || producer_value < minimum_value) {
      if (allow_software_deferral) {
        resolution->needs_deferral = true;
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan queue wait on semaphore %" PRIhsz
          " requires software deferral; only already-signaled waits and waits "
          "backed by a published producer epoch are accepted",
          i);
    }

    if (producer_axis == queue->axis &&
        producer_epoch >= queue->next_epoch_value) {
      if (allow_software_deferral) {
        resolution->needs_deferral = true;
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan queue wait on same-queue future signal requires software "
          "deferral; only prior accepted epochs are accepted");
    }

    iree_async_single_frontier_t producer_frontier;
    iree_async_single_frontier_initialize(&producer_frontier, producer_axis,
                                          producer_epoch);
    if (!iree_async_frontier_merge(
            iree_async_fixed_frontier_as_frontier(pending_frontier),
            IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY,
            iree_async_single_frontier_as_const_frontier(&producer_frontier))) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "Vulkan queue frontier capacity exhausted while resolving waits");
    }

    VkSemaphore wait_handle = VK_NULL_HANDLE;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_semaphore_handle(semaphore, &wait_handle));
    if (resolution->wait_info_count >= resolution->wait_info_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "too many native Vulkan wait semaphores for queue submission");
    }
    resolution->wait_infos[resolution->wait_info_count] =
        (VkSemaphoreSubmitInfo){
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = wait_handle,
            .value = minimum_value,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
        };
    resolution->wait_info_count = resolution->wait_info_count + 1;
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_publish_signals(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  for (iree_host_size_t i = 0; i < submission->signal_semaphore_list.count;
       ++i) {
    iree_hal_vulkan_semaphore_publish_signal(
        submission->signal_semaphore_list.semaphores[i], queue->axis, frontier,
        submission->epoch, submission->signal_semaphore_list.payload_values[i]);
  }
}

static void iree_hal_vulkan_queue_alloca_pool_notification_end_observe(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->alloca.pool_notification_observation_held) {
    submission->alloca.pool_notification_observation_held = false;
    iree_async_notification_end_observe(submission->alloca.pool_notification);
  }
}

static bool iree_hal_vulkan_queue_alloca_has_staged_backing(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  return iree_hal_local_transient_buffer_backing_buffer(
             submission->alloca.buffer) != NULL;
}

static iree_status_t iree_hal_vulkan_queue_stage_alloca_reservation(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* wait_frontier,
    iree_hal_pool_acquire_result_t acquire_result) {
  if (acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT &&
      !iree_all_bits_set(submission->alloca.flags,
                         IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)) {
    iree_hal_pool_release_reservation(submission->alloca.pool, reservation,
                                      wait_frontier);
    iree_hal_vulkan_queue_profile_record_memory_event(
        submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
            IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER,
        IREE_STATUS_RESOURCE_EXHAUSTED, submission->alloca.pool,
        submission->alloca.params, reservation, /*backing_id=*/0,
        submission->alloca.allocation_size,
        wait_frontier ? wait_frontier->entry_count : 0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue_alloca recycled pool memory requires "
                            "IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER");
  }

  iree_hal_buffer_t* backing_buffer = NULL;
  iree_status_t status = iree_hal_pool_materialize_reservation(
      submission->alloca.pool, submission->alloca.params, reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &backing_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_local_transient_buffer_attach_reservation(
        submission->alloca.buffer, submission->alloca.pool, reservation);
    iree_hal_local_transient_buffer_stage_backing(submission->alloca.buffer,
                                                  backing_buffer);
    submission->alloca.wait_frontier =
        acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT ? wait_frontier
                                                              : NULL;
    submission->alloca.memory_wait_kind =
        acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT
            ? IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER
            : IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    iree_hal_vulkan_queue_profile_record_memory_event(
        submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, acquire_result,
        submission->alloca.pool, submission->alloca.params, reservation,
        /*backing_id=*/0, submission->alloca.allocation_size,
        wait_frontier ? wait_frontier->entry_count : 0);
  } else {
    iree_hal_pool_release_reservation(
        submission->alloca.pool, reservation,
        acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT ? wait_frontier
                                                              : NULL);
    iree_hal_profile_memory_event_flags_t release_flags =
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
    if (acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
      release_flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER;
    }
    iree_hal_vulkan_queue_profile_record_memory_event(
        submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        release_flags, iree_status_code(status), submission->alloca.pool,
        submission->alloca.params, reservation, /*backing_id=*/0,
        submission->alloca.allocation_size,
        acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT && wait_frontier
            ? wait_frontier->entry_count
            : 0);
  }
  iree_hal_buffer_release(backing_buffer);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_stage_alloca_sparse_backing(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_t* queue = submission->queue;
  iree_hal_buffer_placement_t placement = {
      .device = (iree_hal_device_t*)queue->device,
      .queue_affinity = submission->alloca.params.queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };

  iree_hal_buffer_t* backing_buffer = NULL;
  iree_host_size_t bind_count = 0;
  VkSparseMemoryBind* binds = NULL;
  iree_status_t status = iree_hal_vulkan_allocator_allocate_queue_sparse_buffer(
      submission->alloca.allocator, placement, submission->alloca.params,
      submission->alloca.allocation_size,
      iree_hal_buffer_byte_length(submission->alloca.buffer),
      queue->host_allocator, &backing_buffer, &bind_count, &binds);

  VkBuffer sparse_buffer_handle = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    VkDeviceMemory sparse_buffer_memory = VK_NULL_HANDLE;
    status = iree_hal_vulkan_sparse_buffer_handle(
        backing_buffer, &sparse_buffer_memory, &sparse_buffer_handle);
    (void)sparse_buffer_memory;
  }
  if (iree_status_is_ok(status) && bind_count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many Vulkan sparse buffer binds for "
                              "queue_alloca");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_local_transient_buffer_stage_backing(submission->alloca.buffer,
                                                  backing_buffer);
    submission->sparse_bind.buffer = sparse_buffer_handle;
    submission->sparse_bind.binds = binds;
    submission->sparse_bind.bind_count = (uint32_t)bind_count;
    binds = NULL;
    submission->alloca.memory_wait_kind =
        IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
  }

  iree_allocator_free(queue->host_allocator, binds);
  iree_hal_buffer_release(backing_buffer);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_prepare_alloca_pool_notification(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_async_frontier_t* requester_frontier) {
  iree_async_notification_t* notification =
      iree_hal_pool_notification(submission->alloca.pool);
  if (IREE_UNLIKELY(!notification)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca exhausted pool did not provide a notification");
  }

  submission->alloca.pool_notification = notification;
  submission->alloca.pool_notification_wait_token =
      iree_async_notification_begin_observe(notification);
  submission->alloca.pool_notification_observation_held = true;

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  iree_status_t status = iree_hal_pool_acquire_reservation(
      submission->alloca.pool, submission->alloca.allocation_size,
      submission->alloca.params.min_alignment
          ? submission->alloca.params.min_alignment
          : 1,
      requester_frontier, submission->alloca.reserve_flags, &reservation,
      &acquire_info, &acquire_result);

  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_profile_record_memory_event(
        submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
        iree_hal_vulkan_queue_profile_pool_reserve_event_flags(acquire_result),
        acquire_result, submission->alloca.pool, submission->alloca.params,
        acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
                acquire_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET
            ? NULL
            : &reservation,
        /*backing_id=*/0, submission->alloca.allocation_size,
        acquire_info.wait_frontier ? acquire_info.wait_frontier->entry_count
                                   : 0);
    switch (acquire_result) {
      case IREE_HAL_POOL_ACQUIRE_OK:
      case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
        iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
        status = iree_hal_vulkan_queue_stage_alloca_reservation(
            submission, &reservation, acquire_info.wait_frontier,
            acquire_result);
        break;
      case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
      case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
        submission->alloca.memory_wait_kind =
            IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION;
        break;
      default:
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "unrecognized pool acquire result %u",
                                  acquire_result);
        break;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_queue_prepare_alloca_backing(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_async_frontier_t* requester_frontier,
    bool* out_needs_memory_wait) {
  *out_needs_memory_wait = false;
  if (iree_hal_vulkan_queue_alloca_has_staged_backing(submission)) {
    *out_needs_memory_wait = submission->alloca.memory_wait_kind !=
                             IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    return iree_ok_status();
  }
  if (submission->alloca.memory_wait_kind !=
      IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE) {
    *out_needs_memory_wait = true;
    return iree_ok_status();
  }

  if (submission->alloca.strategy ==
      IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_SPARSE) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_queue_stage_alloca_sparse_backing(submission));
    *out_needs_memory_wait = false;
    return iree_ok_status();
  }
  if (submission->alloca.strategy !=
      IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_POOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized Vulkan queue_alloca strategy %u",
                            (uint32_t)submission->alloca.strategy);
  }

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservation(
      submission->alloca.pool, submission->alloca.allocation_size,
      submission->alloca.params.min_alignment
          ? submission->alloca.params.min_alignment
          : 1,
      requester_frontier, submission->alloca.reserve_flags, &reservation,
      &acquire_info, &acquire_result));

  iree_hal_vulkan_queue_profile_record_memory_event(
      submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
      iree_hal_vulkan_queue_profile_pool_reserve_event_flags(acquire_result),
      acquire_result, submission->alloca.pool, submission->alloca.params,
      acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
              acquire_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET
          ? NULL
          : &reservation,
      /*backing_id=*/0, submission->alloca.allocation_size,
      acquire_info.wait_frontier ? acquire_info.wait_frontier->entry_count : 0);

  iree_status_t status = iree_ok_status();
  switch (acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      status = iree_hal_vulkan_queue_stage_alloca_reservation(
          submission, &reservation, acquire_info.wait_frontier, acquire_result);
      break;
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      status = iree_hal_vulkan_queue_prepare_alloca_pool_notification(
          submission, requester_frontier);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "unrecognized pool acquire result %u",
                                acquire_result);
      break;
  }
  if (iree_status_is_ok(status)) {
    *out_needs_memory_wait = submission->alloca.memory_wait_kind !=
                             IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
  }
  return status;
}

static iree_status_t iree_hal_vulkan_queue_try_stage_alloca_backing_now(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_frontier_t requester_frontier_storage;
  iree_slim_mutex_lock(&queue->submission_mutex);
  requester_frontier_storage = queue->frontier;
  iree_slim_mutex_unlock(&queue->submission_mutex);
  const iree_async_frontier_t* requester_frontier =
      iree_async_fixed_frontier_as_const_frontier(&requester_frontier_storage);

  if (submission->alloca.strategy ==
      IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_SPARSE) {
    return iree_hal_vulkan_queue_stage_alloca_sparse_backing(submission);
  }
  if (submission->alloca.strategy !=
      IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_POOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized Vulkan queue_alloca strategy %u",
                            (uint32_t)submission->alloca.strategy);
  }

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservation(
      submission->alloca.pool, submission->alloca.allocation_size,
      submission->alloca.params.min_alignment
          ? submission->alloca.params.min_alignment
          : 1,
      requester_frontier, submission->alloca.reserve_flags, &reservation,
      &acquire_info, &acquire_result));

  iree_hal_vulkan_queue_profile_record_memory_event(
      submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
      iree_hal_vulkan_queue_profile_pool_reserve_event_flags(acquire_result),
      acquire_result, submission->alloca.pool, submission->alloca.params,
      acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
              acquire_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET
          ? NULL
          : &reservation,
      /*backing_id=*/0, submission->alloca.allocation_size,
      acquire_info.wait_frontier ? acquire_info.wait_frontier->entry_count : 0);

  switch (acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      return iree_hal_vulkan_queue_stage_alloca_reservation(
          submission, &reservation, acquire_info.wait_frontier, acquire_result);
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized pool acquire result %u",
                              acquire_result);
  }
}

static iree_status_t iree_hal_vulkan_queue_allocate_sparse_wait_arrays(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_vulkan_queue_wait_resolution_t* resolution,
    VkSemaphore** out_wait_semaphores, uint64_t** out_wait_values) {
  *out_wait_semaphores = NULL;
  *out_wait_values = NULL;
  if (resolution->wait_info_count == 0) return iree_ok_status();

  iree_host_size_t wait_semaphore_storage_size = 0;
  if (!iree_host_size_checked_mul(resolution->wait_info_count,
                                  sizeof(**out_wait_semaphores),
                                  &wait_semaphore_storage_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan sparse bind wait semaphore array overflows host storage");
  }
  iree_host_size_t wait_value_storage_size = 0;
  if (!iree_host_size_checked_mul(resolution->wait_info_count,
                                  sizeof(**out_wait_values),
                                  &wait_value_storage_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan sparse bind wait value array overflows host storage");
  }

  iree_status_t status =
      iree_allocator_malloc(queue->host_allocator, wait_semaphore_storage_size,
                            (void**)out_wait_semaphores);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(queue->host_allocator, wait_value_storage_size,
                              (void**)out_wait_values);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(queue->host_allocator, *out_wait_semaphores);
    *out_wait_semaphores = NULL;
  }
  return status;
}

static iree_status_t iree_hal_vulkan_queue_submit_sparse_bind_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_vulkan_queue_wait_resolution_t* resolution) {
  if (!iree_all_bits_set(queue->queue_flags, VK_QUEUE_SPARSE_BINDING_BIT)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue family %u does not support sparse binding",
        queue->queue_family_index);
  }

  VkSemaphore* wait_semaphores = NULL;
  uint64_t* wait_values = NULL;
  iree_status_t status = iree_hal_vulkan_queue_allocate_sparse_wait_arrays(
      queue, resolution, &wait_semaphores, &wait_values);

  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < resolution->wait_info_count; ++i) {
      wait_semaphores[i] = resolution->wait_infos[i].semaphore;
      wait_values[i] = resolution->wait_infos[i].value;
    }
  }

  if (iree_status_is_ok(status)) {
    VkSemaphore signal_semaphore = queue->epoch_semaphore;
    const uint64_t signal_value = submission->epoch;
    VkTimelineSemaphoreSubmitInfo timeline_info = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = resolution->wait_info_count,
        .pWaitSemaphoreValues = wait_values,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal_value,
    };
    VkSparseBufferMemoryBindInfo buffer_bind_info = {
        .buffer = submission->sparse_bind.buffer,
        .bindCount = submission->sparse_bind.bind_count,
        .pBinds = submission->sparse_bind.binds,
    };
    VkBindSparseInfo bind_info = {
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
        .pNext = &timeline_info,
        .waitSemaphoreCount = resolution->wait_info_count,
        .pWaitSemaphores = wait_semaphores,
        .bufferBindCount = 1,
        .pBufferBinds = &buffer_bind_info,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphore,
    };
    iree_slim_mutex_lock(queue->queue_handle_mutex);
    status =
        iree_vkQueueBindSparse(IREE_VULKAN_DEVICE(&queue->syms), queue->queue,
                               /*bindInfoCount=*/1, &bind_info, VK_NULL_HANDLE);
    iree_slim_mutex_unlock(queue->queue_handle_mutex);
  }

  iree_allocator_free(queue->host_allocator, wait_values);
  iree_allocator_free(queue->host_allocator, wait_semaphores);
  return status;
}

static bool iree_hal_vulkan_queue_submission_uses_sparse_bind(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  return submission->kind ==
             IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND ||
         submission->sparse_bind.bind_count != 0;
}

static iree_status_t iree_hal_vulkan_queue_submit_native_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    bool allow_software_deferral,
    iree_hal_vulkan_queue_wait_resolution_t* resolution,
    iree_hal_vulkan_queue_submission_result_t* out_result) {
  out_result->submitted = false;
  out_result->memory_wait_submission = NULL;
  out_result->queue_failure_status = iree_ok_status();
  resolution->wait_info_count = 0;
  resolution->needs_deferral = false;

  iree_status_t status = iree_hal_vulkan_queue_check_error(queue);
  if (iree_status_is_ok(status) && !queue->frontier_tracker) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Vulkan queue frontier is not assigned");
  }
  if (iree_status_is_ok(status) &&
      iree_atomic_load(&queue->stop_requested, iree_memory_order_acquire)) {
    status = iree_make_status(IREE_STATUS_CANCELLED,
                              "Vulkan queue is shutting down");
  }
  if (iree_status_is_ok(status) &&
      iree_hal_vulkan_queue_submission_uses_sparse_bind(submission) &&
      !iree_all_bits_set(queue->queue_flags, VK_QUEUE_SPARSE_BINDING_BIT)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue family %u does not support sparse binding",
        queue->queue_family_index);
  }
  if (iree_status_is_ok(status)) {
    submission->epoch = queue->next_epoch_value;
    submission->frontier = queue->frontier;
    status = iree_hal_vulkan_queue_resolve_waits(
        queue, submission->wait_semaphore_list, &submission->frontier,
        allow_software_deferral, resolution);
    if (iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_profile_set_dependency_resolution(submission,
                                                              resolution);
    }
  }
  if (iree_status_is_ok(status) && resolution->needs_deferral) {
    return iree_ok_status();
  }
  if (iree_status_is_ok(status) &&
      submission->kind == IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA) {
    bool needs_memory_wait = false;
    status = iree_hal_vulkan_queue_prepare_alloca_backing(
        submission,
        iree_async_fixed_frontier_as_const_frontier(&submission->frontier),
        &needs_memory_wait);
    if (iree_status_is_ok(status) && needs_memory_wait) {
      iree_atomic_store(&submission->deferred_state,
                        IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING,
                        iree_memory_order_release);
      iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 1,
                        iree_memory_order_relaxed);
      iree_hal_vulkan_queue_profile_force_software_defer(submission);
      iree_hal_vulkan_queue_append_deferred_submission(queue, submission);
      iree_hal_vulkan_queue_profile_record_submission(submission);
      out_result->memory_wait_submission = submission;
      return iree_ok_status();
    }
  }
  if (iree_status_is_ok(status)) {
    iree_async_single_frontier_t self_frontier;
    iree_async_single_frontier_initialize(&self_frontier, queue->axis,
                                          submission->epoch);
    if (!iree_async_frontier_merge(
            iree_async_fixed_frontier_as_frontier(&submission->frontier),
            IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY,
            iree_async_single_frontier_as_const_frontier(&self_frontier))) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "Vulkan queue frontier capacity exhausted while publishing epoch");
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_publish_descriptor_cache_sets_under_lock(queue,
                                                                   submission);
    if (iree_hal_vulkan_queue_submission_uses_sparse_bind(submission)) {
      status = iree_hal_vulkan_queue_submit_sparse_bind_under_lock(
          queue, submission, resolution);
    } else {
      VkSemaphoreSubmitInfo epoch_signal_info = {
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = queue->epoch_semaphore,
          .value = submission->epoch,
          .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
          .deviceIndex = 0,
      };
      VkCommandBufferSubmitInfo command_buffer_info = {
          .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
          .commandBuffer = submission->native_command_buffer,
          .deviceMask = 0,
      };
      VkSubmitInfo2 submit_info = {
          .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
          .waitSemaphoreInfoCount = resolution->wait_info_count,
          .pWaitSemaphoreInfos = resolution->wait_infos,
          .commandBufferInfoCount = submission->native_command_buffer ? 1u : 0u,
          .pCommandBufferInfos =
              submission->native_command_buffer ? &command_buffer_info : NULL,
          .signalSemaphoreInfoCount = 1,
          .pSignalSemaphoreInfos = &epoch_signal_info,
      };
      iree_slim_mutex_lock(queue->queue_handle_mutex);
      status =
          iree_vkQueueSubmit2(IREE_VULKAN_DEVICE(&queue->syms), queue->queue, 1,
                              &submit_info, VK_NULL_HANDLE);
      iree_slim_mutex_unlock(queue->queue_handle_mutex);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_profile_record_submission(submission);
      iree_hal_vulkan_queue_publish_signals(queue, submission);
      queue->frontier = submission->frontier;
      queue->next_epoch_value = queue->next_epoch_value + 1;
      iree_hal_vulkan_queue_append_pending_submission(queue, submission);
      out_result->submitted = true;
    } else if (iree_status_code(status) == IREE_STATUS_DATA_LOSS) {
      out_result->queue_failure_status =
          iree_hal_vulkan_queue_store_error(queue, iree_status_clone(status));
    }
  }
  return status;
}

static bool iree_hal_vulkan_queue_wait_entry_callback_is_complete(
    const iree_hal_vulkan_queue_wait_entry_t* entry) {
  return iree_atomic_load(&entry->callback_complete,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_vulkan_queue_wait_entry_publish_callback_complete(
    iree_hal_vulkan_queue_wait_entry_t* entry) {
  iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_release);
  iree_notification_post(&entry->submission->callback_notification,
                         IREE_ALL_WAITERS);
}

static bool iree_hal_vulkan_queue_wait_callbacks_are_complete(void* user_data) {
  iree_hal_vulkan_queue_pending_submission_t* submission =
      (iree_hal_vulkan_queue_pending_submission_t*)user_data;
  for (iree_host_size_t i = 0; i < submission->wait_entry_count; ++i) {
    if (!iree_hal_vulkan_queue_wait_entry_callback_is_complete(
            &submission->wait_entries[i])) {
      return false;
    }
  }
  return true;
}

static void iree_hal_vulkan_queue_submission_record_wait_status(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &submission->wait_failure_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static bool iree_hal_vulkan_queue_submission_claim_promotion(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  int32_t expected_state = IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING;
  return iree_atomic_compare_exchange_strong(
      &submission->deferred_state, &expected_state,
      IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PROMOTING, iree_memory_order_acq_rel,
      iree_memory_order_acquire);
}

static void iree_hal_vulkan_queue_deferred_submission_ready(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_t* queue = submission->queue;
  iree_status_t failure_status = iree_ok_status();
  iree_slim_mutex_lock(&queue->submission_mutex);
  const bool was_deferred =
      iree_hal_vulkan_queue_unlink_deferred_submission(queue, submission);
  failure_status = (iree_status_t)iree_atomic_load(&queue->failure_status,
                                                   iree_memory_order_acquire);
  if (was_deferred && iree_status_is_ok(failure_status)) {
    iree_hal_vulkan_queue_append_ready_submission(queue, submission);
    iree_atomic_store(&submission->deferred_state,
                      IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_READY,
                      iree_memory_order_release);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  iree_notification_post(&submission->callback_notification, IREE_ALL_WAITERS);
  if (!was_deferred) {
    iree_hal_vulkan_queue_fail_unsubmitted_submission(
        queue, submission,
        iree_make_status(IREE_STATUS_INTERNAL,
                         "Vulkan deferred submission promotion lost queue "
                         "ownership"));
    return;
  }
  if (!iree_status_is_ok(failure_status)) {
    iree_hal_vulkan_queue_fail_unsubmitted_submission(
        queue, submission, iree_status_clone(failure_status));
    return;
  }

  iree_status_t wake_status = iree_hal_vulkan_queue_signal_wakeup(queue);
  if (!iree_status_is_ok(wake_status)) {
    iree_status_t stored_status =
        iree_hal_vulkan_queue_store_error(queue, wake_status);
    iree_slim_mutex_lock(&queue->submission_mutex);
    const bool recovered_submission =
        iree_hal_vulkan_queue_unlink_ready_submission(queue, submission);
    iree_slim_mutex_unlock(&queue->submission_mutex);
    if (recovered_submission) {
      iree_hal_vulkan_queue_fail_unsubmitted_submission(queue, submission,
                                                        stored_status);
    } else {
      iree_status_free(stored_status);
    }
  }
}

static void iree_hal_vulkan_queue_deferred_wait_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_vulkan_queue_wait_entry_t* entry =
      (iree_hal_vulkan_queue_wait_entry_t*)user_data;
  iree_hal_vulkan_queue_pending_submission_t* submission = entry->submission;
  iree_hal_vulkan_queue_submission_record_wait_status(submission, status);

  const int32_t previous_count = iree_atomic_fetch_sub(
      &submission->wait_count, 1, iree_memory_order_acq_rel);
  const bool owns_promotion =
      previous_count == 1 &&
      iree_hal_vulkan_queue_submission_claim_promotion(submission);

  iree_hal_vulkan_queue_wait_entry_publish_callback_complete(entry);
  if (!owns_promotion) return;

  iree_notification_await(&submission->callback_notification,
                          iree_hal_vulkan_queue_wait_callbacks_are_complete,
                          submission, iree_infinite_timeout());
  iree_hal_vulkan_queue_deferred_submission_ready(submission);
}

static iree_status_t iree_hal_vulkan_queue_prepare_deferred_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  const iree_host_size_t wait_count = submission->wait_semaphore_list.count;
  if (wait_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many software-deferred Vulkan queue waits");
  }
  iree_host_size_t wait_entries_size = 0;
  if (!iree_host_size_checked_mul(wait_count, sizeof(*submission->wait_entries),
                                  &wait_entries_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "software-deferred Vulkan wait list is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(queue->host_allocator, wait_entries_size,
                            (void**)&submission->wait_entries));
  memset(submission->wait_entries, 0, wait_entries_size);
  submission->wait_entry_count = wait_count;
  iree_atomic_store(&submission->wait_count, (int32_t)wait_count,
                    iree_memory_order_release);
  iree_atomic_store(&submission->deferred_state,
                    IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING,
                    iree_memory_order_release);
  for (iree_host_size_t i = 0; i < wait_count; ++i) {
    iree_hal_vulkan_queue_wait_entry_t* entry = &submission->wait_entries[i];
    entry->submission = submission;
    iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_relaxed);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_start_deferred_submission(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_semaphore_list_t wait_semaphores = submission->wait_semaphore_list;
  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_hal_vulkan_queue_wait_entry_t* entry = &submission->wait_entries[i];
    iree_atomic_store(&entry->callback_complete, 0, iree_memory_order_relaxed);
    entry->timepoint.callback = iree_hal_vulkan_queue_deferred_wait_resolved;
    entry->timepoint.user_data = entry;
    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)wait_semaphores.semaphores[i],
        wait_semaphores.payload_values[i], &entry->timepoint);
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_submission_record_wait_status(submission, status);
      iree_atomic_store(&entry->callback_complete, 1,
                        iree_memory_order_release);
      const int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      const int32_t previous_count = iree_atomic_fetch_sub(
          &submission->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered &&
          iree_hal_vulkan_queue_submission_claim_promotion(submission)) {
        iree_notification_await(
            &submission->callback_notification,
            iree_hal_vulkan_queue_wait_callbacks_are_complete, submission,
            iree_infinite_timeout());
        iree_hal_vulkan_queue_deferred_submission_ready(submission);
      }
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  return iree_atomic_load(&submission->alloca.memory_wait_callback_complete,
                          iree_memory_order_acquire) != 0;
}

static bool iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete_thunk(
    void* user_data) {
  return iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
      (const iree_hal_vulkan_queue_pending_submission_t*)user_data);
}

static void iree_hal_vulkan_queue_alloca_memory_wait_publish_complete(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 1,
                    iree_memory_order_release);
  iree_notification_post(&submission->callback_notification, IREE_ALL_WAITERS);
}

static void iree_hal_vulkan_queue_alloca_memory_wait_resolved(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  if (iree_status_is_ok(status)) {
    switch (submission->alloca.memory_wait_kind) {
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER:
        submission->alloca.memory_wait_kind =
            IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
        break;
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
        submission->alloca.memory_wait_kind =
            IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
        submission->alloca.pool_notification = NULL;
        break;
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE:
        break;
    }
  }

  iree_hal_vulkan_queue_submission_record_wait_status(submission, status);
  const bool owns_promotion =
      iree_hal_vulkan_queue_submission_claim_promotion(submission);
  iree_hal_vulkan_queue_alloca_memory_wait_publish_complete(submission);
  if (owns_promotion) {
    iree_hal_vulkan_queue_deferred_submission_ready(submission);
  }
}

static void iree_hal_vulkan_queue_alloca_frontier_wait_resolved(
    void* user_data, iree_status_t status) {
  iree_hal_vulkan_queue_alloca_memory_wait_resolved(
      (iree_hal_vulkan_queue_pending_submission_t*)user_data, status);
}

static void iree_hal_vulkan_queue_alloca_pool_notification_wait_resolved(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_vulkan_queue_alloca_memory_wait_resolved(
      (iree_hal_vulkan_queue_pending_submission_t*)user_data, status);
}

static void iree_hal_vulkan_queue_start_alloca_frontier_wait(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (IREE_UNLIKELY(!submission->alloca.wait_frontier)) {
    iree_hal_vulkan_queue_alloca_memory_wait_resolved(
        submission,
        iree_make_status(
            IREE_STATUS_INTERNAL,
            "queue_alloca waitable pool reservation did not provide a "
            "frontier"));
    return;
  }
  iree_hal_vulkan_queue_profile_record_memory_event(
      submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER,
      UINT32_MAX, submission->alloca.pool, submission->alloca.params,
      /*reservation=*/NULL, /*backing_id=*/0,
      submission->alloca.allocation_size,
      submission->alloca.wait_frontier->entry_count);
  iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 0,
                    iree_memory_order_relaxed);
  iree_status_t status = iree_async_frontier_tracker_wait(
      submission->queue->frontier_tracker, submission->alloca.wait_frontier,
      iree_hal_vulkan_queue_alloca_frontier_wait_resolved, submission,
      &submission->alloca.frontier_waiter);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_alloca_memory_wait_resolved(submission, status);
  }
}

static void iree_hal_vulkan_queue_start_alloca_pool_notification_wait(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_profile_record_memory_event(
      submission, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION,
      UINT32_MAX, submission->alloca.pool, submission->alloca.params,
      /*reservation=*/NULL, /*backing_id=*/0,
      submission->alloca.allocation_size,
      /*frontier_entry_count=*/0);
  iree_async_notification_wait_operation_t* wait_op =
      &submission->alloca.pool_notification_wait_ops
           [submission->alloca.pool_notification_wait_slot];
  iree_async_operation_zero(&wait_op->base, sizeof(*wait_op));
  iree_async_operation_initialize(
      &wait_op->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_vulkan_queue_alloca_pool_notification_wait_resolved, submission);
  wait_op->notification = submission->alloca.pool_notification;
  wait_op->wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op->wait_token = submission->alloca.pool_notification_wait_token;

  iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 0,
                    iree_memory_order_relaxed);
  iree_status_t status = iree_async_proactor_submit_one(
      submission->queue->proactor, &wait_op->base);
  iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_alloca_memory_wait_resolved(submission, status);
  }
}

static iree_status_t iree_hal_vulkan_queue_start_alloca_memory_wait(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  switch (submission->alloca.memory_wait_kind) {
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_vulkan_queue_start_alloca_frontier_wait(submission);
      return iree_ok_status();
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      submission->alloca.pool_notification_wait_slot =
          (uint8_t)((submission->alloca.pool_notification_wait_slot + 1u) & 1u);
      iree_hal_vulkan_queue_start_alloca_pool_notification_wait(submission);
      return iree_ok_status();
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE:
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "Vulkan alloca submission has no memory wait to start");
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unrecognized Vulkan alloca memory wait kind %u",
                          submission->alloca.memory_wait_kind);
}

static iree_status_t iree_hal_vulkan_queue_allocate_wait_infos(
    iree_hal_vulkan_queue_t* queue, iree_host_size_t wait_count,
    VkSemaphoreSubmitInfo** out_wait_infos, uint32_t* out_wait_info_capacity) {
  *out_wait_infos = NULL;
  *out_wait_info_capacity = 0;
  if (wait_count == 0) return iree_ok_status();
  if (wait_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many Vulkan queue wait semaphores");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, wait_count * sizeof(**out_wait_infos),
      (void**)out_wait_infos));
  *out_wait_info_capacity = (uint32_t)wait_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_submission_take_wait_failure(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  return (iree_status_t)iree_atomic_exchange(&submission->wait_failure_status,
                                             0, iree_memory_order_acquire);
}

static iree_hal_vulkan_queue_pending_submission_t*
iree_hal_vulkan_queue_take_cancellable_deferred_submissions_under_lock(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_pending_submission_t* cancellable_head = NULL;
  iree_hal_vulkan_queue_pending_submission_t** cancellable_tail =
      &cancellable_head;
  iree_hal_vulkan_queue_pending_submission_t** link = &queue->deferred_head;
  while (*link) {
    iree_hal_vulkan_queue_pending_submission_t* submission = *link;
    int32_t expected_state = IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING;
    if (iree_atomic_compare_exchange_strong(
            &submission->deferred_state, &expected_state,
            IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_CANCELLING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      *link = submission->next;
      submission->next = NULL;
      *cancellable_tail = submission;
      cancellable_tail = &submission->next;
      continue;
    }
    link = &submission->next;
  }
  return cancellable_head;
}

static void iree_hal_vulkan_queue_cancel_alloca_memory_wait(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->kind != IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA) return;
  if (submission->alloca.memory_wait_kind ==
      IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE) {
    return;
  }

  switch (submission->alloca.memory_wait_kind) {
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER:
      if (iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
              submission)) {
        break;
      }
      if (!iree_async_frontier_tracker_cancel_wait(
              submission->queue->frontier_tracker,
              &submission->alloca.frontier_waiter)) {
        iree_notification_await(
            &submission->callback_notification,
            iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete_thunk,
            submission, iree_infinite_timeout());
      } else {
        iree_hal_vulkan_queue_alloca_memory_wait_publish_complete(submission);
      }
      break;
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
      if (submission->alloca.pool_notification) {
        iree_async_notification_signal(submission->alloca.pool_notification,
                                       INT32_MAX);
      }
      if (!iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
              submission)) {
        iree_notification_await(
            &submission->callback_notification,
            iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete_thunk,
            submission, iree_infinite_timeout());
      }
      break;
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
  submission->alloca.memory_wait_kind =
      IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
}

static void iree_hal_vulkan_queue_cancel_deferred_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  IREE_ASSERT(iree_atomic_load(&submission->deferred_state,
                               iree_memory_order_acquire) ==
                  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_CANCELLING,
              "Vulkan deferred submission must be claimed before "
              "cancellation");

  for (iree_host_size_t i = 0; i < submission->wait_entry_count; ++i) {
    iree_hal_vulkan_queue_wait_entry_t* entry = &submission->wait_entries[i];
    if (iree_hal_vulkan_queue_wait_entry_callback_is_complete(entry)) continue;
    if (iree_async_semaphore_cancel_timepoint(entry->timepoint.semaphore,
                                              &entry->timepoint)) {
      iree_hal_vulkan_queue_wait_entry_publish_callback_complete(entry);
    }
  }
  iree_notification_await(&submission->callback_notification,
                          iree_hal_vulkan_queue_wait_callbacks_are_complete,
                          submission, iree_infinite_timeout());
  iree_hal_vulkan_queue_cancel_alloca_memory_wait(submission);
  iree_hal_vulkan_queue_fail_unsubmitted_submission(queue, submission, status);
}

static void iree_hal_vulkan_queue_cancel_deferred_submissions(
    iree_hal_vulkan_queue_t* queue, iree_status_t status) {
  iree_hal_vulkan_queue_pending_submission_t* deferred_head = NULL;
  iree_slim_mutex_lock(&queue->submission_mutex);
  deferred_head =
      iree_hal_vulkan_queue_take_cancellable_deferred_submissions_under_lock(
          queue);
  iree_slim_mutex_unlock(&queue->submission_mutex);

  while (deferred_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = deferred_head->next;
    deferred_head->next = NULL;
    iree_hal_vulkan_queue_cancel_deferred_submission(queue, deferred_head,
                                                     iree_status_clone(status));
    deferred_head = next;
  }
  iree_status_free(status);
}

static void iree_hal_vulkan_queue_fail_pending_submissions(
    iree_hal_vulkan_queue_t* queue, iree_status_t status) {
  iree_hal_vulkan_queue_pending_submission_t* pending_head = NULL;
  iree_hal_vulkan_queue_pending_submission_t* ready_head = NULL;
  iree_hal_vulkan_queue_pending_submission_t* deferred_head = NULL;
  iree_slim_mutex_lock(&queue->submission_mutex);
  pending_head = queue->pending_head;
  ready_head = queue->ready_head;
  deferred_head =
      iree_hal_vulkan_queue_take_cancellable_deferred_submissions_under_lock(
          queue);
  queue->pending_head = NULL;
  queue->pending_tail = NULL;
  queue->ready_head = NULL;
  queue->ready_tail = NULL;
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (queue->frontier_tracker) {
    iree_async_frontier_tracker_fail_axis(queue->frontier_tracker, queue->axis,
                                          iree_status_clone(status));
  }

  while (pending_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = pending_head->next;
    pending_head->next = NULL;
    iree_hal_vulkan_queue_complete_submission(queue, pending_head, status);
    iree_hal_vulkan_queue_pending_submission_destroy(queue, pending_head);
    pending_head = next;
  }
  while (ready_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = ready_head->next;
    ready_head->next = NULL;
    iree_hal_vulkan_queue_fail_unsubmitted_submission(
        queue, ready_head, iree_status_clone(status));
    ready_head = next;
  }
  while (deferred_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = deferred_head->next;
    deferred_head->next = NULL;
    iree_hal_vulkan_queue_cancel_deferred_submission(queue, deferred_head,
                                                     iree_status_clone(status));
    deferred_head = next;
  }
  iree_status_free(status);
}

static void iree_hal_vulkan_queue_drain_ready_submissions(
    iree_hal_vulkan_queue_t* queue) {
  while (true) {
    iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
    iree_hal_vulkan_queue_submission_result_t submit_result = {
        .submitted = false,
        .queue_failure_status = iree_ok_status(),
    };
    iree_status_t status = iree_ok_status();

    iree_slim_mutex_lock(&queue->submission_mutex);
    submission = iree_hal_vulkan_queue_pop_ready_submission(queue);
    iree_slim_mutex_unlock(&queue->submission_mutex);
    if (!submission) return;

    VkSemaphoreSubmitInfo* wait_infos = NULL;
    uint32_t wait_info_capacity = 0;
    status = iree_hal_vulkan_queue_allocate_wait_infos(
        queue, submission->wait_semaphore_list.count, &wait_infos,
        &wait_info_capacity);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&queue->submission_mutex);
      status = iree_hal_vulkan_queue_submission_take_wait_failure(submission);
      iree_slim_mutex_unlock(&queue->submission_mutex);
    }
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&queue->submission_mutex);
      if (wait_info_capacity < submission->wait_semaphore_list.count) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "too many Vulkan queue wait semaphores for ready submission");
      } else {
        iree_hal_vulkan_queue_wait_resolution_t resolution = {
            .wait_infos = wait_infos,
            .wait_info_capacity = wait_info_capacity,
        };
        status = iree_hal_vulkan_queue_submit_native_under_lock(
            queue, submission, /*allow_software_deferral=*/false, &resolution,
            &submit_result);
      }
      iree_slim_mutex_unlock(&queue->submission_mutex);
    }

    if (wait_infos) {
      iree_allocator_free(queue->host_allocator, wait_infos);
    }

    if (!iree_status_is_ok(submit_result.queue_failure_status)) {
      iree_hal_vulkan_queue_fail_pending_submissions(
          queue, submit_result.queue_failure_status);
    }
    if (submit_result.memory_wait_submission) {
      iree_status_t memory_wait_status =
          iree_hal_vulkan_queue_start_alloca_memory_wait(
              submit_result.memory_wait_submission);
      if (!iree_status_is_ok(memory_wait_status)) {
        iree_hal_vulkan_queue_fail_pending_submissions(queue,
                                                       memory_wait_status);
      }
      iree_status_free(status);
      continue;
    }
    if (submit_result.submitted) {
      iree_status_free(status);
      continue;
    }

    if (iree_status_is_ok(status)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "Vulkan ready submission was neither submitted nor failed");
    }
    iree_hal_vulkan_queue_fail_unsubmitted_submission(queue, submission,
                                                      status);
  }
}

static int iree_hal_vulkan_queue_completion_thread_main(void* entry_arg) {
  iree_hal_vulkan_queue_t* queue = (iree_hal_vulkan_queue_t*)entry_arg;
  uint64_t observed_wakeup_value = (uint64_t)iree_atomic_load(
      &queue->wakeup_value, iree_memory_order_acquire);

  while (true) {
    iree_hal_vulkan_queue_drain_ready_submissions(queue);
    iree_hal_vulkan_queue_drain_completions(queue);
    const bool stop_requested =
        iree_atomic_load(&queue->stop_requested, iree_memory_order_acquire) !=
        0;
    if (stop_requested) {
      iree_hal_vulkan_queue_cancel_deferred_submissions(
          queue, iree_status_from_code(IREE_STATUS_CANCELLED));
      iree_hal_vulkan_queue_drain_ready_submissions(queue);
      iree_hal_vulkan_queue_drain_completions(queue);
    }
    const bool has_pending = iree_hal_vulkan_queue_has_pending(queue);
    if (stop_requested && !has_pending) break;
    iree_status_t failure_status = (iree_status_t)iree_atomic_load(
        &queue->failure_status, iree_memory_order_acquire);
    if (!iree_status_is_ok(failure_status) && !has_pending) break;

    VkSemaphore wait_semaphores[2] = {
        queue->epoch_semaphore,
        queue->wakeup_semaphore,
    };
    uint64_t wait_values[2] = {
        (uint64_t)iree_atomic_load(&queue->last_drained_epoch,
                                   iree_memory_order_acquire) +
            1,
        observed_wakeup_value + 1,
    };
    VkSemaphoreWaitInfo wait_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .flags = VK_SEMAPHORE_WAIT_ANY_BIT,
        .semaphoreCount = IREE_ARRAYSIZE(wait_semaphores),
        .pSemaphores = wait_semaphores,
        .pValues = wait_values,
    };
    VkResult result = iree_vkWaitSemaphores_raw(
        &queue->syms, queue->logical_device, &wait_info, UINT64_MAX);
    if (result == VK_SUCCESS) {
      observed_wakeup_value = (uint64_t)iree_atomic_load(
          &queue->wakeup_value, iree_memory_order_acquire);
      continue;
    }

    iree_status_t status = iree_status_from_vk_result(
        __FILE__, __LINE__, result, "vkWaitSemaphores");
    iree_status_t stored_status =
        iree_hal_vulkan_queue_store_error(queue, status);
    iree_hal_vulkan_queue_fail_pending_submissions(queue, stored_status);
  }

  return 0;
}

iree_status_t iree_hal_vulkan_queue_initialize(
    const iree_hal_vulkan_queue_params_t* params,
    iree_hal_vulkan_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(params->device);
  IREE_ASSERT_ARGUMENT(params->syms);
  IREE_ASSERT_ARGUMENT(params->logical_device);
  IREE_ASSERT_ARGUMENT(params->builtins);
  IREE_ASSERT_ARGUMENT(params->queue);
  IREE_ASSERT_ARGUMENT(params->queue_handle_mutex);
  IREE_ASSERT_ARGUMENT(params->proactor);
  IREE_ASSERT_ARGUMENT(out_queue);
  memset(out_queue, 0, sizeof(*out_queue));

  out_queue->device = params->device;
  out_queue->syms = *params->syms;
  out_queue->logical_device = params->logical_device;
  out_queue->builtins = params->builtins;
  out_queue->queue = params->queue;
  out_queue->queue_flags = params->queue_flags;
  out_queue->timestamp_valid_bits = params->timestamp_valid_bits;
  out_queue->queue_handle_mutex = params->queue_handle_mutex;
  out_queue->proactor = params->proactor;
  out_queue->queue_family_index = params->queue_family_index;
  out_queue->queue_index = params->queue_index;
  out_queue->queue_affinity = params->queue_affinity;
  out_queue->role = params->role;
  out_queue->host_allocator = params->host_allocator;
  out_queue->next_epoch_value = 1;
  iree_slim_mutex_initialize(&out_queue->submission_mutex);
  iree_async_frontier_initialize(
      iree_async_fixed_frontier_as_frontier(&out_queue->frontier),
      /*entry_count=*/0);
  iree_atomic_store(&out_queue->last_drained_epoch, 0,
                    iree_memory_order_release);
  iree_atomic_store(&out_queue->wakeup_value, 0, iree_memory_order_release);

  iree_status_t status = iree_hal_vulkan_queue_create_timeline_semaphore(
      &out_queue->syms, out_queue->logical_device, /*initial_value=*/0,
      &out_queue->epoch_semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_create_timeline_semaphore(
        &out_queue->syms, out_queue->logical_device, /*initial_value=*/0,
        &out_queue->wakeup_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_create_command_pool(
        &out_queue->syms, out_queue->logical_device,
        out_queue->queue_family_index, &out_queue->command_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_descriptor_cache_initialize(out_queue);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_deinitialize(out_queue);
  }
  return status;
}

iree_status_t iree_hal_vulkan_queue_initialize_staging(
    iree_hal_vulkan_queue_t* queue, iree_hal_allocator_t* device_allocator) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(device_allocator);
  if (queue->device_allocator) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue staging is already initialized");
  }

  queue->device_allocator = device_allocator;
  iree_status_t status = iree_hal_vulkan_queue_staging_ring_create(
      queue, IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_SIZE,
      IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_COUNT, &queue->upload_staging_ring);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_staging_ring_create(
        queue, IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_SIZE,
        IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_COUNT,
        &queue->download_staging_ring);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_staging_ring_destroy(queue->download_staging_ring);
    queue->download_staging_ring = NULL;
    iree_hal_vulkan_queue_staging_ring_destroy(queue->upload_staging_ring);
    queue->upload_staging_ring = NULL;
    queue->device_allocator = NULL;
  }
  return status;
}

void iree_hal_vulkan_queue_deinitialize(iree_hal_vulkan_queue_t* queue) {
  if (!queue->logical_device) return;
  iree_hal_vulkan_queue_retire_frontier(queue);
  iree_hal_vulkan_queue_staging_ring_destroy(queue->download_staging_ring);
  queue->download_staging_ring = NULL;
  iree_hal_vulkan_queue_staging_ring_destroy(queue->upload_staging_ring);
  queue->upload_staging_ring = NULL;
  queue->device_allocator = NULL;
  iree_hal_vulkan_queue_descriptor_cache_deinitialize(queue);
  if (queue->command_pool) {
    iree_vkDestroyCommandPool(IREE_VULKAN_DEVICE(&queue->syms),
                              queue->logical_device, queue->command_pool,
                              /*pAllocator=*/NULL);
    queue->command_pool = VK_NULL_HANDLE;
  }
  if (queue->epoch_semaphore) {
    iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(&queue->syms),
                            queue->logical_device, queue->epoch_semaphore,
                            /*pAllocator=*/NULL);
    queue->epoch_semaphore = VK_NULL_HANDLE;
  }
  if (queue->wakeup_semaphore) {
    iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(&queue->syms),
                            queue->logical_device, queue->wakeup_semaphore,
                            /*pAllocator=*/NULL);
    queue->wakeup_semaphore = VK_NULL_HANDLE;
  }
  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &queue->failure_status, iree_memory_order_acquire);
  iree_status_free(failure_status);
  iree_slim_mutex_deinitialize(&queue->submission_mutex);
  memset(queue, 0, sizeof(*queue));
}

iree_status_t iree_hal_vulkan_queue_assign_frontier(
    iree_hal_vulkan_queue_t* queue,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  if (queue->frontier_tracker) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue frontier already assigned");
  }

  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL));
  queue->frontier_tracker = frontier_tracker;
  queue->axis = axis;
  iree_atomic_store(&queue->stop_requested, 0, iree_memory_order_release);

  iree_thread_create_params_t thread_params;
  memset(&thread_params, 0, sizeof(thread_params));
  char thread_name[48] = {0};
  snprintf(thread_name, IREE_ARRAYSIZE(thread_name),
           "iree-hal-vulkan-d%uq%u-complete",
           (unsigned)iree_async_axis_device_index(axis),
           (unsigned)iree_async_axis_queue_index(axis));
  thread_params.name = iree_make_cstring_view(thread_name);
  iree_status_t status = iree_thread_create(
      iree_hal_vulkan_queue_completion_thread_main, queue, thread_params,
      queue->host_allocator, &queue->completion_thread);
  if (!iree_status_is_ok(status)) {
    iree_async_frontier_tracker_retire_axis(
        frontier_tracker, axis, iree_status_from_code(IREE_STATUS_CANCELLED));
    queue->frontier_tracker = NULL;
    queue->axis = 0;
  }
  return status;
}

void iree_hal_vulkan_queue_retire_frontier(iree_hal_vulkan_queue_t* queue) {
  if (!queue->frontier_tracker) return;
  iree_atomic_store(&queue->stop_requested, 1, iree_memory_order_release);
  iree_status_t wake_status = iree_hal_vulkan_queue_signal_wakeup(queue);
  IREE_ASSERT(iree_status_is_ok(wake_status) ||
                  iree_status_code(wake_status) == IREE_STATUS_DATA_LOSS,
              "Vulkan queue completion wakeup failed during teardown");
  iree_status_free(wake_status);
  if (queue->completion_thread) {
    iree_thread_release(queue->completion_thread);
    queue->completion_thread = NULL;
  }
  iree_async_frontier_tracker_retire_axis(
      queue->frontier_tracker, queue->axis,
      iree_status_from_code(IREE_STATUS_CANCELLED));
  queue->frontier_tracker = NULL;
  queue->axis = 0;
}

void iree_hal_vulkan_queue_set_profile_recorder(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t profile_scope,
    iree_atomic_int64_t* submission_counter) {
  if (profile_recorder) {
    queue->profile_scope = profile_scope;
    queue->profile_submission_counter = submission_counter;
    queue->profile_recorder = profile_recorder;
  } else {
    queue->profile_recorder = NULL;
    queue->profile_scope = profile_scope;
    queue->profile_submission_counter = NULL;
  }
}

static iree_status_t iree_hal_vulkan_queue_submit_captured_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_pending_submission_t* deferred_submission = NULL;
  iree_hal_vulkan_queue_pending_submission_t* memory_wait_submission = NULL;
  iree_hal_vulkan_queue_submission_result_t submit_result = {
      .submitted = false,
      .queue_failure_status = iree_ok_status(),
  };

  iree_status_t status = iree_ok_status();
  VkSemaphoreSubmitInfo* wait_infos = NULL;
  uint32_t wait_info_capacity = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_allocate_wait_infos(
        queue, submission->wait_semaphore_list.count, &wait_infos,
        &wait_info_capacity);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    iree_hal_vulkan_queue_wait_resolution_t resolution = {
        .wait_infos = wait_infos,
        .wait_info_capacity = wait_info_capacity,
    };
    status = iree_hal_vulkan_queue_submit_native_under_lock(
        queue, submission, /*allow_software_deferral=*/true, &resolution,
        &submit_result);
    if (iree_status_is_ok(status) && resolution.needs_deferral) {
      status =
          iree_hal_vulkan_queue_prepare_deferred_submission(queue, submission);
      if (iree_status_is_ok(status)) {
        iree_hal_vulkan_queue_append_deferred_submission(queue, submission);
        iree_hal_vulkan_queue_profile_record_submission(submission);
        deferred_submission = submission;
        submission = NULL;
      }
    } else if (submit_result.memory_wait_submission) {
      memory_wait_submission = submit_result.memory_wait_submission;
      submission = NULL;
    } else if (submit_result.submitted) {
      submission = NULL;
    }
    iree_slim_mutex_unlock(&queue->submission_mutex);
  }

  if (wait_infos) {
    iree_allocator_free(queue->host_allocator, wait_infos);
  }
  if (!iree_status_is_ok(submit_result.queue_failure_status)) {
    iree_hal_vulkan_queue_fail_pending_submissions(
        queue, submit_result.queue_failure_status);
  }
  if (deferred_submission) {
    iree_status_t deferred_status =
        iree_hal_vulkan_queue_start_deferred_submission(deferred_submission);
    if (!iree_status_is_ok(deferred_status)) {
      iree_hal_vulkan_queue_fail_pending_submissions(queue, deferred_status);
    }
  }
  if (memory_wait_submission) {
    iree_status_t memory_wait_status =
        iree_hal_vulkan_queue_start_alloca_memory_wait(memory_wait_submission);
    if (!iree_status_is_ok(memory_wait_status)) {
      iree_hal_vulkan_queue_fail_pending_submissions(queue, memory_wait_status);
    }
  }
  if (submission) {
    if (iree_status_is_ok(status)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "Vulkan captured submission was neither submitted nor deferred");
    }
    iree_hal_vulkan_queue_fail_unsubmitted_submission(
        queue, submission, iree_status_clone(status));
  }
  return status;
}

static iree_status_t iree_hal_vulkan_queue_submit_barrier_with_action(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_queue_completion_action_t completion_action) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_vulkan_queue_validate_semaphore_list(
      queue, wait_semaphore_list, IREE_SV("wait"));
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER,
        (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_set_completion_action(submission, completion_action);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_barrier(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  return iree_hal_vulkan_queue_submit_barrier_with_action(
      queue, wait_semaphore_list, signal_semaphore_list,
      iree_hal_vulkan_queue_completion_action_null());
}

static iree_status_t iree_hal_vulkan_queue_create_transient_buffer(
    iree_hal_vulkan_queue_t* queue, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_hal_alloca_flags_t flags, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_placement_t placement = {
      .device = (iree_hal_device_t*)queue->device,
      .queue_affinity = params.queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  if (iree_all_bits_set(flags, IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME)) {
    placement.flags |= IREE_HAL_BUFFER_PLACEMENT_FLAG_INDETERMINATE_LIFETIME;
  }
  return iree_hal_local_transient_buffer_create(
      placement, params, allocation_size, byte_length, queue->host_allocator,
      out_buffer);
}

iree_status_t iree_hal_vulkan_queue_submit_alloca(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_queue_alloca_plan_t allocation_plan,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_alloca_flags_t known_flags =
      IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME |
      IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(flags, ~known_flags)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue alloca flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status) && allocation_size == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue_alloca size must be non-zero");
  }
  if (iree_status_is_ok(status)) {
    switch (allocation_plan.strategy) {
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_POOL:
        if (!allocation_plan.pool) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Vulkan queue_alloca pool strategy requires a pool");
        }
        break;
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_SPARSE:
        if (!allocation_plan.allocator) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Vulkan queue_alloca sparse strategy requires an allocator");
        } else if (!iree_all_bits_set(queue->queue_flags,
                                      VK_QUEUE_SPARSE_BINDING_BIT)) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "Vulkan queue family %u does not support sparse binding",
              queue->queue_family_index);
        }
        break;
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_NONE:
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unrecognized Vulkan queue_alloca strategy "
                                  "%u",
                                  (uint32_t)allocation_plan.strategy);
        break;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_create_transient_buffer(
        queue, params, allocation_size, byte_length, flags, &buffer);
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->alloca.buffer = buffer;
    iree_hal_buffer_retain(buffer);
    submission->alloca.strategy = allocation_plan.strategy;
    submission->alloca.allocator = allocation_plan.allocator;
    submission->alloca.pool = allocation_plan.pool;
    submission->alloca.params = params;
    submission->alloca.allocation_size = allocation_size;
    submission->alloca.flags = flags;
    submission->alloca.reserve_flags =
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER;
    submission->alloca.memory_wait_kind =
        IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    if (wait_semaphore_list.count > 0) {
      status =
          iree_hal_vulkan_queue_try_stage_alloca_backing_now(queue, submission);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
    buffer = NULL;
  }
  iree_hal_buffer_release(buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_dealloca(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_dealloca_flags_t known_flags =
      IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(flags, ~known_flags)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue dealloca flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_local_transient_buffer_isa(buffer)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue_dealloca buffer was not returned "
                              "by Vulkan queue_alloca");
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_local_transient_buffer_begin_dealloca(buffer)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan transient buffer has already been queued for deallocation");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA,
        (iree_hal_host_call_t){0}, /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->dealloca.buffer = buffer;
    iree_hal_buffer_retain(buffer);
    submission->dealloca.flags = flags;
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    iree_hal_local_transient_buffer_abort_dealloca(buffer);
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  } else if (!iree_status_is_ok(status) &&
             iree_hal_local_transient_buffer_isa(buffer)) {
    iree_hal_local_transient_buffer_abort_dealloca(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_sparse_bind(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, VkBuffer buffer,
    iree_host_size_t bind_count, const VkSparseMemoryBind* binds) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)bind_count);

  iree_status_t status = iree_ok_status();
  if (!buffer) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan sparse bind buffer is NULL");
  }
  if (iree_status_is_ok(status) && !binds) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan sparse bind array is NULL");
  }
  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(queue->queue_flags, VK_QUEUE_SPARSE_BINDING_BIT)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue family %u does not support sparse binding",
        queue->queue_family_index);
  }
  if (iree_status_is_ok(status) && bind_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan sparse bind submission must contain at least one bind");
  }
  if (iree_status_is_ok(status) && bind_count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many Vulkan sparse buffer binds");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  VkSparseMemoryBind* copied_binds = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t bind_storage_size = 0;
    if (iree_host_size_checked_mul(bind_count, sizeof(*binds),
                                   &bind_storage_size)) {
      status = iree_allocator_malloc(queue->host_allocator, bind_storage_size,
                                     (void**)&copied_binds);
    } else {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "Vulkan sparse bind array overflows host storage");
    }
  }
  if (iree_status_is_ok(status)) {
    memcpy(copied_binds, binds, bind_count * sizeof(*binds));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND,
        (iree_hal_host_call_t){0}, /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->sparse_bind.buffer = buffer;
    submission->sparse_bind.binds = copied_binds;
    submission->sparse_bind.bind_count = (uint32_t)bind_count;
    copied_binds = NULL;
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }
  iree_allocator_free(queue->host_allocator, copied_binds);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_fill(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_FILL_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported Vulkan queue fill flags: 0x%" PRIx64,
                              flags);
  }
  if (iree_status_is_ok(status) && !pattern) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue fill pattern is NULL");
  }
  if (iree_status_is_ok(status) && pattern_length != 1 && pattern_length != 2 &&
      pattern_length != 4) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue fill pattern length must be 1, 2, or 4 bytes "
        "(got %" PRIhsz ")",
        pattern_length);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(target_buffer),
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->fill.target_buffer = target_buffer;
    iree_hal_buffer_retain(target_buffer);
    submission->fill.target_offset = target_offset;
    submission->fill.length = length;
    memset(submission->fill.pattern, 0, sizeof(submission->fill.pattern));
    memcpy(submission->fill.pattern, pattern, pattern_length);
    submission->fill.pattern_length = pattern_length;
    submission->fill.flags = flags;
  }
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_validate_recordable_backing(
        target_buffer, IREE_SV("fill target"));
  }
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_allocate_native_command_buffer(
        queue, &submission->native_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_profile_prepare_native_timestamps(
          queue, submission);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_record_fill_native(queue, submission);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_update(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_UPDATE_FLAG_NONE) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue update flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status) && length > IREE_HOST_SIZE_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan queue update length exceeds host size");
  }
  if (iree_status_is_ok(status) &&
      source_offset > IREE_HOST_SIZE_MAX - (iree_host_size_t)length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan queue update source range exceeds host "
                              "size");
  }
  if (iree_status_is_ok(status) && length > 0 && !source_buffer) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "queue update source buffer must be non-null");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(target_buffer),
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->update.target_buffer = target_buffer;
    iree_hal_buffer_retain(target_buffer);
    submission->update.target_offset = target_offset;
    submission->update.length = length;
    submission->update.flags = flags;
    if (length > 0) {
      status =
          iree_allocator_malloc(queue->host_allocator, (iree_host_size_t)length,
                                &submission->update.source_data);
      if (iree_status_is_ok(status)) {
        memcpy(submission->update.source_data,
               (const uint8_t*)source_buffer + source_offset,
               (iree_host_size_t)length);
      }
    }
  }
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_validate_recordable_backing(
        target_buffer, IREE_SV("update target"));
  }
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_allocate_native_command_buffer(
        queue, &submission->native_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_profile_prepare_native_timestamps(
          queue, submission);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_record_update_native(queue, submission);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_submit_copy_with_action(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_vulkan_queue_completion_action_t completion_action) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_COPY_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported Vulkan queue copy flags: 0x%" PRIx64,
                              flags);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(source_buffer, source_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(source_buffer),
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(target_buffer),
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->copy.source_buffer = source_buffer;
    iree_hal_buffer_retain(source_buffer);
    submission->copy.source_offset = source_offset;
    submission->copy.target_buffer = target_buffer;
    iree_hal_buffer_retain(target_buffer);
    submission->copy.target_offset = target_offset;
    submission->copy.length = length;
    submission->copy.flags = flags;
    iree_hal_vulkan_queue_set_completion_action(submission, completion_action);
  }
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_validate_recordable_backing(
        source_buffer, IREE_SV("copy source"));
  }
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_validate_recordable_backing(
        target_buffer, IREE_SV("copy target"));
  }
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_allocate_native_command_buffer(
        queue, &submission->native_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_profile_prepare_native_timestamps(
          queue, submission);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_record_copy_native(queue, submission);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_copy(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  return iree_hal_vulkan_queue_submit_copy_with_action(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_buffer, target_offset, length, flags,
      iree_hal_vulkan_queue_completion_action_null());
}

static iree_status_t iree_hal_vulkan_queue_validate_file_range(
    iree_hal_file_t* file, uint64_t offset, iree_device_size_t length,
    iree_string_view_t operation) {
  const uint64_t file_length = iree_hal_file_length(file);
  if (file_length == 0) return iree_ok_status();
  if (length > UINT64_MAX - offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "%.*s range [%" PRIu64 ", overflow) exceeds file length %" PRIu64,
        (int)operation.size, operation.data, offset, file_length);
  }
  const uint64_t end_offset = offset + (uint64_t)length;
  if (end_offset > file_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "%.*s range [%" PRIu64 ", %" PRIu64 ") exceeds file length %" PRIu64,
        (int)operation.size, operation.data, offset, end_offset, file_length);
  }
  return iree_ok_status();
}

typedef enum iree_hal_vulkan_staged_transfer_kind_e {
  // File or host bytes flow through the upload ring into the target buffer.
  IREE_HAL_VULKAN_STAGED_TRANSFER_READ = 0,

  // Source buffer bytes flow through the download ring into the file or host.
  IREE_HAL_VULKAN_STAGED_TRANSFER_WRITE = 1,
} iree_hal_vulkan_staged_transfer_kind_t;

typedef struct iree_hal_vulkan_staged_transfer_t
    iree_hal_vulkan_staged_transfer_t;

typedef struct iree_hal_vulkan_staged_transfer_chunk_t {
  // Owning staged transfer.
  iree_hal_vulkan_staged_transfer_t* transfer;

  // Staging slot owned while this chunk is active.
  iree_hal_vulkan_queue_staging_slot_t* slot;

  // Byte offset from the beginning of the transfer.
  iree_device_size_t transfer_offset;

  // Byte length covered by this chunk.
  iree_device_size_t length;
} iree_hal_vulkan_staged_transfer_chunk_t;

struct iree_hal_vulkan_staged_transfer_t {
  // Resource retained by queue completion actions and ring waiters.
  iree_hal_resource_t resource;

  // Host allocator used for transfer metadata and signal-list storage.
  iree_allocator_t host_allocator;

  // Serializes transfer counters and terminal status ownership.
  iree_slim_mutex_t mutex;

  // Queue owning the staging rings and internal copy submissions.
  iree_hal_vulkan_queue_t* queue;

  // Staging ring used by this transfer.
  iree_hal_vulkan_queue_staging_ring_t* ring;

  // File being read or written.
  iree_hal_file_t* file;

  // Host memory span for memory-file transfers.
  iree_byte_span_t file_contents;

  // User buffer being copied to or from.
  iree_hal_buffer_t* buffer;

  // File byte offset for the first requested byte.
  uint64_t file_offset;

  // User buffer byte offset for the first requested byte.
  iree_device_size_t buffer_offset;

  // Total requested transfer length.
  iree_device_size_t requested_length;

  // Number of bytes assigned to chunks.
  iree_device_size_t submitted_length;

  // Number of bytes fully transferred through all stages.
  iree_device_size_t completed_length;

  // Number of chunks currently owning a staging slot or in-flight copy.
  uint32_t active_chunk_count;

  // Number of chunk records in |chunks|.
  uint32_t chunk_count;

  // Direction of this transfer.
  iree_hal_vulkan_staged_transfer_kind_t kind;

  // Whether terminal completion has started.
  bool finishing;

  // Owned first failure status, or OK if no failure has occurred.
  iree_status_t failure_status;

  // Waiter queued when all staging slots are temporarily unavailable.
  iree_hal_vulkan_queue_staging_waiter_t slot_waiter;

  // Cloned signal list published after the transfer completes.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Chunk records used to pipeline host copies and GPU copies.
  iree_hal_vulkan_staged_transfer_chunk_t* chunks;
};

static void iree_hal_vulkan_staged_transfer_pump(
    iree_hal_vulkan_staged_transfer_t* transfer);

static void iree_hal_vulkan_staged_transfer_try_finish(
    iree_hal_vulkan_staged_transfer_t* transfer);

static void iree_hal_vulkan_staged_transfer_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_vulkan_staged_transfer_t* transfer =
      (iree_hal_vulkan_staged_transfer_t*)resource;
  if (!iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                                 transfer->host_allocator);
  }
  iree_hal_buffer_release(transfer->buffer);
  iree_hal_file_release(transfer->file);
  iree_slim_mutex_deinitialize(&transfer->mutex);
  iree_allocator_free(transfer->host_allocator, transfer);
}

static const iree_hal_resource_vtable_t iree_hal_vulkan_staged_transfer_vtable =
    {
        .destroy = iree_hal_vulkan_staged_transfer_destroy,
};

static void iree_hal_vulkan_staged_transfer_record_failure(
    iree_hal_vulkan_staged_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  iree_slim_mutex_lock(&transfer->mutex);
  if (iree_status_is_ok(transfer->failure_status)) {
    transfer->failure_status = status;
    status = iree_ok_status();
  }
  iree_slim_mutex_unlock(&transfer->mutex);
  iree_status_free(status);
}

static void iree_hal_vulkan_staged_transfer_fail_signals(
    iree_hal_vulkan_staged_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  if (iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    iree_status_free(status);
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
}

static iree_status_t iree_hal_vulkan_staged_transfer_submit_signal_barrier(
    iree_hal_vulkan_staged_transfer_t* transfer) {
  if (iree_hal_semaphore_list_is_empty(transfer->signal_semaphore_list)) {
    return iree_ok_status();
  }
  return iree_hal_vulkan_queue_submit_barrier(transfer->queue,
                                              iree_hal_semaphore_list_empty(),
                                              transfer->signal_semaphore_list);
}

static void iree_hal_vulkan_staged_transfer_complete(
    iree_hal_vulkan_staged_transfer_t* transfer, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_staged_transfer_submit_signal_barrier(transfer);
  }
  iree_hal_vulkan_staged_transfer_fail_signals(transfer, status);
  iree_hal_resource_release(&transfer->resource);
}

static iree_hal_vulkan_staged_transfer_chunk_t*
iree_hal_vulkan_staged_transfer_find_idle_chunk(
    iree_hal_vulkan_staged_transfer_t* transfer) {
  for (uint32_t i = 0; i < transfer->chunk_count; ++i) {
    if (!transfer->chunks[i].slot) return &transfer->chunks[i];
  }
  return NULL;
}

static void iree_hal_vulkan_staged_transfer_slot_available(void* user_data) {
  iree_hal_vulkan_staged_transfer_t* transfer =
      (iree_hal_vulkan_staged_transfer_t*)user_data;
  iree_hal_vulkan_staged_transfer_pump(transfer);
  iree_hal_vulkan_staged_transfer_try_finish(transfer);
}

static void iree_hal_vulkan_staged_transfer_try_finish(
    iree_hal_vulkan_staged_transfer_t* transfer) {
  bool should_complete = false;
  bool should_release_waiter_ref = false;
  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&transfer->mutex);
  const bool has_failure = !iree_status_is_ok(transfer->failure_status);
  const bool is_complete =
      transfer->completed_length == transfer->requested_length;
  if (!transfer->finishing && transfer->active_chunk_count == 0 &&
      (has_failure || is_complete)) {
    transfer->finishing = true;
    status = transfer->failure_status;
    transfer->failure_status = iree_ok_status();
    should_complete = true;
  }
  iree_slim_mutex_unlock(&transfer->mutex);

  if (should_complete && iree_hal_vulkan_queue_staging_ring_cancel_waiter(
                             transfer->ring, &transfer->slot_waiter)) {
    should_release_waiter_ref = true;
  }
  if (should_release_waiter_ref) {
    iree_hal_resource_release(&transfer->resource);
  }
  if (should_complete) {
    iree_hal_vulkan_staged_transfer_complete(transfer, status);
  }
}

static void iree_hal_vulkan_staged_transfer_chunk_finish(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk, bool did_transfer_bytes) {
  iree_hal_vulkan_staged_transfer_t* transfer = chunk->transfer;
  iree_hal_vulkan_queue_staging_slot_t* slot = chunk->slot;
  iree_slim_mutex_lock(&transfer->mutex);
  if (did_transfer_bytes) {
    transfer->completed_length += chunk->length;
  }
  chunk->slot = NULL;
  chunk->transfer_offset = 0;
  chunk->length = 0;
  --transfer->active_chunk_count;
  iree_slim_mutex_unlock(&transfer->mutex);

  iree_hal_vulkan_queue_staging_ring_release(transfer->ring, slot);
  iree_hal_vulkan_staged_transfer_pump(transfer);
  iree_hal_vulkan_staged_transfer_try_finish(transfer);
}

static void iree_hal_vulkan_staged_transfer_chunk_fail(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk, iree_status_t status) {
  iree_hal_vulkan_staged_transfer_record_failure(chunk->transfer, status);
  iree_hal_vulkan_staged_transfer_chunk_finish(chunk,
                                               /*did_transfer_bytes=*/false);
}

static void iree_hal_vulkan_staged_transfer_copy_complete(
    void* user_data, iree_status_t completion_status) {
  iree_hal_vulkan_staged_transfer_chunk_t* chunk =
      (iree_hal_vulkan_staged_transfer_chunk_t*)user_data;
  iree_hal_vulkan_staged_transfer_t* transfer = chunk->transfer;
  iree_status_t status = iree_status_clone(completion_status);
  if (iree_status_is_ok(status) &&
      transfer->kind == IREE_HAL_VULKAN_STAGED_TRANSFER_WRITE) {
    status = iree_hal_buffer_mapping_invalidate_range(
        &transfer->ring->mapping, chunk->slot->buffer_offset, chunk->length);
    if (iree_status_is_ok(status)) {
      memcpy(transfer->file_contents.data + transfer->file_offset +
                 chunk->transfer_offset,
             chunk->slot->host_span.data, (iree_host_size_t)chunk->length);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_staged_transfer_chunk_finish(chunk,
                                                 /*did_transfer_bytes=*/true);
  } else {
    iree_hal_vulkan_staged_transfer_chunk_fail(chunk, status);
  }
}

static iree_status_t iree_hal_vulkan_staged_transfer_submit_copy(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk) {
  iree_hal_vulkan_staged_transfer_t* transfer = chunk->transfer;
  iree_status_t status = iree_ok_status();
  iree_hal_buffer_t* source_buffer = NULL;
  iree_device_size_t source_offset = 0;
  iree_hal_buffer_t* target_buffer = NULL;
  iree_device_size_t target_offset = 0;
  if (transfer->kind == IREE_HAL_VULKAN_STAGED_TRANSFER_READ) {
    memcpy(chunk->slot->host_span.data,
           transfer->file_contents.data + transfer->file_offset +
               chunk->transfer_offset,
           (iree_host_size_t)chunk->length);
    status = iree_hal_buffer_mapping_flush_range(
        &transfer->ring->mapping, chunk->slot->buffer_offset, chunk->length);
    source_buffer = transfer->ring->buffer;
    source_offset = chunk->slot->buffer_offset;
    target_buffer = transfer->buffer;
    target_offset = transfer->buffer_offset + chunk->transfer_offset;
  } else {
    source_buffer = transfer->buffer;
    source_offset = transfer->buffer_offset + chunk->transfer_offset;
    target_buffer = transfer->ring->buffer;
    target_offset = chunk->slot->buffer_offset;
  }
  if (!iree_status_is_ok(status)) return status;

  return iree_hal_vulkan_queue_submit_copy_with_action(
      transfer->queue, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), source_buffer, source_offset,
      target_buffer, target_offset, chunk->length, IREE_HAL_COPY_FLAG_NONE,
      (iree_hal_vulkan_queue_completion_action_t){
          .fn = iree_hal_vulkan_staged_transfer_copy_complete,
          .user_data = chunk,
          .resource = &transfer->resource,
      });
}

static void iree_hal_vulkan_staged_transfer_pump(
    iree_hal_vulkan_staged_transfer_t* transfer) {
  for (;;) {
    iree_slim_mutex_lock(&transfer->mutex);
    const bool can_submit_more =
        !transfer->finishing && iree_status_is_ok(transfer->failure_status) &&
        transfer->submitted_length < transfer->requested_length;
    iree_slim_mutex_unlock(&transfer->mutex);
    if (!can_submit_more) return;

    iree_hal_vulkan_queue_staging_slot_t* slot = NULL;
    if (!iree_hal_vulkan_queue_staging_ring_try_acquire(transfer->ring,
                                                        &slot)) {
      const bool queued = iree_hal_vulkan_queue_staging_ring_queue_waiter(
          transfer->ring, &transfer->slot_waiter,
          iree_hal_vulkan_staged_transfer_slot_available, transfer,
          &transfer->resource);
      if (queued) return;
      continue;
    }

    iree_hal_vulkan_staged_transfer_chunk_t* chunk = NULL;
    iree_slim_mutex_lock(&transfer->mutex);
    if (!transfer->finishing && iree_status_is_ok(transfer->failure_status) &&
        transfer->submitted_length < transfer->requested_length) {
      chunk = iree_hal_vulkan_staged_transfer_find_idle_chunk(transfer);
      if (chunk) {
        const iree_device_size_t remaining_length =
            transfer->requested_length - transfer->submitted_length;
        chunk->transfer = transfer;
        chunk->slot = slot;
        chunk->transfer_offset = transfer->submitted_length;
        chunk->length = iree_min(transfer->ring->slot_size, remaining_length);
        transfer->submitted_length += chunk->length;
        ++transfer->active_chunk_count;
      }
    }
    iree_slim_mutex_unlock(&transfer->mutex);

    if (!chunk) {
      iree_hal_vulkan_queue_staging_ring_release(transfer->ring, slot);
      return;
    }

    iree_status_t status = iree_hal_vulkan_staged_transfer_submit_copy(chunk);
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_staged_transfer_chunk_fail(chunk, status);
      return;
    }
  }
}

static void iree_hal_vulkan_staged_transfer_start(
    void* user_data, iree_status_t completion_status) {
  iree_hal_vulkan_staged_transfer_t* transfer =
      (iree_hal_vulkan_staged_transfer_t*)user_data;
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_staged_transfer_fail_signals(
        transfer, iree_status_clone(completion_status));
    return;
  }
  iree_hal_resource_retain(&transfer->resource);
  iree_hal_vulkan_staged_transfer_pump(transfer);
  iree_hal_vulkan_staged_transfer_try_finish(transfer);
}

static iree_status_t iree_hal_vulkan_staged_transfer_create(
    iree_hal_vulkan_queue_t* queue, iree_hal_vulkan_staged_transfer_kind_t kind,
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_staged_transfer_t** out_transfer) {
  *out_transfer = NULL;
  if (!iree_hal_memory_file_isa(file)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan staged transfers currently require a memory file");
  }
  if (file_offset > IREE_HOST_SIZE_MAX ||
      length > (iree_device_size_t)(IREE_HOST_SIZE_MAX - file_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan staged transfer host file range exceeds addressable size");
  }

  iree_hal_vulkan_queue_staging_ring_t* ring =
      kind == IREE_HAL_VULKAN_STAGED_TRANSFER_READ
          ? queue->upload_staging_ring
          : queue->download_staging_ring;
  if (!ring) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue staging ring is not initialized");
  }

  iree_byte_span_t file_contents = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_memory_file_contents(file, &file_contents));

  iree_host_size_t chunks_size = 0;
  if (!iree_host_size_checked_mul(
          ring->slot_count, sizeof(iree_hal_vulkan_staged_transfer_chunk_t),
          &chunks_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staged transfer chunk table overflows");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_vulkan_staged_transfer_t),
                                  chunks_size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staged transfer allocation overflows");
  }

  iree_hal_vulkan_staged_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator, total_size,
                                             (void**)&transfer));
  memset(transfer, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_vulkan_staged_transfer_vtable,
                               &transfer->resource);
  transfer->host_allocator = queue->host_allocator;
  iree_slim_mutex_initialize(&transfer->mutex);
  transfer->queue = queue;
  transfer->ring = ring;
  transfer->file = file;
  iree_hal_file_retain(transfer->file);
  transfer->file_contents = file_contents;
  transfer->buffer = buffer;
  iree_hal_buffer_retain(transfer->buffer);
  transfer->file_offset = file_offset;
  transfer->buffer_offset = buffer_offset;
  transfer->requested_length = length;
  transfer->kind = kind;
  transfer->chunk_count = ring->slot_count;
  transfer->chunks = (iree_hal_vulkan_staged_transfer_chunk_t*)(transfer + 1);
  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, transfer->host_allocator,
      &transfer->signal_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_transfer = transfer;
  } else {
    iree_hal_resource_release(&transfer->resource);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_queue_submit_staged_transfer(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_staged_transfer_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  iree_hal_vulkan_staged_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_staged_transfer_create(
      queue, kind, file, file_offset, buffer, buffer_offset, length,
      signal_semaphore_list, &transfer));

  iree_status_t status = iree_hal_vulkan_queue_submit_barrier_with_action(
      queue, wait_semaphore_list, iree_hal_semaphore_list_empty(),
      (iree_hal_vulkan_queue_completion_action_t){
          .fn = iree_hal_vulkan_staged_transfer_start,
          .user_data = transfer,
          .resource = &transfer->resource,
      });
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_staged_transfer_fail_signals(transfer,
                                                 iree_status_clone(status));
  }
  iree_hal_resource_release(&transfer->resource);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_read(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(source_file);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_READ_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported Vulkan queue read flags: 0x%" PRIx64,
                              flags);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(target_buffer),
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_file_range(
        source_file, source_offset, length, IREE_SV("read"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }
  if (iree_status_is_ok(status) && length == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_queue_submit_barrier(queue, wait_semaphore_list,
                                                signal_semaphore_list);
  }

  iree_hal_buffer_t* source_storage_buffer =
      iree_status_is_ok(status) ? iree_hal_file_storage_buffer(source_file)
                                : NULL;
  bool source_storage_is_native = false;
  if (iree_status_is_ok(status) && source_storage_buffer) {
    status = iree_hal_vulkan_queue_buffer_is_native(source_storage_buffer,
                                                    &source_storage_is_native);
  }
  bool target_is_native = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_buffer_is_native(target_buffer,
                                                    &target_is_native);
  }
  if (iree_status_is_ok(status) && source_storage_is_native &&
      target_is_native) {
    if (source_offset > IREE_DEVICE_SIZE_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "read source offset %" PRIu64 " exceeds device size", source_offset);
    } else {
      IREE_TRACE_ZONE_END(z0);
      return iree_hal_vulkan_queue_submit_copy(
          queue, wait_semaphore_list, signal_semaphore_list,
          source_storage_buffer, (iree_device_size_t)source_offset,
          target_buffer, target_offset, length, IREE_HAL_COPY_FLAG_NONE);
    }
  }
  if (iree_status_is_ok(status) && iree_hal_memory_file_isa(source_file) &&
      target_is_native) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_queue_submit_staged_transfer(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_STAGED_TRANSFER_READ, source_file, source_offset,
        target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan queue read requires native device-visible file storage and "
        "target buffer storage");
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_write(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_file);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_WRITE_FLAG_NONE) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue write flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_validate_access(target_file,
                                           IREE_HAL_MEMORY_ACCESS_WRITE);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(source_buffer, source_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(source_buffer),
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_file_range(
        target_file, target_offset, length, IREE_SV("write"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }
  if (iree_status_is_ok(status) && length == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_queue_submit_barrier(queue, wait_semaphore_list,
                                                signal_semaphore_list);
  }

  bool source_is_native = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_buffer_is_native(source_buffer,
                                                    &source_is_native);
  }

  iree_hal_buffer_t* target_storage_buffer =
      iree_status_is_ok(status) ? iree_hal_file_storage_buffer(target_file)
                                : NULL;
  bool target_storage_is_native = false;
  if (iree_status_is_ok(status) && target_storage_buffer) {
    status = iree_hal_vulkan_queue_buffer_is_native(target_storage_buffer,
                                                    &target_storage_is_native);
  }
  if (iree_status_is_ok(status) && source_is_native &&
      target_storage_is_native) {
    if (target_offset > IREE_DEVICE_SIZE_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "write target offset %" PRIu64 " exceeds device size", target_offset);
    } else {
      IREE_TRACE_ZONE_END(z0);
      return iree_hal_vulkan_queue_submit_copy(
          queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
          source_offset, target_storage_buffer,
          (iree_device_size_t)target_offset, length, IREE_HAL_COPY_FLAG_NONE);
    }
  }
  if (iree_status_is_ok(status) && source_is_native &&
      iree_hal_memory_file_isa(target_file)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_queue_submit_staged_transfer(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_STAGED_TRANSFER_WRITE, target_file, target_offset,
        source_buffer, source_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan queue write requires native device-visible source buffer and "
        "file storage");
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_prepare_native_execute_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_hal_buffer_binding_table_t binding_table) {
  if (!iree_hal_vulkan_command_buffer_has_native_commands(
          submission->execute.command_buffer)) {
    return iree_ok_status();
  }
  iree_status_t status = iree_hal_vulkan_queue_allocate_native_command_buffer(
      queue, &submission->native_command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_profile_prepare_native_timestamps(
        queue, submission);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_command_buffer_profile_marker_t profile_marker = {
        .query_pool = submission->profile.query_pool,
        .query_count = submission->profile.query_count,
        .queue_start_query = submission->profile.queue_start_query,
        .queue_end_query = submission->profile.queue_end_query,
        .dispatch_base_query = submission->profile.dispatch_base_query,
        .dispatch_query_count = submission->profile.dispatch_query_count,
    };
    status = iree_hal_vulkan_command_buffer_record_native(
        submission->execute.command_buffer, &queue->syms, queue->logical_device,
        queue->builtins, submission->native_command_buffer, binding_table,
        profile_marker.query_pool ? &profile_marker : NULL,
        queue->host_allocator, &submission->native_descriptor_pool);
  }
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_execute(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags,
    iree_hal_profile_queue_event_type_t queue_event_type) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_execute_flags_t known_flags =
      IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(flags, ~known_flags)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue execute flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status) &&
      queue_event_type == IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue execute requires a concrete profile event type");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }
  if (iree_status_is_ok(status) &&
      binding_table.count < command_buffer->binding_count) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "indirect Vulkan command buffer requires at least %u bindings but only "
        "%" PRIhsz " were provided",
        command_buffer->binding_count, binding_table.count);
  }
  if (iree_status_is_ok(status) && command_buffer->binding_count != 0 &&
      !binding_table.bindings) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "indirect Vulkan command buffer binding table storage is NULL for %u "
        "bindings",
        command_buffer->binding_count);
  }
  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE,
        (iree_hal_host_call_t){0}, /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->execute.command_buffer = command_buffer;
    iree_hal_command_buffer_retain(command_buffer);
    submission->execute.flags = flags;
    submission->profile.type = queue_event_type;
    submission->execute.binding_table_count = command_buffer->binding_count;
    if (command_buffer->binding_count != 0) {
      iree_host_size_t binding_table_size = 0;
      if (!iree_host_size_checked_mul(command_buffer->binding_count,
                                      sizeof(*binding_table.bindings),
                                      &binding_table_size)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "Vulkan queue execute binding table capture is too large");
      }
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc(
            queue->host_allocator, binding_table_size,
            (void**)&submission->execute.binding_table_bindings);
      }
      if (iree_status_is_ok(status)) {
        memcpy(submission->execute.binding_table_bindings,
               binding_table.bindings, binding_table_size);
        if (!iree_any_bit_set(
                flags, IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
          for (iree_host_size_t i = 0; i < command_buffer->binding_count; ++i) {
            iree_hal_buffer_retain(
                submission->execute.binding_table_bindings[i].buffer);
          }
        }
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_binding_table_t captured_binding_table = {
        .count = submission->execute.binding_table_count,
        .bindings = submission->execute.binding_table_bindings,
    };
    const uint64_t command_buffer_id =
        iree_hal_vulkan_queue_profile_command_buffer_id(submission);
    status = iree_hal_vulkan_command_buffer_record_profile_metadata(
        command_buffer, queue->profile_recorder, submission->profile.scope,
        command_buffer_id);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_prepare_native_execute_submission(
          queue, submission, captured_binding_table);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_host_call(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_hal_vulkan_queue_validate_host_call(call, args, flags);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }
  if (iree_status_is_ok(status) && wait_semaphore_list.count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many Vulkan queue wait semaphores");
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL, call, args, flags,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_host_size_t iree_hal_vulkan_queue_drain_completions(
    iree_hal_vulkan_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &queue->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure_status)) {
    iree_hal_vulkan_queue_fail_pending_submissions(
        queue, iree_status_clone(failure_status));
    IREE_TRACE_ZONE_END(z0);
    return 0;
  }

  uint64_t completed_epoch = 0;
  VkResult result = iree_vkGetSemaphoreCounterValue_raw(
      &queue->syms, queue->logical_device, queue->epoch_semaphore,
      &completed_epoch);
  if (result != VK_SUCCESS) {
    iree_status_t status = iree_status_from_vk_result(
        __FILE__, __LINE__, result, "vkGetSemaphoreCounterValue");
    iree_status_t stored_status =
        iree_hal_vulkan_queue_store_error(queue, status);
    iree_hal_vulkan_queue_fail_pending_submissions(queue, stored_status);
    IREE_TRACE_ZONE_END(z0);
    return 0;
  }

  iree_hal_vulkan_queue_pending_submission_t* completed_head = NULL;
  iree_slim_mutex_lock(&queue->submission_mutex);
  completed_head =
      iree_hal_vulkan_queue_pop_completed_submissions(queue, completed_epoch);
  iree_slim_mutex_unlock(&queue->submission_mutex);

  iree_host_size_t completed_count = 0;
  while (completed_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = completed_head->next;
    completed_head->next = NULL;
    iree_hal_vulkan_queue_complete_submission(queue, completed_head,
                                              iree_ok_status());
    iree_hal_vulkan_queue_pending_submission_destroy(queue, completed_head);
    completed_head = next;
    ++completed_count;
  }

  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)completed_count);
  IREE_TRACE_ZONE_END(z0);
  return completed_count;
}
