// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/queue.h"

#include <stdio.h>
#include <string.h>

#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/command_buffer.h"
#include "iree/hal/drivers/vulkan/executable.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/local/transient_buffer.h"
#include "iree/hal/utils/memory_file.h"

#define IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_SLOT_ABSENT UINT32_MAX
#define IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_SLOT_RESERVED UINT64_MAX
#define IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX 2
#define IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT UINT32_MAX
#define IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_RESERVED UINT64_MAX
#define IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY 64
#define IREE_HAL_VULKAN_QUEUE_NATIVE_REPLAY_OWNER_RESERVED UINT64_MAX
#define IREE_HAL_VULKAN_QUEUE_NATIVE_DESCRIPTOR_BLOCK_SET_CAPACITY 4096
#define IREE_HAL_VULKAN_QUEUE_BDA_PUBLICATION_BLOCK_SIZE (64ull * 1024ull)
#define IREE_HAL_VULKAN_QUEUE_BDA_PUBLICATION_ALIGNMENT 8ull
#define IREE_HAL_VULKAN_QUEUE_TIMESTAMP_QUERY_BLOCK_CAPACITY (64 * 1024)
#define IREE_HAL_VULKAN_QUEUE_DISPATCH_INLINE_SET_CAPACITY 8
#define IREE_HAL_VULKAN_QUEUE_DISPATCH_INLINE_BINDING_CAPACITY 16

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

struct iree_hal_vulkan_queue_command_buffer_block_t {
  // Next block in the queue-owned command buffer cache.
  iree_hal_vulkan_queue_command_buffer_block_t* next;

  // Command pool backing command_buffers.
  VkCommandPool pool;

  // Primary command buffers leased by one-shot queue submissions.
  VkCommandBuffer
      command_buffers[IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY];

  // Owning queue epoch for each command buffer, 0 when free, or RESERVED.
  uint64_t owner_epochs[IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY];

  // Whether a free command buffer must be reset before its next recording.
  bool needs_reset[IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY];

  // Next slot considered when acquiring from this block.
  uint32_t next_slot;

  // Number of command buffers whose |owner_epochs| entry is 0.
  uint32_t free_count;
};

typedef struct iree_hal_vulkan_queue_command_buffer_lease_t {
  // Command buffer block containing the leased command buffer.
  iree_hal_vulkan_queue_command_buffer_block_t* block;

  // Command buffer slot within |block|.
  uint32_t slot;
} iree_hal_vulkan_queue_command_buffer_lease_t;

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

struct iree_hal_vulkan_queue_native_descriptor_block_t {
  // Next block in the queue-owned native descriptor pool cache.
  iree_hal_vulkan_queue_native_descriptor_block_t* next;

  // Descriptor pool backing all allocations from this block.
  VkDescriptorPool pool;

  // Total descriptor capacity before the pool must be reset.
  iree_hal_vulkan_command_buffer_descriptor_requirements_t capacity;

  // Descriptor capacity allocated since the last pool reset.
  iree_hal_vulkan_command_buffer_descriptor_requirements_t allocated;

  // Number of submissions still using descriptor sets from this block.
  uint32_t active_lease_count;
};

typedef struct iree_hal_vulkan_queue_native_descriptor_lease_t {
  // Native descriptor block containing this submission's descriptor pool.
  iree_hal_vulkan_queue_native_descriptor_block_t* block;
} iree_hal_vulkan_queue_native_descriptor_lease_t;

struct iree_hal_vulkan_queue_bda_publication_block_t {
  // Next block in the queue-owned BDA publication cache.
  iree_hal_vulkan_queue_bda_publication_block_t* next;

  // BDA-capable host-visible buffer backing published tables.
  iree_hal_buffer_t* buffer;

  // Persistent host mapping of |buffer|.
  iree_hal_buffer_mapping_t mapping;

  // Device address of |buffer| byte offset zero.
  VkDeviceAddress device_address;

  // Total byte capacity of |buffer|.
  iree_device_size_t capacity;

  // Bump allocation position within |buffer|.
  iree_device_size_t allocated_length;

  // Number of submissions retaining ranges from this block.
  uint32_t active_lease_count;
};

typedef struct iree_hal_vulkan_queue_bda_publication_lease_t {
  // BDA publication block containing this submission's range.
  iree_hal_vulkan_queue_bda_publication_block_t* block;

  // Byte offset of the leased range within |block|.
  iree_device_size_t offset;

  // Byte length of the leased range.
  iree_device_size_t length;
} iree_hal_vulkan_queue_bda_publication_lease_t;

struct iree_hal_vulkan_queue_timestamp_query_block_t {
  // Next block in the queue-owned timestamp query cache.
  iree_hal_vulkan_queue_timestamp_query_block_t* next;

  // Query pool backing |query_values|.
  VkQueryPool pool;

  // Host storage used for retired timestamp query results.
  uint64_t* query_values;

  // Total query slot capacity of |pool|.
  uint32_t capacity;

  // Bump allocation position within |pool|.
  uint32_t allocated_count;

  // Number of submissions retaining ranges from this block.
  uint32_t active_lease_count;
};

typedef struct iree_hal_vulkan_queue_timestamp_query_lease_t {
  // Timestamp query block containing the leased range.
  iree_hal_vulkan_queue_timestamp_query_block_t* block;

  // First query slot in |block|.
  uint32_t first_query;

  // Number of query slots in the lease.
  uint32_t query_count;
} iree_hal_vulkan_queue_timestamp_query_lease_t;

struct iree_hal_vulkan_queue_native_replay_t {
  // Next native replay instance in the queue-owned cache.
  iree_hal_vulkan_queue_native_replay_t* next;

  // HAL command buffer whose command program was recorded.
  iree_hal_command_buffer_t* command_buffer;

  // Recorded native Vulkan command buffer.
  VkCommandBuffer native_command_buffer;

  // Command-buffer cache lease held for native_command_buffer.
  iree_hal_vulkan_queue_command_buffer_lease_t command_buffer_lease;

  // Stable BDA publication storage baked into native_command_buffer.
  iree_hal_vulkan_queue_bda_publication_lease_t bda_publication_lease;

  // Last resolved BDA slot snapshot for this replay instance.
  iree_hal_vulkan_command_buffer_bda_binding_slot_t* bda_binding_slots;

  // Number of entries in bda_binding_slots.
  iree_host_size_t bda_binding_slot_count;

  // Whether bda_binding_slots matches the current publication contents.
  bool bda_binding_slots_valid;

  // Queue epoch owning this replay instance, 0 when free, or RESERVED.
  uint64_t owner_epoch;
};

typedef iree_status_t(
    IREE_API_PTR* iree_hal_vulkan_queue_record_native_submission_fn_t)(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission);

typedef enum iree_hal_vulkan_queue_submission_kind_e {
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER = 0,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL = 1,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL = 2,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE = 3,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY = 4,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE = 5,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DISPATCH = 6,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA = 7,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA = 8,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND = 9,
} iree_hal_vulkan_queue_submission_kind_t;

typedef enum iree_hal_vulkan_queue_deferred_state_e {
  // Linked on the deferred list and waiting for software dependencies.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING = 0,

  // A dependency wait operation is being armed by the submitting thread.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_ARMING = 1,

  // A dependency callback owns promotion from deferred to ready.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PROMOTING = 2,

  // Linked on the ready list for native submission by the completion thread.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_READY = 3,

  // Cancellation owns the unsubmitted node after unlinking it from deferred.
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_CANCELLING = 4,
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

    // Timestamp query cache lease held by this submission.
    iree_hal_vulkan_queue_timestamp_query_lease_t timestamp_query_lease;

    // Query pool receiving device timestamps, or VK_NULL_HANDLE.
    VkQueryPool query_pool;

    // First query slot leased from query_pool.
    uint32_t first_query;

    // Number of query slots leased from query_pool.
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

  // Optional late native recorder invoked after wait dependencies resolve.
  iree_hal_vulkan_queue_record_native_submission_fn_t record_native_submission;

  // Command buffer cache lease held by native_command_buffer.
  iree_hal_vulkan_queue_command_buffer_lease_t native_command_buffer_lease;

  // Descriptor cache leases held by built-in queue operations.
  iree_hal_vulkan_queue_descriptor_lease_t native_descriptor_leases
      [IREE_HAL_VULKAN_QUEUE_BUILTIN_DESCRIPTOR_SET_COUNT_MAX];

  // Number of entries populated in native_descriptor_leases.
  uint32_t native_descriptor_slot_count;

  // Descriptor-pool cache lease held by native command recording.
  iree_hal_vulkan_queue_native_descriptor_lease_t native_descriptor_lease;

  // BDA publication cache lease held by native command recording.
  iree_hal_vulkan_queue_bda_publication_lease_t bda_publication_lease;

  // Cached native replay instance used by command-buffer execution.
  iree_hal_vulkan_queue_native_replay_t* native_replay;

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

  // Operation payload selected by kind.
  union {
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

      // Set to non-zero after the memory wait callback's final access
      // completes.
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

      // Binding table entries copied into the submission allocation tail.
      iree_hal_buffer_binding_t* binding_table_bindings;

      // Number of entries populated in binding_table_bindings.
      iree_host_size_t binding_table_count;

      // BDA slot-address cache copied into the submission allocation tail.
      iree_hal_vulkan_command_buffer_bda_binding_slot_t* bda_binding_slots;

      // Number of entries populated in bda_binding_slots.
      iree_host_size_t bda_binding_slot_count;

      // HAL execute flags captured from queue_execute.
      iree_hal_execute_flags_t flags;
    } execute;

    // Direct dispatch payload.
    struct {
      // Executable retained until dispatch completion.
      iree_hal_executable_t* executable;

      // Function ordinal captured from queue_dispatch.
      iree_hal_executable_function_t function_ordinal;

      // Dispatch workgroup configuration captured from queue_dispatch.
      iree_hal_dispatch_config_t config;

      // Push constant bytes copied into the submission allocation tail.
      void* constants_data;

      // Number of bytes in constants_data.
      iree_host_size_t constants_data_length;

      // Direct buffer bindings copied into the submission allocation tail.
      iree_hal_buffer_ref_t* bindings;

      // Number of entries populated in bindings.
      iree_host_size_t binding_count;

      // HAL dispatch flags captured from queue_dispatch.
      iree_hal_dispatch_flags_t flags;
    } dispatch;
  };
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
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
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

static bool iree_hal_vulkan_queue_native_descriptor_requirements_are_zero(
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t*
        requirements) {
  return requirements->set_count == 0 && requirements->sampler_count == 0 &&
         requirements->uniform_buffer_count == 0 &&
         requirements->storage_buffer_count == 0;
}

static bool iree_hal_vulkan_queue_native_descriptor_requirements_fit(
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t* capacity,
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t* allocated,
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t*
        requirements) {
  return allocated->set_count <= capacity->set_count &&
         allocated->sampler_count <= capacity->sampler_count &&
         allocated->uniform_buffer_count <= capacity->uniform_buffer_count &&
         allocated->storage_buffer_count <= capacity->storage_buffer_count &&
         requirements->set_count <=
             capacity->set_count - allocated->set_count &&
         requirements->sampler_count <=
             capacity->sampler_count - allocated->sampler_count &&
         requirements->uniform_buffer_count <=
             capacity->uniform_buffer_count - allocated->uniform_buffer_count &&
         requirements->storage_buffer_count <=
             capacity->storage_buffer_count - allocated->storage_buffer_count;
}

static void iree_hal_vulkan_queue_native_descriptor_requirements_add(
    iree_hal_vulkan_command_buffer_descriptor_requirements_t* inout_value,
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t* value) {
  inout_value->set_count = inout_value->set_count + value->set_count;
  inout_value->sampler_count =
      inout_value->sampler_count + value->sampler_count;
  inout_value->uniform_buffer_count =
      inout_value->uniform_buffer_count + value->uniform_buffer_count;
  inout_value->storage_buffer_count =
      inout_value->storage_buffer_count + value->storage_buffer_count;
}

static iree_status_t
iree_hal_vulkan_queue_calculate_dispatch_descriptor_requirements(
    const iree_hal_vulkan_pipeline_t* pipeline,
    iree_hal_vulkan_command_buffer_descriptor_requirements_t*
        out_requirements) {
  memset(out_requirements, 0, sizeof(*out_requirements));
  if (pipeline->push_descriptors.enabled) return iree_ok_status();
  out_requirements->set_count = pipeline->descriptor_requirements.set_count;
  out_requirements->sampler_count =
      pipeline->descriptor_requirements.sampler_count;
  out_requirements->uniform_buffer_count =
      pipeline->descriptor_requirements.uniform_buffer_count;
  out_requirements->storage_buffer_count =
      pipeline->descriptor_requirements.storage_buffer_count;
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_calculate_native_descriptor_block_capacity(
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t*
        requirements,
    iree_hal_vulkan_command_buffer_descriptor_requirements_t* out_capacity) {
  memset(out_capacity, 0, sizeof(*out_capacity));
  if (requirements->set_count == 0) return;

  uint32_t submission_capacity =
      IREE_HAL_VULKAN_QUEUE_NATIVE_DESCRIPTOR_BLOCK_SET_CAPACITY /
      requirements->set_count;
  if (submission_capacity == 0) submission_capacity = 1;
  if (requirements->uniform_buffer_count != 0) {
    submission_capacity = iree_min(
        submission_capacity, UINT32_MAX / requirements->uniform_buffer_count);
  }
  if (requirements->sampler_count != 0) {
    submission_capacity =
        iree_min(submission_capacity, UINT32_MAX / requirements->sampler_count);
  }
  if (requirements->storage_buffer_count != 0) {
    submission_capacity = iree_min(
        submission_capacity, UINT32_MAX / requirements->storage_buffer_count);
  }

  out_capacity->set_count = requirements->set_count * submission_capacity;
  out_capacity->sampler_count =
      requirements->sampler_count * submission_capacity;
  out_capacity->uniform_buffer_count =
      requirements->uniform_buffer_count * submission_capacity;
  out_capacity->storage_buffer_count =
      requirements->storage_buffer_count * submission_capacity;
}

static iree_status_t iree_hal_vulkan_queue_native_descriptor_block_create(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t*
        requirements,
    iree_hal_vulkan_queue_native_descriptor_block_t** out_block) {
  *out_block = NULL;
  iree_hal_vulkan_command_buffer_descriptor_requirements_t capacity;
  iree_hal_vulkan_queue_calculate_native_descriptor_block_capacity(requirements,
                                                                   &capacity);
  if (capacity.set_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan native descriptor block requires descriptor sets");
  }

  iree_hal_vulkan_queue_native_descriptor_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*block), (void**)&block));
  memset(block, 0, sizeof(*block));
  block->capacity = capacity;

  VkDescriptorPoolSize pool_sizes[3];
  uint32_t pool_size_count = 0;
  if (capacity.sampler_count != 0) {
    pool_sizes[pool_size_count++] = (VkDescriptorPoolSize){
        .type = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = capacity.sampler_count,
    };
  }
  if (capacity.uniform_buffer_count != 0) {
    pool_sizes[pool_size_count++] = (VkDescriptorPoolSize){
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = capacity.uniform_buffer_count,
    };
  }
  if (capacity.storage_buffer_count != 0) {
    pool_sizes[pool_size_count++] = (VkDescriptorPoolSize){
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = capacity.storage_buffer_count,
    };
  }
  VkDescriptorPoolCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = capacity.set_count,
      .poolSizeCount = pool_size_count,
      .pPoolSizes = pool_sizes,
  };
  iree_status_t status = iree_vkCreateDescriptorPool(
      IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device, &create_info,
      /*pAllocator=*/NULL, &block->pool);
  if (iree_status_is_ok(status)) {
    *out_block = block;
  } else {
    iree_allocator_free(queue->host_allocator, block);
  }
  return status;
}

static void iree_hal_vulkan_queue_native_descriptor_block_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_native_descriptor_block_t* block) {
  if (!block) return;
  if (block->pool) {
    iree_vkDestroyDescriptorPool(IREE_VULKAN_DEVICE(&queue->syms),
                                 queue->logical_device, block->pool,
                                 /*pAllocator=*/NULL);
  }
  iree_allocator_free(queue->host_allocator, block);
}

static void iree_hal_vulkan_queue_native_descriptor_cache_append_block(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_native_descriptor_block_t* block) {
  block->next = NULL;
  if (queue->native_descriptor_cache.tail) {
    queue->native_descriptor_cache.tail->next = block;
  } else {
    queue->native_descriptor_cache.head = block;
  }
  queue->native_descriptor_cache.tail = block;
  if (!queue->native_descriptor_cache.cursor) {
    queue->native_descriptor_cache.cursor = block;
  }
  queue->native_descriptor_cache.block_count =
      queue->native_descriptor_cache.block_count + 1;
}

static void iree_hal_vulkan_queue_native_descriptor_cache_deinitialize(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_native_descriptor_block_t* block =
      queue->native_descriptor_cache.head;
  while (block) {
    iree_hal_vulkan_queue_native_descriptor_block_t* next = block->next;
    iree_hal_vulkan_queue_native_descriptor_block_destroy(queue, block);
    block = next;
  }
  memset(&queue->native_descriptor_cache, 0,
         sizeof(queue->native_descriptor_cache));
}

static iree_status_t iree_hal_vulkan_queue_native_descriptor_block_reset(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_native_descriptor_block_t* block) {
  IREE_ASSERT(block->active_lease_count == 0,
              "cannot reset a native descriptor block with active leases");
  IREE_RETURN_IF_ERROR(iree_vkResetDescriptorPool(
      IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device, block->pool,
      /*flags=*/0));
  memset(&block->allocated, 0, sizeof(block->allocated));
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_queue_acquire_native_descriptor_pool_from_block(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_hal_vulkan_queue_native_descriptor_block_t* block,
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t*
        requirements,
    VkDescriptorPool* out_descriptor_pool) {
  iree_hal_vulkan_queue_native_descriptor_requirements_add(&block->allocated,
                                                           requirements);
  block->active_lease_count = block->active_lease_count + 1;
  submission->native_descriptor_lease.block = block;
  queue->native_descriptor_cache.cursor = block;
  *out_descriptor_pool = block->pool;
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_queue_try_acquire_native_descriptor_pool_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t*
        requirements,
    VkDescriptorPool* out_descriptor_pool, bool* out_acquired) {
  *out_acquired = false;
  *out_descriptor_pool = VK_NULL_HANDLE;
  const iree_hal_vulkan_command_buffer_descriptor_requirements_t zero = {0};
  iree_hal_vulkan_queue_native_descriptor_block_t* first_block =
      queue->native_descriptor_cache.cursor
          ? queue->native_descriptor_cache.cursor
          : queue->native_descriptor_cache.head;
  iree_hal_vulkan_queue_native_descriptor_block_t* block = first_block;
  while (block) {
    if (iree_hal_vulkan_queue_native_descriptor_requirements_fit(
            &block->capacity, &zero, requirements)) {
      if (!iree_hal_vulkan_queue_native_descriptor_requirements_fit(
              &block->capacity, &block->allocated, requirements) &&
          block->active_lease_count == 0) {
        IREE_RETURN_IF_ERROR(
            iree_hal_vulkan_queue_native_descriptor_block_reset(queue, block));
      }
      if (iree_hal_vulkan_queue_native_descriptor_requirements_fit(
              &block->capacity, &block->allocated, requirements)) {
        IREE_RETURN_IF_ERROR(
            iree_hal_vulkan_queue_acquire_native_descriptor_pool_from_block(
                queue, submission, block, requirements, out_descriptor_pool));
        *out_acquired = true;
        return iree_ok_status();
      }
    }

    block = block->next;
    if (!block && first_block != queue->native_descriptor_cache.head) {
      block = queue->native_descriptor_cache.head;
      first_block = queue->native_descriptor_cache.head;
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_queue_acquire_native_descriptor_pool_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_vulkan_command_buffer_descriptor_requirements_t*
        requirements,
    VkDescriptorPool* out_descriptor_pool) {
  *out_descriptor_pool = VK_NULL_HANDLE;
  if (iree_hal_vulkan_queue_native_descriptor_requirements_are_zero(
          requirements)) {
    return iree_ok_status();
  }
  if (submission->native_descriptor_lease.block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue submission already owns a native descriptor pool");
  }

  for (;;) {
    bool acquired = false;
    iree_status_t status =
        iree_hal_vulkan_queue_try_acquire_native_descriptor_pool_under_lock(
            queue, submission, requirements, out_descriptor_pool, &acquired);
    if (!iree_status_is_ok(status) || acquired) return status;

    iree_hal_vulkan_queue_native_descriptor_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_native_descriptor_block_create(
        queue, requirements, &block));
    iree_hal_vulkan_queue_native_descriptor_cache_append_block(queue, block);
  }
}

static void iree_hal_vulkan_queue_release_native_descriptor_pool(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_native_descriptor_lease_t* lease =
      &submission->native_descriptor_lease;
  if (!lease->block) return;

  iree_slim_mutex_lock(&queue->submission_mutex);
  IREE_ASSERT(lease->block->active_lease_count > 0,
              "native descriptor block active lease count underflow");
  lease->block->active_lease_count = lease->block->active_lease_count - 1;
  queue->native_descriptor_cache.cursor = lease->block;
  lease->block = NULL;
  iree_slim_mutex_unlock(&queue->submission_mutex);
}

static iree_status_t iree_hal_vulkan_queue_bda_publication_block_create(
    iree_hal_vulkan_queue_t* queue, iree_device_size_t minimum_capacity,
    iree_hal_vulkan_queue_bda_publication_block_t** out_block) {
  *out_block = NULL;
  if (!queue->device_allocator) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan BDA publication requires initialized queue staging resources");
  }
  iree_device_size_t capacity = iree_max(
      minimum_capacity,
      (iree_device_size_t)IREE_HAL_VULKAN_QUEUE_BDA_PUBLICATION_BLOCK_SIZE);
  if (!iree_device_size_checked_align(
          capacity, IREE_HAL_VULKAN_QUEUE_BDA_PUBLICATION_ALIGNMENT,
          &capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan BDA publication block size overflows");
  }

  iree_hal_vulkan_queue_bda_publication_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*block), (void**)&block));
  memset(block, 0, sizeof(*block));
  block->capacity = capacity;

  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE,
      .queue_affinity = queue->queue_affinity,
  };
  iree_status_t status = iree_hal_vulkan_allocator_allocate_direct_buffer(
      queue->device_allocator, &params, capacity, &block->buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        block->buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        IREE_HAL_MEMORY_ACCESS_WRITE,
        /*byte_offset=*/0, capacity, &block->mapping);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_buffer_device_address(block->buffer,
                                                   &block->device_address);
  }
  if (iree_status_is_ok(status) && block->device_address == 0) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "Vulkan BDA publication buffer has no device address");
  }
  if (iree_status_is_ok(status) &&
      capacity > UINT64_MAX - block->device_address) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan BDA publication buffer device address range overflows");
  }
  if (iree_status_is_ok(status)) {
    *out_block = block;
  } else {
    iree_hal_buffer_unmap_range(&block->mapping);
    iree_hal_buffer_release(block->buffer);
    iree_allocator_free(queue->host_allocator, block);
  }
  return status;
}

static void iree_hal_vulkan_queue_bda_publication_block_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_bda_publication_block_t* block) {
  if (!block) return;
  iree_hal_buffer_unmap_range(&block->mapping);
  iree_hal_buffer_release(block->buffer);
  iree_allocator_free(queue->host_allocator, block);
}

static void iree_hal_vulkan_queue_bda_publication_cache_append_block(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_bda_publication_block_t* block) {
  block->next = NULL;
  if (queue->bda_publication_cache.tail) {
    queue->bda_publication_cache.tail->next = block;
  } else {
    queue->bda_publication_cache.head = block;
  }
  queue->bda_publication_cache.tail = block;
  if (!queue->bda_publication_cache.cursor) {
    queue->bda_publication_cache.cursor = block;
  }
  queue->bda_publication_cache.block_count =
      queue->bda_publication_cache.block_count + 1;
}

static void iree_hal_vulkan_queue_bda_publication_cache_deinitialize(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_bda_publication_block_t* block =
      queue->bda_publication_cache.head;
  while (block) {
    iree_hal_vulkan_queue_bda_publication_block_t* next = block->next;
    iree_hal_vulkan_queue_bda_publication_block_destroy(queue, block);
    block = next;
  }
  memset(&queue->bda_publication_cache, 0,
         sizeof(queue->bda_publication_cache));
}

static void iree_hal_vulkan_queue_bda_publication_cache_unlink_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_bda_publication_block_t* previous,
    iree_hal_vulkan_queue_bda_publication_block_t* block) {
  if (previous) {
    previous->next = block->next;
  } else {
    queue->bda_publication_cache.head = block->next;
  }
  if (queue->bda_publication_cache.tail == block) {
    queue->bda_publication_cache.tail = previous;
  }
  if (queue->bda_publication_cache.cursor == block) {
    queue->bda_publication_cache.cursor =
        block->next ? block->next : queue->bda_publication_cache.head;
  }
  IREE_ASSERT(queue->bda_publication_cache.block_count > 0,
              "BDA publication block count underflow");
  queue->bda_publication_cache.block_count =
      queue->bda_publication_cache.block_count - 1;
  block->next = NULL;
}

static void iree_hal_vulkan_queue_bda_publication_cache_trim(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_bda_publication_block_t* destroy_head = NULL;
  iree_hal_vulkan_queue_bda_publication_block_t* destroy_tail = NULL;

  iree_slim_mutex_lock(&queue->submission_mutex);
  bool retained_default_idle_block = false;
  iree_hal_vulkan_queue_bda_publication_block_t* previous = NULL;
  iree_hal_vulkan_queue_bda_publication_block_t* block =
      queue->bda_publication_cache.head;
  while (block) {
    iree_hal_vulkan_queue_bda_publication_block_t* next = block->next;
    const bool has_active_leases = block->active_lease_count != 0;
    const bool should_retain_idle =
        !has_active_leases && !retained_default_idle_block &&
        block->capacity <= (iree_device_size_t)
                               IREE_HAL_VULKAN_QUEUE_BDA_PUBLICATION_BLOCK_SIZE;
    if (has_active_leases || should_retain_idle) {
      if (should_retain_idle) {
        block->allocated_length = 0;
        retained_default_idle_block = true;
      }
      previous = block;
    } else {
      iree_hal_vulkan_queue_bda_publication_cache_unlink_under_lock(
          queue, previous, block);
      if (destroy_tail) {
        destroy_tail->next = block;
      } else {
        destroy_head = block;
      }
      destroy_tail = block;
    }
    block = next;
  }
  if (!queue->bda_publication_cache.cursor) {
    queue->bda_publication_cache.cursor = queue->bda_publication_cache.head;
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  while (destroy_head) {
    iree_hal_vulkan_queue_bda_publication_block_t* next = destroy_head->next;
    destroy_head->next = NULL;
    iree_hal_vulkan_queue_bda_publication_block_destroy(queue, destroy_head);
    destroy_head = next;
  }
}

static bool iree_hal_vulkan_queue_bda_publication_block_try_allocate(
    iree_hal_vulkan_queue_bda_publication_block_t* block,
    iree_device_size_t length, iree_device_size_t* out_offset) {
  iree_device_size_t aligned_offset = block->allocated_length;
  if (!iree_device_size_checked_align(
          aligned_offset, IREE_HAL_VULKAN_QUEUE_BDA_PUBLICATION_ALIGNMENT,
          &aligned_offset)) {
    return false;
  }
  if (aligned_offset > block->capacity ||
      length > block->capacity - aligned_offset) {
    return false;
  }
  *out_offset = aligned_offset;
  block->allocated_length = aligned_offset + length;
  block->active_lease_count = block->active_lease_count + 1;
  return true;
}

static iree_status_t
iree_hal_vulkan_queue_try_acquire_bda_publication_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_device_size_t length, bool* out_acquired) {
  *out_acquired = false;
  iree_hal_vulkan_queue_bda_publication_block_t* first_block =
      queue->bda_publication_cache.cursor ? queue->bda_publication_cache.cursor
                                          : queue->bda_publication_cache.head;
  iree_hal_vulkan_queue_bda_publication_block_t* block = first_block;
  while (block) {
    if (length <= block->capacity) {
      if (length > block->capacity - block->allocated_length &&
          block->active_lease_count == 0) {
        block->allocated_length = 0;
      }
      iree_device_size_t offset = 0;
      if (iree_hal_vulkan_queue_bda_publication_block_try_allocate(
              block, length, &offset)) {
        submission->bda_publication_lease =
            (iree_hal_vulkan_queue_bda_publication_lease_t){
                .block = block,
                .offset = offset,
                .length = length,
            };
        queue->bda_publication_cache.cursor = block;
        *out_acquired = true;
        return iree_ok_status();
      }
    }

    block = block->next;
    if (!block && first_block != queue->bda_publication_cache.head) {
      block = queue->bda_publication_cache.head;
      first_block = queue->bda_publication_cache.head;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_acquire_bda_publication_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_device_size_t length, VkDeviceAddress* out_device_address) {
  *out_device_address = 0;
  if (length == 0) return iree_ok_status();
  if (length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan BDA publication length exceeds host addressable size");
  }
  if (submission->bda_publication_lease.block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue submission already owns BDA publication storage");
  }

  for (;;) {
    bool acquired = false;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_queue_try_acquire_bda_publication_under_lock(
            queue, submission, length, &acquired));
    if (acquired) break;

    iree_hal_vulkan_queue_bda_publication_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_bda_publication_block_create(
        queue, length, &block));
    iree_hal_vulkan_queue_bda_publication_cache_append_block(queue, block);
  }

  iree_hal_vulkan_queue_bda_publication_lease_t* lease =
      &submission->bda_publication_lease;
  *out_device_address = lease->block->device_address + lease->offset;
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_queue_acquire_bda_publication_lease_under_lock(
    iree_hal_vulkan_queue_t* queue, iree_device_size_t length,
    iree_hal_vulkan_queue_bda_publication_lease_t* out_lease,
    VkDeviceAddress* out_device_address) {
  *out_lease = (iree_hal_vulkan_queue_bda_publication_lease_t){0};
  *out_device_address = 0;
  iree_hal_vulkan_queue_pending_submission_t synthetic_submission = {0};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_acquire_bda_publication_under_lock(
      queue, &synthetic_submission, length, out_device_address));
  *out_lease = synthetic_submission.bda_publication_lease;
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_release_bda_publication_lease_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_bda_publication_lease_t* lease) {
  if (!lease->block) return;

  IREE_ASSERT(lease->block->active_lease_count > 0,
              "BDA publication block active lease count underflow");
  lease->block->active_lease_count = lease->block->active_lease_count - 1;
  if (lease->block->active_lease_count == 0) {
    lease->block->allocated_length = 0;
  }
  queue->bda_publication_cache.cursor = lease->block;
  memset(lease, 0, sizeof(*lease));
}

static void iree_hal_vulkan_queue_release_bda_publication_lease(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_bda_publication_lease_t* lease) {
  if (!lease->block) return;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_vulkan_queue_release_bda_publication_lease_under_lock(queue, lease);
  iree_slim_mutex_unlock(&queue->submission_mutex);
}

static void iree_hal_vulkan_queue_release_bda_publication(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_release_bda_publication_lease(
      queue, &submission->bda_publication_lease);
}

static iree_hal_vulkan_command_buffer_bda_publication_t
iree_hal_vulkan_queue_bda_publication_for_lease(
    const iree_hal_vulkan_queue_bda_publication_lease_t* lease) {
  if (!lease->block) {
    return (iree_hal_vulkan_command_buffer_bda_publication_t){0};
  }
  IREE_ASSERT(lease->length <= IREE_HOST_SIZE_MAX,
              "BDA publication lease exceeds host addressable size");
  return (iree_hal_vulkan_command_buffer_bda_publication_t){
      .host_span = iree_make_byte_span(
          lease->block->mapping.contents.data + lease->offset,
          (iree_host_size_t)lease->length),
      .device_address = lease->block->device_address + lease->offset,
  };
}

static iree_status_t iree_hal_vulkan_queue_flush_bda_publication_lease(
    const iree_hal_vulkan_queue_bda_publication_lease_t* lease) {
  if (!lease->block || lease->length == 0) return iree_ok_status();
  return iree_hal_buffer_mapping_flush_range(&lease->block->mapping,
                                             lease->offset, lease->length);
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
  if (slot_size > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan staging ring slot size exceeds host addressable size");
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
    iree_hal_buffer_unmap_range(&ring->mapping);
    iree_hal_buffer_release(ring->buffer);
    iree_slim_mutex_deinitialize(&ring->mutex);
    iree_allocator_free(queue->host_allocator, ring);
  }
  return status;
}

static void iree_hal_vulkan_queue_staging_ring_destroy(
    iree_hal_vulkan_queue_staging_ring_t* ring) {
  if (!ring) return;
  iree_hal_buffer_unmap_range(&ring->mapping);
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

static void iree_hal_vulkan_queue_staging_ring_cancel_waiter(
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

static iree_status_t iree_hal_vulkan_queue_command_buffer_block_create(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_command_buffer_block_t** out_block) {
  *out_block = NULL;
  iree_hal_vulkan_queue_command_buffer_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*block), (void**)&block));
  memset(block, 0, sizeof(*block));
  block->free_count = IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY;

  iree_status_t status = iree_hal_vulkan_queue_create_command_pool(
      &queue->syms, queue->logical_device, queue->queue_family_index,
      &block->pool);
  if (iree_status_is_ok(status)) {
    VkCommandBufferAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = block->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount =
            IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY,
    };
    status = iree_vkAllocateCommandBuffers(
        IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device, &allocate_info,
        block->command_buffers);
  }
  if (iree_status_is_ok(status)) {
    *out_block = block;
  } else {
    if (block->pool) {
      iree_vkDestroyCommandPool(IREE_VULKAN_DEVICE(&queue->syms),
                                queue->logical_device, block->pool,
                                /*pAllocator=*/NULL);
    }
    iree_allocator_free(queue->host_allocator, block);
  }
  return status;
}

static void iree_hal_vulkan_queue_command_buffer_block_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_command_buffer_block_t* block) {
  if (!block) return;
  if (block->pool) {
    iree_vkDestroyCommandPool(IREE_VULKAN_DEVICE(&queue->syms),
                              queue->logical_device, block->pool,
                              /*pAllocator=*/NULL);
  }
  iree_allocator_free(queue->host_allocator, block);
}

static void iree_hal_vulkan_queue_command_buffer_cache_append_block(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_command_buffer_block_t* block) {
  block->next = NULL;
  if (queue->command_buffer_cache.tail) {
    queue->command_buffer_cache.tail->next = block;
  } else {
    queue->command_buffer_cache.head = block;
  }
  queue->command_buffer_cache.tail = block;
  if (!queue->command_buffer_cache.cursor) {
    queue->command_buffer_cache.cursor = block;
  }
  queue->command_buffer_cache.block_count =
      queue->command_buffer_cache.block_count + 1;
}

static void iree_hal_vulkan_queue_command_buffer_cache_deinitialize(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_command_buffer_block_t* block =
      queue->command_buffer_cache.head;
  while (block) {
    iree_hal_vulkan_queue_command_buffer_block_t* next = block->next;
    iree_hal_vulkan_queue_command_buffer_block_destroy(queue, block);
    block = next;
  }
  memset(&queue->command_buffer_cache, 0, sizeof(queue->command_buffer_cache));
}

static iree_status_t iree_hal_vulkan_queue_reset_command_buffer_cache_slot(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_command_buffer_block_t* block, uint32_t slot) {
  if (!block->needs_reset[slot]) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_vkResetCommandBuffer(
      IREE_VULKAN_DEVICE(&queue->syms), block->command_buffers[slot],
      /*flags=*/0));
  block->needs_reset[slot] = false;
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_queue_try_acquire_command_buffer_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    bool* out_acquired) {
  *out_acquired = false;
  iree_hal_vulkan_queue_command_buffer_block_t* first_block =
      queue->command_buffer_cache.cursor ? queue->command_buffer_cache.cursor
                                         : queue->command_buffer_cache.head;
  iree_hal_vulkan_queue_command_buffer_block_t* block = first_block;
  while (block) {
    if (block->free_count != 0) {
      for (uint32_t probe_ordinal = 0;
           probe_ordinal < IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY;
           ++probe_ordinal) {
        const uint32_t slot =
            (block->next_slot + probe_ordinal) %
            IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY;
        if (block->owner_epochs[slot] != 0) continue;
        IREE_RETURN_IF_ERROR(
            iree_hal_vulkan_queue_reset_command_buffer_cache_slot(queue, block,
                                                                  slot));
        block->owner_epochs[slot] =
            IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_RESERVED;
        block->next_slot =
            (slot + 1) % IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_BLOCK_CAPACITY;
        block->free_count = block->free_count - 1;
        submission->native_command_buffer_lease =
            (iree_hal_vulkan_queue_command_buffer_lease_t){
                .block = block,
                .slot = slot,
            };
        submission->native_command_buffer = block->command_buffers[slot];
        queue->command_buffer_cache.cursor = block;
        *out_acquired = true;
        return iree_ok_status();
      }
    }
    block = block->next;
    if (!block && first_block != queue->command_buffer_cache.head) {
      block = queue->command_buffer_cache.head;
      first_block = queue->command_buffer_cache.head;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_allocate_native_command_buffer(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->native_command_buffer) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue submission already owns a native command buffer");
  }

  for (;;) {
    bool acquired = false;
    iree_slim_mutex_lock(&queue->submission_mutex);
    iree_status_t status =
        iree_hal_vulkan_queue_try_acquire_command_buffer_under_lock(
            queue, submission, &acquired);
    iree_slim_mutex_unlock(&queue->submission_mutex);
    if (!iree_status_is_ok(status) || acquired) return status;

    const iree_host_size_t drained_count =
        iree_hal_vulkan_queue_drain_completions(queue);
    if (drained_count != 0) continue;

    iree_hal_vulkan_queue_command_buffer_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_queue_command_buffer_block_create(queue, &block));
    iree_slim_mutex_lock(&queue->submission_mutex);
    iree_hal_vulkan_queue_command_buffer_cache_append_block(queue, block);
    iree_slim_mutex_unlock(&queue->submission_mutex);
  }
}

static iree_status_t
iree_hal_vulkan_queue_allocate_native_command_buffer_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->native_command_buffer) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue submission already owns a native command buffer");
  }

  bool acquired = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_queue_try_acquire_command_buffer_under_lock(
          queue, submission, &acquired));
  if (acquired) return iree_ok_status();

  iree_hal_vulkan_queue_command_buffer_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_queue_command_buffer_block_create(queue, &block));
  iree_hal_vulkan_queue_command_buffer_cache_append_block(queue, block);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_queue_try_acquire_command_buffer_under_lock(
          queue, submission, &acquired));
  return acquired
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_INTERNAL,
                                "Vulkan command buffer block had no free slot "
                                "immediately after allocation");
}

static iree_status_t
iree_hal_vulkan_queue_acquire_native_command_buffer_lease_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_command_buffer_lease_t* out_lease,
    VkCommandBuffer* out_native_command_buffer) {
  *out_lease = (iree_hal_vulkan_queue_command_buffer_lease_t){
      .slot = IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT,
  };
  *out_native_command_buffer = VK_NULL_HANDLE;

  iree_hal_vulkan_queue_pending_submission_t synthetic_submission = {
      .native_command_buffer_lease =
          {
              .slot = IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT,
          },
  };
  bool acquired = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_queue_try_acquire_command_buffer_under_lock(
          queue, &synthetic_submission, &acquired));
  if (!acquired) {
    iree_hal_vulkan_queue_command_buffer_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_queue_command_buffer_block_create(queue, &block));
    iree_hal_vulkan_queue_command_buffer_cache_append_block(queue, block);
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_queue_try_acquire_command_buffer_under_lock(
            queue, &synthetic_submission, &acquired));
  }
  if (!acquired) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "Vulkan command buffer block had no free slot "
                            "immediately after allocation");
  }
  *out_lease = synthetic_submission.native_command_buffer_lease;
  *out_native_command_buffer = synthetic_submission.native_command_buffer;
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_publish_native_command_buffer_under_lock(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->native_command_buffer) return;
  iree_hal_vulkan_queue_command_buffer_lease_t lease =
      submission->native_command_buffer_lease;
  if (lease.slot == IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT) return;
  lease.block->owner_epochs[lease.slot] = submission->epoch;
}

static void
iree_hal_vulkan_queue_release_native_command_buffer_lease_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_command_buffer_lease_t* lease,
    VkCommandBuffer* native_command_buffer) {
  if (lease->slot == IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT) return;

  lease->block->owner_epochs[lease->slot] = 0;
  lease->block->needs_reset[lease->slot] = true;
  lease->block->free_count = lease->block->free_count + 1;
  queue->command_buffer_cache.cursor = lease->block;
  lease->block = NULL;
  lease->slot = IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT;
  if (native_command_buffer) *native_command_buffer = VK_NULL_HANDLE;
}

static void iree_hal_vulkan_queue_release_native_command_buffer_lease(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_command_buffer_lease_t* lease,
    VkCommandBuffer* native_command_buffer) {
  if (lease->slot == IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT) return;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_vulkan_queue_release_native_command_buffer_lease_under_lock(
      queue, lease, native_command_buffer);
  iree_slim_mutex_unlock(&queue->submission_mutex);
}

static void iree_hal_vulkan_queue_release_native_command_buffer(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_release_native_command_buffer_lease(
      queue, &submission->native_command_buffer_lease,
      &submission->native_command_buffer);
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

static iree_hal_vulkan_command_buffer_bda_binding_cache_t
iree_hal_vulkan_queue_execute_bda_binding_cache(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  return (iree_hal_vulkan_command_buffer_bda_binding_cache_t){
      .slots = submission->execute.bda_binding_slots,
      .slot_count = submission->execute.bda_binding_slot_count,
  };
}

static iree_status_t iree_hal_vulkan_queue_calculate_native_replay_layout(
    iree_host_size_t bda_binding_slot_count,
    iree_host_size_t* out_allocation_size,
    iree_host_size_t* out_bda_binding_slots_offset) {
  return IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_hal_vulkan_queue_native_replay_t),
      out_allocation_size,
      IREE_STRUCT_FIELD_ALIGNED(
          bda_binding_slot_count,
          iree_hal_vulkan_command_buffer_bda_binding_slot_t,
          iree_alignof(iree_hal_vulkan_command_buffer_bda_binding_slot_t),
          out_bda_binding_slots_offset));
}

static void iree_hal_vulkan_queue_store_native_replay_bda_binding_slots(
    iree_hal_vulkan_queue_native_replay_t* replay,
    const iree_hal_vulkan_command_buffer_bda_binding_cache_t*
        bda_binding_cache) {
  if (replay->bda_binding_slot_count != 0) {
    memcpy(
        replay->bda_binding_slots, bda_binding_cache->slots,
        replay->bda_binding_slot_count * sizeof(replay->bda_binding_slots[0]));
  }
  replay->bda_binding_slots_valid = true;
}

static bool iree_hal_vulkan_queue_native_replay_bda_binding_slots_match(
    const iree_hal_vulkan_queue_native_replay_t* replay,
    const iree_hal_vulkan_command_buffer_bda_binding_cache_t*
        bda_binding_cache) {
  if (!replay->bda_binding_slots_valid) return false;
  if (replay->bda_binding_slot_count != bda_binding_cache->slot_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < replay->bda_binding_slot_count; ++i) {
    const iree_hal_vulkan_command_buffer_bda_binding_slot_t* expected_slot =
        &replay->bda_binding_slots[i];
    const iree_hal_vulkan_command_buffer_bda_binding_slot_t* actual_slot =
        &bda_binding_cache->slots[i];
    if (expected_slot->device_address != actual_slot->device_address ||
        expected_slot->length != actual_slot->length) {
      return false;
    }
  }
  return true;
}

static iree_status_t iree_hal_vulkan_queue_resolve_native_replay_bda_slots(
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_queue_native_replay_t* replay,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache) {
  for (iree_host_size_t i = 0; i < replay->bda_binding_slot_count; ++i) {
    if (replay->bda_binding_slots[i].device_address == 0) continue;
    if (i > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan native replay BDA binding slot ordinal exceeds uint32_t");
    }
    iree_hal_vulkan_command_buffer_bda_binding_slot_t slot = {0};
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_resolve_bda_binding_table_slot(
            binding_table, (uint32_t)i, bda_binding_cache, &slot));
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_native_replay_cache_append_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_native_replay_t* replay) {
  replay->next = NULL;
  if (queue->native_replay_cache.tail) {
    queue->native_replay_cache.tail->next = replay;
  } else {
    queue->native_replay_cache.head = replay;
  }
  queue->native_replay_cache.tail = replay;
  queue->native_replay_cache.instance_count =
      queue->native_replay_cache.instance_count + 1;
  queue->native_replay_cache.publication_bytes =
      queue->native_replay_cache.publication_bytes +
      replay->bda_publication_lease.length;
  queue->native_replay_cache.peak_instance_count =
      iree_max(queue->native_replay_cache.peak_instance_count,
               (uint64_t)queue->native_replay_cache.instance_count);
  queue->native_replay_cache.peak_publication_bytes =
      iree_max(queue->native_replay_cache.peak_publication_bytes,
               queue->native_replay_cache.publication_bytes);
}

static void iree_hal_vulkan_queue_native_replay_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_native_replay_t* replay) {
  if (!replay) return;
  iree_hal_vulkan_queue_release_native_command_buffer_lease(
      queue, &replay->command_buffer_lease, &replay->native_command_buffer);
  iree_hal_vulkan_queue_release_bda_publication_lease(
      queue, &replay->bda_publication_lease);
  iree_hal_command_buffer_release(replay->command_buffer);
  iree_allocator_free(queue->host_allocator, replay);
}

static void iree_hal_vulkan_queue_native_replay_destroy_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_native_replay_t* replay) {
  if (!replay) return;
  iree_hal_vulkan_queue_release_native_command_buffer_lease_under_lock(
      queue, &replay->command_buffer_lease, &replay->native_command_buffer);
  iree_hal_vulkan_queue_release_bda_publication_lease_under_lock(
      queue, &replay->bda_publication_lease);
  iree_hal_command_buffer_release(replay->command_buffer);
  iree_allocator_free(queue->host_allocator, replay);
}

static void iree_hal_vulkan_queue_native_replay_cache_deinitialize(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_native_replay_t* replay =
      queue->native_replay_cache.head;
  while (replay) {
    iree_hal_vulkan_queue_native_replay_t* next = replay->next;
    iree_hal_vulkan_queue_native_replay_destroy(queue, replay);
    replay = next;
  }
  memset(&queue->native_replay_cache, 0, sizeof(queue->native_replay_cache));
}

static void iree_hal_vulkan_queue_publish_native_replay_under_lock(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_native_replay_t* replay = submission->native_replay;
  if (!replay) return;
  IREE_ASSERT(
      replay->owner_epoch == IREE_HAL_VULKAN_QUEUE_NATIVE_REPLAY_OWNER_RESERVED,
      "cached native replay was not reserved before publish");
  replay->owner_epoch = submission->epoch;
}

static void iree_hal_vulkan_queue_release_native_replay(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_native_replay_t* replay = submission->native_replay;
  if (!replay) return;
  iree_slim_mutex_lock(&queue->submission_mutex);
  IREE_ASSERT(replay->owner_epoch == submission->epoch ||
                  replay->owner_epoch ==
                      IREE_HAL_VULKAN_QUEUE_NATIVE_REPLAY_OWNER_RESERVED,
              "cached native replay owner epoch mismatch");
  replay->owner_epoch = 0;
  submission->native_replay = NULL;
  submission->native_command_buffer = VK_NULL_HANDLE;
  iree_slim_mutex_unlock(&queue->submission_mutex);
}

static void iree_hal_vulkan_queue_native_replay_cache_unlink_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_native_replay_t* previous,
    iree_hal_vulkan_queue_native_replay_t* replay) {
  if (previous) {
    previous->next = replay->next;
  } else {
    queue->native_replay_cache.head = replay->next;
  }
  if (queue->native_replay_cache.tail == replay) {
    queue->native_replay_cache.tail = previous;
  }
  queue->native_replay_cache.instance_count =
      queue->native_replay_cache.instance_count - 1;
  queue->native_replay_cache.publication_bytes =
      queue->native_replay_cache.publication_bytes -
      replay->bda_publication_lease.length;
  replay->next = NULL;
}

static bool iree_hal_vulkan_queue_can_cache_native_replay_under_lock(
    iree_hal_vulkan_queue_t* queue, iree_device_size_t publication_length) {
  if (queue->native_replay_cache.max_instance_count == 0) return false;
  if (publication_length > queue->native_replay_cache.max_publication_bytes) {
    return false;
  }
  return true;
}

static iree_status_t iree_hal_vulkan_queue_create_native_replay_under_lock(
    iree_hal_vulkan_queue_t* queue, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    iree_device_size_t publication_length,
    iree_hal_vulkan_queue_native_replay_t** out_replay) {
  *out_replay = NULL;
  if (queue->native_replay_cache.instance_count >=
      queue->native_replay_cache.max_instance_count) {
    return iree_ok_status();
  }
  if (queue->native_replay_cache.publication_bytes >
          queue->native_replay_cache.max_publication_bytes ||
      publication_length > queue->native_replay_cache.max_publication_bytes -
                               queue->native_replay_cache.publication_bytes) {
    return iree_ok_status();
  }

  const iree_host_size_t bda_binding_slot_count = bda_binding_cache->slot_count;
  iree_host_size_t bda_binding_slots_offset = 0;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_calculate_native_replay_layout(
      bda_binding_slot_count, &allocation_size, &bda_binding_slots_offset));

  iree_hal_vulkan_queue_native_replay_t* replay = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             allocation_size, (void**)&replay));
  memset(replay, 0, allocation_size);
  replay->bda_binding_slot_count = bda_binding_slot_count;
  if (bda_binding_slot_count != 0) {
    replay->bda_binding_slots =
        (iree_hal_vulkan_command_buffer_bda_binding_slot_t*)((uint8_t*)replay +
                                                             bda_binding_slots_offset);
  }
  replay->command_buffer_lease.slot =
      IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT;
  replay->owner_epoch = IREE_HAL_VULKAN_QUEUE_NATIVE_REPLAY_OWNER_RESERVED;

  iree_status_t status =
      iree_hal_vulkan_queue_acquire_native_command_buffer_lease_under_lock(
          queue, &replay->command_buffer_lease, &replay->native_command_buffer);
  VkDeviceAddress publication_device_address = 0;
  if (iree_status_is_ok(status) && publication_length != 0) {
    status = iree_hal_vulkan_queue_acquire_bda_publication_lease_under_lock(
        queue, publication_length, &replay->bda_publication_lease,
        &publication_device_address);
  }

  const iree_hal_vulkan_command_buffer_bda_publication_t publication =
      iree_hal_vulkan_queue_bda_publication_for_lease(
          &replay->bda_publication_lease);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_command_buffer_record_native(
        command_buffer, &queue->syms, queue->logical_device,
        &queue->debug_utils, queue->builtins, replay->native_command_buffer,
        /*usage_flags=*/0, VK_NULL_HANDLE, binding_table,
        publication_length != 0 ? &publication : NULL, bda_binding_cache,
        /*profile_marker=*/NULL, queue->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_flush_bda_publication_lease(
        &replay->bda_publication_lease);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_store_native_replay_bda_binding_slots(
        replay, bda_binding_cache);
    iree_hal_command_buffer_retain(command_buffer);
    replay->command_buffer = command_buffer;
    iree_hal_vulkan_queue_native_replay_cache_append_under_lock(queue, replay);
    queue->native_replay_cache.create_count =
        queue->native_replay_cache.create_count + 1;
    *out_replay = replay;
  } else {
    iree_hal_vulkan_queue_native_replay_destroy_under_lock(queue, replay);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_queue_acquire_native_replay_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_hal_vulkan_command_buffer_descriptor_requirements_t
        descriptor_requirements,
    bool has_descriptor_dispatches, iree_device_size_t publication_length,
    bool* out_acquired) {
  *out_acquired = false;
  if (descriptor_requirements.set_count != 0 || has_descriptor_dispatches) {
    queue->native_replay_cache.descriptor_bypass_count =
        queue->native_replay_cache.descriptor_bypass_count + 1;
    return iree_ok_status();
  }
  if (!iree_hal_vulkan_queue_can_cache_native_replay_under_lock(
          queue, publication_length)) {
    queue->native_replay_cache.capacity_bypass_count =
        queue->native_replay_cache.capacity_bypass_count + 1;
    return iree_ok_status();
  }
  if (iree_any_bit_set(
          iree_hal_command_buffer_mode(submission->execute.command_buffer),
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    queue->native_replay_cache.one_shot_bypass_count =
        queue->native_replay_cache.one_shot_bypass_count + 1;
    return iree_ok_status();
  }
  if (iree_hal_vulkan_queue_profile_requests_queue_device_event(submission) ||
      iree_hal_vulkan_queue_profile_requests_dispatch_events(submission)) {
    queue->native_replay_cache.profile_bypass_count =
        queue->native_replay_cache.profile_bypass_count + 1;
    return iree_ok_status();
  }

  iree_hal_vulkan_command_buffer_bda_binding_cache_t bda_binding_cache =
      iree_hal_vulkan_queue_execute_bda_binding_cache(submission);
  iree_hal_buffer_binding_table_t binding_table = {
      .count = submission->execute.binding_table_count,
      .bindings = submission->execute.binding_table_bindings,
  };
  bool has_busy_replay = false;
  bool replay_publication_current = false;
  iree_hal_vulkan_queue_native_replay_t* replay =
      queue->native_replay_cache.head;
  while (replay) {
    if (replay->command_buffer == submission->execute.command_buffer) {
      if (replay->owner_epoch == 0) {
        queue->native_replay_cache.hit_count =
            queue->native_replay_cache.hit_count + 1;
        replay->owner_epoch =
            IREE_HAL_VULKAN_QUEUE_NATIVE_REPLAY_OWNER_RESERVED;
        break;
      }
      has_busy_replay = true;
    }
    replay = replay->next;
  }
  if (!replay) {
    queue->native_replay_cache.miss_count =
        queue->native_replay_cache.miss_count + 1;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_create_native_replay_under_lock(
        queue, submission->execute.command_buffer, binding_table,
        &bda_binding_cache, publication_length, &replay));
    if (!replay) return iree_ok_status();
    replay_publication_current = true;
    if (has_busy_replay) {
      queue->native_replay_cache.fork_count =
          queue->native_replay_cache.fork_count + 1;
    }
  }

  const iree_hal_vulkan_command_buffer_bda_publication_t publication =
      iree_hal_vulkan_queue_bda_publication_for_lease(
          &replay->bda_publication_lease);
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status) && !replay_publication_current) {
    status = iree_hal_vulkan_queue_resolve_native_replay_bda_slots(
        binding_table, replay, &bda_binding_cache);
  }
  if (iree_status_is_ok(status) && !replay_publication_current) {
    replay_publication_current =
        iree_hal_vulkan_queue_native_replay_bda_binding_slots_match(
            replay, &bda_binding_cache);
    if (replay_publication_current && publication_length != 0) {
      queue->native_replay_cache.publication_skip_count =
          queue->native_replay_cache.publication_skip_count + 1;
    }
  }
  if (iree_status_is_ok(status) && !replay_publication_current) {
    replay->bda_binding_slots_valid = false;
    status = iree_hal_vulkan_command_buffer_publish_bda_binding_tables(
        submission->execute.command_buffer, binding_table,
        publication_length != 0 ? &publication : NULL, &bda_binding_cache);
  }
  if (iree_status_is_ok(status) && !replay_publication_current) {
    status = iree_hal_vulkan_queue_flush_bda_publication_lease(
        &replay->bda_publication_lease);
  }
  if (iree_status_is_ok(status) && !replay_publication_current) {
    iree_hal_vulkan_queue_store_native_replay_bda_binding_slots(
        replay, &bda_binding_cache);
    if (publication_length != 0) {
      queue->native_replay_cache.publication_update_count =
          queue->native_replay_cache.publication_update_count + 1;
    }
  }
  if (iree_status_is_ok(status)) {
    submission->native_replay = replay;
    submission->native_command_buffer = replay->native_command_buffer;
    *out_acquired = true;
  } else {
    replay->owner_epoch = 0;
  }
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

static iree_status_t iree_hal_vulkan_queue_timestamp_query_block_create(
    iree_hal_vulkan_queue_t* queue, uint32_t minimum_query_count,
    iree_hal_vulkan_queue_timestamp_query_block_t** out_block) {
  *out_block = NULL;
  const uint32_t capacity =
      iree_max(minimum_query_count,
               (uint32_t)IREE_HAL_VULKAN_QUEUE_TIMESTAMP_QUERY_BLOCK_CAPACITY);

  iree_hal_vulkan_queue_timestamp_query_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*block), (void**)&block));
  memset(block, 0, sizeof(*block));
  block->capacity = capacity;

  iree_status_t status = iree_hal_vulkan_queue_create_timestamp_query_pool(
      queue, capacity, &block->pool);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(queue->host_allocator, capacity,
                                         sizeof(block->query_values[0]),
                                         (void**)&block->query_values);
  }
  if (iree_status_is_ok(status)) {
    *out_block = block;
  } else {
    if (block->pool) {
      iree_vkDestroyQueryPool(IREE_VULKAN_DEVICE(&queue->syms),
                              queue->logical_device, block->pool,
                              /*pAllocator=*/NULL);
    }
    iree_allocator_free(queue->host_allocator, block->query_values);
    iree_allocator_free(queue->host_allocator, block);
  }
  return status;
}

static void iree_hal_vulkan_queue_timestamp_query_block_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_timestamp_query_block_t* block) {
  if (!block) return;
  if (block->pool) {
    iree_vkDestroyQueryPool(IREE_VULKAN_DEVICE(&queue->syms),
                            queue->logical_device, block->pool,
                            /*pAllocator=*/NULL);
  }
  iree_allocator_free(queue->host_allocator, block->query_values);
  iree_allocator_free(queue->host_allocator, block);
}

static void iree_hal_vulkan_queue_timestamp_query_cache_append_block(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_timestamp_query_block_t* block) {
  block->next = NULL;
  if (queue->timestamp_query_cache.tail) {
    queue->timestamp_query_cache.tail->next = block;
  } else {
    queue->timestamp_query_cache.head = block;
  }
  queue->timestamp_query_cache.tail = block;
  if (!queue->timestamp_query_cache.cursor) {
    queue->timestamp_query_cache.cursor = block;
  }
  queue->timestamp_query_cache.block_count =
      queue->timestamp_query_cache.block_count + 1;
}

static void iree_hal_vulkan_queue_timestamp_query_cache_deinitialize(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_timestamp_query_block_t* block =
      queue->timestamp_query_cache.head;
  while (block) {
    iree_hal_vulkan_queue_timestamp_query_block_t* next = block->next;
    iree_hal_vulkan_queue_timestamp_query_block_destroy(queue, block);
    block = next;
  }
  memset(&queue->timestamp_query_cache, 0,
         sizeof(queue->timestamp_query_cache));
}

static void iree_hal_vulkan_queue_timestamp_query_cache_unlink_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_timestamp_query_block_t* previous,
    iree_hal_vulkan_queue_timestamp_query_block_t* block) {
  if (previous) {
    previous->next = block->next;
  } else {
    queue->timestamp_query_cache.head = block->next;
  }
  if (queue->timestamp_query_cache.tail == block) {
    queue->timestamp_query_cache.tail = previous;
  }
  if (queue->timestamp_query_cache.cursor == block) {
    queue->timestamp_query_cache.cursor =
        block->next ? block->next : queue->timestamp_query_cache.head;
  }
  IREE_ASSERT(queue->timestamp_query_cache.block_count > 0,
              "timestamp query block count underflow");
  queue->timestamp_query_cache.block_count =
      queue->timestamp_query_cache.block_count - 1;
  block->next = NULL;
}

static void iree_hal_vulkan_queue_timestamp_query_cache_trim(
    iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_timestamp_query_block_t* destroy_head = NULL;
  iree_hal_vulkan_queue_timestamp_query_block_t* destroy_tail = NULL;

  iree_slim_mutex_lock(&queue->submission_mutex);
  bool retained_default_idle_block = false;
  iree_hal_vulkan_queue_timestamp_query_block_t* previous = NULL;
  iree_hal_vulkan_queue_timestamp_query_block_t* block =
      queue->timestamp_query_cache.head;
  while (block) {
    iree_hal_vulkan_queue_timestamp_query_block_t* next = block->next;
    const bool has_active_leases = block->active_lease_count != 0;
    const bool should_retain_idle =
        !has_active_leases && !retained_default_idle_block &&
        block->capacity <=
            (uint32_t)IREE_HAL_VULKAN_QUEUE_TIMESTAMP_QUERY_BLOCK_CAPACITY;
    if (has_active_leases || should_retain_idle) {
      if (should_retain_idle) {
        block->allocated_count = 0;
        retained_default_idle_block = true;
      }
      previous = block;
    } else {
      iree_hal_vulkan_queue_timestamp_query_cache_unlink_under_lock(
          queue, previous, block);
      if (destroy_tail) {
        destroy_tail->next = block;
      } else {
        destroy_head = block;
      }
      destroy_tail = block;
    }
    block = next;
  }
  if (!queue->timestamp_query_cache.cursor) {
    queue->timestamp_query_cache.cursor = queue->timestamp_query_cache.head;
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  while (destroy_head) {
    iree_hal_vulkan_queue_timestamp_query_block_t* next = destroy_head->next;
    destroy_head->next = NULL;
    iree_hal_vulkan_queue_timestamp_query_block_destroy(queue, destroy_head);
    destroy_head = next;
  }
}

static bool iree_hal_vulkan_queue_timestamp_query_block_try_allocate(
    iree_hal_vulkan_queue_timestamp_query_block_t* block, uint32_t query_count,
    uint32_t* out_first_query) {
  *out_first_query = 0;
  if (query_count > block->capacity) return false;
  if (query_count > block->capacity - block->allocated_count) {
    if (block->active_lease_count != 0) return false;
    block->allocated_count = 0;
  }
  if (query_count > block->capacity - block->allocated_count) return false;
  *out_first_query = block->allocated_count;
  block->allocated_count = block->allocated_count + query_count;
  return true;
}

static bool iree_hal_vulkan_queue_try_acquire_timestamp_queries_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    uint32_t query_count) {
  uint32_t first_query = 0;
  iree_hal_vulkan_queue_timestamp_query_block_t* first_block =
      queue->timestamp_query_cache.cursor ? queue->timestamp_query_cache.cursor
                                          : queue->timestamp_query_cache.head;
  iree_hal_vulkan_queue_timestamp_query_block_t* block = first_block;
  while (block) {
    if (iree_hal_vulkan_queue_timestamp_query_block_try_allocate(
            block, query_count, &first_query)) {
      block->active_lease_count = block->active_lease_count + 1;
      submission->profile.timestamp_query_lease =
          (iree_hal_vulkan_queue_timestamp_query_lease_t){
              .block = block,
              .first_query = first_query,
              .query_count = query_count,
          };
      submission->profile.query_pool = block->pool;
      submission->profile.first_query = first_query;
      submission->profile.query_count = query_count;
      submission->profile.query_values = block->query_values;
      queue->timestamp_query_cache.cursor = block;
      return true;
    }
    block = block->next;
    if (!block && first_block != queue->timestamp_query_cache.head) {
      block = queue->timestamp_query_cache.head;
      first_block = queue->timestamp_query_cache.head;
    }
  }
  return false;
}

static iree_status_t iree_hal_vulkan_queue_acquire_timestamp_queries_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    uint32_t query_count) {
  if (query_count == 0) return iree_ok_status();
  if (submission->profile.timestamp_query_lease.block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue submission already owns timestamp query slots");
  }

  if (iree_hal_vulkan_queue_try_acquire_timestamp_queries_under_lock(
          queue, submission, query_count)) {
    return iree_ok_status();
  }

  iree_hal_vulkan_queue_timestamp_query_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_timestamp_query_block_create(
      queue, query_count, &block));
  iree_hal_vulkan_queue_timestamp_query_cache_append_block(queue, block);
  return iree_hal_vulkan_queue_try_acquire_timestamp_queries_under_lock(
             queue, submission, query_count)
             ? iree_ok_status()
             : iree_make_status(
                   IREE_STATUS_INTERNAL,
                   "Vulkan timestamp query block had no free slots "
                   "immediately after allocation");
}

static void iree_hal_vulkan_queue_release_timestamp_queries(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_timestamp_query_lease_t* lease =
      &submission->profile.timestamp_query_lease;
  if (!lease->block) return;

  iree_slim_mutex_lock(&queue->submission_mutex);
  IREE_ASSERT(lease->block->active_lease_count > 0,
              "timestamp query block active lease count underflow");
  lease->block->active_lease_count = lease->block->active_lease_count - 1;
  queue->timestamp_query_cache.cursor = lease->block;
  lease->block = NULL;
  lease->first_query = 0;
  lease->query_count = 0;
  submission->profile.query_pool = VK_NULL_HANDLE;
  submission->profile.first_query = 0;
  submission->profile.query_count = 0;
  submission->profile.query_values = NULL;
  iree_slim_mutex_unlock(&queue->submission_mutex);
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
  if (semaphore_list.count != 0 &&
      (!semaphore_list.semaphores || !semaphore_list.payload_values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue %.*s semaphore list storage is NULL for %" PRIhsz
        " entries",
        (int)usage.size, usage.data, semaphore_list.count);
  }
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

static iree_status_t iree_hal_vulkan_queue_calculate_submission_layout(
    iree_host_size_t wait_count, iree_host_size_t signal_count,
    iree_host_size_t payload_storage_length, iree_host_size_t* out_total_size,
    iree_host_size_t* out_wait_semaphores_offset,
    iree_host_size_t* out_wait_payload_values_offset,
    iree_host_size_t* out_signal_semaphores_offset,
    iree_host_size_t* out_signal_payload_values_offset,
    iree_host_size_t* out_payload_storage_offset) {
  return IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_hal_vulkan_queue_pending_submission_t),
      out_total_size,
      IREE_STRUCT_FIELD_ALIGNED(wait_count, iree_hal_semaphore_t*,
                                iree_alignof(iree_hal_semaphore_t*),
                                out_wait_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(wait_count, uint64_t, iree_alignof(uint64_t),
                                out_wait_payload_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(signal_count, iree_hal_semaphore_t*,
                                iree_alignof(iree_hal_semaphore_t*),
                                out_signal_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(signal_count, uint64_t, iree_alignof(uint64_t),
                                out_signal_payload_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(payload_storage_length, uint8_t,
                                iree_max_align_t, out_payload_storage_offset));
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
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DISPATCH:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
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
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DISPATCH:
      return true;
    default:
      return false;
  }
}

static bool iree_hal_vulkan_queue_profile_filter_matches_dispatch(
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t command_buffer_id,
    uint32_t command_index, const iree_hal_vulkan_pipeline_t* pipeline) {
  const iree_hal_device_profiling_options_t* options =
      iree_hal_local_profile_recorder_options(profile_recorder);
  if (!options) return false;
  const iree_hal_profile_capture_filter_t* filter = &options->capture_filter;
  if (!iree_hal_profile_capture_filter_matches_location(
          filter, command_buffer_id, command_index,
          scope.physical_device_ordinal, scope.queue_ordinal)) {
    return false;
  }
  if (iree_any_bit_set(
          filter->flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN) &&
      !iree_string_view_match_pattern(pipeline->name,
                                      filter->executable_function_pattern)) {
    return false;
  }
  return true;
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
    if (submission->kind == IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DISPATCH) {
      const iree_hal_vulkan_pipeline_t* pipeline = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
          submission->dispatch.executable,
          submission->dispatch.function_ordinal, &pipeline));
      if (iree_hal_vulkan_queue_profile_filter_matches_dispatch(
              submission->profile.recorder, submission->profile.scope,
              /*command_buffer_id=*/0, /*command_index=*/UINT32_MAX,
              pipeline)) {
        *out_dispatch_count = 1;
      }
    }
    return iree_ok_status();
  }
  return iree_hal_vulkan_command_buffer_count_profiled_dispatches(
      submission->execute.command_buffer, submission->profile.recorder,
      submission->profile.scope,
      iree_hal_vulkan_queue_profile_command_buffer_id(submission),
      out_dispatch_count);
}

static iree_status_t iree_hal_vulkan_queue_append_dispatch_profile_event(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->profile.dispatch_query_count != 1) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan direct dispatch profile event requires exactly one timestamp "
        "range");
  }

  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
      submission->dispatch.executable, submission->dispatch.function_ordinal,
      &pipeline));
  if (!iree_hal_vulkan_queue_profile_filter_matches_dispatch(
          submission->profile.recorder, submission->profile.scope,
          /*command_buffer_id=*/0, /*command_index=*/UINT32_MAX, pipeline)) {
    return iree_ok_status();
  }
  if (pipeline->workgroup_size[0] == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan dispatch profiling requires static SPIR-V LocalSize metadata");
  }

  const uint64_t* ticks =
      &submission->profile
           .query_values[submission->profile.dispatch_base_query];
  if (ticks[1] < ticks[0]) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "Vulkan dispatch profiling timestamp range is not monotonic");
  }

  iree_hal_local_profile_dispatch_event_info_t event_info =
      iree_hal_local_profile_dispatch_event_info_default();
  event_info.scope = submission->profile.scope;
  event_info.submission_id = submission->profile.submission_id;
  event_info.command_buffer_id = 0;
  event_info.executable_id =
      iree_hal_vulkan_executable_profile_id(submission->dispatch.executable);
  event_info.command_index = UINT32_MAX;
  event_info.function_ordinal =
      iree_hal_executable_function_index(submission->dispatch.function_ordinal);
  if (iree_hal_dispatch_uses_indirect_parameters(submission->dispatch.flags)) {
    event_info.flags |=
        IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS;
  } else {
    memcpy(event_info.workgroup_count,
           submission->dispatch.config.workgroup_count,
           sizeof(event_info.workgroup_count));
  }
  memcpy(event_info.workgroup_size, pipeline->workgroup_size,
         sizeof(event_info.workgroup_size));
  event_info.start_tick = ticks[0];
  event_info.end_tick = ticks[1];
  return iree_hal_local_profile_recorder_append_dispatch_event(
      submission->profile.recorder, &event_info, /*out_event_id=*/NULL);
}

static iree_status_t iree_hal_vulkan_queue_record_dispatch_profile_metadata(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_local_profile_recorder_t* profile_recorder =
      submission->queue->profile_recorder;
  if (!iree_hal_local_profile_recorder_is_enabled(
          profile_recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA)) {
    return iree_ok_status();
  }
  return iree_hal_local_profile_recorder_record_executable_with_id(
      profile_recorder, submission->dispatch.executable,
      iree_hal_vulkan_executable_profile_id(submission->dispatch.executable));
}

static iree_status_t
iree_hal_vulkan_queue_profile_prepare_native_timestamps_under_lock(
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
  uint32_t queue_start_query = IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT;
  uint32_t queue_end_query = IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT;
  uint32_t dispatch_base_query = IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT;
  if (needs_queue_device_timestamps) {
    queue_start_query = query_count++;
    queue_end_query = query_count++;
  }
  if (dispatch_query_count != 0) {
    if (dispatch_query_count > (UINT32_MAX - query_count) / 2) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan dispatch profile query count exceeds uint32_t");
    }
    dispatch_base_query = query_count;
    query_count += dispatch_query_count * 2;
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_queue_acquire_timestamp_queries_under_lock(
          queue, submission, query_count));
  const uint32_t first_query = submission->profile.first_query;
  if (queue_start_query != IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    submission->profile.queue_start_query = first_query + queue_start_query;
    submission->profile.queue_end_query = first_query + queue_end_query;
  }
  if (dispatch_base_query != IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    submission->profile.dispatch_base_query = first_query + dispatch_base_query;
    submission->profile.dispatch_query_count = dispatch_query_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_profile_prepare_native_timestamps(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_status_t status =
      iree_hal_vulkan_queue_profile_prepare_native_timestamps_under_lock(
          queue, submission);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  return status;
}

static void iree_hal_vulkan_queue_profile_reset_native_timestamps(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.query_pool || submission->profile.query_count == 0) {
    return;
  }
  iree_vkCmdResetQueryPool(
      IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
      submission->profile.query_pool, submission->profile.first_query,
      submission->profile.query_count);
}

static void iree_hal_vulkan_queue_profile_write_timestamp_begin(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.query_pool ||
      submission->profile.queue_start_query ==
          IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    return;
  }
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

static void iree_hal_vulkan_queue_profile_record_device_tick_range(
    iree_hal_vulkan_queue_t* queue, uint64_t start_tick, uint64_t end_tick) {
  iree_hal_vulkan_profile_clock_alignment_t* clock_alignment =
      queue->profile_clock_alignment;
  if (!clock_alignment) return;

  iree_slim_mutex_lock(&clock_alignment->mutex);
  if (clock_alignment->has_event_ticks) {
    clock_alignment->minimum_event_tick =
        iree_min(clock_alignment->minimum_event_tick, start_tick);
    clock_alignment->maximum_event_tick =
        iree_max(clock_alignment->maximum_event_tick, end_tick);
  } else {
    clock_alignment->minimum_event_tick = start_tick;
    clock_alignment->maximum_event_tick = end_tick;
    clock_alignment->has_event_ticks = true;
  }
  iree_slim_mutex_unlock(&clock_alignment->mutex);
}

static iree_status_t iree_hal_vulkan_queue_profile_read_native_timestamps(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!submission->profile.query_pool) {
    return iree_ok_status();
  }
  VkResult result = iree_vkGetQueryPoolResults_raw(
      &queue->syms, queue->logical_device, submission->profile.query_pool,
      submission->profile.first_query, submission->profile.query_count,
      submission->profile.query_count * sizeof(uint64_t),
      &submission->profile.query_values[submission->profile.first_query],
      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
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
    iree_hal_vulkan_queue_profile_record_device_tick_range(queue, start_tick,
                                                           end_tick);
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
    iree_hal_vulkan_queue_profile_record_device_tick_range(queue, start_tick,
                                                           end_tick);
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
  if (submission->kind == IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DISPATCH) {
    return iree_hal_vulkan_queue_append_dispatch_profile_event(submission);
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

// Captures a retained semaphore list into caller-owned storage.
static void iree_hal_vulkan_queue_capture_semaphore_list(
    iree_hal_semaphore_list_t source_list,
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_value_storage,
    iree_hal_semaphore_list_t* out_list) {
  *out_list = iree_hal_semaphore_list_empty();
  if (source_list.count == 0) return;

  *out_list = (iree_hal_semaphore_list_t){
      .count = source_list.count,
      .semaphores = semaphore_storage,
      .payload_values = payload_value_storage,
  };
  for (iree_host_size_t i = 0; i < source_list.count; ++i) {
    out_list->semaphores[i] = source_list.semaphores[i];
    iree_hal_semaphore_retain(out_list->semaphores[i]);
    out_list->payload_values[i] = source_list.payload_values[i];
  }
}

// Allocates a pending submission with optional trailing payload storage owned
// by the submission allocation.
static iree_status_t iree_hal_vulkan_queue_pending_submission_create(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_queue_submission_kind_t kind, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags,
    iree_host_size_t payload_storage_length,
    iree_byte_span_t* out_payload_storage,
    iree_hal_vulkan_queue_pending_submission_t** out_submission) {
  *out_submission = NULL;
  if (out_payload_storage) {
    *out_payload_storage = iree_byte_span_empty();
  }

  iree_host_size_t wait_semaphores_offset = 0;
  iree_host_size_t wait_payload_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0;
  iree_host_size_t signal_payload_values_offset = 0;
  iree_host_size_t payload_storage_offset = 0;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_calculate_submission_layout(
      wait_semaphore_list.count, signal_semaphore_list.count,
      payload_storage_length, &allocation_size, &wait_semaphores_offset,
      &wait_payload_values_offset, &signal_semaphores_offset,
      &signal_payload_values_offset, &payload_storage_offset));

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, allocation_size, (void**)&submission));
  memset(submission, 0, allocation_size);
  submission->queue = queue;
  submission->kind = kind;
  submission->native_command_buffer_lease.slot =
      IREE_HAL_VULKAN_QUEUE_COMMAND_BUFFER_SLOT_ABSENT;
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
  if (kind == IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA) {
    iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 1,
                      iree_memory_order_relaxed);
  }
  if (kind == IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL) {
    submission->host_call.call = call;
    memcpy(submission->host_call.args, args,
           sizeof(submission->host_call.args));
    submission->host_call.flags = flags;
  }
  iree_hal_vulkan_queue_capture_semaphore_list(
      wait_semaphore_list,
      (iree_hal_semaphore_t**)((uint8_t*)submission + wait_semaphores_offset),
      (uint64_t*)((uint8_t*)submission + wait_payload_values_offset),
      &submission->wait_semaphore_list);
  iree_hal_vulkan_queue_capture_semaphore_list(
      signal_semaphore_list,
      (iree_hal_semaphore_t**)((uint8_t*)submission + signal_semaphores_offset),
      (uint64_t*)((uint8_t*)submission + signal_payload_values_offset),
      &submission->signal_semaphore_list);
  iree_hal_vulkan_queue_profile_submission_initialize(queue, submission);

  if (out_payload_storage && payload_storage_length != 0) {
    *out_payload_storage = iree_make_byte_span(
        (uint8_t*)submission + payload_storage_offset, payload_storage_length);
  }
  *out_submission = submission;
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_pending_submission_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_semaphore_list_release(submission->wait_semaphore_list);
  iree_hal_semaphore_list_release(submission->signal_semaphore_list);
  iree_status_t wait_failure_status = (iree_status_t)iree_atomic_exchange(
      &submission->wait_failure_status, 0, iree_memory_order_acquire);
  iree_status_free(wait_failure_status);
  if (submission->wait_entries) {
    iree_allocator_free(queue->host_allocator, submission->wait_entries);
  }
  iree_allocator_free(queue->host_allocator, submission->sparse_bind.binds);
  iree_hal_vulkan_queue_release_descriptor_cache_sets(queue, submission);
  iree_hal_vulkan_queue_release_native_descriptor_pool(queue, submission);
  iree_hal_vulkan_queue_release_bda_publication(queue, submission);
  iree_hal_vulkan_queue_release_timestamp_queries(queue, submission);
  iree_hal_vulkan_queue_release_completion_action(submission);
  iree_hal_vulkan_queue_release_native_replay(queue, submission);
  iree_hal_vulkan_queue_release_native_command_buffer(queue, submission);

  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_SPARSE_BIND:
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL:
      iree_hal_buffer_release(submission->fill.target_buffer);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE:
      iree_allocator_free(queue->host_allocator,
                          submission->update.source_data);
      iree_hal_buffer_release(submission->update.target_buffer);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY:
      iree_hal_buffer_release(submission->copy.source_buffer);
      iree_hal_buffer_release(submission->copy.target_buffer);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA:
      if (submission->alloca.pool_notification_observation_held) {
        submission->alloca.pool_notification_observation_held = false;
        iree_async_notification_end_observe(
            submission->alloca.pool_notification);
      }
      iree_hal_buffer_release(submission->alloca.buffer);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA:
      iree_hal_buffer_release(submission->dealloca.buffer);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE:
      iree_hal_command_buffer_release(submission->execute.command_buffer);
      if (submission->execute.binding_table_bindings &&
          !iree_any_bit_set(
              submission->execute.flags,
              IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
        for (iree_host_size_t i = 0;
             i < submission->execute.binding_table_count; ++i) {
          iree_hal_buffer_release(
              submission->execute.binding_table_bindings[i].buffer);
        }
      }
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DISPATCH:
      iree_hal_executable_release(submission->dispatch.executable);
      if (submission->dispatch.bindings) {
        for (iree_host_size_t i = 0; i < submission->dispatch.binding_count;
             ++i) {
          iree_hal_buffer_release(submission->dispatch.bindings[i].buffer);
        }
      }
      if (iree_hal_dispatch_uses_indirect_parameters(
              submission->dispatch.flags)) {
        iree_hal_buffer_release(
            submission->dispatch.config.workgroup_count_ref.buffer);
      }
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER:
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL:
      break;
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
  iree_hal_vulkan_queue_profile_reset_native_timestamps(queue, submission);
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
  iree_hal_vulkan_queue_profile_reset_native_timestamps(queue, submission);
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

static iree_status_t iree_hal_vulkan_queue_can_record_update_native(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, bool* out_can_record_native) {
  *out_can_record_native = false;
  if (length == 0) return iree_ok_status();
  if (length > IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE ||
      length % sizeof(uint32_t) != 0) {
    return iree_ok_status();
  }

  iree_hal_buffer_t* target_backing = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(target_buffer, &target_backing));
  iree_device_size_t backing_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      target_buffer, target_backing, target_offset, &backing_offset));
  *out_can_record_native = backing_offset % sizeof(uint32_t) == 0;
  return iree_ok_status();
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
    iree_hal_vulkan_queue_profile_reset_native_timestamps(queue, submission);
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
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DISPATCH:
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
  if (iree_status_is_ok(status) && submission->record_native_submission &&
      !submission->native_command_buffer) {
    status = submission->record_native_submission(queue, submission);
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
    iree_hal_vulkan_queue_publish_native_command_buffer_under_lock(submission);
    iree_hal_vulkan_queue_publish_native_replay_under_lock(submission);
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

static iree_status_t iree_hal_vulkan_queue_alloca_memory_wait_begin_arming(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  int32_t expected_state = IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING;
  if (iree_atomic_compare_exchange_strong(
          &submission->deferred_state, &expected_state,
          IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_ARMING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "Vulkan alloca memory wait cannot arm from deferred state %d",
      expected_state);
}

static bool iree_hal_vulkan_queue_alloca_memory_wait_promote_from_arming(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  int32_t expected_state = IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_ARMING;
  return iree_atomic_compare_exchange_strong(
      &submission->deferred_state, &expected_state,
      IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PROMOTING, iree_memory_order_acq_rel,
      iree_memory_order_acquire);
}

static void iree_hal_vulkan_queue_alloca_memory_wait_finish_arming(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_submission_record_wait_status(submission, status);
    iree_hal_vulkan_queue_alloca_memory_wait_publish_complete(submission);
  }

  if (iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
          submission)) {
    if (iree_hal_vulkan_queue_alloca_memory_wait_promote_from_arming(
            submission)) {
      iree_hal_vulkan_queue_deferred_submission_ready(submission);
    }
    return;
  }

  int32_t expected_state = IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_ARMING;
  if (iree_atomic_compare_exchange_strong(
          &submission->deferred_state, &expected_state,
          IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    if (iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
            submission) &&
        iree_hal_vulkan_queue_submission_claim_promotion(submission)) {
      iree_hal_vulkan_queue_deferred_submission_ready(submission);
    }
  }
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
  iree_status_t status =
      iree_hal_vulkan_queue_alloca_memory_wait_begin_arming(submission);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_alloca_memory_wait_resolved(submission, status);
    return;
  }
  status = iree_async_frontier_tracker_wait(
      submission->queue->frontier_tracker, submission->alloca.wait_frontier,
      iree_hal_vulkan_queue_alloca_frontier_wait_resolved, submission,
      &submission->alloca.frontier_waiter);
  iree_hal_vulkan_queue_alloca_memory_wait_finish_arming(submission, status);
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
  iree_status_t status =
      iree_hal_vulkan_queue_alloca_memory_wait_begin_arming(submission);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
    iree_hal_vulkan_queue_alloca_memory_wait_resolved(submission, status);
    return;
  }
  status = iree_async_proactor_submit_one(submission->queue->proactor,
                                          &wait_op->base);
  iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
  iree_hal_vulkan_queue_alloca_memory_wait_finish_arming(submission, status);
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
    iree_hal_vulkan_queue_t* queue, bool* out_has_deferred_submissions) {
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
  *out_has_deferred_submissions = queue->deferred_head != NULL;
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

static bool iree_hal_vulkan_queue_cancel_deferred_submission_list(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* deferred_head,
    iree_status_t status) {
  bool cancelled_submission = false;
  while (deferred_head) {
    cancelled_submission = true;
    iree_hal_vulkan_queue_pending_submission_t* next = deferred_head->next;
    deferred_head->next = NULL;
    iree_hal_vulkan_queue_cancel_deferred_submission(queue, deferred_head,
                                                     iree_status_clone(status));
    deferred_head = next;
  }
  return cancelled_submission;
}

static void iree_hal_vulkan_queue_cancel_deferred_submissions(
    iree_hal_vulkan_queue_t* queue, iree_status_t status) {
  bool has_deferred_submissions = true;
  while (has_deferred_submissions) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    iree_hal_vulkan_queue_pending_submission_t* deferred_head =
        iree_hal_vulkan_queue_take_cancellable_deferred_submissions_under_lock(
            queue, &has_deferred_submissions);
    iree_slim_mutex_unlock(&queue->submission_mutex);

    const bool cancelled_submission =
        iree_hal_vulkan_queue_cancel_deferred_submission_list(
            queue, deferred_head, status);
    if (has_deferred_submissions && !cancelled_submission) {
      iree_thread_yield();
    }
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
  bool has_deferred_submissions = false;
  deferred_head =
      iree_hal_vulkan_queue_take_cancellable_deferred_submissions_under_lock(
          queue, &has_deferred_submissions);
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
  iree_hal_vulkan_queue_cancel_deferred_submission_list(queue, deferred_head,
                                                        status);
  while (has_deferred_submissions) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    deferred_head =
        iree_hal_vulkan_queue_take_cancellable_deferred_submissions_under_lock(
            queue, &has_deferred_submissions);
    iree_slim_mutex_unlock(&queue->submission_mutex);

    const bool cancelled_submission =
        iree_hal_vulkan_queue_cancel_deferred_submission_list(
            queue, deferred_head, status);
    if (has_deferred_submissions && !cancelled_submission) {
      iree_thread_yield();
    }
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
  IREE_ASSERT_ARGUMENT(params->debug_utils);
  IREE_ASSERT_ARGUMENT(params->logical_device);
  IREE_ASSERT_ARGUMENT(params->builtins);
  IREE_ASSERT_ARGUMENT(params->queue);
  IREE_ASSERT_ARGUMENT(params->queue_handle_mutex);
  IREE_ASSERT_ARGUMENT(params->proactor);
  IREE_ASSERT_ARGUMENT(out_queue);
  memset(out_queue, 0, sizeof(*out_queue));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_dispatch_abis_verify(params->enabled_dispatch_abis));

  out_queue->device = params->device;
  out_queue->syms = *params->syms;
  out_queue->debug_utils = *params->debug_utils;
  out_queue->logical_device = params->logical_device;
  out_queue->builtins = params->builtins;
  out_queue->enabled_dispatch_abis = params->enabled_dispatch_abis;
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
  out_queue->native_replay_cache.max_instance_count =
      params->max_cached_bda_replay_instances;
  out_queue->native_replay_cache.max_publication_bytes =
      params->max_cached_bda_replay_publication_bytes;
  out_queue->native_replay_cache.retained_instance_count =
      params->retained_cached_bda_replay_instances;
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
  iree_hal_vulkan_queue_native_descriptor_cache_deinitialize(queue);
  iree_hal_vulkan_queue_native_replay_cache_deinitialize(queue);
  iree_hal_vulkan_queue_bda_publication_cache_deinitialize(queue);
  iree_hal_vulkan_queue_timestamp_query_cache_deinitialize(queue);
  iree_hal_vulkan_queue_command_buffer_cache_deinitialize(queue);
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

void iree_hal_vulkan_queue_trim(iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_native_replay_t* destroy_head = NULL;
  iree_hal_vulkan_queue_native_replay_t* destroy_tail = NULL;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_vulkan_queue_native_replay_t* previous = NULL;
  iree_hal_vulkan_queue_native_replay_t* replay =
      queue->native_replay_cache.head;
  while (replay) {
    iree_hal_vulkan_queue_native_replay_t* next = replay->next;
    uint32_t retained_idle_count = 0;
    for (iree_hal_vulkan_queue_native_replay_t* retained =
             queue->native_replay_cache.head;
         retained != replay; retained = retained->next) {
      if (retained->owner_epoch == 0 &&
          retained->command_buffer == replay->command_buffer) {
        retained_idle_count = retained_idle_count + 1;
      }
    }
    const bool should_retain =
        replay->owner_epoch != 0 ||
        retained_idle_count <
            queue->native_replay_cache.retained_instance_count;
    if (should_retain) {
      previous = replay;
    } else {
      iree_hal_vulkan_queue_native_replay_cache_unlink_under_lock(
          queue, previous, replay);
      queue->native_replay_cache.trim_count =
          queue->native_replay_cache.trim_count + 1;
      if (destroy_tail) {
        destroy_tail->next = replay;
      } else {
        destroy_head = replay;
      }
      destroy_tail = replay;
    }
    replay = next;
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  while (destroy_head) {
    iree_hal_vulkan_queue_native_replay_t* next = destroy_head->next;
    destroy_head->next = NULL;
    iree_hal_vulkan_queue_native_replay_destroy(queue, destroy_head);
    destroy_head = next;
  }
  iree_hal_vulkan_queue_bda_publication_cache_trim(queue);
  iree_hal_vulkan_queue_timestamp_query_cache_trim(queue);
}

iree_status_t iree_hal_vulkan_queue_prepare_profile_timestamp_queries(
    iree_hal_vulkan_queue_t* queue) {
  iree_slim_mutex_lock(&queue->submission_mutex);
  const bool has_query_block = queue->timestamp_query_cache.head != NULL;
  iree_slim_mutex_unlock(&queue->submission_mutex);
  if (has_query_block) return iree_ok_status();

  iree_hal_vulkan_queue_timestamp_query_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_timestamp_query_block_create(
      queue, IREE_HAL_VULKAN_QUEUE_TIMESTAMP_QUERY_BLOCK_CAPACITY, &block));

  bool should_destroy_block = false;
  iree_slim_mutex_lock(&queue->submission_mutex);
  if (queue->timestamp_query_cache.head) {
    should_destroy_block = true;
  } else {
    iree_hal_vulkan_queue_timestamp_query_cache_append_block(queue, block);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (should_destroy_block) {
    iree_hal_vulkan_queue_timestamp_query_block_destroy(queue, block);
  }
  return iree_ok_status();
}

static int64_t iree_hal_vulkan_queue_i64_saturate_u64(uint64_t value) {
  return value > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)value;
}

bool iree_hal_vulkan_queue_query_i64(iree_hal_vulkan_queue_t* queue,
                                     iree_string_view_t category,
                                     iree_string_view_t key,
                                     int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;

  if (iree_string_view_equal(category,
                             IREE_SV("vulkan.queue.bda_publication_cache"))) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    if (iree_string_view_equal(key, IREE_SV("block_count"))) {
      *out_value = (int64_t)queue->bda_publication_cache.block_count;
    } else {
      iree_slim_mutex_unlock(&queue->submission_mutex);
      return false;
    }
    iree_slim_mutex_unlock(&queue->submission_mutex);
    return true;
  }

  if (iree_string_view_equal(category,
                             IREE_SV("vulkan.queue.timestamp_query_cache"))) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    if (iree_string_view_equal(key, IREE_SV("block_count"))) {
      *out_value = (int64_t)queue->timestamp_query_cache.block_count;
    } else if (iree_string_view_equal(key, IREE_SV("query_capacity"))) {
      uint64_t query_capacity = 0;
      for (iree_hal_vulkan_queue_timestamp_query_block_t* block =
               queue->timestamp_query_cache.head;
           block; block = block->next) {
        query_capacity += block->capacity;
      }
      *out_value = iree_hal_vulkan_queue_i64_saturate_u64(query_capacity);
    } else if (iree_string_view_equal(key, IREE_SV("allocated_count"))) {
      uint64_t allocated_count = 0;
      for (iree_hal_vulkan_queue_timestamp_query_block_t* block =
               queue->timestamp_query_cache.head;
           block; block = block->next) {
        allocated_count += block->allocated_count;
      }
      *out_value = iree_hal_vulkan_queue_i64_saturate_u64(allocated_count);
    } else if (iree_string_view_equal(key, IREE_SV("active_lease_count"))) {
      uint64_t active_lease_count = 0;
      for (iree_hal_vulkan_queue_timestamp_query_block_t* block =
               queue->timestamp_query_cache.head;
           block; block = block->next) {
        active_lease_count += block->active_lease_count;
      }
      *out_value = iree_hal_vulkan_queue_i64_saturate_u64(active_lease_count);
    } else {
      iree_slim_mutex_unlock(&queue->submission_mutex);
      return false;
    }
    iree_slim_mutex_unlock(&queue->submission_mutex);
    return true;
  }

  if (!iree_string_view_equal(category,
                              IREE_SV("vulkan.queue.native_replay_cache"))) {
    return false;
  }

  iree_slim_mutex_lock(&queue->submission_mutex);
  if (iree_string_view_equal(key, IREE_SV("instance_count"))) {
    *out_value = (int64_t)queue->native_replay_cache.instance_count;
  } else if (iree_string_view_equal(key, IREE_SV("max_instance_count"))) {
    *out_value = (int64_t)queue->native_replay_cache.max_instance_count;
  } else if (iree_string_view_equal(key, IREE_SV("retained_instance_count"))) {
    *out_value = (int64_t)queue->native_replay_cache.retained_instance_count;
  } else if (iree_string_view_equal(key, IREE_SV("publication_bytes"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.publication_bytes);
  } else if (iree_string_view_equal(key, IREE_SV("max_publication_bytes"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.max_publication_bytes);
  } else if (iree_string_view_equal(key, IREE_SV("peak_instance_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.peak_instance_count);
  } else if (iree_string_view_equal(key, IREE_SV("peak_publication_bytes"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.peak_publication_bytes);
  } else if (iree_string_view_equal(key, IREE_SV("hit_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.hit_count);
  } else if (iree_string_view_equal(key, IREE_SV("miss_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.miss_count);
  } else if (iree_string_view_equal(key, IREE_SV("create_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.create_count);
  } else if (iree_string_view_equal(key, IREE_SV("fork_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.fork_count);
  } else if (iree_string_view_equal(key, IREE_SV("publication_skip_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.publication_skip_count);
  } else if (iree_string_view_equal(key, IREE_SV("publication_update_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.publication_update_count);
  } else if (iree_string_view_equal(key, IREE_SV("descriptor_bypass_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.descriptor_bypass_count);
  } else if (iree_string_view_equal(key, IREE_SV("profile_bypass_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.profile_bypass_count);
  } else if (iree_string_view_equal(key, IREE_SV("one_shot_bypass_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.one_shot_bypass_count);
  } else if (iree_string_view_equal(key, IREE_SV("capacity_bypass_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.capacity_bypass_count);
  } else if (iree_string_view_equal(key, IREE_SV("trim_count"))) {
    *out_value = iree_hal_vulkan_queue_i64_saturate_u64(
        queue->native_replay_cache.trim_count);
  } else {
    iree_slim_mutex_unlock(&queue->submission_mutex);
    return false;
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);
  return true;
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
    iree_atomic_int64_t* submission_counter,
    iree_hal_vulkan_profile_clock_alignment_t* clock_alignment) {
  if (profile_recorder) {
    queue->profile_scope = profile_scope;
    queue->profile_submission_counter = submission_counter;
    queue->profile_clock_alignment = clock_alignment;
    queue->profile_recorder = profile_recorder;
  } else {
    queue->profile_recorder = NULL;
    queue->profile_scope = profile_scope;
    queue->profile_submission_counter = NULL;
    queue->profile_clock_alignment = NULL;
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
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        /*payload_storage_length=*/0, /*out_payload_storage=*/NULL,
        &submission);
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
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        /*payload_storage_length=*/0, /*out_payload_storage=*/NULL,
        &submission);
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
        /*payload_storage_length=*/0, /*out_payload_storage=*/NULL,
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
        /*payload_storage_length=*/0, /*out_payload_storage=*/NULL,
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
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        /*payload_storage_length=*/0, /*out_payload_storage=*/NULL,
        &submission);
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
    status =
        iree_hal_vulkan_queue_allocate_native_command_buffer(queue, submission);
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
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        /*payload_storage_length=*/0, /*out_payload_storage=*/NULL,
        &submission);
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
    status =
        iree_hal_vulkan_queue_allocate_native_command_buffer(queue, submission);
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

static bool iree_hal_vulkan_queue_file_supports_staged_transfer(
    iree_hal_file_t* file) {
  return iree_hal_memory_file_isa(file) ||
         iree_hal_file_async_handle(file) != NULL;
}

typedef enum iree_hal_vulkan_staged_transfer_kind_e {
  // File or host bytes flow through the upload ring into the target buffer.
  IREE_HAL_VULKAN_STAGED_TRANSFER_READ = 0,

  // Source buffer bytes flow through the download ring into the file or host.
  IREE_HAL_VULKAN_STAGED_TRANSFER_WRITE = 1,
} iree_hal_vulkan_staged_transfer_kind_t;

typedef enum iree_hal_vulkan_staged_transfer_flag_bits_e {
  // No special transfer behavior.
  IREE_HAL_VULKAN_STAGED_TRANSFER_FLAG_NONE = 0u,

  // Capture the host range into transfer-owned storage before returning.
  IREE_HAL_VULKAN_STAGED_TRANSFER_FLAG_CAPTURE_HOST_RANGE = 1u << 0,
} iree_hal_vulkan_staged_transfer_flag_bits_t;

typedef uint32_t iree_hal_vulkan_staged_transfer_flags_t;

typedef enum iree_hal_vulkan_staged_transfer_host_kind_e {
  // Host memory span feeds or receives staged bytes directly.
  IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_MEMORY = 0,

  // Proactor-backed file feeds or receives staged bytes asynchronously.
  IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_ASYNC_FILE = 1,
} iree_hal_vulkan_staged_transfer_host_kind_t;

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

  // Bytes completed by the current partial async file operation.
  iree_host_size_t file_progress;

  // Async read operation storage.
  iree_async_file_read_operation_t read_op;

  // Async write operation storage.
  iree_async_file_write_operation_t write_op;
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

  // Optional file retaining the host endpoint lifetime.
  iree_hal_file_t* file;

  // Async file handle used for proactor-backed transfers. Borrowed from |file|.
  iree_async_file_t* async_file;

  // File byte offset for proactor-backed file transfers.
  uint64_t file_offset;

  // Host bytes read for uploads or written for downloads.
  iree_byte_span_t host_contents;

  // Byte offset into |host_contents| for the first requested byte.
  iree_host_size_t host_offset;

  // User buffer being copied to or from.
  iree_hal_buffer_t* buffer;

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

  // Host-side endpoint strategy used by this transfer.
  iree_hal_vulkan_staged_transfer_host_kind_t host_kind;

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

  if (should_complete) {
    iree_hal_vulkan_queue_staging_ring_cancel_waiter(transfer->ring,
                                                     &transfer->slot_waiter);
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
  chunk->file_progress = 0;
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

static iree_status_t iree_hal_vulkan_staged_transfer_submit_copy(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk);

static iree_status_t iree_hal_vulkan_staged_transfer_submit_next_read(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk);

static iree_status_t iree_hal_vulkan_staged_transfer_submit_next_write(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk);

static void iree_hal_vulkan_staged_transfer_read_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_vulkan_staged_transfer_chunk_t* chunk =
      (iree_hal_vulkan_staged_transfer_chunk_t*)user_data;

  if (iree_status_is_ok(status) && chunk->read_op.bytes_read > 0) {
    chunk->file_progress += chunk->read_op.bytes_read;
    if (chunk->file_progress < (iree_host_size_t)chunk->length) {
      status = iree_hal_vulkan_staged_transfer_submit_next_read(chunk);
      if (iree_status_is_ok(status)) {
        iree_hal_resource_release(&chunk->transfer->resource);
        return;
      }
    }
  } else if (iree_status_is_ok(status) &&
             chunk->file_progress < (iree_host_size_t)chunk->length) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "short read: requested %" PRIhsz " bytes, got %" PRIhsz,
        (iree_host_size_t)chunk->length, chunk->file_progress);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_staged_transfer_submit_copy(chunk);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_staged_transfer_chunk_fail(chunk, status);
  }
  iree_hal_resource_release(&chunk->transfer->resource);
}

static void iree_hal_vulkan_staged_transfer_write_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_vulkan_staged_transfer_chunk_t* chunk =
      (iree_hal_vulkan_staged_transfer_chunk_t*)user_data;

  if (iree_status_is_ok(status) && chunk->write_op.bytes_written > 0) {
    chunk->file_progress += chunk->write_op.bytes_written;
    if (chunk->file_progress < (iree_host_size_t)chunk->length) {
      status = iree_hal_vulkan_staged_transfer_submit_next_write(chunk);
      if (iree_status_is_ok(status)) {
        iree_hal_resource_release(&chunk->transfer->resource);
        return;
      }
    }
  } else if (iree_status_is_ok(status) &&
             chunk->file_progress < (iree_host_size_t)chunk->length) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "short write: requested %" PRIhsz " bytes, wrote %" PRIhsz,
        (iree_host_size_t)chunk->length, chunk->file_progress);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_staged_transfer_chunk_finish(chunk,
                                                 /*did_transfer_bytes=*/true);
  } else {
    iree_hal_vulkan_staged_transfer_chunk_fail(chunk, status);
  }
  iree_hal_resource_release(&chunk->transfer->resource);
}

static iree_status_t iree_hal_vulkan_staged_transfer_submit_next_read(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk) {
  iree_hal_vulkan_staged_transfer_t* transfer = chunk->transfer;
  const iree_host_size_t remaining_length =
      (iree_host_size_t)chunk->length - chunk->file_progress;
  iree_async_operation_zero(&chunk->read_op.base, sizeof(chunk->read_op));
  iree_async_operation_initialize(
      &chunk->read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_vulkan_staged_transfer_read_complete, chunk);
  chunk->read_op.file = transfer->async_file;
  chunk->read_op.offset =
      transfer->file_offset + chunk->transfer_offset + chunk->file_progress;
  chunk->read_op.buffer = iree_async_span_from_ptr(
      chunk->slot->host_span.data + chunk->file_progress, remaining_length);
  iree_hal_resource_retain(&transfer->resource);
  iree_status_t status = iree_async_proactor_submit_one(
      transfer->queue->proactor, &chunk->read_op.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_release(&transfer->resource);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_staged_transfer_submit_next_write(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk) {
  iree_hal_vulkan_staged_transfer_t* transfer = chunk->transfer;
  const iree_host_size_t remaining_length =
      (iree_host_size_t)chunk->length - chunk->file_progress;
  iree_async_operation_zero(&chunk->write_op.base, sizeof(chunk->write_op));
  iree_async_operation_initialize(
      &chunk->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_vulkan_staged_transfer_write_complete, chunk);
  chunk->write_op.file = transfer->async_file;
  chunk->write_op.offset =
      transfer->file_offset + chunk->transfer_offset + chunk->file_progress;
  chunk->write_op.buffer = iree_async_span_from_ptr(
      chunk->slot->host_span.data + chunk->file_progress, remaining_length);
  iree_hal_resource_retain(&transfer->resource);
  iree_status_t status = iree_async_proactor_submit_one(
      transfer->queue->proactor, &chunk->write_op.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_resource_release(&transfer->resource);
  }
  return status;
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
    if (iree_status_is_ok(status) &&
        transfer->host_kind == IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_MEMORY) {
      memcpy(transfer->host_contents.data + transfer->host_offset +
                 chunk->transfer_offset,
             chunk->slot->host_span.data, (iree_host_size_t)chunk->length);
    } else if (iree_status_is_ok(status)) {
      chunk->file_progress = 0;
      status = iree_hal_vulkan_staged_transfer_submit_next_write(chunk);
      if (iree_status_is_ok(status)) return;
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
    if (transfer->host_kind == IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_MEMORY) {
      memcpy(chunk->slot->host_span.data,
             transfer->host_contents.data + transfer->host_offset +
                 chunk->transfer_offset,
             (iree_host_size_t)chunk->length);
    }
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

static iree_status_t iree_hal_vulkan_staged_transfer_submit_chunk(
    iree_hal_vulkan_staged_transfer_chunk_t* chunk) {
  iree_hal_vulkan_staged_transfer_t* transfer = chunk->transfer;
  chunk->file_progress = 0;
  if (transfer->host_kind == IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_ASYNC_FILE &&
      transfer->kind == IREE_HAL_VULKAN_STAGED_TRANSFER_READ) {
    return iree_hal_vulkan_staged_transfer_submit_next_read(chunk);
  }
  return iree_hal_vulkan_staged_transfer_submit_copy(chunk);
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

    iree_status_t status = iree_hal_vulkan_staged_transfer_submit_chunk(chunk);
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
    iree_hal_vulkan_staged_transfer_host_kind_t host_kind,
    iree_byte_span_t host_contents, iree_host_size_t host_offset,
    iree_async_file_t* async_file, uint64_t file_offset,
    iree_hal_file_t* lifetime_file, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    iree_hal_vulkan_staged_transfer_flags_t flags,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_staged_transfer_t** out_transfer) {
  *out_transfer = NULL;
  const iree_hal_vulkan_staged_transfer_flags_t known_flags =
      IREE_HAL_VULKAN_STAGED_TRANSFER_FLAG_CAPTURE_HOST_RANGE;
  if (iree_any_bit_set(flags, ~known_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan staged transfer flags: 0x%" PRIx32, flags);
  }
  const bool captures_host_range = iree_all_bits_set(
      flags, IREE_HAL_VULKAN_STAGED_TRANSFER_FLAG_CAPTURE_HOST_RANGE);
  if (captures_host_range && kind != IREE_HAL_VULKAN_STAGED_TRANSFER_READ) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan staged transfer host capture is only valid for uploads");
  }

  iree_host_size_t host_length = 0;
  switch (host_kind) {
    case IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_MEMORY:
      if (length > IREE_HOST_SIZE_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "Vulkan staged transfer length exceeds host addressable size");
      }
      host_length = (iree_host_size_t)length;
      if (host_offset > host_contents.data_length ||
          host_length > host_contents.data_length - host_offset) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "Vulkan staged transfer host range exceeds available contents");
      }
      if (length != 0 && !host_contents.data) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Vulkan staged transfer host range must be non-null");
      }
      break;
    case IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_ASYNC_FILE:
      if (captures_host_range) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Vulkan async file staged transfers cannot capture host ranges");
      }
      if (!lifetime_file || !async_file) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Vulkan async file staged transfers require an async file handle");
      }
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported Vulkan staged transfer host kind %u",
                              (uint32_t)host_kind);
  }

  iree_hal_vulkan_queue_staging_ring_t* ring =
      kind == IREE_HAL_VULKAN_STAGED_TRANSFER_READ
          ? queue->upload_staging_ring
          : queue->download_staging_ring;
  if (!ring) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue staging ring is not initialized");
  }

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
  const iree_host_size_t captured_host_length =
      captures_host_range ? host_length : 0;
  if (!iree_host_size_checked_add(total_size, captured_host_length,
                                  &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staged transfer allocation overflows");
  }

  iree_hal_vulkan_staged_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator, total_size,
                                             (void**)&transfer));
  memset(transfer, 0, total_size);
  transfer->chunks = (iree_hal_vulkan_staged_transfer_chunk_t*)(transfer + 1);
  uint8_t* captured_host_data = (uint8_t*)transfer->chunks + chunks_size;
  if (captures_host_range && captured_host_length != 0) {
    memcpy(captured_host_data, host_contents.data + host_offset,
           captured_host_length);
    host_contents =
        iree_make_byte_span(captured_host_data, captured_host_length);
    host_offset = 0;
  }

  iree_hal_resource_initialize(&iree_hal_vulkan_staged_transfer_vtable,
                               &transfer->resource);
  transfer->host_allocator = queue->host_allocator;
  iree_slim_mutex_initialize(&transfer->mutex);
  transfer->queue = queue;
  transfer->ring = ring;
  transfer->file = lifetime_file;
  iree_hal_file_retain(transfer->file);
  transfer->async_file = async_file;
  transfer->file_offset = file_offset;
  transfer->host_contents = host_contents;
  transfer->host_offset = host_offset;
  transfer->buffer = buffer;
  iree_hal_buffer_retain(transfer->buffer);
  transfer->buffer_offset = buffer_offset;
  transfer->requested_length = length;
  transfer->kind = kind;
  transfer->host_kind = host_kind;
  transfer->chunk_count = ring->slot_count;
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
  iree_hal_vulkan_staged_transfer_host_kind_t host_kind =
      IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_MEMORY;
  iree_byte_span_t file_contents = iree_byte_span_empty();
  iree_host_size_t host_offset = 0;
  iree_async_file_t* async_file = NULL;
  if (iree_hal_memory_file_isa(file)) {
    if (file_offset > IREE_HOST_SIZE_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan staged transfer host file offset exceeds addressable size");
    }
    host_offset = (iree_host_size_t)file_offset;
    IREE_RETURN_IF_ERROR(iree_hal_memory_file_contents(file, &file_contents));
  } else if ((async_file = iree_hal_file_async_handle(file)) != NULL) {
    host_kind = IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_ASYNC_FILE;
  } else {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan staged transfers require a memory file or proactor-backed "
        "async file handle");
  }

  iree_hal_vulkan_staged_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_staged_transfer_create(
      queue, kind, host_kind, file_contents, host_offset, async_file,
      file_offset, file, buffer, buffer_offset, length,
      IREE_HAL_VULKAN_STAGED_TRANSFER_FLAG_NONE, signal_semaphore_list,
      &transfer));

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

static iree_status_t iree_hal_vulkan_queue_submit_staged_update(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_byte_span_t source_contents = iree_make_byte_span(
      (uint8_t*)source_buffer + source_offset, (iree_host_size_t)length);
  iree_hal_vulkan_staged_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_staged_transfer_create(
      queue, IREE_HAL_VULKAN_STAGED_TRANSFER_READ,
      IREE_HAL_VULKAN_STAGED_TRANSFER_HOST_MEMORY, source_contents,
      /*host_offset=*/0, /*async_file=*/NULL, /*file_offset=*/0,
      /*lifetime_file=*/NULL, target_buffer, target_offset, length,
      IREE_HAL_VULKAN_STAGED_TRANSFER_FLAG_CAPTURE_HOST_RANGE,
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
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_validate_recordable_backing(
        target_buffer, IREE_SV("update target"));
  }

  bool use_native_update = false;
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_can_record_update_native(
        target_buffer, target_offset, length, &use_native_update);
  }
  if (iree_status_is_ok(status) && length != 0 && !use_native_update) {
    status = iree_hal_vulkan_queue_submit_staged_update(
        queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        /*payload_storage_length=*/0, /*out_payload_storage=*/NULL,
        &submission);
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
    status =
        iree_hal_vulkan_queue_allocate_native_command_buffer(queue, submission);
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
  if (iree_status_is_ok(status) && target_is_native &&
      iree_hal_vulkan_queue_file_supports_staged_transfer(source_file)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_queue_submit_staged_transfer(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_STAGED_TRANSFER_READ, source_file, source_offset,
        target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan queue read requires native target buffer storage and either "
        "native device-visible file storage or staged file transfer support");
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
      iree_hal_vulkan_queue_file_supports_staged_transfer(target_file)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_queue_submit_staged_transfer(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_STAGED_TRANSFER_WRITE, target_file, target_offset,
        source_buffer, source_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan queue write requires native source buffer storage and either "
        "native device-visible file storage or staged file transfer support");
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_validate_dispatch_flags(
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_arguments(flags) ||
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan queue_dispatch custom dispatch arguments are unsupported");
  }
  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION |
      IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES;
  if (iree_any_bit_set(flags, ~supported_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue_dispatch flags: 0x%" PRIx64, flags);
  }
  const iree_hal_dispatch_flags_t indirect_parameter_flags =
      flags & (IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
               IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
  if (indirect_parameter_flags ==
      (IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
       IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue_dispatch cannot use both static and dynamic indirect "
        "workgroup parameters");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_validate_dispatch_shape(
    const iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags) {
  if (config.workgroup_size[0] != 0 || config.workgroup_size[1] != 0 ||
      config.workgroup_size[2] != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan queue_dispatch workgroup size overrides are unsupported");
  }
  if (config.dynamic_workgroup_local_memory != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan queue_dispatch dynamic workgroup local memory is unsupported");
  }
  (void)flags;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_validate_dispatch_storage_usage(
    iree_hal_buffer_t* buffer) {
  const iree_hal_buffer_usage_t allowed_usage =
      iree_hal_buffer_allowed_usage(buffer);
  if (iree_any_bit_set(allowed_usage, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
    return iree_ok_status();
  }
#if IREE_STATUS_MODE
  iree_bitfield_string_temp_t allowed_temp;
  iree_bitfield_string_temp_t required_temp;
  iree_string_view_t allowed_usage_string =
      iree_hal_buffer_usage_format(allowed_usage, &allowed_temp);
  iree_string_view_t required_usage_string = iree_hal_buffer_usage_format(
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE, &required_temp);
  return iree_make_status(
      IREE_STATUS_PERMISSION_DENIED,
      "requested usage was not specified when the buffer was allocated; buffer "
      "allows %.*s, operation requires one of %.*s",
      (int)allowed_usage_string.size, allowed_usage_string.data,
      (int)required_usage_string.size, required_usage_string.data);
#else
  return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
}

static iree_status_t iree_hal_vulkan_queue_validate_dispatch_binding(
    const iree_hal_buffer_ref_t* binding, VkDescriptorType descriptor_type) {
  if (binding->reserved != 0 || binding->buffer_slot != 0 || !binding->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue_dispatch bindings must be direct non-null buffer "
        "references");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding->buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  switch (descriptor_type) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
      IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
          iree_hal_buffer_allowed_usage(binding->buffer),
          IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ));
      IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
          iree_hal_buffer_allowed_access(binding->buffer),
          IREE_HAL_MEMORY_ACCESS_READ));
      break;
    }
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_queue_validate_dispatch_storage_usage(
              binding->buffer));
      break;
    }
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Vulkan queue_dispatch descriptor type %u is unsupported",
          (uint32_t)descriptor_type);
  }
  iree_device_size_t descriptor_offset = 0;
  iree_device_size_t descriptor_length = 0;
  return iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(binding->buffer),
      binding->offset, binding->length, &descriptor_offset, &descriptor_length);
}

static iree_status_t
iree_hal_vulkan_queue_validate_dispatch_indirect_parameters(
    const iree_hal_buffer_ref_t* workgroup_count_ref) {
  const iree_device_size_t workgroup_count_length = sizeof(uint32_t[3]);
  if (workgroup_count_ref->reserved != 0 ||
      workgroup_count_ref->buffer_slot != 0 || !workgroup_count_ref->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue_dispatch indirect workgroup parameters must use a direct "
        "non-null buffer reference");
  }
  if ((workgroup_count_ref->offset % sizeof(uint32_t)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue_dispatch indirect workgroup parameter offset must be "
        "4-byte aligned");
  }
  if (workgroup_count_ref->length != IREE_HAL_WHOLE_BUFFER &&
      workgroup_count_ref->length < workgroup_count_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan queue_dispatch indirect workgroup "
                            "parameter buffer must contain "
                            "at least uint32_t[3]");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(workgroup_count_ref->buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(workgroup_count_ref->buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(workgroup_count_ref->buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  return iree_hal_buffer_validate_range(workgroup_count_ref->buffer,
                                        workgroup_count_ref->offset,
                                        workgroup_count_length);
}

static iree_status_t iree_hal_vulkan_queue_validate_dispatch_descriptor(
    iree_hal_vulkan_queue_t* queue, const iree_hal_vulkan_pipeline_t* pipeline,
    const iree_hal_buffer_ref_list_t bindings) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(queue->enabled_dispatch_abis,
                         IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan descriptor dispatch ABI is disabled for this queue");
  }
  if (iree_status_is_ok(status) &&
      bindings.count != pipeline->descriptor_binding_count) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Vulkan queue_dispatch provides %" PRIhsz
                         " bindings but descriptor pipeline expects %" PRIhsz,
                         bindings.count, pipeline->descriptor_binding_count);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < pipeline->descriptor_binding_count;
       ++i) {
    const VkDescriptorType descriptor_type =
        pipeline->descriptor_bindings[i].descriptor_type;
    status = iree_hal_vulkan_queue_validate_dispatch_binding(
        &bindings.values[i], descriptor_type);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "binding[%" PRIhsz "]", i);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_validate_dispatch_bda(
    iree_hal_vulkan_queue_t* queue, const iree_hal_vulkan_pipeline_t* pipeline,
    iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(queue->enabled_dispatch_abis,
                         IREE_HAL_VULKAN_DISPATCH_ABI_BDA)) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "Vulkan BDA dispatch ABI is disabled for this queue");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_pipeline_validate_bda_dispatch_abi(
        pipeline, constants, bindings.count, IREE_SV("Vulkan queue_dispatch"));
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < bindings.count;
       ++i) {
    status = iree_hal_vulkan_queue_validate_dispatch_binding(
        &bindings.values[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "binding[%" PRIhsz "]", i);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_validate_dispatch_abi(
    iree_hal_vulkan_queue_t* queue, const iree_hal_vulkan_pipeline_t* pipeline,
    iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings) {
  switch (pipeline->dispatch_abi) {
    case IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR:
      return iree_hal_vulkan_queue_validate_dispatch_descriptor(queue, pipeline,
                                                                bindings);
    case IREE_HAL_VULKAN_DISPATCH_ABI_BDA:
      return iree_hal_vulkan_queue_validate_dispatch_bda(queue, pipeline,
                                                         constants, bindings);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan pipeline has invalid dispatch ABI 0x%08x",
                              pipeline->dispatch_abi);
  }
}

static iree_status_t iree_hal_vulkan_queue_resolve_dispatch_descriptor_binding(
    const iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_vulkan_descriptor_binding_t* descriptor_binding,
    iree_host_size_t binding_ordinal, VkDescriptorBufferInfo* out_buffer_info) {
  if (descriptor_binding->descriptor_type !=
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
      descriptor_binding->descriptor_type !=
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Vulkan queue_dispatch binding %" PRIhsz
                            " uses unsupported descriptor type %u",
                            binding_ordinal,
                            (uint32_t)descriptor_binding->descriptor_type);
  }
  const iree_hal_buffer_ref_t* binding =
      &submission->dispatch.bindings[binding_ordinal];
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_validate_dispatch_binding(
      binding, descriptor_binding->descriptor_type));

  iree_device_size_t descriptor_offset = 0;
  iree_device_size_t descriptor_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(binding->buffer),
      binding->offset, binding->length, &descriptor_offset,
      &descriptor_length));
  if (descriptor_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan queue_dispatch binding %" PRIhsz
                            " resolved to an empty buffer range",
                            binding_ordinal);
  }

  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(binding->buffer, &backing_buffer));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(backing_buffer);
  if (!iree_hal_vulkan_buffer_isa(allocated_buffer) &&
      !iree_hal_vulkan_sparse_buffer_isa(allocated_buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue_dispatch binding %" PRIhsz
                            " buffer is not backed by the Vulkan HAL",
                            binding_ordinal);
  }

  iree_device_size_t backing_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      binding->buffer, backing_buffer, descriptor_offset, &backing_offset));

  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_handle(backing_buffer, &memory, &buffer));
  (void)memory;

  *out_buffer_info = (VkDescriptorBufferInfo){
      .buffer = buffer,
      .offset = (VkDeviceSize)backing_offset,
      .range = (VkDeviceSize)descriptor_length,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_resolve_dispatch_bda_binding(
    const iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_vulkan_pipeline_t* pipeline,
    iree_host_size_t binding_ordinal, VkDeviceAddress* out_device_address) {
  *out_device_address = 0;
  const iree_hal_buffer_ref_t* binding =
      &submission->dispatch.bindings[binding_ordinal];

  iree_device_size_t binding_offset = 0;
  iree_device_size_t binding_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(binding->buffer),
      binding->offset, binding->length, &binding_offset, &binding_length));
  if (binding_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan queue_dispatch binding %" PRIhsz
                            " resolved to an empty buffer range",
                            binding_ordinal);
  }

  VkDeviceAddress buffer_address = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_device_address(binding->buffer, &buffer_address));
  if (buffer_address == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue_dispatch binding %" PRIhsz
                            " buffer has no device address",
                            binding_ordinal);
  }
  if (binding_offset > UINT64_MAX - buffer_address) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan queue_dispatch binding %" PRIhsz
                            " device address overflows",
                            binding_ordinal);
  }
  const VkDeviceAddress device_address = buffer_address + binding_offset;
  if (binding_length > UINT64_MAX - device_address) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan queue_dispatch binding %" PRIhsz
                            " device range overflows",
                            binding_ordinal);
  }
  if (binding_ordinal < pipeline->bda.binding_requirement_count) {
    const iree_hal_vulkan_bda_binding_requirement_t* requirement =
        &pipeline->bda.binding_requirements[binding_ordinal];
    if (requirement->minimum_alignment > 1 &&
        (device_address & (requirement->minimum_alignment - 1)) != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue_dispatch binding %" PRIhsz
                              " device address 0x%" PRIx64
                              " does not satisfy BDA alignment %u",
                              binding_ordinal, (uint64_t)device_address,
                              requirement->minimum_alignment);
    }
    if (binding_length < requirement->minimum_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan queue_dispatch binding %" PRIhsz " has length %" PRIdsz
          " but BDA pipeline requires at least %" PRIu64 " bytes",
          binding_ordinal, binding_length, requirement->minimum_length);
    }
  }
  *out_device_address = device_address;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_resolve_dispatch_indirect_parameters(
    const iree_hal_vulkan_queue_pending_submission_t* submission,
    VkBuffer* out_handle, VkDeviceSize* out_offset) {
  *out_handle = VK_NULL_HANDLE;
  *out_offset = 0;
  const iree_hal_buffer_ref_t* workgroup_count_ref =
      &submission->dispatch.config.workgroup_count_ref;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_queue_validate_dispatch_indirect_parameters(
          workgroup_count_ref));

  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing(
      workgroup_count_ref->buffer, &backing_buffer));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(backing_buffer);
  if (!iree_hal_vulkan_buffer_isa(allocated_buffer) &&
      !iree_hal_vulkan_sparse_buffer_isa(allocated_buffer)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan queue_dispatch indirect workgroup parameter buffer is not "
        "backed by the Vulkan HAL");
  }

  VkDeviceMemory memory = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_handle(backing_buffer, &memory, out_handle));
  (void)memory;
  iree_device_size_t backing_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      workgroup_count_ref->buffer, backing_buffer, workgroup_count_ref->offset,
      &backing_offset));
  *out_offset = (VkDeviceSize)backing_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_record_dispatch_descriptor_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_vulkan_pipeline_t* pipeline) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  status = iree_hal_vulkan_queue_allocate_native_command_buffer_under_lock(
      queue, submission);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_profile_prepare_native_timestamps_under_lock(
        queue, submission);
  }

  iree_hal_vulkan_command_buffer_descriptor_requirements_t
      descriptor_requirements;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_calculate_dispatch_descriptor_requirements(
        pipeline, &descriptor_requirements);
  }

  VkDescriptorSet inline_descriptor_sets
      [IREE_HAL_VULKAN_QUEUE_DISPATCH_INLINE_SET_CAPACITY] = {0};
  VkDescriptorSet* descriptor_sets = inline_descriptor_sets;
  if (iree_status_is_ok(status) && descriptor_requirements.set_count >
                                       IREE_ARRAYSIZE(inline_descriptor_sets)) {
    status = iree_allocator_malloc_array(
        queue->host_allocator, descriptor_requirements.set_count,
        sizeof(descriptor_sets[0]), (void**)&descriptor_sets);
  }
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  if (iree_status_is_ok(status) && descriptor_requirements.set_count != 0) {
    status = iree_hal_vulkan_queue_acquire_native_descriptor_pool_under_lock(
        queue, submission, &descriptor_requirements, &descriptor_pool);
  }
  if (iree_status_is_ok(status) && descriptor_requirements.set_count != 0) {
    VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = descriptor_requirements.set_count,
        .pSetLayouts = pipeline->descriptor_set_layouts,
    };
    status = iree_vkAllocateDescriptorSets(IREE_VULKAN_DEVICE(&queue->syms),
                                           queue->logical_device,
                                           &allocate_info, descriptor_sets);
  }

  VkDescriptorBufferInfo inline_buffer_infos
      [IREE_HAL_VULKAN_QUEUE_DISPATCH_INLINE_BINDING_CAPACITY] = {0};
  VkDescriptorBufferInfo* buffer_infos = inline_buffer_infos;
  VkWriteDescriptorSet inline_write_infos
      [IREE_HAL_VULKAN_QUEUE_DISPATCH_INLINE_BINDING_CAPACITY] = {0};
  VkWriteDescriptorSet* write_infos = inline_write_infos;
  if (iree_status_is_ok(status) && pipeline->descriptor_binding_count >
                                       IREE_ARRAYSIZE(inline_buffer_infos)) {
    status = iree_allocator_malloc_array(
        queue->host_allocator, pipeline->descriptor_binding_count,
        sizeof(buffer_infos[0]), (void**)&buffer_infos);
  }
  if (iree_status_is_ok(status) &&
      pipeline->descriptor_binding_count > IREE_ARRAYSIZE(inline_write_infos)) {
    status = iree_allocator_malloc_array(
        queue->host_allocator, pipeline->descriptor_binding_count,
        sizeof(write_infos[0]), (void**)&write_infos);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < pipeline->descriptor_binding_count;
       ++i) {
    const iree_hal_vulkan_descriptor_binding_t* descriptor_binding =
        &pipeline->descriptor_bindings[i];
    status = iree_hal_vulkan_queue_resolve_dispatch_descriptor_binding(
        submission, descriptor_binding, i, &buffer_infos[i]);
    if (iree_status_is_ok(status)) {
      write_infos[i] = (VkWriteDescriptorSet){
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = pipeline->push_descriptors.enabled
                        ? VK_NULL_HANDLE
                        : descriptor_sets[descriptor_binding->set_ordinal],
          .dstBinding = descriptor_binding->binding,
          .dstArrayElement = descriptor_binding->array_element,
          .descriptorCount = 1,
          .descriptorType = descriptor_binding->descriptor_type,
          .pBufferInfo = &buffer_infos[i],
      };
    }
  }
  if (iree_status_is_ok(status) && pipeline->descriptor_binding_count != 0 &&
      !pipeline->push_descriptors.enabled) {
    iree_vkUpdateDescriptorSets(
        IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device,
        (uint32_t)pipeline->descriptor_binding_count, write_infos,
        /*descriptorCopyCount=*/0, /*pDescriptorCopies=*/NULL);
  }

  VkBuffer indirect_parameters_handle = VK_NULL_HANDLE;
  VkDeviceSize indirect_parameters_offset = 0;
  if (iree_status_is_ok(status) &&
      iree_hal_dispatch_uses_indirect_parameters(submission->dispatch.flags)) {
    status = iree_hal_vulkan_queue_resolve_dispatch_indirect_parameters(
        submission, &indirect_parameters_handle, &indirect_parameters_offset);
  }

  if (iree_status_is_ok(status)) {
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    status = iree_vkBeginCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                       submission->native_command_buffer,
                                       &begin_info);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_profile_reset_native_timestamps(queue, submission);
    iree_hal_vulkan_queue_profile_write_timestamp_begin(queue, submission);
    if (pipeline->push_descriptors.enabled &&
        pipeline->descriptor_binding_count != 0) {
      iree_vkCmdPushDescriptorSetKHR(
          IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
          VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout,
          pipeline->push_descriptors.set_ordinal,
          (uint32_t)pipeline->descriptor_binding_count, write_infos);
    } else if (descriptor_requirements.set_count != 0) {
      iree_vkCmdBindDescriptorSets(
          IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
          VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, /*firstSet=*/0,
          descriptor_requirements.set_count, descriptor_sets,
          /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/NULL);
    }
    if (submission->dispatch.constants_data_length != 0) {
      iree_vkCmdPushConstants(
          IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
          pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
          (uint32_t)submission->dispatch.constants_data_length,
          submission->dispatch.constants_data);
    }
    iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&queue->syms),
                           submission->native_command_buffer,
                           VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
    if (submission->profile.query_pool &&
        submission->profile.dispatch_base_query !=
            IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
      iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(&queue->syms),
                                submission->native_command_buffer,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                submission->profile.query_pool,
                                submission->profile.dispatch_base_query);
    }
    if (indirect_parameters_handle) {
      iree_vkCmdDispatchIndirect(
          IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
          indirect_parameters_handle, indirect_parameters_offset);
    } else {
      iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&queue->syms),
                         submission->native_command_buffer,
                         submission->dispatch.config.workgroup_count[0],
                         submission->dispatch.config.workgroup_count[1],
                         submission->dispatch.config.workgroup_count[2]);
    }
    if (submission->profile.query_pool &&
        submission->profile.dispatch_base_query !=
            IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
      iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(&queue->syms),
                                submission->native_command_buffer,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                submission->profile.query_pool,
                                submission->profile.dispatch_base_query + 1);
    }
    iree_hal_vulkan_queue_profile_write_timestamp_end(queue, submission);
    status = iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                     submission->native_command_buffer);
  }

  if (write_infos != inline_write_infos) {
    iree_allocator_free(queue->host_allocator, write_infos);
  }
  if (buffer_infos != inline_buffer_infos) {
    iree_allocator_free(queue->host_allocator, buffer_infos);
  }
  if (descriptor_sets != inline_descriptor_sets) {
    iree_allocator_free(queue->host_allocator, descriptor_sets);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_record_dispatch_bda_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_vulkan_pipeline_t* pipeline) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  status = iree_hal_vulkan_queue_allocate_native_command_buffer_under_lock(
      queue, submission);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_profile_prepare_native_timestamps_under_lock(
        queue, submission);
  }

  const iree_host_size_t binding_count =
      pipeline->bda.binding_count_known ? pipeline->binding_count
                                        : submission->dispatch.binding_count;
  iree_device_size_t binding_table_length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)binding_count,
                                    pipeline->bda.binding_table_entry_length,
                                    &binding_table_length)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan BDA binding table length overflows");
  }

  VkDeviceAddress binding_table_address = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_acquire_bda_publication_under_lock(
        queue, submission, binding_table_length, &binding_table_address);
  }
  const iree_hal_vulkan_command_buffer_bda_publication_t publication =
      iree_hal_vulkan_queue_bda_publication_for_lease(
          &submission->bda_publication_lease);
  if (iree_status_is_ok(status) && binding_count != 0) {
    uint64_t* binding_table = (uint64_t*)publication.host_span.data;
    for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < binding_count;
         ++i) {
      VkDeviceAddress device_address = 0;
      status = iree_hal_vulkan_queue_resolve_dispatch_bda_binding(
          submission, pipeline, i, &device_address);
      if (iree_status_is_ok(status)) {
        binding_table[i] = device_address;
      }
    }
  }
  if (iree_status_is_ok(status) && binding_table_length != 0) {
    status = iree_hal_vulkan_queue_flush_bda_publication_lease(
        &submission->bda_publication_lease);
  }

  VkBuffer indirect_parameters_handle = VK_NULL_HANDLE;
  VkDeviceSize indirect_parameters_offset = 0;
  if (iree_status_is_ok(status) &&
      iree_hal_dispatch_uses_indirect_parameters(submission->dispatch.flags)) {
    status = iree_hal_vulkan_queue_resolve_dispatch_indirect_parameters(
        submission, &indirect_parameters_handle, &indirect_parameters_offset);
  }

  if (iree_status_is_ok(status)) {
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    status = iree_vkBeginCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                       submission->native_command_buffer,
                                       &begin_info);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_profile_reset_native_timestamps(queue, submission);
    if (binding_table_length != 0) {
      VkMemoryBarrier2 memory_barrier = {
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
          .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
          .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
      };
      VkDependencyInfo dependency_info = {
          .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
          .memoryBarrierCount = 1,
          .pMemoryBarriers = &memory_barrier,
      };
      iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(&queue->syms),
                                 submission->native_command_buffer,
                                 &dependency_info);
    }
    iree_hal_vulkan_queue_profile_write_timestamp_begin(queue, submission);
    iree_hal_vulkan_bda_dispatch_root_v1_t root = {
        .binding_table_address = binding_table_address,
        .constants_address = 0,
        .binding_base = 0,
        .constant_base = 0,
        .flags = 0,
        .reserved0 = 0,
    };
    iree_vkCmdPushConstants(
        IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
        pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
        pipeline->bda.root_push_constant_offset, sizeof(root), &root);
    if (submission->dispatch.constants_data_length != 0) {
      iree_vkCmdPushConstants(
          IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
          pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
          pipeline->bda.constant_push_constant_offset,
          (uint32_t)submission->dispatch.constants_data_length,
          submission->dispatch.constants_data);
    }
    iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&queue->syms),
                           submission->native_command_buffer,
                           VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
    if (submission->profile.query_pool &&
        submission->profile.dispatch_base_query !=
            IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
      iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(&queue->syms),
                                submission->native_command_buffer,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                submission->profile.query_pool,
                                submission->profile.dispatch_base_query);
    }
    if (indirect_parameters_handle) {
      iree_vkCmdDispatchIndirect(
          IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
          indirect_parameters_handle, indirect_parameters_offset);
    } else {
      iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&queue->syms),
                         submission->native_command_buffer,
                         submission->dispatch.config.workgroup_count[0],
                         submission->dispatch.config.workgroup_count[1],
                         submission->dispatch.config.workgroup_count[2]);
    }
    if (submission->profile.query_pool &&
        submission->profile.dispatch_base_query !=
            IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
      iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(&queue->syms),
                                submission->native_command_buffer,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                submission->profile.query_pool,
                                submission->profile.dispatch_base_query + 1);
    }
    iree_hal_vulkan_queue_profile_write_timestamp_end(queue, submission);
    status = iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                     submission->native_command_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_record_dispatch_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
      submission->dispatch.executable, submission->dispatch.function_ordinal,
      &pipeline));
  switch (pipeline->dispatch_abi) {
    case IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR:
      return iree_hal_vulkan_queue_record_dispatch_descriptor_native(
          queue, submission, pipeline);
    case IREE_HAL_VULKAN_DISPATCH_ABI_BDA:
      return iree_hal_vulkan_queue_record_dispatch_bda_native(queue, submission,
                                                              pipeline);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan pipeline has invalid dispatch ABI 0x%08x",
                              pipeline->dispatch_abi);
  }
}

static iree_status_t iree_hal_vulkan_queue_calculate_execute_payload_layout(
    iree_host_size_t binding_table_count,
    iree_host_size_t bda_binding_slot_count,
    iree_host_size_t* out_payload_storage_length,
    iree_host_size_t* out_binding_table_offset,
    iree_host_size_t* out_bda_binding_slots_offset) {
  return IREE_STRUCT_LAYOUT(
      0, out_payload_storage_length,
      IREE_STRUCT_FIELD_ALIGNED(binding_table_count, iree_hal_buffer_binding_t,
                                iree_alignof(iree_hal_buffer_binding_t),
                                out_binding_table_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          bda_binding_slot_count,
          iree_hal_vulkan_command_buffer_bda_binding_slot_t,
          iree_alignof(iree_hal_vulkan_command_buffer_bda_binding_slot_t),
          out_bda_binding_slots_offset));
}

static iree_status_t iree_hal_vulkan_queue_calculate_dispatch_payload_layout(
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    iree_host_size_t* out_payload_storage_length,
    iree_host_size_t* out_constants_data_offset,
    iree_host_size_t* out_bindings_offset) {
  return IREE_STRUCT_LAYOUT(
      0, out_payload_storage_length,
      IREE_STRUCT_FIELD(constants.data_length, uint8_t,
                        out_constants_data_offset),
      IREE_STRUCT_FIELD_ALIGNED(binding_count, iree_hal_buffer_ref_t,
                                iree_alignof(iree_hal_buffer_ref_t),
                                out_bindings_offset));
}

iree_status_t iree_hal_vulkan_queue_submit_dispatch(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t function_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  iree_status_t status = iree_hal_vulkan_queue_validate_dispatch_flags(flags);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_dispatch_shape(config, flags);
  }
  if (iree_status_is_ok(status) &&
      constants.data_length % sizeof(uint32_t) != 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue_dispatch constants must be 4-byte aligned");
  }
  if (iree_status_is_ok(status) && constants.data_length != 0 &&
      !constants.data) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue_dispatch constant data must be non-null when length is "
        "non-zero");
  }
  if (iree_status_is_ok(status) && constants.data_length > UINT32_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan queue_dispatch constants exceed Vulkan limit %u", UINT32_MAX);
  }
  if (iree_status_is_ok(status) && bindings.count != 0 && !bindings.values) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue_dispatch binding storage is NULL");
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_executable_isa(executable)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue_dispatch executable is not a Vulkan executable");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_executable_lookup_pipeline(
        executable, function_ordinal, &pipeline);
  }
  if (iree_status_is_ok(status) &&
      constants.data_length >
          (iree_host_size_t)pipeline->constant_count * sizeof(uint32_t)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan queue_dispatch provides %" PRIhsz
        " constant bytes but pipeline accepts at most %u",
        constants.data_length,
        (uint32_t)pipeline->constant_count * (uint32_t)sizeof(uint32_t));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_dispatch_abi(queue, pipeline,
                                                         constants, bindings);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_dispatch_uses_indirect_parameters(flags)) {
    status = iree_hal_vulkan_queue_validate_dispatch_indirect_parameters(
        &config.workgroup_count_ref);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_host_size_t dispatch_payload_storage_length = 0;
  iree_host_size_t dispatch_constants_data_offset = 0;
  iree_host_size_t dispatch_bindings_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_calculate_dispatch_payload_layout(
        constants, bindings.count, &dispatch_payload_storage_length,
        &dispatch_constants_data_offset, &dispatch_bindings_offset);
  }

  iree_byte_span_t dispatch_payload_storage = iree_byte_span_empty();
  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DISPATCH,
        (iree_hal_host_call_t){0}, /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        dispatch_payload_storage_length, &dispatch_payload_storage,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->dispatch.executable = executable;
    iree_hal_executable_retain(executable);
    submission->dispatch.function_ordinal = function_ordinal;
    submission->dispatch.config = config;
    submission->dispatch.constants_data_length = constants.data_length;
    submission->dispatch.binding_count = bindings.count;
    submission->dispatch.flags = flags;
  }
  if (iree_status_is_ok(status) &&
      iree_hal_dispatch_uses_indirect_parameters(flags)) {
    iree_hal_buffer_retain(
        submission->dispatch.config.workgroup_count_ref.buffer);
  }
  if (iree_status_is_ok(status) && constants.data_length != 0) {
    submission->dispatch.constants_data =
        dispatch_payload_storage.data + dispatch_constants_data_offset;
    memcpy(submission->dispatch.constants_data, constants.data,
           constants.data_length);
  }
  if (iree_status_is_ok(status) && bindings.count != 0) {
    submission->dispatch.bindings =
        (iree_hal_buffer_ref_t*)(dispatch_payload_storage.data +
                                 dispatch_bindings_offset);
    memcpy(submission->dispatch.bindings, bindings.values,
           bindings.count * sizeof(submission->dispatch.bindings[0]));
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      iree_hal_buffer_retain(submission->dispatch.bindings[i].buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_record_dispatch_profile_metadata(submission);
  }
  if (iree_status_is_ok(status)) {
    submission->record_native_submission =
        iree_hal_vulkan_queue_record_dispatch_native;
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

static iree_status_t iree_hal_vulkan_queue_record_execute_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (!iree_hal_vulkan_command_buffer_has_native_commands(
          submission->execute.command_buffer)) {
    return iree_ok_status();
  }
  iree_hal_vulkan_command_buffer_descriptor_requirements_t
      descriptor_requirements;
  iree_status_t status =
      iree_hal_vulkan_command_buffer_native_descriptor_pool_requirements(
          submission->execute.command_buffer, &descriptor_requirements);
  iree_device_size_t bda_publication_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_command_buffer_native_bda_publication_length(
        submission->execute.command_buffer, &bda_publication_length);
  }
  const bool has_descriptor_dispatches =
      iree_hal_vulkan_command_buffer_has_descriptor_dispatches(
          submission->execute.command_buffer);
  bool acquired_native_replay = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_acquire_native_replay_under_lock(
        queue, submission, descriptor_requirements, has_descriptor_dispatches,
        bda_publication_length, &acquired_native_replay);
  }
  if (!iree_status_is_ok(status) || acquired_native_replay) {
    return status;
  }

  status = iree_hal_vulkan_queue_allocate_native_command_buffer_under_lock(
      queue, submission);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_profile_prepare_native_timestamps_under_lock(
        queue, submission);
  }
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  if (iree_status_is_ok(status) && descriptor_requirements.set_count != 0) {
    status = iree_hal_vulkan_queue_acquire_native_descriptor_pool_under_lock(
        queue, submission, &descriptor_requirements, &descriptor_pool);
  }
  iree_hal_vulkan_command_buffer_bda_publication_t bda_publication = {0};
  const iree_hal_vulkan_command_buffer_bda_publication_t* bda_publication_ptr =
      NULL;
  if (iree_status_is_ok(status) && bda_publication_length != 0) {
    status = iree_hal_vulkan_queue_acquire_bda_publication_under_lock(
        queue, submission, bda_publication_length,
        &bda_publication.device_address);
  }
  if (iree_status_is_ok(status) && bda_publication_length != 0) {
    bda_publication = iree_hal_vulkan_queue_bda_publication_for_lease(
        &submission->bda_publication_lease);
    bda_publication_ptr = &bda_publication;
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_binding_table_t binding_table = {
        .count = submission->execute.binding_table_count,
        .bindings = submission->execute.binding_table_bindings,
    };
    iree_hal_vulkan_command_buffer_bda_binding_cache_t bda_binding_cache =
        iree_hal_vulkan_queue_execute_bda_binding_cache(submission);
    iree_hal_vulkan_command_buffer_profile_marker_t profile_marker = {
        .query_pool = submission->profile.query_pool,
        .first_query = submission->profile.first_query,
        .query_count = submission->profile.query_count,
        .queue_start_query = submission->profile.queue_start_query,
        .queue_end_query = submission->profile.queue_end_query,
        .dispatch_base_query = submission->profile.dispatch_base_query,
        .dispatch_query_count = submission->profile.dispatch_query_count,
        .recorder = submission->profile.recorder,
        .scope = submission->profile.scope,
        .command_buffer_id =
            iree_hal_vulkan_queue_profile_command_buffer_id(submission),
    };
    status = iree_hal_vulkan_command_buffer_record_native(
        submission->execute.command_buffer, &queue->syms, queue->logical_device,
        &queue->debug_utils, queue->builtins, submission->native_command_buffer,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, descriptor_pool,
        binding_table, bda_publication_ptr, &bda_binding_cache,
        profile_marker.query_pool ? &profile_marker : NULL,
        queue->host_allocator);
  }
  if (iree_status_is_ok(status) && bda_publication_length != 0) {
    status = iree_hal_vulkan_queue_flush_bda_publication_lease(
        &submission->bda_publication_lease);
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
  iree_device_size_t execute_bda_publication_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_command_buffer_native_bda_publication_length(
        command_buffer, &execute_bda_publication_length);
  }
  const iree_host_size_t execute_bda_binding_slot_count =
      execute_bda_publication_length != 0 ? command_buffer->binding_count : 0;
  iree_host_size_t execute_payload_storage_length = 0;
  iree_host_size_t execute_binding_table_offset = 0;
  iree_host_size_t execute_bda_binding_slots_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_calculate_execute_payload_layout(
        command_buffer->binding_count, execute_bda_binding_slot_count,
        &execute_payload_storage_length, &execute_binding_table_offset,
        &execute_bda_binding_slots_offset);
  }

  iree_byte_span_t execute_payload_storage = iree_byte_span_empty();
  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE,
        (iree_hal_host_call_t){0}, /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        execute_payload_storage_length, &execute_payload_storage, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->execute.command_buffer = command_buffer;
    iree_hal_command_buffer_retain(command_buffer);
    submission->execute.flags = flags;
    submission->profile.type = queue_event_type;
    submission->execute.binding_table_count = command_buffer->binding_count;
    if (command_buffer->binding_count != 0) {
      submission->execute.binding_table_bindings =
          (iree_hal_buffer_binding_t*)(execute_payload_storage.data +
                                       execute_binding_table_offset);
      memcpy(submission->execute.binding_table_bindings, binding_table.bindings,
             command_buffer->binding_count * sizeof(binding_table.bindings[0]));
      if (!iree_any_bit_set(
              flags, IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
        for (iree_host_size_t i = 0; i < command_buffer->binding_count; ++i) {
          iree_hal_buffer_retain(
              submission->execute.binding_table_bindings[i].buffer);
        }
      }
    }
    submission->execute.bda_binding_slot_count = execute_bda_binding_slot_count;
    if (execute_bda_binding_slot_count != 0) {
      uint8_t* const bda_binding_slots_storage =
          execute_payload_storage.data + execute_bda_binding_slots_offset;
      submission->execute.bda_binding_slots =
          (iree_hal_vulkan_command_buffer_bda_binding_slot_t*)
              bda_binding_slots_storage;
    }
  }
  if (iree_status_is_ok(status)) {
    const uint64_t command_buffer_id =
        iree_hal_vulkan_queue_profile_command_buffer_id(submission);
    status = iree_hal_vulkan_command_buffer_record_profile_metadata(
        command_buffer, queue->profile_recorder, submission->profile.scope,
        command_buffer_id);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_vulkan_command_buffer_has_native_commands(command_buffer)) {
    submission->record_native_submission =
        iree_hal_vulkan_queue_record_execute_native;
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
        /*payload_storage_length=*/0, /*out_payload_storage=*/NULL,
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
