// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_INTERNAL_H_
#define IREE_EXPERIMENTAL_STREAMING_INTERNAL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Compiler support
//===----------------------------------------------------------------------===//

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201102L) && \
    !__STDC_NO_THREADS__
#define iree_thread_local _Thread_local
#elif defined(IREE_COMPILER_MSVC)
#define iree_thread_local __declspec(thread)
#else
#define iree_thread_local static
#endif  // __STDC_NO_THREADS__

typedef uint64_t iree_hal_streaming_deviceptr_t;
typedef iree_host_size_t iree_hal_streaming_device_ordinal_t;

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_buffer_table_t
    iree_hal_streaming_buffer_table_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;
typedef struct iree_hal_streaming_device_registry_t
    iree_hal_streaming_device_registry_t;
typedef struct iree_hal_streaming_event_t iree_hal_streaming_event_t;
typedef struct iree_hal_streaming_graph_t iree_hal_streaming_graph_t;
typedef struct iree_hal_streaming_graph_exec_t iree_hal_streaming_graph_exec_t;
typedef struct iree_hal_streaming_graph_node_t iree_hal_streaming_graph_node_t;
typedef struct iree_hal_streaming_mem_pool_t iree_hal_streaming_mem_pool_t;
typedef struct iree_hal_streaming_module_t iree_hal_streaming_module_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_symbol_t iree_hal_streaming_symbol_t;

//===----------------------------------------------------------------------===//
// Context types
//===----------------------------------------------------------------------===//

// Scheduling policy.
typedef enum iree_hal_streaming_scheduling_mode_e {
  // Automatic scheduling.
  IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO = 0,
  // Spin wait (busy wait).
  IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN,
  // Yield to OS scheduler.
  IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD,
  // Blocking synchronization.
  IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC,
} iree_hal_streaming_scheduling_mode_t;

// Context scheduling and behavior flags.
typedef struct iree_hal_streaming_context_flags_t {
  // Scheduling policy.
  iree_hal_streaming_scheduling_mode_t scheduling_mode;

  // Memory mapping: can map host memory.
  uint64_t map_host_memory : 1;
  // Memory mapping: resize local memory to max.
  uint64_t resize_local_mem_to_max : 1;
} iree_hal_streaming_context_flags_t;

// Context resource limits.
typedef struct iree_hal_streaming_limits_t {
  size_t stack_size;                        // Stack size per GPU thread.
  size_t printf_fifo_size;                  // Printf FIFO buffer size.
  size_t malloc_heap_size;                  // Device malloc heap size.
  size_t dev_runtime_sync_depth;            // Device runtime sync depth.
  size_t dev_runtime_pending_launch_count;  // Pending launch count.
  size_t max_l2_fetch_granularity;          // L2 cache fetch granularity.
  size_t persisting_l2_cache_size;          // Persistent L2 cache size.
} iree_hal_streaming_limits_t;

// Stream context mapped to HAL device.
typedef struct iree_hal_streaming_context_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // Associated device.
  iree_hal_device_t* device;
  iree_hal_streaming_device_ordinal_t device_ordinal;
  iree_hal_streaming_device_t* device_entry;

  // HAL resources.
  iree_hal_allocator_t* device_allocator;
  iree_hal_executable_cache_t* executable_cache;
  iree_status_t loop_status;

  // Context flags.
  iree_hal_streaming_context_flags_t flags;

  // Default stream for this context (always created during context
  // initialization).
  iree_hal_streaming_stream_t* default_stream;

  // Peer access list.
  iree_hal_streaming_context_t** peer_contexts;
  iree_host_size_t peer_count;
  iree_host_size_t peer_capacity;

  // Buffer mapping table.
  iree_hal_streaming_buffer_table_t* buffer_table;

  // Context resource limits.
  iree_hal_streaming_limits_t limits;

  // Synchronization.
  iree_slim_mutex_t mutex;

  // Host allocator.
  iree_allocator_t host_allocator;

  // Stream tracking (non-owning references - streams are not retained).
  // NOTE: Streams are NOT retained by this list to avoid reference cycles.
  // Streams must unregister themselves before destruction.
  iree_hal_streaming_stream_t** streams;  // Non-owning pointers.
  iree_host_size_t stream_count;
  iree_host_size_t stream_capacity;

  // Dedicated mutex for stream list access.
  iree_slim_mutex_t stream_list_mutex;

  // Global context list node pointers for cleanup tracking.
  // These are used to link all contexts in a global list for proper cleanup.
  // Guarded by the context list mutex.
  struct {
    iree_hal_streaming_context_t* next;
    iree_hal_streaming_context_t* prev;
  } context_list_entry;
} iree_hal_streaming_context_t;

//===----------------------------------------------------------------------===//
// Device types
//===----------------------------------------------------------------------===//

// Maximum number of devices supported by the stream HAL.
// This avoids dynamic enumeration overhead during initialization.
#define IREE_HAL_STREAMING_MAX_DEVICES 64

// P2P link information between two devices.
typedef struct iree_hal_streaming_p2p_link_t {
  iree_host_size_t src_device;
  iree_host_size_t dst_device;

  // P2P attributes.
  bool access_supported;             // Basic P2P access.
  bool native_atomic_supported;      // Native atomic operations.
  bool cuda_array_access_supported;  // CUDA/HIP array access.
  int32_t performance_rank;          // Performance ranking (higher is better).

  // Additional link properties.
  uint64_t bandwidth_mbps;  // Estimated bandwidth in MB/s.
  uint64_t latency_ns;      // Estimated latency in nanoseconds.
} iree_hal_streaming_p2p_link_t;

// Device registry entry for multi-device support.
typedef struct iree_hal_streaming_device_t {
  // Device ordinal in the global registry.
  iree_host_size_t ordinal;

  iree_hal_driver_t* driver;
  iree_hal_device_t* hal_device;
  iree_hal_device_info_t info;

  // Device capabilities.
  uint32_t compute_capability_major;
  uint32_t compute_capability_minor;
  iree_device_size_t total_memory;
  iree_device_size_t free_memory;
  bool supports_cooperative_launch;

  // Device properties cache.
  uint32_t max_threads_per_block;
  uint32_t max_block_dim[3];
  uint32_t max_grid_dim[3];
  uint32_t warp_size;
  uint32_t multiprocessor_count;

  // Occupancy calculation properties.
  uint32_t max_threads_per_multiprocessor;
  uint32_t max_blocks_per_multiprocessor;
  uint32_t max_registers_per_multiprocessor;
  uint32_t max_shared_memory_per_multiprocessor;
  uint32_t max_registers_per_block;
  uint32_t max_shared_memory_per_block;

  // Arena block pool for transient host allocations.
  // Shared by all graphs created from this device.
  iree_arena_block_pool_t block_pool;

  // Primary context flags.
  iree_hal_streaming_context_flags_t primary_context_flags;

  // Primary context mutex for thread-safe lazy initialization.
  iree_slim_mutex_t primary_context_mutex;

  // Primary context (lazily created on first access).
  iree_hal_streaming_context_t* primary_context;

  // Primary context reference count.
  // When > 0, the primary context is retained and must not be destroyed.
  // When reaches 0, the primary context is destroyed.
  // Protected by primary_context_mutex.
  int32_t primary_context_ref_count;

  // Memory pools.
  iree_hal_streaming_mem_pool_t* default_mem_pool;
  iree_hal_streaming_mem_pool_t* current_mem_pool;
} iree_hal_streaming_device_t;

// Global device registry for multi-device management.
typedef struct iree_hal_streaming_device_registry_t {
  // Host allocator for internal allocations.
  iree_allocator_t host_allocator;

  // Global initialization state.
  bool initialized;

  iree_slim_mutex_t mutex;

  // HAL driver registry.
  iree_hal_driver_registry_t* driver_registry;

  // P2P topology: array of links between all device pairs.
  iree_hal_streaming_p2p_link_t* p2p_topology;
  // Total size of the topology: device_count * device_count.
  iree_host_size_t p2p_link_count;

  // Fixed-size array of registered devices.
  iree_hal_streaming_device_t devices[IREE_HAL_STREAMING_MAX_DEVICES];
  iree_host_size_t device_count;

  // Global context tracking for cleanup.
  // All created contexts are tracked here to ensure proper cleanup.
  struct {
    iree_slim_mutex_t mutex;
    iree_hal_streaming_context_t* head;
    iree_hal_streaming_context_t* tail;
  } context_list;
} iree_hal_streaming_device_registry_t;

//===----------------------------------------------------------------------===//
// Stream types
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_stream_flag_bits_e {
  IREE_HAL_STREAMING_STREAM_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING = 1ull << 0,
} iree_hal_streaming_stream_flags_t;

// Stream capture status enum (matching CUDA/HIP).
typedef enum iree_hal_streaming_capture_status_e {
  IREE_HAL_STREAMING_CAPTURE_STATUS_NONE = 0,
  IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE = 1,
  IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED = 2,
} iree_hal_streaming_capture_status_t;

// Stream capture mode.
typedef enum iree_hal_streaming_capture_mode_e {
  IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL = 0,
  IREE_HAL_STREAMING_CAPTURE_MODE_THREAD_LOCAL = 1,
  IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED = 2,
} iree_hal_streaming_capture_mode_t;

// Stream capture dependencies update mode.
typedef enum iree_hal_streaming_capture_dependencies_mode_e {
  // Replace the current dependencies with new ones.
  IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET = 0,
  // Add new dependencies to existing ones.
  IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD = 1,
} iree_hal_streaming_capture_dependencies_mode_t;

// Stream for asynchronous execution.
typedef struct iree_hal_streaming_stream_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // Parent context, unowned (to avoid cycles).
  iree_hal_streaming_context_t* context;

  // Stream properties.
  iree_hal_streaming_stream_flags_t flags;
  int priority;

  // Command buffer for batching operations.
  iree_hal_command_buffer_t* command_buffer;

  // Semaphore chain for synchronization.
  iree_hal_semaphore_t* timeline_semaphore;
  uint64_t pending_value;
  uint64_t completed_value;

  // Queue affinity.
  iree_hal_queue_affinity_t queue_affinity;

  // Recorded events on this stream.
  iree_hal_streaming_event_t** recorded_events;
  iree_host_size_t event_count;
  iree_host_size_t event_capacity;

  // Stream capture state.
  iree_hal_streaming_capture_status_t capture_status;
  iree_hal_streaming_capture_mode_t capture_mode;
  iree_hal_streaming_graph_t* capture_graph;
  unsigned long long capture_id;
  iree_hal_streaming_graph_node_t** capture_dependencies;
  iree_host_size_t capture_dependency_count;
  iree_host_size_t capture_dependency_capacity;

  // Synchronization.
  iree_slim_mutex_t mutex;

  // Host allocator.
  iree_allocator_t host_allocator;
} iree_hal_streaming_stream_t;

//===----------------------------------------------------------------------===//
// Module types
//===----------------------------------------------------------------------===//

// Symbol type enumeration.
typedef enum iree_hal_streaming_symbol_type_e {
  IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION = 0,
  IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL = 1,
  IREE_HAL_STREAMING_SYMBOL_TYPE_DATA = 2,
} iree_hal_streaming_symbol_type_t;

// Copy operation: memcpy(dst_offset, src_offset, size).
typedef struct iree_hal_streaming_parameter_copy_op_t {
  // Size in bytes of the copy operation.
  uint16_t size;
  uint16_t reserved;
  // Source offset in parameters buffer, in bytes.
  uint16_t src_offset;
  // Destination offset in constants, in bytes.
  uint16_t dst_offset;
} iree_hal_streaming_parameter_copy_op_t;

// Binding resolve operation: lookup and construct iree_hal_buffer_ref_t.
typedef struct iree_hal_streaming_parameter_resolve_op_t {
  uint32_t reserved;
  // Source offset in parameters buffer, in bytes.
  uint16_t src_offset;
  // Destination binding ordinal.
  uint16_t dst_ordinal;
} iree_hal_streaming_parameter_resolve_op_t;

typedef union iree_hal_streaming_parameter_op_t {
  iree_hal_streaming_parameter_copy_op_t copy;
  iree_hal_streaming_parameter_resolve_op_t resolve;
} iree_hal_streaming_parameter_op_t;

// Function parameter information used for unpacking.
// CUDA-style kernel parameters (usually a list of pointers, but can be packed)
// need to be converted into IREE constants and bindings for dispatch. For
// compatibility we allow bindless constant-only parameters by directly copying
// buffer pointers to the constant storage. The preferred mode for compatibility
// is to resolve buffer pointers to their HAL buffers and pass them in as
// bindings. Though it takes longer to resolve in the dispatch path when using
// graphs we capture only when the graph is recorded and not when it is
// launched. The binding approach is also required in order to use the IREE
// async allocation support.
typedef struct iree_hal_streaming_parameter_info_t {
  // Total size, in bytes, of the final parameter pack.
  uint16_t buffer_size;
  // Total size of constants, in bytes. Includes raw buffer pointers.
  // May include padding/alignment.
  uint16_t constant_bytes;
  // Total number of HAL bindings in the parameters (and resolve ops).
  uint16_t binding_count;
  // Total number of parameter copy operations to perform during unpacking.
  uint16_t copy_count;
  // Copy and resolve ops.
  // Ordered by copies first (copy_count) followed by bindings (binding_count).
  iree_hal_streaming_parameter_op_t* ops;
} iree_hal_streaming_parameter_info_t;

// Symbol metadata structure.
typedef struct iree_hal_streaming_symbol_t {
  // Parent module. Unowned.
  iree_hal_streaming_module_t* module;
  iree_string_view_t name;
  iree_hal_streaming_symbol_type_t type;
  iree_hal_executable_export_ordinal_t export_ordinal;

  // Function attributes (only valid for FUNCTION type).
  iree_hal_occupancy_info_t occupancy_info;
  // TODO(benvanik): replace with occupancy info?
  uint32_t max_threads_per_block;
  uint32_t shared_size_bytes;
  uint32_t local_size_bytes;
  uint32_t num_regs;
  uint32_t max_dynamic_shared_size_bytes;

  // Function parameter information used for unpacking.
  iree_hal_streaming_parameter_info_t parameters;

  // Global/data attributes (only valid for GLOBAL/DATA types).
  iree_hal_streaming_deviceptr_t device_address;
  iree_device_size_t size_bytes;
} iree_hal_streaming_symbol_t;

// Module containing compiled kernels.
typedef struct iree_hal_streaming_module_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // HAL executable resources.
  iree_hal_executable_cache_t* cache;
  iree_hal_executable_t* executable;

  // Symbol metadata.
  iree_hal_streaming_symbol_t* symbols;
  iree_host_size_t symbol_count;

  // File mapping if loaded from file.
  iree_io_file_mapping_t* file_mapping;

  // Context that loaded this module.
  iree_hal_streaming_context_t* context;

  // Host allocator.
  iree_allocator_t host_allocator;
} iree_hal_streaming_module_t;

//===----------------------------------------------------------------------===//
// Event types
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_event_flag_bits_e {
  IREE_HAL_STREAMING_EVENT_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_EVENT_FLAG_BLOCKING_SYNC = 1ull << 0,
  IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING = 1ull << 1,
  IREE_HAL_STREAMING_EVENT_FLAG_INTERPROCESS = 1ull << 2,
} iree_hal_streaming_event_flags_t;

// Event for synchronization.
typedef struct iree_hal_streaming_event_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // Event properties.
  iree_hal_streaming_event_flags_t flags;

  // HAL semaphore.
  iree_hal_semaphore_t* semaphore;
  uint64_t signal_value;

  // Recording stream and context.
  iree_hal_streaming_stream_t* recording_stream;
  iree_hal_streaming_context_t* context;

  // Timing information.
  iree_time_t record_time_ns;

  // Platform-specific IPC handle, if the event is IPC enabled.
  void* ipc_handle;

  // Host allocator.
  iree_allocator_t host_allocator;
} iree_hal_streaming_event_t;

//===----------------------------------------------------------------------===//
// Memory types
//===----------------------------------------------------------------------===//

// Host memory registration flags.
typedef enum iree_hal_streaming_host_register_flag_bits_e {
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT = 0ull,
  // Memory is portable across devices.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE = 1ull << 0,
  // Memory is mapped for device access.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED = 1ull << 1,
  // Write-combined memory.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_WRITE_COMBINED = 1ull << 2,
  // Read-only from device.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_READ_ONLY = 1ull << 3,
} iree_hal_streaming_host_register_flags_t;

// Buffer wrapper for device memory.
typedef struct iree_hal_streaming_buffer_t {
  // Device address obtained from the buffer handle.
  iree_hal_streaming_deviceptr_t device_ptr;

  // Host address, if available.
  void* host_ptr;

  // Total size in bytes of the buffer.
  iree_device_size_t size;

  // HAL buffer.
  iree_hal_buffer_t* buffer;

  // Owning context.
  iree_hal_streaming_context_t* context;

  // Platform-specific memory type.
  int memory_type;

  // Host registration flags (if registered host memory).
  iree_hal_streaming_host_register_flags_t host_register_flags;

  // Platform-specific IPC handle, if the buffer is IPC enabled.
  void* ipc_handle;

  // Read-mostly hint for optimizing memory duplication across devices.
  bool read_mostly_hint;

  // Preferred location device ID for memory residency.
  // -1 indicates CPU preference, >= 0 indicates device ID.
  int32_t preferred_location;

  // Last prefetch location for this memory range.
  // -1 indicates CPU, -2 indicates never prefetched, >= 0 indicates device ID.
  int32_t last_prefetch_location;
} iree_hal_streaming_buffer_t;

// A buffer and an offset into it resolved from a device pointer.
// Device pointers may reference any offset within a buffer.
// The original device pointer is `buffer->device_ptr + offset`.
typedef struct iree_hal_streaming_buffer_ref_t {
  iree_hal_streaming_buffer_t* buffer;
  iree_device_size_t offset;
} iree_hal_streaming_buffer_ref_t;

static inline iree_hal_buffer_ref_t iree_hal_streaming_convert_buffer_ref(
    iree_hal_streaming_buffer_ref_t ref) {
  return iree_hal_make_buffer_ref(ref.buffer->buffer, ref.offset,
                                  IREE_HAL_WHOLE_BUFFER);
}

static inline iree_hal_buffer_ref_t iree_hal_streaming_convert_range_buffer_ref(
    iree_hal_streaming_buffer_ref_t ref, iree_device_size_t length) {
  return iree_hal_make_buffer_ref(ref.buffer->buffer, ref.offset, length);
}

//===----------------------------------------------------------------------===//
// Dispatch types
//===----------------------------------------------------------------------===//

// Dispatch flags for kernel launches.
typedef enum iree_hal_streaming_dispatch_flag_bits_e {
  IREE_HAL_STREAMING_DISPATCH_FLAG_NONE = 0ull,
  // Cooperative kernel launch.
  IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE = 1ull << 0,
} iree_hal_streaming_dispatch_flags_t;

// Dispatch parameters for kernel launches.
typedef struct iree_hal_streaming_dispatch_params_t {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t shared_memory_bytes;
  void* buffer;
  iree_hal_streaming_dispatch_flags_t flags;
} iree_hal_streaming_dispatch_params_t;

//===----------------------------------------------------------------------===//
// Graph types
//===----------------------------------------------------------------------===//

// Graph node types.
enum iree_hal_streaming_graph_node_type_e {
  // Bit indicating the node type is recordable in command buffers.
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE = 1u << 7,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY = 0,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL =
      1 | IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY =
      2 | IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET =
      3 | IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL = 4,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH = 5,
};
typedef uint8_t iree_hal_streaming_graph_node_type_t;

// Returns true if the node type can be recorded into a command buffer.
// Nodes without this bit set will be queue operations.
static bool iree_hal_streaming_graph_node_is_recordable(
    iree_hal_streaming_graph_node_type_t type) {
  return (type & IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE) != 0;
}

// Graph node attribute structures.
typedef struct iree_hal_streaming_graph_kernel_node_attrs_t {
  iree_hal_streaming_symbol_t* symbol;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t shared_memory_bytes;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
} iree_hal_streaming_graph_kernel_node_attrs_t;

typedef struct iree_hal_streaming_graph_memcpy_node_attrs_t {
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_hal_streaming_buffer_ref_t src_ref;
  iree_device_size_t size;
  iree_hal_copy_flags_t flags;
} iree_hal_streaming_graph_memcpy_node_attrs_t;

typedef struct iree_hal_streaming_graph_memset_node_attrs_t {
  iree_hal_streaming_buffer_ref_t dst_ref;
  uint32_t pattern;
  uint8_t pattern_size;
  iree_device_size_t count;
  iree_hal_copy_flags_t flags;
} iree_hal_streaming_graph_memset_node_attrs_t;

typedef struct iree_hal_streaming_graph_host_call_node_attrs_t {
  void (*fn)(void* user_data);
  void* user_data;
} iree_hal_streaming_graph_host_call_node_attrs_t;

// Graph node structure.
// Memory layout:
// [iree_hal_streaming_graph_node_t]
// [dependencies array (dependency_count * sizeof(node*))]
// [padding to iree_max_align_t]
// [extra_data (e.g., packed kernel arguments)]
typedef struct iree_hal_streaming_graph_node_t {
  // Type of the node indicating which attribute data is valid.
  iree_hal_streaming_graph_node_type_t type;
  // Unique index assigned when added to graph.
  uint32_t node_index;
  uint32_t dependency_count;

  // Node-specific data.
  union {
    iree_hal_streaming_graph_kernel_node_attrs_t kernel;
    iree_hal_streaming_graph_memcpy_node_attrs_t memcpy;
    iree_hal_streaming_graph_memset_node_attrs_t memset;
    iree_hal_streaming_graph_host_call_node_attrs_t host;
  } attrs;

  // Variable-length array of dependencies follows the struct.
  // TODO(benvanik): use uint32_t instead as it'd shrink the size of the struct
  // and be faster to look up later. It means having a single node pointer is
  // not sufficient to get a dependency node pointer (need the table) but that's
  // not really useful anyway.
  iree_hal_streaming_graph_node_t* dependencies[];
} iree_hal_streaming_graph_node_t;

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_init_flag_bits_e {
  IREE_HAL_STREAMING_INIT_FLAG_NONE = 0ull,
} iree_hal_streaming_init_flags_t;

// Initializes global state.
// Synchronization: none (one-time initialization).
iree_status_t iree_hal_streaming_init_global(
    iree_hal_streaming_init_flags_t flags, iree_allocator_t host_allocator);

// Cleans up global state and releases all resources.
// Synchronization: all contexts (synchronizes all active contexts).
void iree_hal_streaming_cleanup_global(void);

// Accessor for the global device registry.
// Synchronization: none (read-only access).
iree_hal_streaming_device_registry_t* iree_hal_streaming_device_registry(void);

// Global context list management.
// Synchronization: none (thread-safe internal locking).
void iree_hal_streaming_register_context(iree_hal_streaming_context_t* context);
void iree_hal_streaming_unregister_context(
    iree_hal_streaming_context_t* context);

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

// Synchronization: none (queries static device count).
iree_status_t iree_hal_streaming_device_count(iree_host_size_t* out_count);

// Synchronization: none (returns device entry).
iree_hal_streaming_device_t* iree_hal_streaming_device_entry(
    iree_hal_streaming_device_ordinal_t ordinal);

// Synchronization: none (queries device properties).
iree_status_t iree_hal_streaming_device_name(
    iree_hal_streaming_device_ordinal_t ordinal, char* name,
    iree_host_size_t name_size);

// Synchronization: none (queries current memory info).
iree_status_t iree_hal_streaming_device_memory_info(
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_device_size_t* out_free_memory, iree_device_size_t* out_total_memory);

// Synchronization: none (queries P2P capability).
iree_status_t iree_hal_streaming_device_can_access_peer(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    iree_hal_streaming_device_ordinal_t peer_device_ordinal, bool* can_access);

// Looks up a P2P link between two devices.
// Returns NULL if no link exists.
// Synchronization: none (queries static link info).
iree_hal_streaming_p2p_link_t* iree_hal_streaming_device_lookup_p2p_link(
    iree_hal_streaming_device_ordinal_t src_device,
    iree_hal_streaming_device_ordinal_t dst_device);

// Synchronization: none (queries context state).
iree_status_t iree_hal_streaming_device_primary_context_state(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    iree_hal_streaming_context_flags_t* out_flags, bool* out_active);

// Gets or creates the primary context for a device (thread-safe).
// This performs lazy initialization of the primary context on first access.
// Synchronization: none (thread-safe creation with internal locking).
iree_status_t iree_hal_streaming_device_get_or_create_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context);

// Retains the primary context, creating it if necessary.
// Increments device-level reference count.
// Returns the retained context.
// Synchronization: none (protected by internal mutex).
iree_status_t iree_hal_streaming_device_retain_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context);

// Releases the primary context.
// Decrements device-level reference count.
// Destroys context when count reaches 0.
// Synchronization: context (waits for idle when destroying).
iree_status_t iree_hal_streaming_device_release_primary_context(
    iree_hal_streaming_device_t* device);

// Synchronization: none (sets flags for future context creation).
iree_status_t iree_hal_streaming_device_set_primary_context_flags(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    const iree_hal_streaming_context_flags_t* flags);

// Synchronization: none (queries kernel occupancy).
iree_status_t iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t block_size, uint32_t dynamic_shared_mem_size,
    uint32_t* out_max_blocks);

// Callback type for dynamic shared memory size calculation.
// This callback is invoked during occupancy calculation to determine how much
// dynamic shared memory a kernel needs for a specific block size.
//
// Parameters:
//   block_size: The number of threads per block being tested.
//   user_data: Optional user-provided context passed through from the caller.
//
// Returns:
//   The number of bytes of dynamic shared memory required for the given block
//   size.
//
// Example implementation for a matrix multiplication kernel that uses shared
// memory based on tile size derived from block dimensions:
// ```c
// uint32_t matmul_dynamic_smem_callback(uint32_t block_size, void* user_data) {
//   // Assume square blocks (e.g., 16x16 = 256 threads)
//   uint32_t tile_size = (uint32_t)sqrt(block_size);
//   // Need shared memory for two tiles (A and B matrices)
//   return 2 * tile_size * tile_size * sizeof(float);
// }
// ```
typedef uint32_t (*iree_hal_streaming_block_to_dynamic_smem_fn_t)(
    uint32_t block_size);

// Calculates optimal block size for a kernel with optional dynamic shared
// memory callback. If smem_callback is NULL, dynamic_shared_mem_size is used as
// a fixed value. If smem_callback is provided, it will be called for each block
// size tested to determine the dynamic shared memory requirement.
// Synchronization: none (queries kernel occupancy).
iree_status_t iree_hal_streaming_calculate_optimal_block_size(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t dynamic_shared_mem_size,
    iree_hal_streaming_block_to_dynamic_smem_fn_t dynamic_shared_mem_callback,
    uint32_t block_size_limit, uint32_t* out_block_size,
    uint32_t* out_min_grid_size);

// Returns the maximum number of blocks that can be launched for a cooperative
// kernel on the device. If the device does not support cooperative launch,
// returns OK with out_max_blocks set to 0.
// Synchronization: none (queries kernel occupancy).
iree_status_t iree_hal_streaming_calculate_max_cooperative_blocks(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t block_size, uint32_t dynamic_shared_mem_size,
    uint32_t* out_max_blocks);

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

// Synchronization: none (creates new context).
iree_status_t iree_hal_streaming_context_create(
    iree_hal_streaming_device_t* device_entry,
    iree_hal_streaming_context_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_context_t** out_context);

// Synchronization: none (reference counting).
void iree_hal_streaming_context_retain(iree_hal_streaming_context_t* context);
void iree_hal_streaming_context_release(iree_hal_streaming_context_t* context);

// Synchronization: none (queries flags).
iree_hal_streaming_context_flags_t iree_hal_streaming_context_flags(
    iree_hal_streaming_context_t* context);

// Synchronization: none (thread-local access).
iree_hal_streaming_context_t* iree_hal_streaming_context_current(void);

// Synchronization: none (thread-local modification).
iree_status_t iree_hal_streaming_context_set_current(
    iree_hal_streaming_context_t* context);

// Synchronization: none (thread-local stack operation).
iree_status_t iree_hal_streaming_context_push(
    iree_hal_streaming_context_t* context);

// Synchronization: none (thread-local stack operation).
iree_status_t iree_hal_streaming_context_pop(
    iree_hal_streaming_context_t** out_context);

// Limit types for context resource limits.
typedef enum iree_hal_streaming_context_limit_e {
  IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE = 0,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE,
} iree_hal_streaming_context_limit_t;

// Synchronization: none (queries limit value).
iree_status_t iree_hal_streaming_context_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t* out_value);

// Synchronization: none (sets limit value).
iree_status_t iree_hal_streaming_context_set_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t value);

// Synchronization: none (configures peer access).
iree_status_t iree_hal_streaming_context_enable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context);

// Synchronization: none (disables peer access).
iree_status_t iree_hal_streaming_context_disable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context);

// Registers a stream with the context (non-owning).
// Called during stream creation. Does NOT retain the stream.
// Synchronization: none (thread-safe internal locking).
iree_status_t iree_hal_streaming_context_register_stream(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);

// Unregisters a stream from the context.
// Called during stream destruction.
// Synchronization: none (thread-safe internal locking).
void iree_hal_streaming_context_unregister_stream(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);

// Waits for all streams in the context to become idle.
// Synchronization: all streams in context (blocking wait).
iree_status_t iree_hal_streaming_context_wait_idle(
    iree_hal_streaming_context_t* context, iree_timeout_t timeout);

// Synchronization: default stream (blocks until stream idle).
iree_status_t iree_hal_streaming_context_synchronize(
    iree_hal_streaming_context_t* context);

//===----------------------------------------------------------------------===//
// Module management
//===----------------------------------------------------------------------===//

// Loads module from a binary image in memory.
// Synchronization: none (creates new module).
iree_status_t iree_hal_streaming_module_create_from_memory(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t image, iree_allocator_t host_allocator,
    iree_hal_streaming_module_t** out_module);

// Loads module from a file at the given path.
// Synchronization: none (creates new module).
iree_status_t iree_hal_streaming_module_create_from_file(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_caching_mode_t caching_mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_hal_streaming_module_t** out_module);

// Synchronization: none (reference counting).
void iree_hal_streaming_module_retain(iree_hal_streaming_module_t* module);
void iree_hal_streaming_module_release(iree_hal_streaming_module_t* module);

// Synchronization: none (queries symbol metadata).
iree_status_t iree_hal_streaming_module_symbol(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_type_t expected_type,
    iree_hal_streaming_symbol_t** out_symbol);

// Synchronization: none (queries function metadata).
iree_status_t iree_hal_streaming_module_function(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_t** out_function);

// Synchronization: none (queries global metadata).
iree_status_t iree_hal_streaming_module_global(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_deviceptr_t* out_device_ptr,
    iree_device_size_t* out_size);

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

// Synchronization: none (creates new stream).
iree_status_t iree_hal_streaming_stream_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_flags_t flags, int priority,
    iree_allocator_t host_allocator, iree_hal_streaming_stream_t** out_stream);

// Synchronization: none (reference counting).
void iree_hal_streaming_stream_retain(iree_hal_streaming_stream_t* stream);
void iree_hal_streaming_stream_release(iree_hal_streaming_stream_t* stream);

// Begins command buffer recording.
// Synchronization: none (begins recording).
iree_status_t iree_hal_streaming_stream_begin(
    iree_hal_streaming_stream_t* stream);

// Flushes pending commands.
// Synchronization: none (submits to queue, non-blocking).
iree_status_t iree_hal_streaming_stream_flush(
    iree_hal_streaming_stream_t* stream);

// Synchronization: none (queries stream status, non-blocking).
iree_status_t iree_hal_streaming_stream_query(
    iree_hal_streaming_stream_t* stream, int* status);

// Synchronization: stream (blocks until stream idle).
iree_status_t iree_hal_streaming_stream_synchronize(
    iree_hal_streaming_stream_t* stream);

// Waits for an event on a stream.
// Synchronization: none (enqueues wait operation, non-blocking).
iree_status_t iree_hal_streaming_stream_wait_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event);

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

// Unpacks parameters from a CUDA-style kernel parameter buffer into a constant
// buffer and binding list. Some dispatches may use raw device buffer pointers
// and others may use bindings that can be resolved to HAL buffers.
// Callers must ensure sufficient storage in |out_constants| and |out_bindings|
// based on the symbol constant size and binding count.
// Synchronization: none (data packing utility).
iree_status_t iree_hal_streaming_unpack_parameters(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_parameter_info_t* parameters,
    const void* parameter_buffer, void* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings);

// Synchronization: none (enqueues kernel launch, non-blocking).
iree_status_t iree_hal_streaming_launch_kernel(
    iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_stream_t* stream);

// Launches a host function on the stream.
// The function will be called with user_data when the stream reaches this
// point. The stream will be flushed before enqueueing the host call to ensure
// proper ordering with device operations.
// Synchronization: stream flush (flushes stream before enqueue).
iree_status_t iree_hal_streaming_launch_host_function(
    iree_hal_streaming_stream_t* stream, void (*fn)(void*), void* user_data);

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

// Synchronization: none (creates new event).
iree_status_t iree_hal_streaming_event_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_event_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_event_t** out_event);

// Synchronization: none (reference counting).
void iree_hal_streaming_event_retain(iree_hal_streaming_event_t* event);
void iree_hal_streaming_event_release(iree_hal_streaming_event_t* event);

// Synchronization: none (queries event status, non-blocking).
iree_status_t iree_hal_streaming_event_query(iree_hal_streaming_event_t* event,
                                             int* status);

// Synchronization: stream flush (flushes stream before recording).
iree_status_t iree_hal_streaming_event_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream);

// Synchronization: event (blocks until event signaled).
iree_status_t iree_hal_streaming_event_synchronize(
    iree_hal_streaming_event_t* event);

// Synchronization: both events (waits for both events to complete).
iree_status_t iree_hal_streaming_event_elapsed_time(
    float* ms, iree_hal_streaming_event_t* start,
    iree_hal_streaming_event_t* stop);

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_memory_flag_bits_e {
  IREE_HAL_STREAMING_MEMORY_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_MEMORY_FLAG_PINNED = 1ull << 0,
  IREE_HAL_STREAMING_MEMORY_FLAG_PORTABLE = 1ull << 1,
  IREE_HAL_STREAMING_MEMORY_FLAG_WRITE_COMBINED = 1ull << 2,
} iree_hal_streaming_memory_flags_t;

// Synchronization: none (returns pointer value).
iree_hal_streaming_deviceptr_t iree_hal_streaming_buffer_device_pointer(
    iree_hal_streaming_buffer_t* buffer);

// Looks up a buffer by device pointer.
// Returns a borrowed reference to the buffer (does not transfer ownership).
// Returns an error if the device pointer is not found.
// Synchronization: none (table lookup).
iree_status_t iree_hal_streaming_memory_lookup(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr,
    iree_hal_streaming_buffer_ref_t* out_ref);

// Looks up a buffer that contains the specified address range.
// Returns a borrowed reference to the buffer (does not transfer ownership).
// Returns an error if no buffer contains the entire range
// `[device_ptr, device_ptr + size)`.
// Synchronization: none (table lookup).
iree_status_t iree_hal_streaming_memory_lookup_range(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_ref_t* out_ref);

// Synchronization: none (allocates memory).
iree_status_t iree_hal_streaming_memory_allocate_device(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: none (allocates pitched memory).
iree_status_t iree_hal_streaming_memory_allocate_device_pitched(
    iree_hal_streaming_context_t* context, iree_device_size_t width_bytes,
    iree_device_size_t height, iree_device_size_t element_size_bytes,
    iree_device_size_t* out_pitch, iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: context (waits for all operations to complete).
iree_status_t iree_hal_streaming_memory_free_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr);

// Synchronization: none (allocates host memory).
iree_status_t iree_hal_streaming_memory_allocate_host(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: context (waits for all operations to complete).
iree_status_t iree_hal_streaming_memory_free_host(
    iree_hal_streaming_context_t* context, void* ptr);

// Synchronization: none (registers existing memory).
iree_status_t iree_hal_streaming_memory_register_host(
    iree_hal_streaming_context_t* context, void* ptr, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: context (waits for all operations to complete).
iree_status_t iree_hal_streaming_memory_unregister_host(
    iree_hal_streaming_context_t* context, void* ptr);

// Synchronization: none (queries address range).
iree_status_t iree_hal_streaming_memory_address_range(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_deviceptr_t* out_base, iree_device_size_t* out_size);

// Synchronization: none (queries registration flags).
iree_status_t iree_hal_streaming_memory_host_flags(
    iree_hal_streaming_context_t* context, void* ptr,
    iree_hal_streaming_host_register_flags_t* out_flags);

// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memory_memset(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_streaming_stream_t* stream);

// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memory_memcpy(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Performs P2P memory transfer.
// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_peer(
    iree_hal_streaming_context_t* dst_context,
    iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_context_t* src_context,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Memory copy helpers for different transfer types.
// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_host_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    const void* src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_device_to_host(
    iree_hal_streaming_context_t* context, void* dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_device_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

//===----------------------------------------------------------------------===//
// Memory pool management
//===----------------------------------------------------------------------===//

// Memory allocation handle types for IPC.
typedef enum iree_hal_streaming_mem_handle_type_e {
  IREE_HAL_STREAMING_MEM_HANDLE_TYPE_NONE = 0,
  IREE_HAL_STREAMING_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
  IREE_HAL_STREAMING_MEM_HANDLE_TYPE_WIN32,
  IREE_HAL_STREAMING_MEM_HANDLE_TYPE_WIN32_KMT,
} iree_hal_streaming_mem_handle_type_t;

// Memory pool location types.
typedef enum iree_hal_streaming_mem_location_type_e {
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID = 0,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_DEVICE,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
} iree_hal_streaming_mem_location_type_t;

// Memory access flags for memory pools.
typedef enum iree_hal_streaming_mem_access_flag_bits_e {
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE = 0ull,
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ = 1ull << 0,
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READWRITE =
      (1ull << 1) | IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ,
} iree_hal_streaming_mem_access_flags_t;

// Memory pool attributes.
typedef enum iree_hal_streaming_mem_pool_attr_e {
  // Allow reuse with internal dependencies.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 0,
  // Allow reuse following event dependencies.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
  // Allow opportunistic reuse.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
  // Release threshold in bytes.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD,
  // Current reserved memory.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
  // High watermark of reserved memory.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_HIGH,
  // Current used memory.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_CURRENT,
  // High watermark of used memory.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_HIGH,
} iree_hal_streaming_mem_pool_attr_t;

// Memory pool properties.
typedef struct iree_hal_streaming_mem_pool_props_t {
  // Allocation type for IPC.
  iree_hal_streaming_mem_handle_type_t alloc_handle_type;
  // Memory location.
  iree_hal_streaming_mem_location_type_t location_type;
  // Location ID (device ordinal or NUMA node).
  int location_id;
} iree_hal_streaming_mem_pool_props_t;

// Memory pool structure.
typedef struct iree_hal_streaming_mem_pool_t {
  iree_atomic_ref_count_t ref_count;

  // Owning context.
  iree_hal_streaming_context_t* context;

  // Pool properties.
  iree_hal_streaming_mem_pool_props_t props;

  // Pool attributes.
  uint64_t release_threshold;
  bool reuse_allow_internal_dependencies;
  bool reuse_follow_event_dependencies;
  bool reuse_allow_opportunistic;

  // Statistics.
  uint64_t reserved_mem_current;
  uint64_t reserved_mem_high;
  uint64_t used_mem_current;
  uint64_t used_mem_high;

  // Platform-specific handle, if IPC is enabled.
  void* platform_handle;

  // Synchronization.
  iree_slim_mutex_t mutex;

  // Host allocator.
  iree_allocator_t host_allocator;
} iree_hal_streaming_mem_pool_t;

// Creates a memory pool with the specified properties.
// Synchronization: none (creates new pool).
iree_status_t iree_hal_streaming_mem_pool_create(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_mem_pool_props_t* props,
    iree_allocator_t host_allocator, iree_hal_streaming_mem_pool_t** out_pool);

// Retains a reference to the memory pool.
// Synchronization: none (reference counting).
void iree_hal_streaming_mem_pool_retain(iree_hal_streaming_mem_pool_t* pool);

// Releases a reference to the memory pool.
// Synchronization: none (reference counting).
void iree_hal_streaming_mem_pool_release(iree_hal_streaming_mem_pool_t* pool);

// Sets an attribute of the memory pool.
// Synchronization: none (sets attribute value).
iree_status_t iree_hal_streaming_mem_pool_set_attribute(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_mem_pool_attr_t attr, uint64_t value);

// Gets an attribute of the memory pool.
// Synchronization: none (queries attribute value).
iree_status_t iree_hal_streaming_mem_pool_get_attribute(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_mem_pool_attr_t attr, uint64_t* out_value);

// Trims the memory pool to the specified size.
// Synchronization: none (trims free memory).
iree_status_t iree_hal_streaming_mem_pool_trim_to(
    iree_hal_streaming_mem_pool_t* pool, iree_device_size_t min_bytes_to_keep);

// Gets the default memory pool for a device.
iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_default_mem_pool(
    iree_hal_streaming_device_t* device);

// Gets the current memory pool for a device.
iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_mem_pool(
    iree_hal_streaming_device_t* device);

// Sets the current memory pool for a device.
iree_status_t iree_hal_streaming_device_set_mem_pool(
    iree_hal_streaming_device_t* device, iree_hal_streaming_mem_pool_t* pool);

// Allocates memory asynchronously from the current pool.
// Synchronization: stream (async allocation on stream).
iree_status_t iree_hal_streaming_memory_allocate_async(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_deviceptr_t* out_ptr);

// Allocates memory asynchronously from a specific pool.
// Synchronization: stream (async allocation on stream).
iree_status_t iree_hal_streaming_memory_allocate_from_pool_async(
    iree_hal_streaming_mem_pool_t* pool, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_deviceptr_t* out_ptr);

// Frees memory asynchronously.
// Synchronization: stream (async free on stream, no implicit sync).
iree_status_t iree_hal_streaming_memory_free_async(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_stream_t* stream);

// IPC functions for memory pools.
iree_status_t iree_hal_streaming_mem_pool_export_to_shareable_handle(
    iree_hal_streaming_mem_pool_t* pool, void* shared_handle);

iree_status_t iree_hal_streaming_mem_pool_import_from_shareable_handle(
    iree_hal_streaming_context_t* context, void* shared_handle,
    iree_hal_streaming_mem_pool_t** out_pool);

iree_status_t iree_hal_streaming_mem_pool_export_pointer(
    iree_hal_streaming_deviceptr_t ptr, iree_hal_streaming_mem_pool_t* pool,
    void* shared_handle);

iree_status_t iree_hal_streaming_mem_pool_import_pointer(
    iree_hal_streaming_mem_pool_t* pool, void* shared_handle,
    iree_hal_streaming_deviceptr_t* out_ptr);

//===----------------------------------------------------------------------===//
// Graph management
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_graph_flag_bits_e {
  IREE_HAL_STREAMING_GRAPH_FLAG_NONE = 0ull,
} iree_hal_streaming_graph_flags_t;

typedef enum iree_hal_streaming_graph_instantiate_flag_bits_e {
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1ull << 0,
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_UPLOAD = 1ull << 1,
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 1ull << 2,
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 1ull << 3,
} iree_hal_streaming_graph_instantiate_flags_t;

// Synchronization: none (creates new graph).
iree_status_t iree_hal_streaming_graph_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_graph_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_graph_t** out_graph);

// Synchronization: none (reference counting).
void iree_hal_streaming_graph_retain(iree_hal_streaming_graph_t* graph);
void iree_hal_streaming_graph_release(iree_hal_streaming_graph_t* graph);

iree_status_t iree_hal_streaming_graph_add_empty_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_kernel_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_memcpy_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_memset_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_deviceptr_t dst,
    uint32_t pattern, iree_host_size_t pattern_size, iree_device_size_t count,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_host_call_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void (*fn)(void*), void* user_data,
    iree_hal_streaming_graph_node_t** out_node);

// Synchronization: none (creates executable graph).
iree_status_t iree_hal_streaming_graph_instantiate(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_hal_streaming_graph_exec_t** out_exec);

// Synchronization: none (reference counting).
void iree_hal_streaming_graph_exec_retain(
    iree_hal_streaming_graph_exec_t* exec);
void iree_hal_streaming_graph_exec_release(
    iree_hal_streaming_graph_exec_t* exec);

// Synchronization: stream (launches graph async on stream).
iree_status_t iree_hal_streaming_graph_exec_launch(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream);

// Synchronization: none (updates graph structure).
iree_status_t iree_hal_streaming_graph_exec_update(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph);

//===----------------------------------------------------------------------===//
// Stream capture
//===----------------------------------------------------------------------===//

// Synchronization: none (begins capture mode).
iree_status_t iree_hal_streaming_begin_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_mode_t mode);

// Synchronization: none (ends capture mode, creates graph).
iree_status_t iree_hal_streaming_end_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_t** out_graph);

// Synchronization: none (queries capture status).
iree_status_t iree_hal_streaming_capture_status(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_status_t* out_status,
    unsigned long long* out_id);

// Synchronization: none (queries capture state).
iree_status_t iree_hal_streaming_is_capturing(
    iree_hal_streaming_stream_t* stream, bool* out_is_capturing);

// Synchronization: none (updates dependencies).
iree_status_t iree_hal_streaming_update_capture_dependencies(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_capture_dependencies_mode_t mode);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_INTERNAL_H_
