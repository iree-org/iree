# Graph Instantiation Phase 0 - Node Recording and Partitioning

## Overview

Transform graph nodes (kernel, memcpy, memset, host) into executable blocks that can be launched via HAL queue operations. The key insight is that we need a topologically sorted flat array of node pointers for efficient indexing, since the chained node blocks cannot be directly indexed.

## Core Design Principles

1. **No interleaving**: Command buffers cannot be interleaved with queue operations
2. **Single-node optimization**: Skip command buffer overhead for single recordable nodes
3. **Efficient indexing**: Use sorted node array instead of chained blocks
4. **Memory efficiency**: Transient data in temp arena, persistent in exec arena

## Implementation Plan

### 1. Data Flow

```
Node Blocks (chained) → Linearize → Sort → Partition → Create Blocks
     ↓                      ↓         ↓        ↓            ↓
  Linked list           Flat array  Topo   Ranges      Exec blocks
                        of ptrs     order
```

### 2. Memory Management

**Transient allocations (temp arena)**:
- `sorted_nodes[]` - flattened array of node pointers
- `partitions` linked list - partition descriptors (may grow as we partition, use a partition_block_t like the node_block_t approach in graph.c)
- Analysis metadata (bitmaps, etc.)

**Persistent allocations (exec arena)**:
- `exec->blocks[]` - final executable blocks
- Command buffers
- Semaphore arrays
- Block-specific data

### 3. Main Instantiation Flow

```c
iree_status_t iree_hal_streaming_graph_exec_instantiate_locked(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_node_block_t* node_blocks,
    iree_host_size_t node_count) {

  // 1. Create temporary arena for transient allocations
  iree_arena_allocator_t temp_arena;
  iree_arena_initialize(pool, &temp_arena);

  // 2. Linearize nodes from chained blocks into flat array
  iree_hal_streaming_graph_node_t** sorted_nodes;
  iree_arena_allocate(&temp_arena,
                      node_count * sizeof(*sorted_nodes),
                      (void**)&sorted_nodes);
  linearize_node_blocks(node_blocks, sorted_nodes, node_count);

  // 3. Perform topological sort (if needed)
  // For now, assume nodes are added in dependency order
  // We could linearize and sort (at least a first-pass) at the same time to
  // avoid needing to walk the whole list multiple times.

  // 4. Analyze partitions on sorted array
  partition_t* partitions;
  iree_host_size_t partition_count;
  analyze_partitions(sorted_nodes, node_count, &temp_arena,
                    &partitions, &partition_count);

  // 5. Create blocks using sorted node array and partition ranges
  create_blocks_from_partitions(exec, sorted_nodes, partitions,
                                partition_count);

  // 6. Create semaphores for synchronization
  create_internal_semaphores(exec, partition_count);

  // 7. Cleanup temp arena (includes sorted_nodes and partitions)
  iree_arena_deinitialize(&temp_arena);
}
```

### 4. Partition Analysis

```c
typedef enum iree_hal_streaming_graph_partition_type_e {
  // Can go in command buffer (count 1 may also be optimizable into a queue op).
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE,
  // Must be separate host call.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL,
  // Barrier node.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_EMPTY,
} iree_hal_streaming_graph_partition_type_t;
typedef struct iree_hal_streaming_graph_partition_t {
  uint32_t start_index;  // Index into sorted_nodes array
  uint32_t count;
  iree_hal_streaming_graph_partition_type_t type;
} iree_hal_streaming_graph_partition_t;

static iree_status_t analyze_partitions(
    iree_hal_streaming_graph_node_t** sorted_nodes,
    iree_host_size_t node_count,
    iree_arena_allocator_t* temp_arena,
    partition_t** out_partitions,
    iree_host_size_t* out_count) {
  // Walk sorted nodes and identify partition boundaries
  // Recordable: KERNEL, MEMCPY, MEMSET
  // Non-recordable: HOST, EMPTY, GRAPH

  // Partition breaks at:
  // - Non-recordable nodes
  // - Transitions between recordable and non-recordable
}
```

### 5. Block Creation with Optimizations

```c
// Single-node recordable partition optimization
if (partition->type == PARTITION_RECORDABLE && partition->count == 1) {
  node = sorted_nodes[partition->start_index];
  switch(node->type) {
    case KERNEL:
      // Direct dispatch - avoid command buffer overhead
      create_dispatch_block_from_kernel(exec, node, ...);
      break;
    case MEMCPY:
      // Direct copy - avoid command buffer overhead
      create_copy_block_from_memcpy(exec, node, ...);
      break;
    case MEMSET:
      // Direct fill - avoid command buffer overhead
      create_fill_block_from_memset(exec, node, ...);
      break;
  }
} else if (partition->type == PARTITION_RECORDABLE) {
  // Multiple nodes - use command buffer
  create_execute_block_with_recording(exec, sorted_nodes,
                                      partition->start_index,
                                      partition->count, ...);
}
```

### 6. Command Buffer Recording

```c
static iree_status_t iree_hal_streaming_graph_create_execute_block(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t** sorted_nodes,  // Flat array
    uint32_t node_start_index, uint32_t node_count,
    uint16_t wait_semaphore_count, uint16_t signal_semaphore_count,
    iree_hal_streaming_graph_block_t** out_block) {

  // Create and begin command buffer
  iree_hal_command_buffer_t* cmd_buffer;
  iree_hal_command_buffer_create(..., &cmd_buffer);
  iree_hal_command_buffer_begin(cmd_buffer);

  // Record nodes using direct array access (O(1) indexing)
  for (uint32_t i = 0; i < node_count; i++) {
    iree_hal_streaming_graph_node_t* node = sorted_nodes[node_start_index + i];
    switch(node->type) {
      case KERNEL: record_kernel_dispatch(cmd_buffer, node); break;
      case MEMCPY: record_memcpy(cmd_buffer, node); break;
      case MEMSET: record_memset(cmd_buffer, node); break;
    }
  }

  iree_hal_command_buffer_end(cmd_buffer);

  // Allocate block and store command buffer
  // ... block allocation ...
}
```

## Optimization Opportunities

### 1. Bitmap Acceleration Structures

Maybe use bitmaps to avoid redundant node type checking and preform scans/range analysis (if a bitmap indicates whether a node is recordable then doing a find of the next unset bit after N returns the range [N,N+c) that is the commands that can be partitioned together without needing to inspect each command):

```c
typedef struct partition_analysis_t {
  // Bitmaps for O(1) property queries
  iree_bitmap_t* recordable_nodes;   // Set if KERNEL|MEMCPY|MEMSET
  iree_bitmap_t* barrier_nodes;      // Set if EMPTY
  iree_bitmap_t* host_nodes;         // Set if HOST

  // Derived bitmaps for partition boundaries
  iree_bitmap_t* partition_starts;   // Set at partition boundaries
} partition_analysis_t;

// Single-pass analysis to build all bitmaps
static void analyze_nodes_with_bitmaps(
    iree_hal_streaming_graph_node_t** sorted_nodes,
    iree_host_size_t node_count,
    partition_analysis_t* analysis) {

  for (i = 0; i < node_count; i++) {
    switch(sorted_nodes[i]->type) {
      case KERNEL:
      case MEMCPY:
      case MEMSET:
        iree_bitmap_set(analysis->recordable_nodes, i);
        break;
      case HOST:
        iree_bitmap_set(analysis->host_nodes, i);
        // Force partition boundary
        if (i > 0) iree_bitmap_set(analysis->partition_starts, i);
        if (i + 1 < node_count) iree_bitmap_set(analysis->partition_starts, i + 1);
        break;
    }
  }
}

// Find recordable ranges efficiently
static void find_recordable_partitions(partition_analysis_t* analysis) {
  // Use bitmap_find_first_set/clear_bit for O(1) word scans
  // Can find runs of 64 recordable nodes in single operation
}
```

### 2. Single-Pass Partition Analysis

Combine linearization, analysis, and partition identification:

```c
static iree_status_t linearize_and_partition(
    iree_hal_streaming_node_block_t* node_blocks,
    iree_arena_allocator_t* temp_arena,
    iree_hal_streaming_graph_node_t*** out_sorted_nodes,
    partition_t** out_partitions,
    iree_host_size_t* out_partition_count) {

  // Single walk through nodes:
  // 1. Copy to sorted array
  // 2. Identify partition boundaries
  // 3. Build metadata

  bool in_recordable = false;
  uint32_t partition_start = 0;

  for (block = node_blocks; block; block = block->next) {
    for (i = 0; i < block->count; i++) {
      node = block->nodes[i];
      sorted_nodes[index] = node;

      bool is_recordable = (node->type == KERNEL ||
                            node->type == MEMCPY ||
                            node->type == MEMSET);

      if (is_recordable != in_recordable) {
        // Partition boundary
        if (index > partition_start) {
          create_partition(partition_start, index - partition_start, ...);
        }
        partition_start = index;
        in_recordable = is_recordable;
      }
      index++;
    }
  }
}
```

### 3. Cache-Friendly Traversal

Prefetch nodes during linearization:

```c
// Prefetch next block while processing current
for (block = node_blocks; block; block = block->next) {
  if (block->next) {
    __builtin_prefetch(block->next, 0, 1);  // Prefetch for read
  }
  // Process current block
}
```

### 4. Future: Dependency-Based Partitioning

For concurrent execution (phase 1):

```c
// Use dependency information to identify independent partitions
// Nodes with no shared dependencies can execute concurrently
// This requires more complex analysis but enables parallelism
```

## TODO List

- [x] Create PLAN.graphs.p0.md with implementation plan
- [ ] Add sorted_nodes parameter to create_execute_block
- [ ] Implement linearize_node_blocks helper
- [ ] Implement partition analysis (simple version)
- [ ] Implement record_kernel_dispatch
- [ ] Implement record_memcpy
- [ ] Implement record_memset
- [ ] Update instantiate_locked with full flow
- [ ] Test with simple graphs
- [ ] Add bitmap acceleration (optimization)
- [ ] Add single-pass analysis (optimization)
- [ ] Add dependency-based partitioning (future)

## Testing Strategy

### Test Case 1: Simple Linear Graph
```
KERNEL → MEMCPY → KERNEL
Expected: Single command buffer with 3 operations
```

### Test Case 2: Host Call Interruption
```
KERNEL → HOST → KERNEL
Expected: 3 blocks (dispatch, host_call, dispatch)
```

### Test Case 3: Single Node Optimization
```
Single MEMCPY node
Expected: Direct QUEUE_COPY block (no command buffer)
```

### Test Case 4: Complex Graph
```
KERNEL → MEMCPY → HOST → MEMSET → KERNEL → MEMCPY
Expected: Mixed blocks with proper partitioning
```

## Notes

- The sorted_nodes array enables O(1) indexing vs O(n) for chained blocks
- Partitioning must respect recordable/non-recordable boundaries
- Single-node optimization avoids command buffer overhead
- Bitmap acceleration can make analysis O(n/64) instead of O(n)
- All transient data uses temp arena for automatic cleanup
- Use naming and style consistent with graph.c/graph_exec.c
- Analysis and partitioning logic should in graph_analysis.c
- graph_exec.c should be used for instantiation (recording command buffers, allocating graph blocks, etc) and launching (executing graph blocks)
- use graph.h for any internal types/enums/functions used only by the various graph files
