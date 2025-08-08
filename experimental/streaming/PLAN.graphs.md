# Graph Instantiation to Command Buffer Conversion Plan

## Overview

This document outlines the design and implementation plan for converting `iree_hal_streaming_graph_t` (graph templates) into executable `iree_hal_streaming_graph_exec_t` instances that can be efficiently launched on device queues. The system will intelligently partition graph nodes into blocks that execute as either queue operations or command buffers, with sophisticated concurrency detection and optimization strategies.

## Requirements

### Functional Requirements
1. Convert arbitrary graph structures to executable format
2. Support mixed operation types (kernels, memory ops, host calls, child graphs)
3. Detect and exploit concurrency between independent nodes
4. Efficient semaphore-based synchronization between blocks
5. Support graphs from 1 to 10,000+ nodes
6. Immutable block structure after instantiation for fast launches
7. Multiple optimization modes (latency, throughput, memory)

### Performance Requirements
- **100-node graph instantiation**: < 100μs (target < 50μs)
- **1000-node graph instantiation**: < 1ms (target < 500μs)
- **10000-node graph instantiation**: < 10ms
- **Launch overhead**: < 10μs for pre-instantiated graphs
- **Memory overhead**: < 100 bytes per node in block structure

### Design Constraints
- All graph-specific types remain internal to graph.c
- Use device block pool arena allocator (no reallocation after initial sizing)
- Minimize allocations during launch for predictable performance
- Support both reusable and fresh semaphore strategies

## Architecture

### File Organization

The implementation is split across two files for better separation of concerns:

- **graph.c**: Graph template management (creation, node addition, destruction)
  - `iree_hal_streaming_graph_t` structure and operations
  - Node creation and dependency tracking
  - Recording-time metadata collection
  - Graph capture functionality

- **graph_exec.c**: Graph execution management (instantiation, analysis, launch)
  - `iree_hal_streaming_graph_exec_t` structure and operations
  - Graph analysis and topological sorting
  - Block partitioning and command buffer creation
  - Semaphore management and launch logic
  - All optimization algorithms

### Core Data Structures (Internal to graph_exec.c)

#### Graph Block
Represents an executable unit - either a queue operation or command buffer containing multiple operations.

```c
typedef enum iree_hal_streaming_graph_block_type_e {
  // iree_hal_device_queue_barrier
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER = 0,
  // iree_hal_device_queue_execute
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_COMMAND_BUFFER,
  // iree_hal_device_queue_host_call
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
  // iree_hal_device_queue_dispatch
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
  // iree_hal_device_queue_copy
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY,
  // iree_hal_device_queue_fill
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL,
} iree_hal_streaming_graph_block_type_t;

typedef struct iree_hal_streaming_graph_block_t {
  iree_hal_streaming_graph_block_type_t type;

  // Range in sorted node list.
  uint32_t node_start_index;  // First node in sorted array
  uint32_t node_count;         // Number of nodes in this block

  // Semaphore synchronization.
  uint16_t wait_semaphore_count;
  uint16_t signal_semaphore_count;

  // Variable-length data follows:
  // - uint16_t wait_semaphore_indices[wait_semaphore_count]
  // - uint32_t wait_payload_deltas[wait_semaphore_count]
  // - uint16_t signal_semaphore_indices[signal_semaphore_count]
  // - uint32_t signal_payload_deltas[signal_semaphore_count]
  // - union { command_buffer, queue_op_data } (based on type)
} iree_hal_streaming_graph_block_t;
```

#### Analysis State
Temporary state used during graph analysis and partitioning.

```c
typedef struct iree_hal_streaming_graph_analysis_t {
  // Node trait bitmaps (allocated from arena).
  iree_bitmap_t can_command_buffer;      // Node can go in command buffer
  iree_bitmap_t is_queue_only;           // Must be queue operation
  iree_bitmap_t has_dependencies;        // Has upstream dependencies
  iree_bitmap_t is_concurrent_with_next; // Can run parallel with next node

  // Sorted node array (if sort was needed).
  iree_hal_streaming_graph_node_t** sorted_nodes;

  // Graph properties.
  bool already_sorted;      // Input was topologically sorted
  bool is_linear;          // Each node depends only on previous
  uint32_t max_concurrency; // Maximum parallel branches
  uint32_t dependency_depth; // Longest dependency chain
} iree_hal_streaming_graph_analysis_t;
```

#### Executable Graph Extension
```c
struct iree_hal_streaming_graph_exec_t {
  // ... existing fields ...

  // Immutable block list created during instantiate.
  iree_hal_streaming_graph_block_t** blocks;
  uint32_t block_count;

  // Semaphore pool for internal synchronization.
  uint32_t semaphore_count;
  iree_hal_semaphore_t** semaphores;      // Pooled semaphores
  uint64_t* semaphore_base_values;        // Current base values

  // Optimization mode used during instantiation.
  iree_hal_streaming_graph_algorithm_t algorithm;
};
```

## Semaphore Payload Management

### Delta Encoding Strategy

Blocks store **relative delta values** instead of absolute semaphore payloads. This enables:
- Reuse of semaphores across multiple launches
- Efficient encoding (smaller values)
- Simple increment tracking

#### Example Fork-Join Pattern
```
Initial state: All semaphores at base value 0

Block A (root):
  - Waits: stream_timeline @ current_value
  - Signals: sem[0] @ delta +1 (absolute: 0+1=1)

Block B (fork 1):
  - Waits: sem[0] @ delta +1 (absolute: 1)
  - Signals: sem[1] @ delta +1 (absolute: 0+1=1)

Block C (fork 2):
  - Waits: sem[0] @ delta +1 (absolute: 1)
  - Signals: sem[2] @ delta +1 (absolute: 0+1=1)

Block D (join):
  - Waits: sem[1] @ delta +1 (absolute: 1), sem[2] @ delta +1 (absolute: 1)
  - Signals: stream_timeline @ pending_value
```

### Launch-Time Calculation

```c
iree_status_t iree_hal_streaming_graph_exec_launch(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_stream_t* stream) {

  // Strategy A: Reuse semaphores - query current values.
  for (i = 0; i < exec->semaphore_count; i++) {
    exec->semaphore_base_values[i] = iree_hal_semaphore_query(exec->semaphores[i]);
  }

  // Strategy B: Fresh semaphores - start at 0
  // memset(exec->semaphore_base_values, 0, sizeof(uint64_t) * exec->semaphore_count);

  // Submit each block.
  for (block_idx = 0; block_idx < exec->block_count; block_idx++) {
    block = exec->blocks[block_idx];

    // Build absolute wait values.
    for (i = 0; i < block->wait_count; i++) {
      sem_idx = block->wait_semaphore_indices[i];
      wait_values[i] = exec->semaphore_base_values[sem_idx] +
                       block->wait_payload_deltas[i];
    }

    // Build absolute signal values and update base.
    for (i = 0; i < block->signal_count; i++) {
      sem_idx = block->signal_semaphore_indices[i];
      signal_values[i] = exec->semaphore_base_values[sem_idx] +
                         block->signal_payload_deltas[i];
      exec->semaphore_base_values[sem_idx] = signal_values[i];
    }

    // Submit based on block type.
    switch (block->type) {
      case GRAPH_BLOCK_TYPE_COMMAND_BUFFER:
        iree_hal_device_queue_execute(...);
        break;
      case GRAPH_BLOCK_TYPE_QUEUE_DISPATCH:
        iree_hal_device_queue_dispatch(...);
        break;
      // ... other queue operations
    }
  }
}
```

### Semaphore Strategy Tradeoffs

#### Strategy A: Reuse Semaphores
**Pros:**
- Minimal allocation per launch
- Predictable memory usage
- Good cache locality

**Cons:**
- Must track base values
- Complex with concurrent launches
- Potential value overflow over time

#### Strategy B: Fresh Semaphores per Launch
**Pros:**
- Simple implementation (always start at 0)
- Concurrent launch friendly
- No state tracking needed

**Cons:**
- Allocation overhead per launch
- Semaphore creation/destruction cost
- More memory churn

## Graph Analysis Pipeline

### Graph Recording Optimizations

During graph recording, maintain metadata to avoid analysis passes:

```c
// Extended graph structure with recording-time analysis.
struct iree_hal_streaming_graph_t {
  // ... existing fields ...

  // Maintained during recording.
  bool is_linear;              // True until non-sequential dependency added
  bool is_sorted;              // True until out-of-order dependency added
  uint32_t max_depth_seen;     // Maximum dependency depth observed

  // Leaf nodes (no dependents) - maintained incrementally.
  iree_hal_streaming_graph_node_t** leaf_nodes;
  iree_host_size_t leaf_count;

  // Node metadata array - parallel to nodes array.
  iree_hal_streaming_node_metadata_t* node_metadata;
};

// Per-node metadata populated during recording.
typedef struct iree_hal_streaming_node_metadata_t {
  uint16_t in_degree;          // Dependencies (parents)
  uint16_t out_degree;         // Dependents (children)
  uint16_t depth;              // Distance from root
  uint16_t height;             // Distance to leaf
  uint32_t dependent_mask;     // Bitmap of direct dependents (for small graphs)
} iree_hal_streaming_node_metadata_t;
```

### Phase 1: Single-Pass Topological Sort & Analysis

```c
typedef struct iree_hal_streaming_sort_context_t {
  // Output arrays (allocated from arena).
  iree_hal_streaming_graph_node_t** sorted_nodes;
  uint32_t* node_levels;           // Concurrency level per node
  uint32_t* level_starts;          // Index where each level starts

  // Analysis results computed during sort.
  uint32_t max_width;              // Maximum concurrent nodes
  uint32_t critical_path_length;  // Longest dependency chain
  bool has_diamond_patterns;      // Detected fork-join patterns
  bool has_wide_fan_out;          // Nodes with many dependents

  // Working state.
  uint32_t* in_degrees;            // Mutable copy for Kahn's algorithm
  uint32_t* queue;                 // BFS queue for level-based traversal
} iree_hal_streaming_sort_context_t;

iree_status_t analyze_graph_single_pass(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_analysis_t* analysis) {

  // Fast path: Already sorted and linear.
  if (graph->is_sorted && graph->is_linear) {
    analysis->already_sorted = true;
    analysis->is_linear = true;
    analysis->max_concurrency = 1;
    // No sort needed, use original array.
    analysis->sorted_nodes = graph->nodes;
    return iree_ok_status();
  }

  // Fast path: Already sorted but not linear.
  if (graph->is_sorted) {
    analysis->already_sorted = true;
    analysis->sorted_nodes = graph->nodes;
    // Still need one pass for concurrency analysis.
    return analyze_concurrency_presorted(graph, analysis);
  }

  // Full sort needed - but do everything in one pass.
  iree_hal_streaming_sort_context_t sort_ctx = {0};
  allocate_sort_context(graph->node_count, &sort_ctx);

  // Modified Kahn's algorithm that computes everything during sort.
  level_based_topological_sort(graph, &sort_ctx);

  // Transfer results to analysis struct.
  analysis->sorted_nodes = sort_ctx.sorted_nodes;
  analysis->max_concurrency = sort_ctx.max_width;
  analysis->is_linear = (sort_ctx.max_width == 1);

  // Set trait bitmaps based on sorted order and metadata.
  populate_trait_bitmaps_from_sort(graph, &sort_ctx, analysis);

  return iree_ok_status();
}

// Level-based sort that computes all properties in one pass
iree_status_t level_based_topological_sort(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_sort_context_t* ctx) {

  // Initialize in-degrees from pre-computed metadata
  memcpy(ctx->in_degrees, graph->node_metadata.in_degrees,
         sizeof(uint32_t) * graph->node_count);

  // Start with root nodes (in-degree 0)
  uint32_t queue_head = 0, queue_tail = 0;
  uint32_t current_level = 0;
  uint32_t level_node_count = 0;
  uint32_t sorted_count = 0;

  // Enqueue all roots
  for (i = 0; i < graph->node_count; i++) {
    if (ctx->in_degrees[i] == 0) {
      ctx->queue[queue_tail++] = i;
      ctx->node_levels[i] = 0;
      level_node_count++;
    }
  }

  ctx->level_starts[0] = 0;
  ctx->max_width = level_node_count;

  // Process nodes level by level.
  while (queue_head < queue_tail) {
    uint32_t level_end = queue_tail;
    uint32_t next_level_count = 0;

    // Process all nodes at current level.
    while (queue_head < level_end) {
      uint32_t node_idx = ctx->queue[queue_head++];
      ctx->sorted_nodes[sorted_count++] = graph->nodes[node_idx];

      // Update dependents and enqueue ready nodes.
      node_deps = graph->nodes[node_idx]->dependencies;
      for (j = 0; j < graph->nodes[node_idx]->dependency_count; j++) {
        uint32_t dep_idx = find_node_index(node_deps[j]);
        if (--ctx->in_degrees[dep_idx] == 0) {
          ctx->queue[queue_tail++] = dep_idx;
          ctx->node_levels[dep_idx] = current_level + 1;
          next_level_count++;
        }
      }
    }

    // Track maximum width.
    if (next_level_count > ctx->max_width) {
      ctx->max_width = next_level_count;
    }

    // Move to next level.
    if (next_level_count > 0) {
      current_level++;
      ctx->level_starts[current_level] = sorted_count;
      level_node_count = next_level_count;
    }
  }

  ctx->critical_path_length = current_level;

  // Detect patterns during sort (no extra pass).
  if (ctx->max_width > 2) {
    ctx->has_diamond_patterns = true;
  }

  return iree_ok_status();
}
```

### Phase 1B: Reverse Traversal for Partitioning Hints

For large concurrent graphs, traverse from leaves backwards:

```c
iree_status_t analyze_from_leaves(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_analysis_t* analysis) {

  // Start from pre-tracked leaf nodes.
  for (i = 0; i < graph->leaf_count; i++) {
    node = graph->leaf_nodes[i];

    // Walk backwards to find convergence points.
    trace_back_to_convergence(node, analysis);
  }

  // This identifies natural partition points where parallel work converges.
  return iree_ok_status();
}
```

### Phase 2: Block Partitioning

```c
iree_status_t partition_into_blocks(
    iree_hal_streaming_graph_analysis_t* analysis,
    iree_hal_streaming_graph_algorithm_t algorithm,
    iree_hal_streaming_graph_block_list_t* blocks) {

  iree_host_size_t node_idx = 0;

  while (node_idx < node_count) {
    // Find next queue-only operation.
    iree_host_size_t queue_only_idx =
        iree_bitmap_find_first_set(analysis->is_queue_only, node_idx);

    if (queue_only_idx > node_idx) {
      // Create command buffer block for range [node_idx, queue_only_idx).
      create_command_buffer_block(node_idx, queue_only_idx - node_idx);
      node_idx = queue_only_idx;
    }

    if (queue_only_idx < node_count) {
      // Create queue operation block for single node.
      create_queue_operation_block(queue_only_idx);
      node_idx = queue_only_idx + 1;
    }
  }

  // Apply algorithm-specific optimizations.
  switch (algorithm) {
    case LATENCY:
      // Minimal processing, leave as-is.
      break;
    case THROUGHPUT:
      // Split blocks to maximize concurrency.
      split_blocks_for_concurrency(blocks, analysis);
      break;
    case MEMORY:
      // Merge blocks to minimize concurrent execution.
      merge_blocks_for_memory(blocks, analysis);
      break;
  }
}
```

### Phase 3: Block Creation & Recording

```c
iree_status_t create_command_buffer_block(
    iree_host_size_t start_idx,
    iree_host_size_t count,
    iree_hal_streaming_graph_block_t** out_block) {

  // Calculate memory needed.
  size_t block_size = sizeof(iree_hal_streaming_graph_block_t) +
                     dependency_arrays_size +
                     sizeof(iree_hal_command_buffer_t*);

  // Allocate from arena.
  iree_hal_streaming_graph_block_t* block;
  iree_arena_allocate(arena, block_size, &block);

  // Initialize block.
  block->type = GRAPH_BLOCK_TYPE_COMMAND_BUFFER;
  block->node_start_index = start_idx;
  block->node_count = count;

  // Create and record command buffer.
  iree_hal_command_buffer_t* cmd_buffer;
  iree_hal_command_buffer_create(..., &cmd_buffer);

  // Record operations from node range.
  for (i = start_idx; i < start_idx + count; i++) {
    record_node_to_command_buffer(nodes[i], cmd_buffer);
  }

  iree_hal_command_buffer_end(cmd_buffer);

  // Store command buffer in block.
  block->command_buffer = cmd_buffer;

  *out_block = block;
}
```

## Optimization Algorithms

### Latency-Optimized Algorithm
- **Goal**: Minimize instantiation time (< 50μs for 100 nodes)
- **Strategy**: Simple linear partitioning, minimal analysis
- **When to use**: Real-time graph creation, small graphs

### Throughput-Optimized Algorithm
- **Goal**: Maximum parallel execution
- **Strategy**: Aggressive block splitting, maximize concurrent branches
- **When to use**: Large batch processing, GPU-heavy workloads

### Memory-Optimized Algorithm
- **Goal**: Minimize memory usage and allocations
- **Strategy**: Merge sequential blocks, limit concurrent execution
- **When to use**: Memory-constrained environments, very large graphs

## Scaling Considerations

### Small Graphs (1-100 nodes)
- Stack-allocated bitmaps (avoid allocation)
- Simple linear scan algorithms
- Single command buffer when possible
- Target: < 50μs instantiation

### Medium Graphs (100-1000 nodes)
- Arena-allocated bitmaps
- Efficient bitmap operations for range detection
- Balanced partitioning with concurrency limits
- Target: < 500μs instantiation

### Large Graphs (1000-10000 nodes)
- Hierarchical analysis (subgraph partitioning)
- Streaming instantiation (process in chunks)
- Aggressive memory reuse
- Concurrency capping to prevent semaphore explosion
- Target: < 10ms instantiation

## Implementation Work Items

### Phase 1: Infrastructure
1. **Bitmap Utility Migration**
   - Copy `runtime/src/iree/hal/drivers/amdgpu/util/bitmap.h/c` to `runtime/src/iree/base/internal/`
   - Rename types (remove amdgpu prefix)
   - Add comprehensive unit tests
   - Create benchmarks for bitmap operations

2. **Graph Recording Benchmark**
   - File: `experimental/streaming/graph_recording_benchmark.cc`
   - Measure node addition performance
   - Profile arena allocator usage
   - Test dependency tracking overhead

### Phase 2: Core Implementation
1. **Graph Management (graph.c)**
   - Maintain recording-time metadata
   - Track is_linear and is_sorted flags
   - Incremental leaf node tracking
   - Node metadata array management

2. **Graph Execution (graph_exec.c)**
   - Graph analysis pipeline implementation
   - Topological sort with trait detection
   - Block partitioning algorithms
   - Command buffer recording from nodes

3. **Semaphore Management (graph_exec.c)**
   - Delta encoding implementation
   - Launch-time payload calculation
   - Semaphore pooling strategies
   - Fork-join pattern optimization

### Phase 3: Optimization
1. **Graph Translation Benchmark**
   - File: `experimental/streaming/graph_translation_benchmark.cc`
   - Profile instantiation performance
   - Compare algorithm modes
   - Measure memory usage

2. **Algorithm Tuning**
   - Implement three optimization modes
   - Auto-selection heuristics
   - Performance profiling

### Phase 4: Testing & Validation
1. **Unit Tests**
   - Internal data structure tests
   - Topological sort validation
   - Semaphore payload calculation

2. **Integration Tests**
   - Real graph patterns (linear, fork-join, diamond)
   - Correctness validation
   - Performance regression tests

3. **Stress Tests**
   - 10,000+ node graphs
   - Maximum concurrency scenarios
   - Memory limit testing

## Success Metrics

### Performance
- 100-node instantiation: < 100μs (target < 50μs)
- 1000-node instantiation: < 1ms (target < 500μs)
- 10000-node instantiation: < 10ms
- Launch overhead: < 10μs
- Memory per node: < 100 bytes

### Functionality
- Correct topological ordering maintained
- All node types supported
- Concurrency correctly detected and exploited
- No deadlocks or race conditions
- Deterministic execution order

### Quality
- Zero memory leaks (arena cleanup)
- Comprehensive test coverage
- Benchmark-driven optimization
- Clear documentation and examples

## Future Optimization Exploration

### Advanced Techniques for Minimizing Graph Traversals

#### 1. Incremental Graph Properties
Track and update properties incrementally during recording:
```c
// When adding node N with dependencies D1, D2, ..., Dk:
if (all_deps_are_previous_node) {
  // Maintain linear property
  graph->is_linear = graph->is_linear && (k == 1 && deps[0] == nodes[n-1]);
}

if (all_deps_have_lower_indices) {
  // Maintain sorted property.
  graph->is_sorted = graph->is_sorted && true;
}

// Update leaf tracking.
for (dep in dependencies) {
  remove_from_leaf_set(dep);  // No longer a leaf
}
add_to_leaf_set(new_node);     // Initially a leaf until someone depends on it
```

#### 2. Hybrid Forward-Backward Analysis
Combine forward sort with backward convergence detection:
```c
// During single-pass sort, also track reverse edges
struct node_bidir_metadata {
  uint16_t forward_level;   // Distance from roots
  uint16_t backward_level;  // Distance from leaves
  uint32_t convergence_id;  // Which join point this leads to
};
```

#### 3. Compressed Dependency Representation
For large graphs, use compressed formats:
- **Interval encoding**: If nodes depend on ranges [5-10], store as interval
- **Delta encoding**: Store dependency indices as deltas from node index
- **Bit vectors**: For dense dependencies, use bitmaps instead of lists

#### 4. Level-Set Representation
Maintain nodes grouped by concurrency level:
```c
struct level_set_graph {
  uint32_t num_levels;
  uint32_t* level_sizes;      // Number of nodes at each level
  uint32_t** level_nodes;     // Node indices at each level
  uint64_t* level_bitmaps;    // Bitmap of active nodes per level
};
```

### Benchmarks for Measuring Optimization Success

#### 1. Graph Traversal Microbenchmarks
`experimental/streaming/graph_traversal_benchmark.cc`:
```c
// Measure individual operations.
BENCHMARK(CheckTopologicalOrder_Linear_100);
BENCHMARK(CheckTopologicalOrder_Diamond_100);
BENCHMARK(TopologicalSort_Random_1000);
BENCHMARK(ConcurrencyDetection_Wide_500);
BENCHMARK(LeafTracking_Incremental_1000);
```

#### 2. Recording Overhead Benchmark
Track overhead of maintaining metadata during recording:
```c
BENCHMARK(RecordNode_WithoutMetadata);
BENCHMARK(RecordNode_WithLinearCheck);
BENCHMARK(RecordNode_WithSortedCheck);
BENCHMARK(RecordNode_WithLeafTracking);
BENCHMARK(RecordNode_WithFullMetadata);
```

#### 3. Cache Efficiency Metrics
Measure cache behavior during analysis:
- L1/L2 cache misses during sort
- Memory bandwidth utilization
- Prefetch effectiveness
- TLB misses for large graphs

### Additional Optimization Ideas

#### 1. SIMD Bitmap Operations
Use AVX2/AVX512 for bitmap operations:
```c
// Find multiple set bits at once.
__m256i mask = _mm256_load_si256(bitmap_word);
int leading_zeros = _lzcnt_u64(_mm256_movemask_epi8(mask));
```

#### 2. Parallel Analysis for Large Graphs
Partition graph into regions for parallel analysis:
- Identify cut points (nodes with few dependencies)
- Process regions independently
- Merge results with minimal overhead

#### 3. Graph Pattern Templates
Pre-compute common patterns:
- Linear chains → single block
- Perfect diamonds → known semaphore pattern
- Map-reduce patterns → optimized fork-join

#### 4. Adaptive Algorithm Selection
Choose algorithm based on graph characteristics:
```c
if (graph->is_linear) {
  return fast_linear_instantiate(graph);
} else if (graph->max_concurrency < 4) {
  return simple_concurrent_instantiate(graph);
} else {
  return full_analysis_instantiate(graph);
}
```

#### 5. Memory Layout Optimizations
- **Node clustering**: Place frequently accessed nodes together
- **Hot/cold separation**: Separate metadata by access frequency
- **Prefetch hints**: Use `__builtin_prefetch` for predictable access

#### 6. Incremental Re-instantiation
For graphs that change slightly between uses:
- Track which nodes changed
- Re-analyze only affected subgraphs
- Reuse unchanged blocks

### Test Coverage Goals

#### Unit Tests
- Graph property detection accuracy
- Metadata consistency during recording
- Sort correctness for all topologies
- Bitmap operation correctness

#### Integration Tests
- Real ML model graphs (transformer, CNN, RNN)
- Procedurally generated stress patterns
- Adversarial cases (maximum width, depth)

#### Performance Tests
- Regression detection (<5% tolerance)
- Scaling validation (O(n) for linear, O(n log n) worst case)
- Memory usage bounds verification
