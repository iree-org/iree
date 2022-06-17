// RUN: iree-opt %s

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_matmul_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.generic"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  // Step 1: simple strategy, each thread does a full reduction.
  // transform.structured.canonicalized_sequence %arg0 {
  // ^bb1(%arg1: !pdl.operation):
  //   %0 = pdl_match @pdl_matmul_target in %arg1
  //   %tiling_1_result:2 = tile_to_foreach_thread_op %0 {num_threads = [7]}
  //   transform.iree.bufferize
  //   transform.iree.foreach_thread_to_gpu_and_translation_info
  // }

  // Step 2: split k, tile 1st split to scf.foreach_thread by 1x32.
  // This uses a special version of split-reduction that linearizes the split
  // expression and avoids the need for reshape ops.
  // transform.sequence %arg0 {
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_matmul_target in %arg1
    %fill_op, %more_parallel_op, %combiner_op = transform.structured.split_reduction %0
      { split_factor = 4, insert_split_dimension = 1}
    
    // %1 = pdl_match @pdl_matmul_target in %arg1
    tile_to_foreach_thread_op %fill_op {num_threads = [4, 4]}
    tile_to_foreach_thread_op %more_parallel_op {num_threads = [4, 4]}
    tile_to_foreach_thread_op %combiner_op {num_threads = [4]}

    transform.iree.bufferize
    transform.iree.foreach_thread_to_gpu_and_translation_info
  }
  
  // Step 3: tile to scf.for reduction by 1x32, vectorize 1x32, 
  // distribute vector along 32 on threadidx.x.
  // TODO: connect vector distribution to transform dialect. 

  // Step n: vectorize for vector load/store.
}
