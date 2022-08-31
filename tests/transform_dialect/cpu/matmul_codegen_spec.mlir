// RUN: iree-opt %s 

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1

    %foreach_thread, %tiled_generic = 
      transform.structured.tile_to_foreach_thread_op %0 num_threads [2]
    
    transform.iree.bufferize
    
    %func = transform.structured.match ops{["func.func"]} in %arg1
    transform.iree.foreach_thread_to_workgroup %func
  }
}
