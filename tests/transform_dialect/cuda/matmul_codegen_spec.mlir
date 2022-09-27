// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op

  // Create phantom loop just to start offload
  %foreach_thread, %tiled_generic =
    transform.iree.tile_to_foreach_thread_and_workgroup_count_region %matmul tile_sizes [16, 8]
  
  // Bufferize
  %variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op   
  %func = transform.structured.match ops{["func.func"]} in %variant_op_2
  
  // Assign phantom loop to workgroups
  %func2 = transform.iree.foreach_thread_to_workgroup %func

  // Vectorize
  %isolated = transform.get_closest_isolated_parent %func2
  %isolated2 = transform.structured.vectorize %isolated
  
  // Vector -> WMMA
  transform.iree.target_vector_to "NVGPU" %isolated2
}
