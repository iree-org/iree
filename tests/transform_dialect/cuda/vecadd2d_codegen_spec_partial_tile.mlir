transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  // Tile only one dimension, skip the other one.
  %forall_grid, %_ = transform.structured.tile_to_forall_op %generics 
                  tile_sizes [0, 3] ( mapping = [#gpu.block<z>])
                   : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!pdl.operation) -> ()


  // Late canonicalizations to cleanup and pass the checks.
  // Needs to occur on the whole variant to perform cse on the workgroup_count region
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation              ) -> ()
}
