transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op 
    : (!transform.any_op) -> !transform.any_op
  // Tile only one dimension, skip the other one.
  %forall_grid, %_ = transform.structured.tile_to_forall_op %generics 
                  tile_sizes [0, 3] ( mapping = [#gpu.block<z>])
                   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()


  // Late canonicalizations to cleanup and pass the checks.
  // Needs to occur on the whole variant to perform cse on the workgroup_count region
  transform.apply_patterns to %variant_op {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
  } : !transform.any_op
  transform.iree.apply_patterns %variant_op
    { canonicalization, licm, cse } : (!transform.any_op) -> ()
}
