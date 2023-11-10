transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op 
    : (!transform.any_op) -> !transform.any_op
  // Tile only one dimension, skip the other one.
  %_, %forall_grid = transform.structured.tile_using_forall %generics 
                  tile_sizes [0, 3] ( mapping = [#gpu.block<z>])
                   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()


  // Late canonicalizations to cleanup and pass the checks.
  // Needs to occur on the whole variant to perform cse on the workgroup_count region
  %func_op = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func_op {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_licm %func_op : !transform.any_op
  transform.iree.apply_cse %func_op : !transform.any_op
}
