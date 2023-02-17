transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  // Tile only one dimension, skip the other one.
  transform.iree.tile_to_forall_and_workgroup_count_region %generics 
                  tile_sizes [0, 3] ( mapping = [#gpu.block<z>])
}
