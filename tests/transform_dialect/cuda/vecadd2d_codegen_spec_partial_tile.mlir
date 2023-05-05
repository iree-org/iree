transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  // Tile only one dimension, skip the other one.
  transform.iree.tile_to_forall_and_workgroup_count_region %generics 
                  tile_sizes [0, 3] ( mapping = [#gpu.block<z>])

  // Late canonicalizations to cleanup and pass the checks.
  // Needs to occur on the whole variant to perform cse on the workgroup_count region
  transform.iree.apply_patterns %variant_op
    { canonicalization, tiling_canonicalization, licm, cse } : (!pdl.operation              ) -> ()
}
