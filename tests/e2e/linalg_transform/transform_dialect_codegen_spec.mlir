transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %variant_op_2 = transform.iree.bufferize %variant_op
  %mfunc_0 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
  %mfunc_1 = transform.apply_registered_pass "buffer-deallocation" to %mfunc_0 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %mfunc_1 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  %mfunc_2 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
  %mfunc_3 = transform.apply_registered_pass "buffer-deallocation-simplification" to %mfunc_2 : (!transform.any_op) -> !transform.any_op
  %mfunc_4 = transform.apply_registered_pass "bufferization-lower-deallocations" to %mfunc_3 : (!transform.any_op) -> !transform.any_op
  transform.apply_cse to %mfunc_4 : !transform.any_op
  %mfunc_5 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %mfunc_5 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
}
