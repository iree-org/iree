transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()
}
