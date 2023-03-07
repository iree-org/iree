transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %variant_op_2 = transform.iree.bufferize %variant_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func
}
