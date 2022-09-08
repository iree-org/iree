transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%variant_op: !pdl.operation):
    transform.iree.bufferize %variant_op
  }
}
