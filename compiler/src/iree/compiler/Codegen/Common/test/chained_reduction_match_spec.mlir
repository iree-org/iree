// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.iree.register_match_callbacks
  %0:7 = transform.iree.match_callback failures(propagate) "chained_reduction"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  transform.iree.emit_remark "leading_1" at %0#0 : !pdl.operation
  transform.iree.emit_remark "fill_1" at %0#1 : !pdl.operation
  transform.iree.emit_remark "reduction_1" at %0#2 : !pdl.operation
  transform.iree.emit_remark "middle" at %0#3 : !pdl.operation
  transform.iree.emit_remark "fill_2" at %0#4 : !pdl.operation
  transform.iree.emit_remark "reduction_2" at %0#5 : !pdl.operation
  transform.iree.emit_remark "trailing_2" at %0#6 : !pdl.operation
}
