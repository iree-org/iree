// RUN: iree-opt %s

// Dispatch elementwise(fusion(elementwise)) into the same region. The elementwise
// feeding the reduction is automatically fused by IREE, so we disable that fusion
// and form the dispatch region manually here to avoid that.
transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %variant_op
  %fill, %leading_eltwise, %reduction, %trailing_eltwise = transform.split_handles %ops in [4]
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %region_op1 = transform.iree.wrap_in_dispatch_region %trailing_eltwise
  %others = transform.merge_handles %fill, %leading_eltwise, %reduction : !pdl.operation
  %region_op2 = transform.iree.move_preceding_op_into_dispatch_region %others into %region_op1
  transform.iree.region_to_workgroups %region_op2
}
