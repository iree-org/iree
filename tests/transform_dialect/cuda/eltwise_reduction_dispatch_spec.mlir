// RUN: iree-opt %s

// Dispatch elementwise followed by fusion into the same region.
transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} in %variant_op
  %fill, %eltwise, %reduction = transform.split_handles %ops in [3]
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  %region_op1 = transform.iree.wrap_in_dispatch_region %reduction
  %others = transform.merge_handles %fill, %eltwise : !pdl.operation
  %region_op2 = transform.iree.move_preceding_op_into_dispatch_region %others into %region_op1
  transform.iree.region_to_workgroups %region_op2
}
