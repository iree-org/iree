// RUN: iree-opt %s

// Dispatch softmax.
transform.sequence failures(propagate){
^bb1(%variant_op: !pdl.operation):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
    in %variant_op : (!pdl.operation) -> !pdl.operation

  %input_max_fill, %input_max, %exps_sum_fill, %exps, %exps_sum, %div =
    transform.split_handles %ops in [6]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation,
                             !pdl.operation, !pdl.operation, !pdl.operation)

  /// This must be used with the custom dispatch region formation
  /// because IREE's does not fuse the 6 ops softmax version even with
  /// --iree-flow-enable-aggressive-fusion.
  %region_op = transform.iree.wrap_in_dispatch_region %div { generateWorkload = false }

  %non_div = transform.merge_handles %input_max_fill, %input_max, %exps_sum_fill, %exps, %exps_sum
    : !pdl.operation
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_div into %region_op

  %empty = transform.structured.match ops{["tensor.empty"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %empty into %region_op_2
  transform.iree.region_to_workgroups %region_op_3
}
