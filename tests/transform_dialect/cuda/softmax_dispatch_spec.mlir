// RUN: iree-opt %s

// Dispatch softmax.
transform.sequence failures(propagate){
^bb1(%variant_op: !transform.any_op):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
    in %variant_op : (!transform.any_op) -> !transform.any_op

  %input_max_fill, %input_max, %exps_sum_fill, %exps, %exps_sum, %div =
    transform.split_handle %ops
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op,
                             !transform.any_op, !transform.any_op, !transform.any_op)

  /// This must be used with the custom dispatch region formation
  /// because IREE's does not fuse the 6 ops softmax version even with
  /// --iree-flow-fuse-multi-use.
  %region_op = transform.iree.wrap_in_dispatch_region %div { generateWorkload = false } : (!transform.any_op) -> !transform.any_op

  %non_div = transform.merge_handles %input_max_fill, %input_max, %exps_sum_fill, %exps, %exps_sum
    : !transform.any_op
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_div into %region_op : (!transform.any_op, !transform.any_op) -> !transform.any_op

  %empty = transform.structured.match ops{["tensor.empty"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %empty into %region_op_2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
  transform.iree.region_to_workgroups %region_op_3 : (!transform.any_op) -> !transform.any_op
}
