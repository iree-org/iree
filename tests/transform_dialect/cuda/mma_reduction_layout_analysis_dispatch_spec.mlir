// RUN: iree-opt %s

transform.sequence failures(propagate){
^bb1(%variant_op: !transform.any_op):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul_transpose_b", "linalg.generic"]}
    in %variant_op : (!transform.any_op) -> !transform.any_op

  %fill0, %fill1, %matmul, %reduce, %broadcast =
    transform.split_handle %ops
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op,
                             !transform.any_op, !transform.any_op)

  %region_op = transform.iree.wrap_in_dispatch_region %broadcast { generateWorkload = false } : (!transform.any_op) -> !transform.any_op

  %non_broadcast = transform.merge_handles %fill0, %fill1, %matmul, %reduce : !transform.any_op
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_broadcast into %region_op : (!transform.any_op, !transform.any_op) -> !transform.any_op

  %empty = transform.structured.match ops{["tensor.empty"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %empty into %region_op_2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
  transform.iree.region_to_workgroups %region_op_3 : (!transform.any_op) -> !transform.any_op
}
