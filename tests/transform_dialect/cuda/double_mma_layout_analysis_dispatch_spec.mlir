// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%func: !transform.any_op) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul"]}
      in %func : (!transform.any_op) -> !transform.any_op

    %fill0, %matmul0, %fill1, %matmul1 =
      transform.split_handle %ops
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                               !transform.any_op, !transform.any_op)

    %region_op = transform.iree.wrap_in_dispatch_region %matmul1 { generateWorkload = false } : (!transform.any_op) -> !transform.any_op

    %non_matmul1 = transform.merge_handles %fill0, %matmul0, %fill1 : !transform.any_op
    %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_matmul1 into %region_op : (!transform.any_op, !transform.any_op) -> !transform.any_op

    %empty = transform.structured.match ops{["tensor.empty"]} in %func : (!transform.any_op) -> !transform.any_op
    %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %empty into %region_op_2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.iree.region_to_workgroups %region_op_3 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
