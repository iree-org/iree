// RUN: iree-opt %s

transform.sequence failures(propagate){
^bb1(%variant_op: !pdl.operation):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul", "linalg.generic"]}
    in %variant_op : (!pdl.operation) -> !pdl.operation

  %fill0, %fill1, %matmul, %reduce, %broadcast =
    transform.split_handles %ops in [5]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation,
                             !pdl.operation, !pdl.operation)

  %region_op = transform.iree.wrap_in_dispatch_region %broadcast { generateWorkload = false }

  %non_broadcast = transform.merge_handles %fill0, %fill1, %matmul, %reduce : !pdl.operation
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_broadcast into %region_op

  %empty = transform.structured.match ops{["tensor.empty"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %empty into %region_op_2
  transform.iree.region_to_workgroups %region_op_3
}
