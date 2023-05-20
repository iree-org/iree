// RUN: iree-opt %s

transform.sequence failures(propagate){
^bb1(%variant_op: !pdl.operation):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul"]}
    in %variant_op : (!pdl.operation) -> !pdl.operation

  %fill0, %matmul0, %fill1, %matmul1 =
    transform.split_handle %ops
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation,
                             !pdl.operation, !pdl.operation)

  %region_op = transform.iree.wrap_in_dispatch_region %matmul1 { generateWorkload = false }

  %non_matmul1 = transform.merge_handles %fill0, %matmul0, %fill1 : !pdl.operation
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_matmul1 into %region_op

  %empty = transform.structured.match ops{["tensor.empty"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %empty into %region_op_2
  transform.iree.region_to_workgroups %region_op_3
}
