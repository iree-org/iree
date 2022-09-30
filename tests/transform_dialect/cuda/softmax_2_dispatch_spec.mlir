// RUN: iree-opt %s

// Dispatch softmax.
transform.structured.canonicalized_sequence failures(propagate){
^bb1(%variant_op: !pdl.operation):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]} 
    in %variant_op

  %input_max_fill, %input_max, %exps_sum_fill, %exps, %exps_sum, %div = 
    transform.split_handles %ops in [6] 
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation,
                             !pdl.operation, !pdl.operation, !pdl.operation) 
  %non_div = transform.merge_handles %input_max_fill, %input_max, %exps_sum_fill, %exps, %exps_sum
    : !pdl.operation

  // This must be used with the custom dispatch region formation because IREE's
  // pulls in tensor.empty by default. This results in threadprivate allocations
  // and prevents vector distribution down the line.
  %region_op = transform.iree.wrap_in_dispatch_region %div
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_div into %region_op
  transform.iree.region_to_workgroups %region_op_2
}
