transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %region_op = transform.iree.wrap_in_dispatch_region %0
    transform.iree.region_to_workgroups %region_op
  }
}
