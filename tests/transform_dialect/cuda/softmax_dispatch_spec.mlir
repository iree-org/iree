// RUN: iree-opt %s

// Dispatch softmax.
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 failures(propagate){
  ^bb1(%arg1: !pdl.operation):
    %root = transform.structured.match interface{LinalgOp}
      attributes{iterator_types = ["parallel", "parallel", "parallel"]} in %arg1
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1
    %red = transform.structured.match interface{LinalgOp}
      attributes{iterator_types = ["parallel", "parallel", "reduction"]} in %arg1

    // TODO: this could be replaced by a C++ only version.
    // Atm the IR produced is not the same so all pieces do not connect.
    %region_op = transform.iree.wrap_in_dispatch_region %root
    %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %red into %region_op
    %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %fill into %region_op_2
    transform.iree.region_to_workgroups %region_op_3
  }
}
