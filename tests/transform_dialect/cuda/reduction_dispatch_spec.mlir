// RUN: iree-opt %s

// Dispatch reduction.
transform.structured.canonicalized_sequence failures(propagate){
^bb1(%variant_op: !pdl.operation):
  %root = transform.structured.match interface{LinalgOp}
    attributes{iterator_types = ["parallel", "reduction"]} in %variant_op
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op

  // TODO: this could be replaced by a C++ only version.
  // Atm the IR produced is not the same so all pieces do not connect.
  %region_op = transform.iree.wrap_in_dispatch_region %root
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %fill into %region_op
  transform.iree.region_to_workgroups %region_op_2
}
