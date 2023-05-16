// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  print %arg0 : !transform.any_op
}
