// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  print %arg0 : !pdl.operation
}
