// RUN: iree-dialects-opt --transform-dialect-interpreter %s | FileCheck %s

// CHECK-LABEL: IR printer: test print
// CHECK-NEXT:  module
// CHECK-NEXT:  transform.structured.canonicalized_sequence
transform.structured.canonicalized_sequence {
^bb0(%arg0: !pdl.operation):
  print {name = "test print"}
}
