// RUN: iree-dialects-opt --linalg-interp-transforms %s | FileCheck %s

// CHECK-LABEL: IR printer: test print
// CHECK-NEXT:  module
// CHECK-NEXT:  iree_linalg_transform.sequence
iree_linalg_transform.sequence {
  print {name = "test print"}
}
