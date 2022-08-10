// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

// CHECK-LABEL: @select_cmp_eq_select
//       CHECK:   return %arg1
func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi eq, %arg0, %arg1 : i64
  %1 = arith.select %0, %arg0, %arg1 : i64
  return %1 : i64
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.iree.apply_patterns %0 { canonicalization }
  }
}
