// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-materialize-exported-reflection -iree-flow-merge-exported-reflection %s | IreeFileCheck %s

// -----
// CHECK-LABEL: func @notExported
// CHECK-NOT: iree.reflection
func @notExported(%arg0 : tensor<4x4xi64>) -> tensor<4x4xi64> {
  return %arg0 : tensor<4x4xi64>
}

// -----
// CHECK-LABEL: func @emptyWithVersion
// CHECK-SAME: iree.reflection = {f = "I1!R1!", fv = "1"}
func @emptyWithVersion() -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @exportedTensor
// CHECK-SAME: iree.reflection = {f = "I27!I10!B7!t7d4d4I10!B7!t7d5d5R14!R10!B7!t7d5d5", fv = "1"}
func @exportedTensor(%arg0 : tensor<4x4xi64>, %arg1 : tensor<5x5xi64>) -> tensor<5x5xi64>
    attributes {iree.module.export}
{
  return %arg1 : tensor<5x5xi64>
}

// -----
// expected-error @+1 {{Illegal partial reflection attribute: '' on argument 0}}
func @unsupportedType(%arg0 : i1) -> ()
    attributes {iree.module.export}
{
  return
}
