// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-merge-exported-reflection %s | IreeFileCheck %s

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
// expected-error @+1 {{Reflection signature missing for argument 0}}
func @missingArgumentPartial(%arg0 : tensor<4x4xi64>, %arg1 : tensor<5x5xi64>) -> tensor<5x5xi64>
    attributes {iree.module.export}
{
  return %arg1 : tensor<5x5xi64>
}

// -----
// expected-error @+1 {{Reflection signature missing for result 0}}
func @missingResultPartial(%arg0 : i1 {iree.reflection = {f_partial = "I6!B3!d1"}}) -> (i1)
    attributes {iree.module.export}
{
  return %arg0 : i1
}
