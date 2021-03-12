// RUN: iree-opt -split-input-file -verify-diagnostics -iree-sip-materialize-reflection-attrs %s | IreeFileCheck %s

// CHECK-LABEL: func @notExported
// CHECK-NOT: iree.reflection
func @notExported(%arg0 : tensor<4x4xi64>) -> tensor<4x4xi64> {
  return %arg0 : tensor<4x4xi64>
}

// -----

// CHECK-LABEL: func @emptyWithVersion
// CHECK-SAME: iree.reflection = {f = "I1!R1!", fv = "1"}
func @emptyWithVersion() -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @exportedTensor
// CHECK-SAME: iree.reflection = {f = "I19!B7!t7d4d4B7!t7d5d5R10!B7!t7d5d5", fv = "1"}
func @exportedTensor(%arg0 : tensor<4x4xi64>, %arg1 : tensor<5x5xi64>) -> tensor<5x5xi64> attributes {
  iree.module.export
} {
  return %arg1 : tensor<5x5xi64>
}

// -----

// CHECK-LABEL: func @dynamicDim
// CHECK-SAME: iree.reflection = {f = "I11!B8!t7d-1d4R1!", fv = "1"}
func @dynamicDim(%arg0 : tensor<?x4xi64>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @scalari32
// CHECK-SAME: iree.reflection = {f = "I6!S3!t6R1!", fv = "1"}
func @scalari32(%arg0 : i32) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @tensorFloat32
// CHECK-SAME: iree.reflection = {f = "I6!B3!d1R1!", fv = "1"}
func @tensorFloat32(%arg0 : tensor<1xf32>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @tensorFloat64
// CHECK-SAME: iree.reflection = {f = "I8!B5!t2d1R1!", fv = "1"}
func @tensorFloat64(%arg0 : tensor<1xf64>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @tensorFloat16
// CHECK-SAME: iree.reflection = {f = "I8!B5!t1d1R1!", fv = "1"}
func @tensorFloat16(%arg0 : tensor<1xf16>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @tensorBfloat16
// CHECK-SAME: iree.reflection = {f = "I8!B5!t3d1R1!", fv = "1"}
func @tensorBfloat16(%arg0 : tensor<1xbf16>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @tensorSint8
// CHECK-SAME: iree.reflection = {f = "I8!B5!t4d1R1!", fv = "1"}
func @tensorSint8(%arg0 : tensor<1xi8>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @tensorSint16
// CHECK-SAME: iree.reflection = {f = "I8!B5!t5d1R1!", fv = "1"}
func @tensorSint16(%arg0 : tensor<1xi16>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @tensorSint32
// CHECK-SAME: iree.reflection = {f = "I8!B5!t6d1R1!", fv = "1"}
func @tensorSint32(%arg0 : tensor<1xi32>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @tensorSint64
// CHECK-SAME: iree.reflection = {f = "I8!B5!t7d1R1!", fv = "1"}
func @tensorSint64(%arg0 : tensor<1xi64>) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @noReflectionOnAbiNone
// CHECK-NOT: iree.reflection
func @noReflectionOnAbiNone(%arg0 : tensor<4x4xi64>, %arg1 : tensor<5x5xi64>) -> tensor<5x5xi64> attributes {
  iree.module.export,
  iree.abi.none
} {
  return %arg1 : tensor<5x5xi64>
}

// -----

// CHECK-LABEL: @unsupportedTypeOnAbiNone
// Should not generate warning
func @unsupportedTypeOnAbiNone(%arg0 : i1) -> () attributes {
  iree.module.export,
  iree.abi.none
} {
  return
}

// -----

// CHECK-LABEL: @reflectionOnBool
// CHECK-SAME: iree.reflection = {f = "I6!S3!t4R1!", fv = "1"}
func @reflectionOnBool(%arg0 : i1) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// expected-warning @+1 {{Argument #0 of function unsupportedType is not a recognized public ABI type and the function may not be invokable by standard tools}}
func @unsupportedType(%arg0 : i3) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @unrecognizedArgument
// CHECK-SAME: iree.reflection = {f = "I4!U1!R1!", fv = "1"}
// expected-warning @+1 {{Argument #0 of function unrecognizedArgument is not a recognized public ABI type and the function may not be invokable by standard tools}}
func @unrecognizedArgument(%arg0 : i3) -> () attributes {
  iree.module.export
} {
  return
}

// -----

// CHECK-LABEL: func @unrecognizedResult
// CHECK-SAME: iree.reflection = {f = "I1!R4!U1!", fv = "1"}
// expected-warning @+1 {{Result #0 of function unrecognizedResult is not a recognized public ABI type and the function may not be invokable by standard tools}}
func @unrecognizedResult() -> (i3) attributes {
  iree.module.export
} {
  %0 = constant 0 : i3
  return %0 : i3
}
