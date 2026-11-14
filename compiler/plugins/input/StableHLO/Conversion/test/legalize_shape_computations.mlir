// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-shape-computations))" \
// RUN:          --split-input-file %s | FileCheck %s
// RUN: iree-compile --iree-input-type=stablehlo --compile-to=input %s | FileCheck %s --check-prefix=INPUT

// CHECK-LABEL: func @get_dimension_size
func.func @get_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>) {
  %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[FROM:.+]] = tensor.from_elements %[[IDX]]
// CHECK: return %[[FROM]] : tensor<i32>

// -----

// CHECK-LABEL: func @reshape_dimension_size
func.func @reshape_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<1xi32>) {
  %0 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[FROM:.+]] = tensor.from_elements %[[IDX]]
// CHECK: return %[[FROM]] : tensor<1xi32>

// -----

// CHECK-LABEL: func @multiply_dimension_size
func.func @multiply_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>) {
  %0 = stablehlo.constant dense<2> : tensor<i32>
  %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %2 = "stablehlo.multiply"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}


// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[C2:.+]] = arith.constant 2
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[MUL:.+]] = arith.muli %[[IDX]], %[[C2]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[MUL]]
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: func @concat_dimension_size
func.func @concat_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<2xi32>) {
  %0 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
  %2 = stablehlo.constant dense<2> : tensor<1xi32>
  %3 = "stablehlo.concatenate"(%1, %2) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %3 : tensor<2xi32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[C2:.+]] = arith.constant 2
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[IDX]], %[[C2]]
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: func @compare_dimension_size
func.func @compare_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i1>) {
  %0 = stablehlo.constant dense<3> : tensor<i32>
  %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %2 = stablehlo.compare GE, %1, %0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[C3:.+]] = arith.constant 3
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[CMP:.+]] = arith.cmpi sge, %[[IDX]], %[[C3]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[CMP]]
// CHECK: return %[[RES]]

// -----

// JAX computes a derived dimension's quotient with floor-division helpers.
// CHECK-LABEL: func @floor_remainder_dimension_size
func.func @floor_remainder_dimension_size(%arg0: tensor<4x?x?xf32>) -> (tensor<i32>, tensor<i32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %0 = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<4x?x?xf32>) -> tensor<i32>
  %1 = stablehlo.remainder %0, %c3 : tensor<i32>
  %2 = stablehlo.compare LT, %1, %c0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.add %1, %c3 : tensor<i32>
  %4 = stablehlo.select %2, %3, %1 : tensor<i1>, tensor<i32>
  %5 = stablehlo.sign %0 : tensor<i32>
  func.return %4, %5 : tensor<i32>, tensor<i32>
}

// CHECK-NOT: stablehlo.remainder
// CHECK-NOT: stablehlo.select
// CHECK-NOT: stablehlo.sign
// CHECK: %[[DIM:.+]] = tensor.dim %arg0
// CHECK: %[[SIZE:.+]] = arith.index_cast %[[DIM]] : index to i32
// CHECK: %[[REM:.+]] = arith.remsi %[[SIZE]]
// CHECK: %[[NEG:.+]] = arith.cmpi slt, %[[REM]]
// CHECK: %[[ADD:.+]] = arith.addi %[[REM]]
// CHECK: %[[SEL:.+]] = arith.select %[[NEG]], %[[ADD]], %[[REM]] : i32
// CHECK: %[[QUOT:.+]] = tensor.from_elements %[[SEL]] : tensor<i32>
// CHECK: %[[SIGN:.+]] = tensor.from_elements %{{.+}} : tensor<i32>
// CHECK: return %[[QUOT]], %[[SIGN]]

// -----

// A scalar shape predicate does not imply a statically sized result.
// CHECK-LABEL: func @select_dynamic_vector
// CHECK: %[[PRED:.+]] = tensor.from_elements %arg0 : tensor<i1>
// CHECK: %[[SELECT:.+]] = stablehlo.select %[[PRED]], %arg1, %arg2
// CHECK: return %[[SELECT]] : tensor<?xi32>
// INPUT-LABEL: util.func public @select_dynamic_vector
// INPUT: arith.select
// INPUT-NOT: stablehlo.
// INPUT: util.return
func.func @select_dynamic_vector(%predicate: i1, %lhs: tensor<?xi32>, %rhs: tensor<?xi32>) -> tensor<?xi32> {
  %pred = tensor.from_elements %predicate : tensor<i1>
  %result = stablehlo.select %pred, %lhs, %rhs : tensor<i1>, tensor<?xi32>
  return %result : tensor<?xi32>
}

// -----

// CHECK-LABEL: func @select_static_vector
// CHECK: %[[SELECT:.+]] = arith.select %arg0,
// CHECK: %[[RESULT:.+]] = tensor.from_elements %[[SELECT]] : tensor<1xi32>
// CHECK: return %[[RESULT]]
func.func @select_static_vector(%predicate: i1, %lhs: tensor<1xi32>, %rhs: tensor<1xi32>) -> tensor<1xi32> {
  %pred = tensor.from_elements %predicate : tensor<i1>
  %result = stablehlo.select %pred, %lhs, %rhs : tensor<i1>, tensor<1xi32>
  return %result : tensor<1xi32>
}

// -----

// CHECK-LABEL: func @select_scalar
// CHECK: %[[SELECT:.+]] = arith.select %arg0,
// CHECK: %[[RESULT:.+]] = tensor.from_elements %[[SELECT]] : tensor<i32>
// CHECK: return %[[RESULT]]
func.func @select_scalar(%predicate: i1, %lhs: tensor<i32>, %rhs: tensor<i32>) -> tensor<i32> {
  %pred = tensor.from_elements %predicate : tensor<i1>
  %result = stablehlo.select %pred, %lhs, %rhs : tensor<i1>, tensor<i32>
  return %result : tensor<i32>
}

// -----

// CHECK-LABEL: func @select_empty_vector
// CHECK-NOT: stablehlo.select
// CHECK-NOT: tensor.extract
// CHECK: %[[RESULT:.+]] = {{.*}} : tensor<0xi32>
// CHECK: return %[[RESULT]]
func.func @select_empty_vector(%predicate: i1, %lhs: tensor<0xi32>, %rhs: tensor<0xi32>) -> tensor<0xi32> {
  %pred = tensor.from_elements %predicate : tensor<i1>
  %result = stablehlo.select %pred, %lhs, %rhs : tensor<i1>, tensor<0xi32>
  return %result : tensor<0xi32>
}

// -----

// Explicit unsigned integers need the tensor conversion's type normalisation.
// CHECK-LABEL: func @compare_unsigned_shape_scalar
// CHECK: %[[TENSOR:.+]] = tensor.from_elements %arg0 : tensor<ui32>
// CHECK: %[[RESULT:.+]] = stablehlo.compare GT, %[[TENSOR]],
// CHECK: return %[[RESULT]] : tensor<i1>
// INPUT-LABEL: util.func public @compare_unsigned_shape_scalar
// INPUT: arith.cmpi ugt,
// INPUT-NOT: stablehlo.
// INPUT: util.return
func.func @compare_unsigned_shape_scalar(%value: ui32) -> tensor<i1> {
  %lhs = tensor.from_elements %value : tensor<ui32>
  %rhs = stablehlo.constant dense<0> : tensor<ui32>
  %result = stablehlo.compare GT, %lhs, %rhs : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
  return %result : tensor<i1>
}

// -----

// The result type also retains unsigned semantics until tensor conversion.
// CHECK-LABEL: func @remainder_unsigned_shape_scalar
// CHECK: %[[TENSOR:.+]] = tensor.from_elements %arg0 : tensor<ui32>
// CHECK: %[[RESULT:.+]] = stablehlo.remainder %[[TENSOR]],
// CHECK: return %[[RESULT]] : tensor<ui32>
// INPUT-LABEL: util.func public @remainder_unsigned_shape_scalar
// INPUT: arith.remui
// INPUT-NOT: stablehlo.
// INPUT: util.return
func.func @remainder_unsigned_shape_scalar(%value: ui32) -> tensor<ui32> {
  %lhs = tensor.from_elements %value : tensor<ui32>
  %rhs = stablehlo.constant dense<2> : tensor<ui32>
  %result = stablehlo.remainder %lhs, %rhs : tensor<ui32>
  return %result : tensor<ui32>
}

// -----

// Unsigned tensor extracts crossing a helper boundary need type conversion too.
// CHECK-LABEL: func @extract_unsigned_shape_scalar
// CHECK: %[[VALUE:.+]] = tensor.extract %arg0[] : tensor<ui32>
// CHECK: return %[[VALUE]] : ui32
// INPUT-LABEL: util.func public @extract_unsigned_shape_scalar
// INPUT: %[[VALUE:.+]] = tensor.extract %arg0[] : tensor<i32>
// INPUT: util.return %[[VALUE]] : i32
func.func @extract_unsigned_shape_scalar(%input: tensor<ui32>) -> ui32 {
  %value = tensor.extract %input[] : tensor<ui32>
  return %value : ui32
}
