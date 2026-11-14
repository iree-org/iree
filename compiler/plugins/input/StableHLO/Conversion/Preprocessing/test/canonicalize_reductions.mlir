// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-stablehlo-canonicalize))" %s | FileCheck %s --check-prefix=CANON
// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s --check-prefix=LOWER --implicit-check-not=stablehlo.

// The region declares the arithmetic type; the source operand still determines
// whether widening uses zero or sign extension.
// CANON-LABEL: func.func @promoted_inputs
// CANON-DAG: %[[P:.+]] = stablehlo.convert %arg0 : (tensor<2xi1>) -> tensor<2xi32>
// CANON-DAG: %[[S:.+]] = stablehlo.convert %arg1 : (tensor<2xi8>) -> tensor<2xi32>
// CANON-DAG: %[[U:.+]] = stablehlo.convert %arg2 : (tensor<2xui8>) -> tensor<2xi32>
// CANON-DAG: stablehlo.reduce(%[[P]] init: %arg3)
// CANON-DAG: stablehlo.reduce(%[[S]] init: %arg3)
// CANON-DAG: stablehlo.reduce(%[[U]] init: %arg3)
// LOWER-LABEL: func.func @promoted_inputs
// LOWER-DAG: arith.extui %{{.+}} : i1 to i32
// LOWER-DAG: arith.extsi %{{.+}} : i8 to i32
// LOWER-DAG: arith.extui %{{.+}} : i8 to i32
// LOWER: return %{{.+}}, %{{.+}}, %{{.+}} : tensor<i32>, tensor<i32>, tensor<i32>
func.func @promoted_inputs(%p: tensor<2xi1>, %s: tensor<2xi8>, %u: tensor<2xui8>, %init: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %a = stablehlo.reduce(%p init: %init) across dimensions = [0] : (tensor<2xi1>, tensor<i32>) -> tensor<i32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  %b = stablehlo.reduce(%s init: %init) across dimensions = [0] : (tensor<2xi8>, tensor<i32>) -> tensor<i32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  %c = stablehlo.reduce(%u init: %init) across dimensions = [0] : (tensor<2xui8>, tensor<i32>) -> tensor<i32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %a, %b, %c : tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

// CANON-LABEL: func.func @promoted_initializer
// CANON: %[[INIT:.+]] = stablehlo.convert %arg1 : (tensor<f16>) -> tensor<f32>
// CANON: stablehlo.reduce(%arg0 init: %[[INIT]])
// LOWER-LABEL: func.func @promoted_initializer
// LOWER: arith.extf %{{.+}} : f16 to f32
// LOWER: linalg.fill
// LOWER: arith.addf
func.func @promoted_initializer(%x: tensor<?xf32>, %init: tensor<f16>) -> tensor<f32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [0] : (tensor<?xf32>, tensor<f16>) -> tensor<f32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %v : tensor<f32>
    }
  return %r : tensor<f32>
}

// -----

// Shapes and encodings survive while normalising element types.
// CANON-LABEL: func.func @promoted_bounded_input
// CANON: %[[INPUT:.+]] = stablehlo.convert %arg0 : (tensor<?x2xui8, #stablehlo.bounds<16, ?>>) -> tensor<?x2xi32, #stablehlo.bounds<16, ?>>
// CANON: stablehlo.reduce(%[[INPUT]] init: %arg1)
// LOWER-LABEL: func.func @promoted_bounded_input
// LOWER: util.assume.int %{{.+}}<umax = 16>
// LOWER: arith.extui %{{.+}} : i8 to i32
func.func @promoted_bounded_input(%x: tensor<?x2xui8, #stablehlo.bounds<16, ?>>, %init: tensor<i32>) -> tensor<?xi32, #stablehlo.bounds<16>> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [1] : (tensor<?x2xui8, #stablehlo.bounds<16, ?>>, tensor<i32>) -> tensor<?xi32, #stablehlo.bounds<16>>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %r : tensor<?xi32, #stablehlo.bounds<16>>
}

// -----

// CANON-LABEL: func.func @zero_axes_promoted
// CANON: %[[RESULT:.+]] = stablehlo.convert %arg0 : (tensor<2xui8>) -> tensor<2xi32>
// CANON-NOT: stablehlo.reduce
// CANON: return %[[RESULT]]
// LOWER-LABEL: func.func @zero_axes_promoted
// LOWER: arith.extui %{{.+}} : i8 to i32
// LOWER: return %{{.+}} : tensor<2xi32>
func.func @zero_axes_promoted(%x: tensor<2xui8>, %init: tensor<i32>) -> tensor<2xi32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [] : (tensor<2xui8>, tensor<i32>) -> tensor<2xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %r : tensor<2xi32>
}

// -----

// CANON-LABEL: func.func @zero_axes_dynamic_result
// CANON: %[[RESULT:.+]] = tensor.cast %arg0 : tensor<2xi32> to tensor<?xi32>
// CANON: return %[[RESULT]]
// LOWER-LABEL: func.func @zero_axes_dynamic_result
// LOWER: return %{{.+}} : tensor<?xi32>
func.func @zero_axes_dynamic_result(%x: tensor<2xi32>, %init: tensor<i32>) -> tensor<?xi32> {
  %r = stablehlo.reduce(%x init: %init) applies stablehlo.add across dimensions = [] : (tensor<2xi32>, tensor<i32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}

// -----

// Empty reductions broadcast the promoted initializer, regardless of pattern
// order. The scalar cannot be broadcast to a different element type directly.
// CANON-LABEL: func.func @empty_promoted_initializer
// CANON: %[[INIT:.+]] = stablehlo.convert %arg1 : (tensor<i8>) -> tensor<i32>
// CANON: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %[[INIT]], dims = [] : (tensor<i32>) -> tensor<3xi32>
// CANON: return %[[RESULT]]
// LOWER-LABEL: func.func @empty_promoted_initializer
// LOWER: arith.extsi %{{.+}} : i8 to i32
// LOWER: return %{{.+}} : tensor<3xi32>
func.func @empty_promoted_initializer(%x: tensor<0x3xi8>, %init: tensor<i8>) -> tensor<3xi32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [0] : (tensor<0x3xi8>, tensor<i8>) -> tensor<3xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %r : tensor<3xi32>
}

// -----

// CANON-LABEL: func.func @dynamic_empty_promoted_initializer
// CANON: %[[INIT:.+]] = stablehlo.convert %arg1 : (tensor<i8>) -> tensor<i32>
// CANON: stablehlo.dynamic_broadcast_in_dim %[[INIT]]
// LOWER-LABEL: func.func @dynamic_empty_promoted_initializer
// LOWER: arith.extsi %{{.+}} : i8 to i32
// LOWER: return %{{.+}} : tensor<?xi32>
func.func @dynamic_empty_promoted_initializer(%x: tensor<?x0xi8>, %init: tensor<i8>) -> tensor<?xi32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [1] : (tensor<?x0xi8>, tensor<i8>) -> tensor<?xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      %v = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %v : tensor<i32>
    }
  return %r : tensor<?xi32>
}

// -----

// Every lane uses its own input and initializer type.
// CANON-LABEL: func.func @variadic_promotions
// CANON-DAG: stablehlo.convert %arg0 : (tensor<2xi8>) -> tensor<2xi32>
// CANON-DAG: stablehlo.convert %arg1 : (tensor<2xui8>) -> tensor<2xi64>
// CANON-DAG: stablehlo.convert %arg2 : (tensor<i16>) -> tensor<i32>
// CANON-DAG: stablehlo.convert %arg3 : (tensor<i32>) -> tensor<i64>
// CANON: return %{{.+}}#0, %{{.+}}#1 : tensor<i32>, tensor<i64>
// LOWER-LABEL: func.func @variadic_promotions
// LOWER-DAG: arith.extsi %{{.+}} : i16 to i32
// LOWER-DAG: arith.extsi %{{.+}} : i32 to i64
// LOWER-DAG: arith.extui %{{.+}} : i8 to i64
// LOWER: return %{{.+}} : tensor<i32>, tensor<i64>
func.func @variadic_promotions(%s: tensor<2xi8>, %u: tensor<2xui8>, %init0: tensor<i16>, %init1: tensor<i32>) -> (tensor<i32>, tensor<i64>) {
  %r:2 = "stablehlo.reduce"(%s, %u, %init0, %init1) ({
    ^bb0(%a: tensor<i32>, %b: tensor<i64>, %x: tensor<i32>, %y: tensor<i64>):
      %v0 = stablehlo.add %a, %x : tensor<i32>
      %v1 = stablehlo.add %b, %y : tensor<i64>
      stablehlo.return %v0, %v1 : tensor<i32>, tensor<i64>
  }) {dimensions = array<i64: 0>} : (tensor<2xi8>, tensor<2xui8>, tensor<i16>, tensor<i32>) -> (tensor<i32>, tensor<i64>)
  return %r#0, %r#1 : tensor<i32>, tensor<i64>
}

// -----

// A captured scalar cannot replace a partially reduced tensor directly.
// CANON-LABEL: func.func @captured_scalar_partial_reduction
// CANON: %[[RESULT:.+]] = stablehlo.reduce
// CANON: return %[[RESULT]] : tensor<3xi32>
// LOWER-LABEL: func.func @captured_scalar_partial_reduction
// LOWER: return %{{.+}} : tensor<3xi32>
func.func @captured_scalar_partial_reduction(%x: tensor<2x3xi32>, %capture: tensor<i32>, %init: tensor<i32>) -> tensor<3xi32> {
  %r = stablehlo.reduce(%x init: %init) across dimensions = [0] : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
    reducer(%lhs: tensor<i32>, %rhs: tensor<i32>) {
      stablehlo.return %capture : tensor<i32>
    }
  return %r : tensor<3xi32>
}
