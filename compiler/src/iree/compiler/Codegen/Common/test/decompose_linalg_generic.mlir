// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-linalg-generic))" %s | FileCheck %s

// CHECK-LABEL: @parallel_with_broadcast_dynamic
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<64xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x64xf32>
func.func @parallel_with_broadcast_dynamic(
    %arg0 : tensor<64xf32>, %arg1 : tensor<?x64xf32>) -> tensor<?x64xf32> {
  // Verify that the decomposition pattern splits the generic into two operations:
  // First generic computes the add, second generic computes the multiply using
  // the result of the first.
  //       CHECK: %[[EMPTY:.+]] = tensor.empty
  //       CHECK: %[[ADD:.+]] = linalg.generic
  //  CHECK-SAME:   ins(%[[ARG0]] :
  //  CHECK-SAME:   outs(%[[EMPTY]] :
  //       CHECK:   %[[ADDF:.+]] = arith.addf
  //       CHECK:   linalg.yield %[[ADDF]]
  //       CHECK: %[[MUL:.+]] = linalg.generic
  //  CHECK-SAME:   ins(%[[ARG0]], %[[ADD]] :
  //  CHECK-SAME:   outs(%[[ARG1]] :
  //       CHECK:   %[[MULF:.+]] = arith.mulf
  //       CHECK:   linalg.yield %[[MULF]]
  //       CHECK: return %[[MUL]]
  %12 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<64xf32>) outs(%arg1 : tensor<?x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %13 = arith.addf %arg2, %arg2 : f32
    %14 = arith.mulf %13, %arg2 : f32
    linalg.yield %14 : f32
  } -> tensor<?x64xf32>
  func.return %12 : tensor<?x64xf32>
}

// -----

// CHECK-LABEL: @read_output_argument
func.func @read_output_argument(%x: tensor<4xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
  // Decomposing the producer must not enqueue its consumer with a read output.
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.yield
  // CHECK: %[[PRODUCER:.*]] = linalg.generic
  // CHECK: arith.mulf
  // CHECK: linalg.yield
  %empty = tensor.empty() : tensor<4xf32>
  %producer = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%x : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%element: f32, %unused: f32):
    %twice = arith.addf %element, %element : f32
    %product = arith.mulf %twice, %element : f32
    linalg.yield %product : f32
  } -> tensor<4xf32>
  // Reading the initial output must retain its value after decomposition.
  // CHECK: %[[RESULT:.*]] = linalg.generic
  // CHECK-SAME: ins(%[[PRODUCER]] :
  // CHECK: ^bb0(%[[X:.*]]: f32, %[[INIT:.*]]: f32):
  // CHECK: %[[TWICE:.*]] = arith.addf %[[X]], %[[X]] : f32
  // CHECK: %[[SUM:.*]] = arith.addf %[[TWICE]], %[[INIT]] : f32
  // CHECK: linalg.yield %[[SUM]] : f32
  // CHECK-NOT: linalg.generic
  // CHECK: return %[[RESULT]]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%producer : tensor<4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%element: f32, %initial: f32):
    %twice = arith.addf %element, %element : f32
    %sum = arith.addf %twice, %initial : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: @masked_score
func.func @masked_score(%mask: tensor<4xi8>, %scores: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant 0 : i8
  %bias = arith.constant -1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32
  // The comparison yields i1; it must not replace the f32 score argument.
  // CHECK: %[[RESULT:.*]] = linalg.generic
  // CHECK: ^bb0(%[[MASK:.*]]: i8, %[[SCORE:.*]]: f32):
  // CHECK: %[[VISIBLE:.*]] = arith.cmpi ne, %[[MASK]], %{{.*}} : i8
  // CHECK: %[[BIAS:.*]] = arith.select %[[VISIBLE]], %{{.*}}, %{{.*}} : f32
  // CHECK: %[[SUM:.*]] = arith.addf %[[SCORE]], %[[BIAS]] : f32
  // CHECK: linalg.yield %[[SUM]] : f32
  // CHECK-NOT: linalg.generic
  // CHECK: return %[[RESULT]]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%mask : tensor<4xi8>) outs(%scores : tensor<4xf32>) {
  ^bb0(%element: i8, %score: f32):
    %visible = arith.cmpi ne, %element, %zero : i8
    %selected = arith.select %visible, %zero_f32, %bias : f32
    %sum = arith.addf %score, %selected : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: @nested_scalar_region
func.func @nested_scalar_region(%x: tensor<4xf32>, %condition: i1) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic
  // CHECK: ^bb0(%[[ELEMENT:.*]]: f32, %{{.*}}: f32):
  // CHECK: scf.if
  // CHECK: scf.yield %[[ELEMENT]] : f32
  // CHECK: arith.negf %[[ELEMENT]] : f32
  // CHECK-NOT: linalg.generic
  // CHECK: return %[[RESULT]]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%x : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%element: f32, %unused: f32):
    %selected = scf.if %condition -> f32 {
      scf.yield %element : f32
    } else {
      %negative = arith.negf %element : f32
      scf.yield %negative : f32
    }
    %sum = arith.addf %selected, %selected : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: @complex_output
func.func @complex_output(%x: tensor<4xf32>) -> tensor<4xcomplex<f32>> {
  // The decomposition's zero initialiser only handles int/index/float values.
  // CHECK: %[[RESULT:.*]] = linalg.generic
  // CHECK: arith.negf
  // CHECK: complex.create
  // CHECK-NOT: linalg.generic
  // CHECK: return %[[RESULT]]
  %empty = tensor.empty() : tensor<4xcomplex<f32>>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%x : tensor<4xf32>) outs(%empty : tensor<4xcomplex<f32>>) {
  ^bb0(%element: f32, %unused: complex<f32>):
    %negative = arith.negf %element : f32
    %complex = complex.create %negative, %element : complex<f32>
    linalg.yield %complex : complex<f32>
  } -> tensor<4xcomplex<f32>>
  return %result : tensor<4xcomplex<f32>>
}

// -----

func.func private @external_effect(f32) -> f32

// CHECK-LABEL: @effectful_body
func.func @effectful_body(%x: tensor<4xf32>) -> tensor<4xf32> {
  // Splitting the loop must not change the ordering of payload side effects.
  // CHECK: %[[RESULT:.*]] = linalg.generic
  // CHECK: func.call @external_effect
  // CHECK: func.call @external_effect
  // CHECK-NOT: linalg.generic
  // CHECK: return %[[RESULT]]
  %empty = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%x : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%element: f32, %unused: f32):
    %first = func.call @external_effect(%element) : (f32) -> f32
    %second = func.call @external_effect(%first) : (f32) -> f32
    linalg.yield %second : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}
