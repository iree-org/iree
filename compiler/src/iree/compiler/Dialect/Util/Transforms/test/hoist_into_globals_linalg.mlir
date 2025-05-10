// RUN: iree-opt --split-input-file --iree-util-hoist-into-globals %s | FileCheck %s

// Spot verification that policies for linalg ops is respected.

// CHECK-LABEL: @compute_hoisted
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @compute_hoisted {
  // CHECK: util.global private @[[HOISTED:.*]] : tensor<5x6xf32>
  // CHECK: util.initializer
  // CHECK: util.func public @main
  util.func public @main() -> (tensor<5x6xf32>) {
    %cst_0 = arith.constant dense<1.270000e+02> : tensor<f32>

    // A non-leaf broadcast.
    %0 = tensor.empty() : tensor<5x6xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<f32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<5x6xf32>

    // A leaf-compute.
    %2 = tensor.empty() : tensor<5x6xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%2 : tensor<5x6xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %42 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %42 : f32
    } -> tensor<5x6xf32>

    // CHECK: %[[RESULT:.*]] = util.global.load immutable @[[HOISTED]] : tensor<5x6xf32>
    // CHECK: util.return %[[RESULT]]
    util.return %3 : tensor<5x6xf32>
  }
}

// -----

// Verifies that projected permutations (broadcasts) will never be materialized
// as a leaf. Also verifies that empty operands, which can be considered
// const-expr, are not materialized as a leaf.

// CHECK-LABEL: @broadcast_treated_as_leaf
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @broadcast_treated_as_leaf {
  // CHECK-NOT: util.global
  // CHECK-NOT: util.initializer
  // CHECK: util.func public @main
  util.func public @main() -> (tensor<5x6xf32>) {
    %cst_0 = arith.constant dense<1.270000e+02> : tensor<f32>
    // CHECK: tensor.empty()
    %0 = tensor.empty() : tensor<5x6xf32>
    // A broadcast.
    // CHECK: linalg.generic
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<f32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<5x6xf32>
    // CHECK: util.return
    util.return %1 : tensor<5x6xf32>
  }
}

// -----

// Checks that a ConstantOp which only has a consumer within a nest gets hoisted.
module @nested_consumer {
  // CHECK: util.global private @[[HOISTED:.*]] : tensor<2xf32>
  // CHECK: util.initializer
  // CHECK: util.func public @main
  util.func @main(%arg0: tensor<2xindex>) -> tensor<1x2xf32> {
    %const_t = arith.constant dense<[0.0, 1.0]> : tensor<2xf32>
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<2xf32>
    %add_one = linalg.generic { //constOp
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%const_t : tensor<2xf32>) outs(%empty: tensor<2xf32>) {
      ^bb0(%in : f32, %out : f32):
        %added = arith.addf %in, %one : f32
        linalg.yield %added : f32
    } -> tensor<2xf32>
    // CHECK: %[[CONST:.*]] = util.global.load immutable @[[HOISTED]] : tensor<2xf32>
    %empty2 = tensor.empty() : tensor<2xf32>
    %loaded = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<2xindex>) outs(%empty2 : tensor<2xf32>) {
      ^bb0(%in: index, %out: f32):
        //CHECK %[[.*]] = tensor.extract %[[CONST]][%in] : tensor<2xf32>
        %extracted = tensor.extract %add_one[%in] : tensor<2xf32> // consumer
        linalg.yield %extracted : f32
    } -> tensor<2xf32>
    %reshaped = tensor.expand_shape %loaded [[0, 1]] output_shape[1, 2]: tensor<2xf32> into tensor<1x2xf32>
    util.return %reshaped : tensor<1x2xf32>
  }
}

// -----

// CHECK-LABEL: @do_not_hoist_sequence_basic
module @do_not_hoist_sequence_basic {
  // CHECK-NOT: util.initializer
  //     CHECK: util.func public @main
  //     CHECK:   linalg.generic
  //     CHECK:   linalg.generic
  util.func @main(%arg0 : tensor<128xf64>) -> (tensor<128xf64>){
    %0 = arith.constant dense<0> : tensor<128xi64>
    %1 = tensor.empty() : tensor<128xf64>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%0 : tensor<128xi64>) {
    ^bb0(%out: i64):
      %1870 = linalg.index 0 : index
      %1871 = arith.index_cast %1870 : index to i64
      linalg.yield %1871 : i64
    } -> tensor<128xi64>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<128xi64>) outs(%1 : tensor<128xf64>) {
    ^bb0(%ins: i64 , %outs: f64):
      %4 = arith.index_cast %ins : i64 to index
      %5 = tensor.extract %arg0 [%4] : tensor<128xf64>
      linalg.yield %5 : f64
    } -> tensor<128xf64>
    util.return %3 : tensor<128xf64>
  }
}

// -----

// CHECK-LABEL: @do_not_hoist_sequence_cst_from_above
module @do_not_hoist_sequence_cst_from_above {
  // CHECK-NOT: util.initializer
  //     CHECK: util.func public @main
  //     CHECK:   %[[V0:.+]] = linalg.generic
  //     CHECK:   util.return %[[V0]]
  util.func @main() -> (tensor<128xi64>){
    %0 = arith.constant dense<0> : tensor<128xi64>
    %cst = arith.constant 10 : i64
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%0 : tensor<128xi64>) {
    ^bb0(%out: i64):
      %00 = linalg.index 0 : index
      %01 = arith.index_cast %00 : index to i64
      %02 = arith.addi %cst, %01 : i64
      linalg.yield %02 : i64
    } -> tensor<128xi64>
    util.return %1 : tensor<128xi64>
  }
}

// -----

// CHECK-LABEL: @do_not_hoist_sequence_cst_argument
module @do_not_hoist_sequence_cst_argument {
  // CHECK-NOT: util.initializer
  //     CHECK: util.func public @main
  //     CHECK:   %[[V0:.+]] = linalg.generic
  //     CHECK:   util.return %[[V0]]
  util.func @main() -> (tensor<128xi64>){
    %0 = arith.constant dense<0> : tensor<128xi64>
    %cst = arith.constant 10 : i64
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%cst : i64) outs(%0 : tensor<128xi64>) {
    ^bb0(%in : i64, %out : i64):
      %00 = linalg.index 0 : index
      %01 = arith.index_cast %00 : index to i64
      %02 = arith.addi %cst, %01 : i64
      linalg.yield %02 : i64
    } -> tensor<128xi64>
    util.return %1 : tensor<128xi64>
  }
}

// -----

// CHECK-LABEL: @do_not_hoist_bit_extend
#map = affine_map<(d0) -> (d0)>
module @do_not_hoist_bit_extend {
  // CHECK-NOT: util.initializer
  //     CHECK: util.func public @main
  //     CHECK:   %[[V0:.+]] = linalg.generic
  //     CHECK:   util.return %[[V0]]
  util.func @main() -> (tensor<1024xf32>){
    %0 = arith.constant dense<3.14> : tensor<1024xf16>
    %1 = tensor.empty() : tensor<1024xf32>
    %cst = arith.constant 10 : i64
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<1024xf16>) outs(%1 : tensor<1024xf32>) {
    ^bb0(%in: f16, %out: f32):
      %4 = arith.extf %in : f16 to f32
      linalg.yield %4 : f32
    } -> tensor<1024xf32>
    util.return %2 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL: @do_hoist_bit_truncate
#map = affine_map<(d0) -> (d0)>
module @do_hoist_bit_truncate {
  //     CHECK: util.initializer
  //     CHECK: util.func public @main
  //     CHECK:   %[[V0:.+]] = util.global.load
  //     CHECK:   util.return %[[V0]]
  util.func @main() -> (tensor<1024xf16>){
    %0 = arith.constant dense<3.14> : tensor<1024xf32>
    %1 = tensor.empty() : tensor<1024xf16>
    %cst = arith.constant 10 : i64
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<1024xf32>) outs(%1 : tensor<1024xf16>) {
    ^bb0(%in: f32, %out: f16):
      %4 = arith.truncf %in : f32 to f16
      linalg.yield %4 : f16
    } -> tensor<1024xf16>
    util.return %2 : tensor<1024xf16>
  }
}
