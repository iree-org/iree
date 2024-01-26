// RUN: iree-opt --split-input-file --iree-consteval-jit-target-backend=vmvx --verify-diagnostics --iree-consteval-jit-debug --iree-consteval-jit-globals  %s | FileCheck %s

// TODO(laurenzo): Full type matrix for tests.

module @no_uninitialized {
  util.global private @hoisted : tensor<5x6xf32> = dense<4.0> : tensor<5x6xf32>
  func.func @main() -> tensor<5x6xf32> {
    %hoisted = util.global.load @hoisted : tensor<5x6xf32>
    return %hoisted : tensor<5x6xf32>
  }
}

// -----
// CHECK-LABEL: @linalg_tensor_jit
// CHECK: util.global private @{{.*}} = dense<4.000000e+04> : tensor<5x6xf32>
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @linalg_tensor_jit {
  util.global private @hoisted : tensor<5x6xf32>
  func.func @main() -> tensor<5x6xf32> {
    %hoisted = util.global.load @hoisted : tensor<5x6xf32>
    return %hoisted : tensor<5x6xf32>
  }
  // CHECK-NOT: util.initializer
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<2.0e+02> : tensor<f32>
    %0 = tensor.empty() : tensor<5x6xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
      linalg.yield %arg0 : f32
    } -> tensor<5x6xf32>
    %2 = tensor.empty() : tensor<5x6xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%2 : tensor<5x6xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
      %4 = arith.mulf %arg0, %arg1 : f32
      linalg.yield %4 : f32
    } -> tensor<5x6xf32>
    util.global.store %3, @hoisted : tensor<5x6xf32>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_splat_detection
// CHECK: util.global private @{{.*}} = dense<2> : tensor<2xi32>
module @eval_splat_detection {
  util.global private @hoisted : tensor<2xi32>
  func.func @main() -> tensor<2xi32> {
    %hoisted = util.global.load @hoisted : tensor<2xi32>
    return %hoisted : tensor<2xi32>
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
    util.global.store %cst, @hoisted : tensor<2xi32>
    util.return
  }
}


// -----
// CHECK-LABEL: @eval_f16_tensor
module @eval_f16_tensor {
  util.global private @hoisted : tensor<5x6xf16>
  func.func @main() -> tensor<5x6xf16> {
    %hoisted = util.global.load @hoisted : tensor<5x6xf16>
    return %hoisted : tensor<5x6xf16>
  }
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<2.0e+2> : tensor<5x6xf16>
    util.global.store %cst, @hoisted : tensor<5x6xf16>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_bf16_tensor
// Not currently supported (initializer should remain)
module @eval_bf16_tensor {
  util.global private @hoisted : tensor<5x6xbf16>
  func.func @main() -> tensor<5x6xbf16> {
    %hoisted = util.global.load @hoisted : tensor<5x6xbf16>
    return %hoisted : tensor<5x6xbf16>
  }
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<2.0e+2> : tensor<5x6xbf16>
    util.global.store %cst, @hoisted : tensor<5x6xbf16>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_f32_tensor
// CHECK: util.global private @{{.*}} = dense<[2.000000e+02, 3.200000e+03]> : tensor<2xf32>
module @eval_f32_tensor {
  util.global private @hoisted : tensor<2xf32>
  func.func @main() -> tensor<2xf32> {
    %hoisted = util.global.load @hoisted : tensor<2xf32>
    return %hoisted : tensor<2xf32>
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[2.0e+2, 3.2e+3]> : tensor<2xf32>
    util.global.store %cst, @hoisted : tensor<2xf32>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_f64_tensor
module @eval_f64_tensor {
  util.global private @hoisted : tensor<2xf64>
  func.func @main() -> tensor<2xf64> {
    %hoisted = util.global.load @hoisted : tensor<2xf64>
    return %hoisted : tensor<2xf64>
  }
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[2.0e+2, 3.2e+3]> : tensor<2xf64>
    util.global.store %cst, @hoisted : tensor<2xf64>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_i1_tensor
// CHECK: util.global private @{{.*}} = dense<[false, true, false, true, true, false]> : tensor<6xi1>
module @eval_i1_tensor {
  util.global private @hoisted : tensor<6xi1>
  func.func @main() -> tensor<6xi1> {
    %hoisted = util.global.load @hoisted : tensor<6xi1>
    return %hoisted : tensor<6xi1>
  }
  util.initializer attributes {iree.compiler.consteval} {
    // Note that the level we are testing at is a bit odd in the way i1 vs
    // i8 are handled.
    %cst = arith.constant dense<[0, 1, 0, 1, 1, 0]> : tensor<6xi8>
    %casted = arith.trunci %cst : tensor<6xi8> to tensor<6xi1>
    util.global.store %casted, @hoisted : tensor<6xi1>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_i4_tensor
module @eval_i4_tensor {
  util.global private @hoisted : tensor<5x6xi4>
  func.func @main() -> tensor<5x6xi4> {
    %hoisted = util.global.load @hoisted : tensor<5x6xi4>
    return %hoisted : tensor<5x6xi4>
  }
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<3> : tensor<5x6xi4>
    util.global.store %cst, @hoisted : tensor<5x6xi4>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_i8_tensor
// CHECK: util.global private @{{.*}} = dense<[2, 3]> : tensor<2xi8>
module @eval_i8_tensor {
  util.global private @hoisted : tensor<2xi8>
  func.func @main() -> tensor<2xi8> {
    %hoisted = util.global.load @hoisted : tensor<2xi8>
    return %hoisted : tensor<2xi8>
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi8>
    util.global.store %cst, @hoisted : tensor<2xi8>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_i16_tensor
// CHECK: util.global private @{{.*}} = dense<[2, 3]> : tensor<2xi16>
module @eval_i16_tensor {
  util.global private @hoisted : tensor<2xi16>
  func.func @main() -> tensor<2xi16> {
    %hoisted = util.global.load @hoisted : tensor<2xi16>
    return %hoisted : tensor<2xi16>
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi16>
    util.global.store %cst, @hoisted : tensor<2xi16>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_i32_tensor
// CHECK: util.global private @{{.*}} = dense<[2, 3]> : tensor<2xi32>
module @eval_i32_tensor {
  util.global private @hoisted : tensor<2xi32>
  func.func @main() -> tensor<2xi32> {
    %hoisted = util.global.load @hoisted : tensor<2xi32>
    return %hoisted : tensor<2xi32>
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi32>
    util.global.store %cst, @hoisted : tensor<2xi32>
    util.return
  }
}

// -----
// CHECK-LABEL: @eval_i64_tensor
// CHECK: util.global private @{{.*}} = dense<[2, 3]> : tensor<2xi64>
module @eval_i64_tensor {
  util.global private @hoisted : tensor<2xi64>
  func.func @main() -> tensor<2xi64> {
    %hoisted = util.global.load @hoisted : tensor<2xi64>
    return %hoisted : tensor<2xi64>
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi64>
    util.global.store %cst, @hoisted : tensor<2xi64>
    util.return
  }
}

// -----
// Splat of an 8byte value ensures that large fills are possible.
// CHECK-LABEL: @eval_i64_tensor_splat
// CHECK: util.global private @{{.*}} = dense<2> : tensor<2xi64>
module @eval_i64_tensor_splat {
  util.global private @hoisted : tensor<2xi64>
  func.func @main() -> tensor<2xi64> {
    %hoisted = util.global.load @hoisted : tensor<2xi64>
    return %hoisted : tensor<2xi64>
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant dense<2> : tensor<2xi64>
    util.global.store %cst, @hoisted : tensor<2xi64>
    util.return
  }
}

// -----
// CHECK-LABEL: @serializable_attrs
// CHECK: util.global private @{{.*}} = dense<2> : tensor<5x6xi8>
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @serializable_attrs {
  util.global private @hoisted : tensor<5x6xi8>
  func.func @main() -> tensor<5x6xi8> {
    %hoisted = util.global.load @hoisted : tensor<5x6xi8>
    return %hoisted : tensor<5x6xi8>
  }
  util.global private @constant = #util.byte_pattern<1> : tensor<5x6xi8>
  // CHECK-NOT: util.initializer
  util.initializer attributes {iree.compiler.consteval} {
    %cst = util.global.load @constant : tensor<5x6xi8>
    %0 = tensor.empty() : tensor<5x6xi8>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst, %cst : tensor<5x6xi8>, tensor<5x6xi8>) outs(%0 : tensor<5x6xi8>) {
    ^bb0(%arg0: i8, %arg1: i8, %arg2: i8):  // no predecessors
      %2 = arith.addi %arg0, %arg1 : i8
      linalg.yield %2 : i8
    } -> tensor<5x6xi8>
    util.global.store %1, @hoisted : tensor<5x6xi8>
    util.return
  }
}
