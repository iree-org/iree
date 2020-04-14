// RUN: iree-opt -split-input-file  -xla-legalize-to-std -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  func @const_f32(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>)
  attributes {iree.dispatch_fn_name = "const_f32"} {
    // CHECK: [[CONST:%.*]] = spv.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf32> : !spv.array<6 x f32, stride=4>
    // CHECK: [[VAR:%.*]] = spv.Variable init([[CONST]])
    // CHECK: [[NUMPTR:%.*]] = spv.AccessChain [[VAR]]
    // CHECK: spv.Load "Function" [[NUMPTR]]
    %0 = iree.load_input(%arg0 : memref<2x3xf32>) : tensor<2x3xf32>
    %1 = "xla_hlo.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> (tensor<2x3xf32>)
    %2 = "xla_hlo.add"(%0, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    iree.store_output(%2 : tensor<2x3xf32>, %arg1 : memref<2x3xf32>)
    return
  }
}

// -----

module {
  func @splat_const_f32(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>)
  attributes {iree.dispatch_fn_name = "splat_const_f32"} {
    // CHECK: spv.constant 1.000000e+00 : f32
    %0 = iree.load_input(%arg0 : memref<2x3xf32>) : tensor<2x3xf32>
    %1 = "xla_hlo.constant"() {value = dense<1.0> : tensor<2x3xf32>} : () -> (tensor<2x3xf32>)
    %2 = "xla_hlo.add"(%0, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    iree.store_output(%2 : tensor<2x3xf32>, %arg1 : memref<2x3xf32>)
    return
  }
}

// -----

module {
  func @const_i32(%arg0: memref<2x3xi32>, %arg1: memref<2x3xi32>)
  attributes {iree.dispatch_fn_name = "const_i32"} {
    // CHECK: [[CONST:%.*]] = spv.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
    // CHECK: [[VAR:%.*]] = spv.Variable init([[CONST]])
    // CHECK: [[NUMPTR:%.*]] = spv.AccessChain [[VAR]]
    // CHECK: spv.Load "Function" [[NUMPTR]] : i32
    %0 = iree.load_input(%arg0 : memref<2x3xi32>) : tensor<2x3xi32>
    %1 = "xla_hlo.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>} : () -> (tensor<2x3xi32>)
    %2 = "xla_hlo.add"(%0, %1) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    iree.store_output(%2 : tensor<2x3xi32>, %arg0 : memref<2x3xi32>)
    return
  }
}
