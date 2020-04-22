// RUN: iree-opt -split-input-file -iree-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

// CHECK_LABEL: @pad_cst
func @pad_cst(%arg0 : memref<12x4xf32>, %arg1 : memref<18x12xf32>)
attributes {iree.dispatch_fn_name = ""} {
  // CHECK: linalg.indexed_generic
  %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.pad"(%0, %1) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
  iree.store_output(%2 : tensor<18x12xf32>, %arg1 : memref<18x12xf32>)
  return
}

// -----

// CHECK_LABEL: @pad_memref
func @pad_memref(%arg0 : memref<12x4xf32>,
          %arg1 : memref<f32>,
          %arg2 : memref<18x12xf32>) attributes {iree.dispatch_fn_name = ""} {
  // CHECK: linalg.indexed_generic
  %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
  %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
  %2 = "xla_hlo.pad"(%0, %1) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
  iree.store_output(%2 : tensor<18x12xf32>, %arg2 : memref<18x12xf32>)
  return
}

// -----

// CHECK_LABEL: @pad_no_op
func @pad_no_op(%arg0 : memref<12x4xf32>, %arg1 : memref<12x4xf32>)
attributes {iree.dispatch_fn_name = ""} {
  // CHECK: linalg.indexed_generic
  %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
  %1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.pad"(%0, %1) {
    edge_padding_high = dense<0> : tensor<2xi64>,
    edge_padding_low = dense<0> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<12x4xf32>
  iree.store_output(%2 : tensor<12x4xf32>, %arg1 : memref<12x4xf32>)
  return
}
