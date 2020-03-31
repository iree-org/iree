// RUN: iree-opt -iree-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

// CHECK: func @dot
func @dot(%arg0: memref<2x3xf32>, %arg1: memref<3x2xf32>,
          %arg2: memref<2x2xf32>) attributes {iree.dispatch_fn_name = ""} {
  %0 = iree.load_input(%arg0 : memref<2x3xf32>) : tensor<2x3xf32>
  %1 = iree.load_input(%arg1 : memref<3x2xf32>) : tensor<3x2xf32>
  // CHECK: linalg.matmul(%arg0, %arg1, %arg2) : memref<2x3xf32>, memref<3x2xf32>, memref<2x2xf32>
  %result = "xla_hlo.dot"(%0, %1) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  iree.store_output(%result: tensor<2x2xf32>, %arg2 : memref<2x2xf32>)
  return
}
