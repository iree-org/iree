// RUN: iree-opt -split-input-file -iree-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

module {
  func @reduce_window_min(%arg0: memref<1x18x18x64xf32>, %arg1: memref<f32>, %arg2: memref<1x8x8x64xf32>)
  attributes {iree.dispatch_fn_name = ""} {
    %0 = iree.load_input(%arg0 : memref<1x18x18x64xf32>) : tensor<1x18x18x64xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    // CHECK: linalg.pooling_min
    %2 = "xla_hlo.reduce_window"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.minimum %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
        window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x18x18x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
    iree.store_output(%2 : tensor<1x8x8x64xf32>, %arg2 : memref<1x8x8x64xf32>)
    return
  }
}

// -----

module {
  func @reduce_window_max(%arg0: memref<1x18x18x64xf32>, %arg1: memref<f32>, %arg2: memref<1x8x8x64xf32>)
  attributes {iree.dispatch_fn_name = ""} {
    %0 = iree.load_input(%arg0 : memref<1x18x18x64xf32>) : tensor<1x18x18x64xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    // CHECK: linalg.pooling_max
    %2 = "xla_hlo.reduce_window"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.maximum %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
        window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x18x18x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
    iree.store_output(%2 : tensor<1x8x8x64xf32>, %arg2 : memref<1x8x8x64xf32>)
    return
  }
}

// -----

module {
  func @reduce_window_add(%arg0: memref<1x18x18x64xf32>, %arg1: memref<f32>, %arg2: memref<1x8x8x64xf32>)
  attributes {iree.dispatch_fn_name = ""} {
    %0 = iree.load_input(%arg0 : memref<1x18x18x64xf32>) : tensor<1x18x18x64xf32>
    %1 = iree.load_input(%arg1 : memref<f32>) : tensor<f32>
    // CHECK: linalg.pooling_sum
    %2 = "xla_hlo.reduce_window"(%0, %1) ( {
    ^bb0(%arg3: tensor<f32>, %arg4 : tensor<f32>):
      %3 = xla_hlo.add %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%3) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
        window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x18x18x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
    iree.store_output(%2 : tensor<1x8x8x64xf32>, %arg2 : memref<1x8x8x64xf32>)
    return
  }
}
