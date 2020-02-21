// RUN: iree-opt -split-input-file -iree-linalg-tensor-to-buffer %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {
  // CHECK: func @element_wise
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<2x2xf32>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: memref<2x2xf32>
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: memref<2x2xf32>
  func @element_wise(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>)
  attributes {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 :i32} {
    %0 = iree.load_input(%arg0 : memref<2x2xf32>) : tensor<2x2xf32>
    %1 = iree.load_input(%arg1 : memref<2x2xf32>) : tensor<2x2xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: [[ARG0]], [[ARG1]], [[ARG2]]
    %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %1 {
    // CHECK: ^{{[a-zA-z0-9_]*}}
    // CHECK-SAME: [[ARG3:%[a-zA-z0-9_]*]]: f32
    // CHECK-SAME: [[ARG4:%[a-zA-z0-9_]*]]: f32
    // CHECK-SAME: [[ARG5:%[a-zA-z0-9_]*]]: f32
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      // CHECK: addf [[ARG3]], [[ARG4]]
      %3 = addf %arg3, %arg4 : f32
      linalg.yield %3 : f32
    }: tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    iree.store_output(%2 : tensor<2x2xf32>, %arg2 : memref<2x2xf32>)
    // CHECK: return
    iree.return
  }
}
