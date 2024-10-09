// RUN: iree-opt --split-input-file %s | FileCheck %s

func.func @sort_tensor(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  %0 = iree_linalg_ext.sort
    dimension(0)
    outs(%arg0 : tensor<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %1 = arith.cmpi sgt, %arg1, %arg2 : i32
    iree_linalg_ext.yield %1 : i1
  } -> tensor<128xi32>
  return %0 : tensor<128xi32>
}
// CHECK-LABEL: func.func @sort_tensor(
// CHECK:         iree_linalg_ext.sort
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      outs({{.*}})
// CHECK:           iree_linalg_ext.yield

// -----

func.func @sort_memref(%arg0: memref<128xi32>) {
  iree_linalg_ext.sort dimension(0)
    outs(%arg0 : memref<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %0 = arith.cmpi sgt, %arg1, %arg2 : i32
    iree_linalg_ext.yield %0 : i1
  }
  return
}
// CHECK-LABEL: func.func @sort_memref(
// CHECK:         iree_linalg_ext.sort
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      outs({{.*}})
// CHECK:           iree_linalg_ext.yield

// -----

func.func @sort_multi_result_tensor(
    %arg0: tensor<?x?xi32>, %arg1: tensor<?x?xf32>)
    -> (tensor<?x?xi32>, tensor<?x?xf32>) {
  %0:2 = iree_linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %1 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %1 : i1
      } -> tensor<?x?xi32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?x?xi32>, tensor<?x?xf32>
}
// CHECK-LABEL: func.func @sort_multi_result_tensor(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]]:2 = iree_linalg_ext.sort dimension(0)
//  CHECK-SAME:      outs(%[[ARG0]], %[[ARG1]]
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @sort_multi_result_memref(
    %arg0: memref<?x?xi32>, %arg1: memref<?x?xf32>) {
  iree_linalg_ext.sort dimension(0)
     outs(%arg0, %arg1 : memref<?x?xi32>, memref<?x?xf32>) {
     ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
       %1 = arith.cmpf ogt, %arg4, %arg5 : f32
       iree_linalg_ext.yield %1 : i1
     }
  return
}
// CHECK-LABEL: func.func @sort_multi_result_memref(
//  CHECK-SAME:   %[[ARG0:.+]]: memref<?x?xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: memref<?x?xf32>
//       CHECK:   iree_linalg_ext.sort dimension(0)
//  CHECK-SAME:      outs(%[[ARG0]], %[[ARG1]]

// -----

func.func @scatter_tensor_dynamic(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original: tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @scatter_tensor_dynamic(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : f32
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_repeated_tensor_dynamic(
    %original: tensor<?x?xf32>, %indices: tensor<?x1xi32>,
    %update: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(false)
    ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%original: tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @scatter_repeated_tensor_dynamic(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?x1xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(false)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : f32
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_tensor_static(
    %original: tensor<128x3xf32>, %indices: tensor<48x1xi32>,
    %update: tensor<48x3xf32>) -> tensor<128x3xf32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%update, %indices : tensor<48x3xf32>, tensor<48x1xi32>)
    outs(%original: tensor<128x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<128x3xf32>
  return %0 : tensor<128x3xf32>
}
// CHECK-LABEL: func.func @scatter_tensor_static(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<128x3xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<48x1xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: tensor<48x3xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//       CHECK:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : f32
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_tensor_multi_index_depth(
    %original: tensor<1x128x3xf32>, %indices: tensor<48x2xi32>,
    %update: tensor<48x3xf32>) -> tensor<1x128x3xf32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0, 1]
    unique_indices(true)
    ins(%update, %indices : tensor<48x3xf32>, tensor<48x2xi32>)
    outs(%original: tensor<1x128x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<1x128x3xf32>
  return %0 : tensor<1x128x3xf32>
}
// CHECK-LABEL: func.func @scatter_tensor_multi_index_depth(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<1x128x3xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<48x2xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: tensor<48x3xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0, 1]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : f32
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_memref_dynamic(
    %original: memref<?x?xf32>, %indices: memref<?x1xi32>,
    %update: memref<?x?xf32>) {
  iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%update, %indices : memref<?x?xf32>, memref<?x1xi32>)
    outs(%original: memref<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    }
  return
}
// CHECK-LABEL: func.func @scatter_memref_dynamic(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: memref<?x1xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//       CHECK:   iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : f32
//       CHECK:   return

// -----

func.func @scatter_memref_static(
    %original: memref<128x3xf32>, %indices: memref<48x1xi32>,
    %update: memref<48x3xf32>) {
  iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%update, %indices : memref<48x3xf32>, memref<48x1xi32>)
    outs(%original: memref<128x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    }
  return
}
// CHECK-LABEL: func.func @scatter_memref_static(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: memref<128x3xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: memref<48x1xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: memref<48x3xf32>
//       CHECK:   iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : f32
//       CHECK:   return

// -----

func.func @scatter_memref_multi_index_depth(
    %original: memref<1x128x3xf32>, %indices: memref<48x2xi32>,
    %update: memref<48x3xf32>) {
  iree_linalg_ext.scatter
    dimension_map = [0, 1]
    unique_indices(true)
    ins(%update, %indices : memref<48x3xf32>, memref<48x2xi32>)
    outs(%original: memref<1x128x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    }
  return
}
// CHECK-LABEL: func.func @scatter_memref_multi_index_depth(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: memref<1x128x3xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: memref<48x2xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: memref<48x3xf32>
//       CHECK:   iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0, 1]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : f32
//       CHECK:   return

// -----

func.func @scatter_update_scalar_1D(
    %original: tensor<8xi32>, %indices: tensor<3x1xi32>,
    %updates: tensor<3xi32>) -> tensor<8xi32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%updates, %indices : tensor<3xi32>, tensor<3x1xi32>)
    outs(%original : tensor<8xi32>)  {
    ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
      iree_linalg_ext.yield %arg0 : i32
    } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: func.func @scatter_update_scalar_1D(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : i32
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_update_i64_scalar_1D(
    %original: tensor<8xi32>, %indices: tensor<3x1xi64>,
    %updates: tensor<3xi32>) -> tensor<8xi32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%updates, %indices : tensor<3xi32>, tensor<3x1xi64>)
    outs(%original : tensor<8xi32>)  {
    ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
      iree_linalg_ext.yield %arg0 : i32
    } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL: func.func @scatter_update_i64_scalar_1D(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : i32
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_update_scalar_2D(
    %original: tensor<4x3xi32>, %indices: tensor<3x2xi32>,
    %updates: tensor<3xi32>) -> tensor<4x3xi32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0, 1]
    unique_indices(true)
    ins(%updates, %indices : tensor<3xi32>, tensor<3x2xi32>)
    outs(%original : tensor<4x3xi32>)  {
    ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
      iree_linalg_ext.yield %arg0 : i32
    } -> tensor<4x3xi32>
  return %0 : tensor<4x3xi32>
}
// CHECK-LABEL: func.func @scatter_update_scalar_2D(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0, 1]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : i32
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_update_slice_2D(
    %original: tensor<4x3xi32>, %indices: tensor<1x1xi32>,
    %updates: tensor<1x3xi32>) -> tensor<4x3xi32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%updates, %indices : tensor<1x3xi32>, tensor<1x1xi32>)
    outs(%original : tensor<4x3xi32>)  {
    ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
      iree_linalg_ext.yield %arg0 : i32
    } -> tensor<4x3xi32>
  return %0 : tensor<4x3xi32>
}
// CHECK-LABEL: func.func @scatter_update_slice_2D(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : i32
//       CHECK:   return %[[RESULT]]

// -----

func.func @scatter_update_slice_2D(
    %original: tensor<4x?xi32>, %indices: tensor<1x1xi32>,
    %updates: tensor<1x3xi32>) -> tensor<4x?xi32> {
  %0 = iree_linalg_ext.scatter
    dimension_map = [0]
    unique_indices(true)
    ins(%updates, %indices : tensor<1x3xi32>, tensor<1x1xi32>)
    outs(%original : tensor<4x?xi32>)  {
    ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
      iree_linalg_ext.yield %arg0 : i32
    } -> tensor<4x?xi32>
  return %0 : tensor<4x?xi32>
}
// CHECK-LABEL: func.func @scatter_update_slice_2D(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:     dimension_map = [0]
//  CHECK-SAME:     unique_indices(true)
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     iree_linalg_ext.yield %{{.+}} : i32
//       CHECK:   return %[[RESULT]]

// -----

func.func @fft_tensor(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>)
    -> (tensor<1024xf32>, tensor<1024xf32>) {
  %cst1 = arith.constant 1 : index
  %0:2 = iree_linalg_ext.fft
    ins(%cst1: index)
    outs(%arg0, %arg1: tensor<1024xf32>, tensor<1024xf32>)
  : tensor<1024xf32>, tensor<1024xf32>
  return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>
}
// CHECK-LABEL: func.func @fft_tensor(
//  CHECK-SAME:   %[[REAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[IMAG:[a-zA-Z0-9_]+]]
//       CHECK:   %[[CST:.+]] = arith.constant 1 : index
//       CHECK:   %[[RES:.+]]:2 = iree_linalg_ext.fft
//  CHECK-SAME:     ins(%[[CST]] : index)
//  CHECK-SAME:    outs(%[[REAL]], %[[IMAG]] : tensor<1024xf32>, tensor<1024xf32>)
//  CHECK-SAME:   : tensor<1024xf32>, tensor<1024xf32>
//       CHECK:   return %[[RES]]#0, %[[RES]]#1

// -----

func.func @fft_memref(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
  %cst1 = arith.constant 1 : index
  iree_linalg_ext.fft
    ins(%cst1: index)
    outs(%arg0, %arg1: memref<1024xf32>, memref<1024xf32>)
  return
}
// CHECK-LABEL: func.func @fft_memref(
//  CHECK-SAME:   %[[REAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[IMAG:[a-zA-Z0-9_]+]]
//       CHECK:   %[[CST:.+]] = arith.constant 1 : index
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     ins(%[[CST]] : index)
//  CHECK-SAME:    outs(%[[REAL]], %[[IMAG]] : memref<1024xf32>, memref<1024xf32>)
//       CHECK:   return

// -----

func.func @fft_tensor_coef(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>,
    %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
  %cst1 = arith.constant 1 : index
  %0:2 = iree_linalg_ext.fft
    ins(%cst1, %arg2, %arg3: index, tensor<1xf32>, tensor<1xf32>)
    outs(%arg0, %arg1: tensor<1024xf32>, tensor<1024xf32>)
  : tensor<1024xf32>, tensor<1024xf32>
  return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>
}
// CHECK-LABEL: func.func @fft_tensor_coef(
//  CHECK-SAME:   %[[REAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[IMAG:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
//       CHECK:   %[[CST:.+]] = arith.constant 1 : index
//       CHECK:   %[[RES:.+]]:2 = iree_linalg_ext.fft
//  CHECK-SAME:     ins(%[[CST]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, tensor<1xf32>, tensor<1xf32>)
//  CHECK-SAME:    outs(%[[REAL]], %[[IMAG]] : tensor<1024xf32>, tensor<1024xf32>)
//  CHECK-SAME:   : tensor<1024xf32>, tensor<1024xf32>
//       CHECK:   return %[[RES]]#0, %[[RES]]#1

// -----

func.func @fft_memref_coef(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>,
                 %arg2: memref<1xf32>, %arg3: memref<1xf32>) {
  %cst1 = arith.constant 1 : index
  iree_linalg_ext.fft
    ins(%cst1, %arg2, %arg3: index, memref<1xf32>, memref<1xf32>)
    outs(%arg0, %arg1: memref<1024xf32>, memref<1024xf32>)
  return
}
// CHECK-LABEL: func.func @fft_memref_coef(
//  CHECK-SAME:   %[[REAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[IMAG:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
//       CHECK:   %[[CST:.+]] = arith.constant 1 : index
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     ins(%[[CST]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, memref<1xf32>, memref<1xf32>)
//  CHECK-SAME:    outs(%[[REAL]], %[[IMAG]] : memref<1024xf32>, memref<1024xf32>)
//       CHECK:   return

// -----

// The size of coefficient tensor is 2^(stage-1).
func.func @fft_tensor_coef_stage_5(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>,
    %arg2: tensor<16xf32>, %arg3: tensor<16xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
  %cst1 = arith.constant 5 : index
  %0:2 = iree_linalg_ext.fft
    ins(%cst1, %arg2, %arg3: index, tensor<16xf32>, tensor<16xf32>)
    outs(%arg0, %arg1: tensor<1024xf32>, tensor<1024xf32>)
  : tensor<1024xf32>, tensor<1024xf32>
  return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>
}
// CHECK-LABEL: func.func @fft_tensor_coef_stage_5(
//  CHECK-SAME:   %[[REAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[IMAG:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[COEF_REAL:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[COEF_IMAG:[a-zA-Z0-9_]+]]
//       CHECK:   %[[CST:.+]] = arith.constant 5 : index
//       CHECK:   %[[RES:.+]]:2 = iree_linalg_ext.fft
//  CHECK-SAME:     ins(%[[CST]], %[[COEF_REAL]], %[[COEF_IMAG]] : index, tensor<16xf32>, tensor<16xf32>)
//  CHECK-SAME:    outs(%[[REAL]], %[[IMAG]] : tensor<1024xf32>, tensor<1024xf32>)
//  CHECK-SAME:   : tensor<1024xf32>, tensor<1024xf32>
//       CHECK:   return %[[RES]]#0, %[[RES]]#1

// -----

func.func @topk_tensor(%input_values: tensor<20x10x8x4xf32>, %input_indices: tensor<20x10x8x4xi32>) -> (tensor<20x10x3x4xf32>, tensor<20x10x3x4xi32>) {
  %out_values = tensor.empty() : tensor<20x10x3x4xf32>
  %out_indices = tensor.empty() : tensor<20x10x3x4xi32>
  %0:2 = iree_linalg_ext.topk
        dimension(2)
        ins(%input_values, %input_indices : tensor<20x10x8x4xf32> , tensor<20x10x8x4xi32>)
        outs(%out_values, %out_indices : tensor<20x10x3x4xf32>, tensor<20x10x3x4xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<20x10x3x4xf32>, tensor<20x10x3x4xi32>
  return %0#0, %0#1 : tensor<20x10x3x4xf32>, tensor<20x10x3x4xi32>
}
// CHECK-LABEL: func.func @topk_tensor(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<20x10x8x4xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<20x10x8x4xi32>
//       CHECK:   %[[OUT_VALUES:.+]] = tensor.empty()
//       CHECK:   %[[OUT_INDICES:.+]] = tensor.empty()
//       CHECK:   %[[RESULT:.+]]:2 = iree_linalg_ext.topk
//  CHECK-SAME:      dimension(2)
//  CHECK-SAME:      ins(%[[ARG0]], %[[ARG1]]
//  CHECK-SAME:      outs(%[[OUT_VALUES]], %[[OUT_INDICES]]
//       CHECK:      iree_linalg_ext.yield
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @topk_memref(%input_values: memref<4x10xf32>, %input_indices: memref<4x10xi32>, %out_values: memref<4x3xf32>, %out_indices: memref<4x3xi32>) {
  iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : memref<4x10xf32> , memref<4x10xi32>)
        outs(%out_values, %out_indices : memref<4x3xf32>, memref<4x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        }
  return
}
// CHECK-LABEL: func.func @topk_memref(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: memref<4x10xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: memref<4x10xi32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: memref<4x3xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: memref<4x3xi32>
//       CHECK:   iree_linalg_ext.topk
//  CHECK-SAME:      dimension(1)
//  CHECK-SAME:      ins(%[[ARG0]], %[[ARG1]]
//  CHECK-SAME:      outs(%[[ARG2]], %[[ARG3]]
//       CHECK:      iree_linalg_ext.yield

// -----

func.func @topk_dynamic_tensor(%input_values: tensor<?x?xf32>, %input_indices: tensor<?x?xi32>, %out_values: tensor<?x?xf32>, %out_indices: tensor<?x?xi32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)  {
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<?x?xf32> , tensor<?x?xi32>)
        outs(%out_values, %out_indices : tensor<?x?xf32>, tensor<?x?xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<?x?xf32>, tensor<?x?xi32>
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}
// CHECK-LABEL: func.func @topk_dynamic_tensor(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: tensor<?x?xi32>
//       CHECK:   %[[RESULT:.+]]:2 = iree_linalg_ext.topk
//  CHECK-SAME:      dimension(1)
//  CHECK-SAME:      ins(%[[ARG0]], %[[ARG1]]
//  CHECK-SAME:      outs(%[[ARG2]], %[[ARG3]]
//       CHECK:      iree_linalg_ext.yield
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @topk_tensor_optional(%input_values: tensor<20x10x8x4xf32>) -> (tensor<20x10x3x4xf32>, tensor<20x10x3x4xi32>) {
  %out_values = tensor.empty() : tensor<20x10x3x4xf32>
  %out_indices = tensor.empty() : tensor<20x10x3x4xi32>
  %0:2 = iree_linalg_ext.topk
        dimension(2)
        ins(%input_values : tensor<20x10x8x4xf32>)
        outs(%out_values, %out_indices : tensor<20x10x3x4xf32>, tensor<20x10x3x4xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<20x10x3x4xf32>, tensor<20x10x3x4xi32>
  return %0#0, %0#1 : tensor<20x10x3x4xf32>, tensor<20x10x3x4xi32>
}

// CHECK-LABEL: func.func @topk_tensor_optional(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x10x8x4xf32>
//       CHECK:   %[[OUT_VALUES:.+]] = tensor.empty()
//       CHECK:   %[[OUT_INDICES:.+]] = tensor.empty()
//       CHECK:   %[[RESULT:.+]]:2 = iree_linalg_ext.topk
//  CHECK-SAME:      dimension(2)
//  CHECK-SAME:      ins(%[[ARG0]]
//  CHECK-SAME:      outs(%[[OUT_VALUES]], %[[OUT_INDICES]]
//       CHECK:      iree_linalg_ext.yield
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @pack(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3x1x1xf32>) -> tensor<3x3x1x1xf32> {
  %1 = iree_linalg_ext.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %arg1 : (tensor<3x3xf32> tensor<3x3x1x1xf32>) -> tensor<3x3x1x1xf32>
  return %1 : tensor<3x3x1x1xf32>
}

// CHECK-LABEL: func.func @pack(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x3xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: tensor<3x3x1x1xf32>
// CHECK:         %[[RES:.*]] = iree_linalg_ext.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %[[ARG1]] : (tensor<3x3xf32> tensor<3x3x1x1xf32>) -> tensor<3x3x1x1xf32>
// CHECK:         return %[[RES]] : tensor<3x3x1x1xf32>

// -----

func.func @pack(%arg0: memref<3x3xf32>, %arg1: memref<3x3x1x1xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %arg1 : (memref<3x3xf32> memref<3x3x1x1xf32>)
  return
}

// CHECK-LABEL: func.func @pack(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<3x3xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: memref<3x3x1x1xf32>
// CHECK:         iree_linalg_ext.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %[[ARG1]] : (memref<3x3xf32> memref<3x3x1x1xf32>)

// -----

func.func @extra_pad_and_pack(%input: tensor<13x15xf32>, %output: tensor<3x8x8x2xf32>, %pad: f32) -> tensor<3x8x8x2xf32> {
  // expected-error@+1 {{infered type do not match provided output type. Expected 'tensor<2x8x8x2xf32>' but got: 'tensor<3x8x8x2xf32>}}
  %0 = iree_linalg_ext.pack %input padding_value(%pad: f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<13x15xf32> tensor<3x8x8x2xf32>) -> tensor<3x8x8x2xf32>
  return %0 : tensor<3x8x8x2xf32>
}
// CHECK-LABEL: func @extra_pad_and_pack(
// CHECK-SAME:    %[[INPUT:.+]]: tensor<13x15xf32>
// CHECK-SAME:    %[[OUTPUT:.+]]: tensor<3x8x8x2xf32>
// CHECK-SAME:    %[[PAD:.+]]: f32
// CHECK:         %[[RES:.+]] = iree_linalg_ext.pack %[[INPUT]]
// CHECK-SAME:      padding_value(%[[PAD]] : f32)
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [8, 2]
// CHECK-SAME:      into %[[OUTPUT]]
// CHECK:         return %[[RES]]

// -----

func.func @pad_and_pack_static(%input: tensor<13x15xf32>, %output: tensor<2x8x8x2xf32>, %pad: f32) -> tensor<2x8x8x2xf32> {
  %0 = iree_linalg_ext.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<13x15xf32> tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}
// CHECK-LABEL: func.func @pad_and_pack_static(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]: tensor<13x15xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9_]+]]: tensor<2x8x8x2xf32>
// CHECK-SAME:    %[[PAD:[a-zA-Z0-9_]+]]: f32
// CHECK:         %[[RES:.+]] = iree_linalg_ext.pack %[[INPUT]]
// CHECK-SAME:      padding_value(%[[PAD]] : f32)
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [8, 2]
// CHECK-SAME:      into %[[OUTPUT]]
// CHECK:         return %[[RES]]

// -----

func.func @pad_and_pack_partially_dynamic(%input: tensor<?x?xf32>, %output: tensor<?x?x8x2xf32>, %pad: f32) -> tensor<?x?x8x2xf32> {
  %0 = iree_linalg_ext.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<?x?xf32> tensor<?x?x8x2xf32>) -> tensor<?x?x8x2xf32>
  return %0 : tensor<?x?x8x2xf32>
}
// CHECK-LABEL: func.func @pad_and_pack_partially_dynamic(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9_]+]]: tensor<?x?x8x2xf32>
// CHECK-SAME:    %[[PAD:[a-zA-Z0-9_]+]]: f32
// CHECK:         %[[RES:.+]] = iree_linalg_ext.pack %[[INPUT]]
// CHECK-SAME:      padding_value(%[[PAD]] : f32)
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [8, 2]
// CHECK-SAME:      into %[[OUTPUT]]
// CHECK:         return %[[RES]]

// -----

func.func @pad_and_pack_fully_dynamic(%input: tensor<?x?xf32>, %output: tensor<?x?x?x?xf32>, %pad: f32, %tile_n : index, %tile_m : index) -> tensor<?x?x?x?xf32> {
  %0 = iree_linalg_ext.pack %input padding_value(%pad : f32)
    inner_dims_pos = [0, 1] inner_tiles = [%tile_n, %tile_m] into %output : (tensor<?x?xf32> tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func.func @pad_and_pack_fully_dynamic(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:    %[[PAD:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME:    %[[TILE_N:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[TILE_M:[a-zA-Z0-9_]+]]: index
// CHECK:         %[[RES:.+]] = iree_linalg_ext.pack %[[INPUT]]
// CHECK-SAME:      padding_value(%[[PAD]] : f32)
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [%[[TILE_N]], %[[TILE_M]]]
// CHECK-SAME:      into %[[OUTPUT]]
// CHECK:         return %[[RES]]

// -----

func.func @unpack(%arg0: memref<3x3xf32>, %arg1: memref<3x3x1x1xf32>) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %arg0 : (memref<3x3x1x1xf32> memref<3x3xf32>)
  return
}

// CHECK-LABEL: func.func @unpack(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<3x3xf32>,
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: memref<3x3x1x1xf32>) {
// CHECK:         iree_linalg_ext.unpack %[[ARG1]] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %[[ARG0]] : (memref<3x3x1x1xf32> memref<3x3xf32>)

// -----

func.func @unpack_static(%input: tensor<8x8x32x16xf32>, %output: tensor<256x128xf32>) -> tensor<256x128xf32> {
  %0 = iree_linalg_ext.unpack %input inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %output : (tensor<8x8x32x16xf32> tensor<256x128xf32>) -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// CHECK-LABEL: func.func @unpack_static(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9_]+]]
// CHECK:         %[[UNPACK:.+]] = iree_linalg_ext.unpack
// CHECK-SAME:      %[[INPUT]]
// CHECK-SAME       dim_pos = [0, 1]
// CHECK-SAME       inner_pos = [32, 16]
// CHECK-SAME:      into %[[OUTPUT]]
// CHECK:         return %[[UNPACK]]

// -----

func.func @unpack_undo_padding(%input: tensor<2x8x8x2xf32>, %output: tensor<13x15xf32>) -> tensor<13x15xf32> {
  %0 = iree_linalg_ext.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<2x8x8x2xf32> tensor<13x15xf32>) -> tensor<13x15xf32>
  return %0 : tensor<13x15xf32>
}
// CHECK-LABEL: func.func @unpack_undo_padding(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9_]+]]
// CHECK:         %[[UNPACK:.+]] = iree_linalg_ext.unpack
// CHECK-SAME:      %[[INPUT]]
// CHECK-SAME       dim_pos = [0, 1]
// CHECK-SAME       inner_pos = [32, 16]
// CHECK-SAME:      into %[[OUTPUT]]
// CHECK:         return %[[UNPACK]]

// -----

func.func @unpack(%arg0: memref<3x3xf32>, %arg1: memref<3x3x1x1xf32>) {
  iree_linalg_ext.unpack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %arg0 : (memref<3x3x1x1xf32> memref<3x3xf32>)
  return
}

// CHECK-LABEL: func.func @unpack(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<3x3xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: memref<3x3x1x1xf32>
// CHECK:         iree_linalg_ext.unpack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %[[ARG0]] : (memref<3x3x1x1xf32> memref<3x3xf32>)

// -----

func.func @pack(%arg0: memref<128x256xf32>, %arg1: memref<32x4x32x8xf32>) {
  iree_linalg_ext.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<32x4x32x8xf32>)
  return
}

// CHECK-LABEL: func.func @pack(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: memref<32x4x32x8xf32>
// CHECK:         iree_linalg_ext.pack %[[ARG0]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %[[ARG1]] : (memref<128x256xf32> memref<32x4x32x8xf32>)

// -----

func.func @pack(%arg0: memref<128x256xf32>, %arg1: memref<4x32x32x8xf32>) {
  iree_linalg_ext.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<4x32x32x8xf32>)
  return
}

// CHECK-LABEL: func.func @pack(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: memref<4x32x32x8xf32>
// CHECK:         iree_linalg_ext.pack %[[ARG0]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %[[ARG1]] : (memref<128x256xf32> memref<4x32x32x8xf32>)

// -----

func.func @unpack(%arg0: memref<128x256xf32>, %arg1: memref<4x32x32x8xf32>) {
  iree_linalg_ext.unpack %arg1 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg0 : (memref<4x32x32x8xf32> memref<128x256xf32>)
  return
}

// CHECK-LABEL: func.func @unpack(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: memref<4x32x32x8xf32>
// CHECK:         iree_linalg_ext.unpack %[[ARG1]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %[[ARG0]] : (memref<4x32x32x8xf32> memref<128x256xf32>)

// -----

func.func @unpack(%arg0: memref<128x256xf32>, %arg1: memref<32x4x32x8xf32>) {
  iree_linalg_ext.unpack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg0 : (memref<32x4x32x8xf32> memref<128x256xf32>)
  return
}

// CHECK-LABEL: func.func @unpack(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: memref<32x4x32x8xf32>
// CHECK:         iree_linalg_ext.unpack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %[[ARG0]] : (memref<32x4x32x8xf32> memref<128x256xf32>)

// -----

func.func @im2col(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
           m_offset = [0] * [1] k_offset = [0] * [1]
           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<2x34x34x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}
// CHECK-LABEL: func.func @im2col(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<2x1024x5760xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:      m_offset = [0] * [1] k_offset = [0] * [1]
// CHECK-SAME:      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:      ins(%[[ARG0]] : tensor<2x34x34x640xf32>)
// CHECK-SAME:      outs(%[[D0]] : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
// CHECK:         return %[[D1]] : tensor<2x1024x5760xf32>

// -----

func.func @im2col_dynamic(%arg0: tensor<?x?x?x?xf32>, %s0: index, %s1: index, %s2: index,
                          %mOffset: index, %kOffset: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%s0, %s1, %s2) : tensor<?x?x?xf32>
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
           m_offset = [%mOffset] * [1] k_offset = [%kOffset] * [1]
           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<?x?x?x?xf32>)
           outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func.func @im2col_dynamic(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:    %{{.+}}: index, %{{.+}}: index, %{{.+}}: index, %[[MOFFSET:.+]]: index, %[[KOFFSET:.+]]: index
// CHECK:         %[[D0:.+]] = tensor.empty({{.+}}) : tensor<?x?x?xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:      m_offset = [%[[MOFFSET]]] * [1] k_offset = [%[[KOFFSET]]] * [1]
// CHECK-SAME:      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:      ins(%[[ARG0]] : tensor<?x?x?x?xf32>)
// CHECK-SAME:      outs(%[[D0]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         return %[[D1]] : tensor<?x?x?xf32>

// -----

func.func @im2col_strided(%arg0: tensor<2x65x96x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  %1 = iree_linalg_ext.im2col strides = [2, 3] dilations = [1, 1] kernel_size = [3, 3]
           m_offset = [0] * [1] k_offset = [0] * [1]
           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<2x65x96x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}
// CHECK-LABEL: func.func @im2col_strided(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x65x96x640xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<2x1024x5760xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.im2col strides = [2, 3] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:      m_offset = [0] * [1] k_offset = [0] * [1]
// CHECK-SAME:      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:      ins(%[[ARG0]] : tensor<2x65x96x640xf32>)
// CHECK-SAME:      outs(%[[D0]] : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
// CHECK:         return %[[D1]] : tensor<2x1024x5760xf32>

// -----

func.func @im2col_dilated(%arg0: tensor<2x44x46x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [6, 7] kernel_size = [3, 3]
           m_offset = [0] * [1] k_offset = [0] * [1]
           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<2x44x46x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}
// CHECK-LABEL: func.func @im2col_dilated(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x44x46x640xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<2x1024x5760xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.im2col strides = [1, 1] dilations = [6, 7] kernel_size = [3, 3]
// CHECK-SAME:      m_offset = [0] * [1] k_offset = [0] * [1]
// CHECK-SAME:      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:      ins(%[[ARG0]] : tensor<2x44x46x640xf32>)
// CHECK-SAME:      outs(%[[D0]] : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
// CHECK:         return %[[D1]] : tensor<2x1024x5760xf32>

// -----

func.func @im2col_strided_dilated_mixed_kernel(%arg0: tensor<2x172x101x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  %1 = iree_linalg_ext.im2col strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2]
           m_offset = [0] * [1] k_offset = [0] * [1]
           batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<2x172x101x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}
// CHECK-LABEL: func.func @im2col_strided_dilated_mixed_kernel(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x172x101x640xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<2x1024x5760xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.im2col strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2]
// CHECK-SAME:      m_offset = [0] * [1] k_offset = [0] * [1]
// CHECK-SAME:      batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:      ins(%[[ARG0]] : tensor<2x172x101x640xf32>)
// CHECK-SAME:      outs(%[[D0]] : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
// CHECK:         return %[[D1]] : tensor<2x1024x5760xf32>

// -----

func.func @im2col_transposed_m_pos(%arg0: tensor<640x2x101x172xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  %1 = iree_linalg_ext.im2col strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2]
           m_offset = [0] * [1] k_offset = [0] * [1]
           batch_pos = [1] m_pos = [3, 2] k_pos = [0]
           ins(%arg0 : tensor<640x2x101x172xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}
// CHECK-LABEL: func.func @im2col_transposed_m_pos(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<640x2x101x172xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<2x1024x5760xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.im2col strides = [5, 3] dilations = [4, 7] kernel_size = [5, 2]
// CHECK-SAME:      m_offset = [0] * [1] k_offset = [0] * [1]
// CHECK-SAME:      batch_pos = [1] m_pos = [3, 2] k_pos = [0]
// CHECK-SAME:      ins(%[[ARG0]] : tensor<640x2x101x172xf32>)
// CHECK-SAME:      outs(%[[D0]] : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
// CHECK:         return %[[D1]] : tensor<2x1024x5760xf32>

// -----

func.func @im2col_expanded(%arg0: tensor<2x3x34x34x640xf32>) -> tensor<2x3x128x8x90x64xf32> {
  %0 = tensor.empty() : tensor<2x3x128x8x90x64xf32>
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
           m_offset = [0, 0] * [8, 1] k_offset = [0, 0] * [64, 1]
           batch_pos = [0, 1] m_pos = [2, 3] k_pos = [4]
           ins(%arg0 : tensor<2x3x34x34x640xf32>)
           outs(%0 : tensor<2x3x128x8x90x64xf32>) -> tensor<2x3x128x8x90x64xf32>
  return %1 : tensor<2x3x128x8x90x64xf32>
}
// CHECK-LABEL: func.func @im2col_expanded(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x3x34x34x640xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<2x3x128x8x90x64xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:      m_offset = [0, 0] * [8, 1] k_offset = [0, 0] * [64, 1]
// CHECK-SAME:      batch_pos = [0, 1] m_pos = [2, 3] k_pos = [4]
// CHECK-SAME:      ins(%[[ARG0]] : tensor<2x3x34x34x640xf32>)
// CHECK-SAME:      outs(%[[D0]] : tensor<2x3x128x8x90x64xf32>) -> tensor<2x3x128x8x90x64xf32>
// CHECK:         return %[[D1]] : tensor<2x3x128x8x90x64xf32>

// -----

func.func @winograd_filter_transform(%arg0: tensor<3x3x64x128xf32>) -> tensor<8x8x64x128xf32> {
  %0 = tensor.empty() : tensor<8x8x64x128xf32>
  %1 = iree_linalg_ext.winograd.filter_transform
    output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
    ins(%arg0 : tensor<3x3x64x128xf32>) outs(%0 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %1 : tensor<8x8x64x128xf32>
}
// CHECK-LABEL: func.func @winograd_filter_transform(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<3x3x64x128xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<8x8x64x128xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      kernel_dimensions([0, 1]) ins(%[[ARG0]] : tensor<3x3x64x128xf32>) outs(%[[D0]] :
// CHECK-SAME:      tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
// CHECK:         return %[[D1]] : tensor<8x8x64x128xf32>

// -----

func.func @winograd_filter_transform_dynamic(%arg0: tensor<3x3x?x?xf32>, %arg1: tensor<8x8x?x?xf32>) -> tensor<8x8x?x?xf32> {
  %1 = iree_linalg_ext.winograd.filter_transform
    output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
    ins(%arg0 : tensor<3x3x?x?xf32>) outs(%arg1 : tensor<8x8x?x?xf32>) -> tensor<8x8x?x?xf32>
  return %1 : tensor<8x8x?x?xf32>
}
// CHECK-LABEL: func.func @winograd_filter_transform_dynamic(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<3x3x?x?xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<8x8x?x?xf32>
// CHECK:         %[[D0:.+]] = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      kernel_dimensions([0, 1]) ins(%[[ARG0]] : tensor<3x3x?x?xf32>) outs(%[[ARG1]] :
// CHECK-SAME:      tensor<8x8x?x?xf32>) -> tensor<8x8x?x?xf32>
// CHECK:         return %[[D0]] : tensor<8x8x?x?xf32>

// -----

func.func @winograd_filter_transform_fchw(%arg0: tensor<128x64x3x3xf32>) -> tensor<8x8x64x128xf32> {
  %0 = tensor.empty() : tensor<8x8x64x128xf32>
  %1 = iree_linalg_ext.winograd.filter_transform
    output_tile_size(6) kernel_size(3) kernel_dimensions([2, 3])
    ins(%arg0 : tensor<128x64x3x3xf32>) outs(%0 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %1 : tensor<8x8x64x128xf32>
}
// CHECK-LABEL: func.func @winograd_filter_transform_fchw(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<128x64x3x3xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<8x8x64x128xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      kernel_dimensions([2, 3]) ins(%[[ARG0]] : tensor<128x64x3x3xf32>) outs(%[[D0]] :
// CHECK-SAME:      tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
// CHECK:         return %[[D1]] : tensor<8x8x64x128xf32>

// -----

func.func @winograd_input_transform(%arg0: tensor<1x10x10x1280xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<1x10x10x1280xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}
// CHECK-LABEL: func.func @winograd_input_transform(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<1x10x10x1280xf32>) outs(%[[D0]] :
// CHECK-SAME:      tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
// CHECK:         return %[[D1]] : tensor<8x8x1x2x2x1280xf32>

// -----

func.func @winograd_input_transform_dynamic(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<8x8x?x?x?x?xf32>) -> tensor<8x8x?x?x?x?xf32> {
  %1 = iree_linalg_ext.winograd.input_transform
    output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<?x?x?x?xf32>) outs(%arg1 : tensor<8x8x?x?x?x?xf32>) -> tensor<8x8x?x?x?x?xf32>
  return %1 : tensor<8x8x?x?x?x?xf32>
}
// CHECK-LABEL: func.func @winograd_input_transform_dynamic(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<8x8x?x?x?x?xf32>
// CHECK:         %[[D0:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<?x?x?x?xf32>) outs(%[[ARG1]] :
// CHECK-SAME:      tensor<8x8x?x?x?x?xf32>) -> tensor<8x8x?x?x?x?xf32>
// CHECK:         return %[[D0]] : tensor<8x8x?x?x?x?xf32>

// -----

func.func @winograd_input_transform_nchw(%arg0: tensor<1x1280x10x10xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
    ins(%arg0 : tensor<1x1280x10x10xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}
// CHECK-LABEL: func.func @winograd_input_transform_nchw(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1280x10x10xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      image_dimensions([2, 3]) ins(%[[ARG0]] : tensor<1x1280x10x10xf32>) outs(%[[D0]] :
// CHECK-SAME:      tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
// CHECK:         return %[[D1]] : tensor<8x8x1x2x2x1280xf32>

// -----

func.func @winograd_output_transform(%arg0: tensor<8x8x1x2x2x1280xf32>) -> tensor<1x12x12x1280xf32> {
  %0 = tensor.empty() : tensor<1x12x12x1280xf32>
  %1 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<8x8x1x2x2x1280xf32>) outs(%0 : tensor<1x12x12x1280xf32>) -> tensor<1x12x12x1280xf32>
  return %1 : tensor<1x12x12x1280xf32>
}
// CHECK-LABEL: func.func @winograd_output_transform(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x1280xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<1x12x12x1280xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<8x8x1x2x2x1280xf32>) outs(%[[D0]] :
// CHECK-SAME:      tensor<1x12x12x1280xf32>) -> tensor<1x12x12x1280xf32>
// CHECK:         return %[[D1]] : tensor<1x12x12x1280xf32>

// -----

func.func @winograd_output_transform_dynamic(%arg0: tensor<8x8x?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %1 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<8x8x?x?x?x?xf32>) outs(%arg1 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func.func @winograd_output_transform_dynamic(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x?x?x?x?xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
// CHECK:         %[[D0:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<8x8x?x?x?x?xf32>) outs(%[[ARG1]] :
// CHECK-SAME:      tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         return %[[D0]] : tensor<?x?x?x?xf32>

// -----

func.func @winograd_output_transform_nchw(%arg0: tensor<8x8x1x2x2x1280xf32>) -> tensor<1x1280x12x12xf32> {
  %0 = tensor.empty() : tensor<1x1280x12x12xf32>
  %1 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
    ins(%arg0 : tensor<8x8x1x2x2x1280xf32>) outs(%0 : tensor<1x1280x12x12xf32>) -> tensor<1x1280x12x12xf32>
  return %1 : tensor<1x1280x12x12xf32>
}
// CHECK-LABEL: func.func @winograd_output_transform_nchw(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x1280xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<1x1280x12x12xf32>
// CHECK:         %[[D1:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:      image_dimensions([2, 3]) ins(%[[ARG0]] : tensor<8x8x1x2x2x1280xf32>) outs(%[[D0]] :
// CHECK-SAME:      tensor<1x1280x12x12xf32>) -> tensor<1x1280x12x12xf32>
// CHECK:         return %[[D1]] : tensor<1x1280x12x12xf32>

// -----

func.func @attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32)
                     outs(%0 : tensor<192x1024x64xf32>) {
                        ^bb0(%arg0: f32):
                        iree_linalg_ext.yield %arg0 : f32
                     } -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}

// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG: #[[$MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK-LABEL: func.func @attention(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:         %[[SCALE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:         %[[D1:.+]] = iree_linalg_ext.attention
// CHECK-SAME:                 {indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_S]], #[[$MAP_O]]]}
// CHECK-SAME:                 ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] :
// CHECK-SAME:      tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32) outs(%[[D0]] :
// CHECK-SAME:      tensor<192x1024x64xf32>) {
// CHECK:          ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK:          iree_linalg_ext.yield %[[SCORE]] : f32
// CHECK:        } -> tensor<192x1024x64xf32>
// CHECK:         return %[[D1]] : tensor<192x1024x64xf32>

// -----

func.func @cross_attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x2048x64xf32>, %value: tensor<192x2048x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x2048x64xf32>, tensor<192x2048x64xf32>, f32)
                     outs(%0 : tensor<192x1024x64xf32>) {
                        ^bb0(%arg0: f32):
                        iree_linalg_ext.yield %arg0 : f32
                     }
                      -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG: #[[$MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK-LABEL: func.func @cross_attention(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<192x2048x64xf32>
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x2048x64xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:         %[[SCALE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:         %[[D1:.+]] = iree_linalg_ext.attention
// CHECK-SAME:                 {indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_S]], #[[$MAP_O]]]}
// CHECK-SAME:                 ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] :
// CHECK-SAME:      tensor<192x1024x64xf32>, tensor<192x2048x64xf32>, tensor<192x2048x64xf32>, f32) outs(%[[D0]] :
// CHECK-SAME:      tensor<192x1024x64xf32>) {
// CHECK:          ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK:          iree_linalg_ext.yield %[[SCORE]] : f32
// CHECK:        } -> tensor<192x1024x64xf32>
// CHECK:         return %[[D1]] : tensor<192x1024x64xf32>

// -----

// transpose_V is detected through indexingMap.

func.func @cross_attention_transposev(%query: tensor<192x1024x64xf32>, %key: tensor<192x2048x64xf32>, %value: tensor<192x64x2048xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x2048x64xf32>, tensor<192x64x2048xf32>, f32) outs(%0 : tensor<192x1024x64xf32>) {
                        ^bb0(%arg0: f32):
                        iree_linalg_ext.yield %arg0 : f32
                     } -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>
// CHECK-DAG: #[[$MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK-LABEL: func.func @cross_attention_transposev(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<192x2048x64xf32>
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x64x2048xf32>
// CHECK:         %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:         %[[SCALE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:         %[[D1:.+]] = iree_linalg_ext.attention
// CHECK-SAME:                 {indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_S]], #[[$MAP_O]]]}
// CHECK-SAME:                 ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] :
// CHECK-SAME:      tensor<192x1024x64xf32>, tensor<192x2048x64xf32>, tensor<192x64x2048xf32>, f32) outs(%[[D0]] :
// CHECK-SAME:      tensor<192x1024x64xf32>) {
// CHECK:          ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK:          iree_linalg_ext.yield %[[SCORE]] : f32
// CHECK:        } -> tensor<192x1024x64xf32>
// CHECK:         return %[[D1]] : tensor<192x1024x64xf32>

// -----

func.func @cross_attention_transposev_dyn(%query: tensor<?x?x?xf32>, %key: tensor<?x?x?xf32>, %value: tensor<?x?x?xf32>, %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %scale = arith.constant 1.0 : f32
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32) outs(%init : tensor<?x?x?xf32>) {
                        ^bb0(%arg0: f32):
                        iree_linalg_ext.yield %arg0 : f32
                     } -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>
// CHECK-DAG: #[[$MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK-LABEL: func.func @cross_attention_transposev_dyn(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:    %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK:         %[[SCALE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:         %[[D1:.+]] = iree_linalg_ext.attention
// CHECK-SAME:                 {indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_S]], #[[$MAP_O]]]}
// CHECK-SAME:                 ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] :
// CHECK-SAME:      tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32) outs(%[[ARG3]] :
// CHECK-SAME:      tensor<?x?x?xf32>) {
// CHECK:          ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK:          iree_linalg_ext.yield %[[SCORE]] : f32
// CHECK:        } -> tensor<?x?x?xf32>
// CHECK:         return %[[D1]] : tensor<?x?x?xf32>

// -----

func.func @custom_op_default(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @custom_op_default(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.custom_op
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP]]]
//  CHECK-SAME:       iterator_types = [#iree_linalg_ext.iterator_type<parallel>]
//  CHECK-SAME:       ins(%[[ARG0]] : tensor<?xf32>) outs(%[[ARG1]] : tensor<?xf32>)
//  CHECK-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: tensor<?xf32>, %[[B1:[a-zA-Z0-9]+]]: tensor<?xf32>)
//  CHECK-NEXT:       iree_linalg_ext.yield %[[B0]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @custom_op_scalar_arg(%arg0 : tensor<?xf32>, %arg1 : f32, %arg2 : tensor<?xf32>, %arg3 : index) -> tensor<?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, f32) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : f32, %b2 : tensor<?xf32>):
      %1 = tensor.insert %b1 into %b0[%arg3] : tensor<?xf32>
      iree_linalg_ext.yield %1 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
//       CHECK: #[[$MAP:.+]] = affine_map<(d0) -> ()>
// CHECK-LABEL: func @custom_op_scalar_arg(
//  CHECK-SAME:     %[[SCALAR_ARG:[a-zA-Z0-9]+]]: f32
//       CHECK:   iree_linalg_ext.custom_op
//  CHECK-SAME:       indexing_maps = [#{{.+}}, #[[$MAP]], #{{.+}}]
//  CHECK-SAME:       ins(%{{.+}}, %[[SCALAR_ARG]] : tensor<?xf32>, f32)
//  CHECK-NEXT:     %[[B1:.+]]: f32

// -----

func.func @custom_op_empty_affine_map(%arg0 : tensor<?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>, %arg3 : index) -> tensor<?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<() -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?x?xf32>, %b2 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
//       CHECK: #[[$MAP:.+]] = affine_map<() -> ()>
// CHECK-LABEL: func @custom_op_empty_affine_map(
//       CHECK:   iree_linalg_ext.custom_op
//  CHECK-SAME:       indexing_maps = [#{{.+}}, #[[$MAP]], #{{.+}}]

// -----

func.func @custom_op_static_args(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<10xf32>) outs(%arg1 : tensor<10xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
// CHECK-LABEL: func @custom_op_static_args(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<10xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10xf32>
//       CHECK:   iree_linalg_ext.custom_op
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<10xf32>) outs(%[[ARG1]] : tensor<10xf32>)
//  CHECK-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: tensor<?xf32>, %[[B1:[a-zA-Z0-9]+]]: tensor<?xf32>)

// -----

func.func @custom_op_reduction(%arg0 : tensor<?x?xf32>, %arg1 : f32,
    %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> ()>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, f32) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : f32, %b2 : tensor<?xf32>):
      %1 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> ()>,
                           affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%b0, %b1 : tensor<?x?xf32>, f32)
          outs(%b2 : tensor<?xf32>) {
        ^bb0(%b00 : f32, %b01 : f32, %b02 : f32):
          %2 = arith.addf %b00, %b01 : f32
          linalg.yield %2 : f32
      } -> tensor<?xf32>
      iree_linalg_ext.yield %1 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @custom_op_reduction(
//       CHECK:   iree_linalg_ext.custom_op
//  CHECK-SAME:     iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]
//  CHECK-NEXT:   ^bb0
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//       CHECK:     iree_linalg_ext.yield %[[GENERIC]]

// -----

func.func @custom_op_multiple_results(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>) {
  %0:2 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1, %arg1 : tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>, %b2 : tensor<?xf32>):
      iree_linalg_ext.yield %b0, %b0 : tensor<?xf32>, tensor<?xf32>
  } -> tensor<?xf32>, tensor<?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}
// CHECK-LABEL: func @custom_op_multiple_results(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?xf32>
//       CHECK:   %[[RESULT:.+]]:2 = iree_linalg_ext.custom_op
//  CHECK-SAME:       outs(%[[ARG1]], %[[ARG1]] : tensor<?xf32>, tensor<?xf32>)
//  CHECK-NEXT:     ^bb0(%[[B0:[a-zA-Z0-9]+]]: tensor<?xf32>, %[[B1:[a-zA-Z0-9]+]]: tensor<?xf32>, %[[B2:[a-zA-Z0-9]+]]: tensor<?xf32>)
//  CHECK-NEXT:       iree_linalg_ext.yield %[[B0]], %[[B0]]
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1

// -----

func.func @custom_op_symbolic_dims(%lhs1 : tensor<1000000x?xf32>,
    %rhs1 : tensor<?x?xf32>, %rhs2 : tensor<?x?xf32>,
    %outs1 : tensor<1000000x?xf32>, %outs2 : tensor<1000000x?xf32>)
    -> (tensor<1000000x?xf32>, tensor<1000000x?xf32>) {
  %0:2 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (d0, s0)>,
                         affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                         affine_map<(d0, d1)[s0, s1] -> (s1, d1)>,
                         affine_map<(d0, d1)[s0, s1] -> (d0, s1)>,
                         affine_map<(d0, d1)[s0, s1] -> (d0, d1)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                          #iree_linalg_ext.iterator_type<parallel>]}
        ins(%lhs1, %rhs1, %rhs2
            : tensor<1000000x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%outs1, %outs2 : tensor<1000000x?xf32>, tensor<1000000x?xf32>) {
      ^bb0(%t0 : tensor<?x?xf32>, %t1 : tensor<?x?xf32>, %t2 : tensor<?x?xf32>,
           %t3 : tensor<?x?xf32>, %t4 : tensor<?x?xf32>) :
        %0 = linalg.matmul ins(%t0, %t1 : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%t3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %1 = linalg.matmul ins(%0, %t2 : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%t4 : tensor<?x?xf32>) -> tensor<?x?xf32>
        iree_linalg_ext.yield %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
    } -> tensor<1000000x?xf32>, tensor<1000000x?xf32>
  return %0#0, %0#1 : tensor<1000000x?xf32>, tensor<1000000x?xf32>
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0, s0)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (s0, s1)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1)[s0, s1] -> (s1, d1)>
//   CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0, s1)>
//   CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0, d1)>
// CHECK-LABEL: func @custom_op_symbolic_dims(
//       CHECK:   iree_linalg_ext.custom_op
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]]]

// -----

func.func @custom_op_index(%arg0 : tensor<?x?xindex>) -> tensor<?x?xindex> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>]}
      outs(%arg0: tensor<?x?xindex>) {
    ^bb0(%b0 : tensor<?x?xindex>):
      %1 = iree_linalg_ext.index 0 : index
      %2 = iree_linalg_ext.index 1 : index
      %3 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          outs(%b0 : tensor<?x?xindex>) {
        ^bb1(%bb0 : index):
          %4 = arith.addi %bb0, %1 : index
          %5 = arith.addi %4, %2 : index
          linalg.yield %5 : index
      } -> tensor<?x?xindex>
      iree_linalg_ext.yield %3 : tensor<?x?xindex>
  } -> tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}
// CHECK-LABEL: func @custom_op_index
//       CHECK:   iree_linalg_ext.custom_op
//       CHECK:     iree_linalg_ext.index 0
//       CHECK:     iree_linalg_ext.index 1
//       CHECK:     iree_linalg_ext.yield
