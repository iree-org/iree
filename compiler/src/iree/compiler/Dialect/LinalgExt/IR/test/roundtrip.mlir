// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @sort_tensor
// CHECK:         iree_linalg_ext.sort
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      outs({{.*}})
// CHECK:           iree_linalg_ext.yield
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

// -----

// CHECK-LABEL: func.func @sort_memref
// CHECK:         iree_linalg_ext.sort
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      outs({{.*}})
// CHECK:           iree_linalg_ext.yield
func.func @sort_memref(%arg0: memref<128xi32>) {
  iree_linalg_ext.sort dimension(0)
    outs(%arg0 : memref<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %0 = arith.cmpi sgt, %arg1, %arg2 : i32
    iree_linalg_ext.yield %0 : i1
  }
  return
}

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
// CHECK-LABEL: func.func @sort_multi_result_tensor
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
// CHECK-LABEL: func.func @sort_multi_result_memref
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

func.func @reverse_tensor(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  %init = tensor.empty() : tensor<3x5xi32>
  %0 = iree_linalg_ext.reverse
         dimensions(dense<0> : tensor<1xi64>)
         ins(%arg0 : tensor<3x5xi32>)
         outs(%init : tensor<3x5xi32>) : tensor<3x5xi32>
  return %0 : tensor<3x5xi32>
}
// CHECK-LABEL: func.func @reverse_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x5xi32>
//       CHECK:   %[[INIT:.+]] = tensor.empty()
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.reverse
//  CHECK-SAME:      dimensions(dense<0> : tensor<1xi64>)
//  CHECK-SAME:      ins(%[[ARG0]]
//  CHECK-SAME:      outs(%[[INIT]]

// -----

func.func @reverse_memref(%arg0: memref<3x5xi32>, %arg1: memref<3x5xi32>) {
  iree_linalg_ext.reverse
    dimensions(dense<0> : tensor<1xi64>)
    ins(%arg0 : memref<3x5xi32>)
    outs(%arg1 : memref<3x5xi32>)
  return
}
// CHECK-LABEL: func.func @reverse_memref
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: memref<3x5xi32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: memref<3x5xi32>
//       CHECK:   iree_linalg_ext.reverse
//  CHECK-SAME:      dimensions(dense<0> : tensor<1xi64>)
//  CHECK-SAME:      ins(%[[ARG0]]
//  CHECK-SAME:      outs(%[[ARG1]]

// -----

func.func @reverse_dynamic_tensor(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %0 = iree_linalg_ext.reverse
         dimensions(dense<1> : tensor<1xi64>)
         ins(%arg0 : tensor<?x?xi32>)
         outs(%init : tensor<?x?xi32>) : tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @reverse_dynamic_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]])
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.reverse
//  CHECK-SAME:      dimensions(dense<1> : tensor<1xi64>)
//  CHECK-SAME:      ins(%[[ARG0]]
//  CHECK-SAME:      outs(%[[INIT]]

// -----

func.func @reverse_static_dynamic_tensor(%arg0: tensor<3x5xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<3x5xi32>
  %d1 = tensor.dim %arg0, %c1 : tensor<3x5xi32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %0 = iree_linalg_ext.reverse
         dimensions(dense<1> : tensor<1xi64>)
         ins(%arg0 : tensor<3x5xi32>)
         outs(%init : tensor<?x?xi32>) : tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @reverse_static_dynamic_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x5xi32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]])
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.reverse
//  CHECK-SAME:      dimensions(dense<1> : tensor<1xi64>)
//  CHECK-SAME:      ins(%[[ARG0]]
//  CHECK-SAME:      outs(%[[INIT]]

// -----

func.func @reverse_multi_dims(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  %init = tensor.empty() : tensor<3x5xi32>
  %0 = iree_linalg_ext.reverse
         dimensions(dense<[0, 1]> : tensor<2xi64>)
         ins(%arg0 : tensor<3x5xi32>)
         outs(%init : tensor<3x5xi32>) : tensor<3x5xi32>
  return %0 : tensor<3x5xi32>
}
// CHECK-LABEL: func.func @reverse_multi_dims
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x5xi32>
//       CHECK:   %[[INIT:.+]] = tensor.empty()
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.reverse
//  CHECK-SAME:      dimensions(dense<[0, 1]> : tensor<2xi64>)
//  CHECK-SAME:      ins(%[[ARG0]]
//  CHECK-SAME:      outs(%[[INIT]]

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

// CHECK-LABEL: func.func @topk_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x10x8x4xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x10x8x4xi32>
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
// CHECK-LABEL: func.func @topk_memref
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
// CHECK-LABEL: func.func @topk_dynamic_tensor
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

// CHECK-LABEL: func.func @topk_tensor
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

// CHECK: func.func @pack(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x3xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: tensor<3x3x1x1xf32>) -> tensor<3x3x1x1xf32>
// CHECK: %[[RES:.*]] = iree_linalg_ext.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %[[ARG1]] : (tensor<3x3xf32> tensor<3x3x1x1xf32>) -> tensor<3x3x1x1xf32>
// CHECK: return %[[RES]] : tensor<3x3x1x1xf32>

// -----

func.func @pack(%arg0: memref<3x3xf32>, %arg1: memref<3x3x1x1xf32>) {
  iree_linalg_ext.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %arg1 : (memref<3x3xf32> memref<3x3x1x1xf32>)
  return
}

// CHECK: func.func @pack(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<3x3xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<3x3x1x1xf32>) {
// CHECK: iree_linalg_ext.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %[[ARG1]] : (memref<3x3xf32> memref<3x3x1x1xf32>)

// -----

func.func @extra_pad_and_pack(%input: tensor<13x15xf32>, %output: tensor<3x8x8x2xf32>, %pad: f32) -> tensor<3x8x8x2xf32> {
  // expected-error@+1 {{infered type do not match provided output type. Expected 'tensor<2x8x8x2xf32>' but got: 'tensor<3x8x8x2xf32>}}
  %0 = iree_linalg_ext.pack %input padding_value(%pad: f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<13x15xf32> tensor<3x8x8x2xf32>) -> tensor<3x8x8x2xf32>
  return %0 : tensor<3x8x8x2xf32>
}
// CHECK:      func @extra_pad_and_pack(
// CHECK-SAME:   %[[INPUT:.+]]: tensor<13x15xf32>
// CHECK-SAME:   %[[OUTPUT:.+]]: tensor<3x8x8x2xf32>
// CHECK-SAME:   %[[PAD:.+]]: f32
// CHECK:        %[[RES:.+]] = iree_linalg_ext.pack %[[INPUT]]
// CHECK-SAME:     padding_value(%[[PAD]] : f32)
// CHECK-SAME:     inner_dims_pos = [0, 1]
// CHECK-SAME:     inner_tiles = [8, 2]
// CHECK-SAME:     into %[[OUTPUT]]
// CHECK:       return %[[RES]]

// -----

func.func @pad_and_pack_static(%input: tensor<13x15xf32>, %output: tensor<2x8x8x2xf32>, %pad: f32) -> tensor<2x8x8x2xf32> {
  %0 = iree_linalg_ext.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<13x15xf32> tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}
// CHECK:      func.func @pad_and_pack_static
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]: tensor<13x15xf32>
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9_]+]]: tensor<2x8x8x2xf32>
// CHECK-SAME:   %[[PAD:[a-zA-Z0-9_]+]]: f32
// CHECK:        %[[RES:.+]] = iree_linalg_ext.pack %[[INPUT]]
// CHECK-SAME:     padding_value(%[[PAD]] : f32)
// CHECK-SAME:     inner_dims_pos = [0, 1]
// CHECK-SAME:     inner_tiles = [8, 2]
// CHECK-SAME:     into %[[OUTPUT]]
// CHECK:        return %[[RES]]

// -----

func.func @pad_and_pack_partially_dynamic(%input: tensor<?x?xf32>, %output: tensor<?x?x8x2xf32>, %pad: f32) -> tensor<?x?x8x2xf32> {
  %0 = iree_linalg_ext.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<?x?xf32> tensor<?x?x8x2xf32>) -> tensor<?x?x8x2xf32>
  return %0 : tensor<?x?x8x2xf32>
}
// CHECK:      func.func @pad_and_pack_partially_dynamic
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9_]+]]: tensor<?x?x8x2xf32>
// CHECK-SAME:   %[[PAD:[a-zA-Z0-9_]+]]: f32
// CHECK:        %[[RES:.+]] = iree_linalg_ext.pack %[[INPUT]]
// CHECK-SAME:     padding_value(%[[PAD]] : f32)
// CHECK-SAME:     inner_dims_pos = [0, 1]
// CHECK-SAME:     inner_tiles = [8, 2]
// CHECK-SAME:     into %[[OUTPUT]]
// CHECK:        return %[[RES]]

// -----

func.func @pad_and_pack_fully_dynamic(%input: tensor<?x?xf32>, %output: tensor<?x?x?x?xf32>, %pad: f32, %tile_n : index, %tile_m : index) -> tensor<?x?x?x?xf32> {
  %0 = iree_linalg_ext.pack %input padding_value(%pad : f32)
    inner_dims_pos = [0, 1] inner_tiles = [%tile_n, %tile_m] into %output : (tensor<?x?xf32> tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK:      func.func @pad_and_pack_fully_dynamic
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:   %[[PAD:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME:   %[[TILE_N:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[TILE_M:[a-zA-Z0-9_]+]]: index
// CHECK:        %[[RES:.+]] = iree_linalg_ext.pack %[[INPUT]]
// CHECK-SAME:     padding_value(%[[PAD]] : f32)
// CHECK-SAME:     inner_dims_pos = [0, 1]
// CHECK-SAME:     inner_tiles = [%[[TILE_N]], %[[TILE_M]]]
// CHECK-SAME:     into %[[OUTPUT]]
// CHECK:        return %[[RES]]

// -----

func.func @unpack(%arg0: memref<3x3xf32>, %arg1: memref<3x3x1x1xf32>) {
  iree_linalg_ext.unpack %arg1 inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %arg0 : (memref<3x3x1x1xf32> memref<3x3xf32>)
  return
}

// CHECK: func.func @unpack(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<3x3xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<3x3x1x1xf32>) {
// CHECK: iree_linalg_ext.unpack %[[ARG1]] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %[[ARG0]] : (memref<3x3x1x1xf32> memref<3x3xf32>)

// -----

func.func @unpack_static(%input: tensor<8x8x32x16xf32>, %output: tensor<256x128xf32>) -> tensor<256x128xf32> {
  %0 = iree_linalg_ext.unpack %input inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %output : (tensor<8x8x32x16xf32> tensor<256x128xf32>) -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// CHECK:      func.func @unpack_static
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9_]+]]
// CHECK:        %[[UNPACK:.+]] = iree_linalg_ext.unpack
// CHECK-SAME:     %[[INPUT]]
// CHECK-SAME      dim_pos = [0, 1]
// CHECK-SAME      inner_pos = [32, 16]
// CHECK-SAME:     into %[[OUTPUT]]
// CHECK:        return %[[UNPACK]]

// -----

func.func @unpack_undo_padding(%input: tensor<2x8x8x2xf32>, %output: tensor<13x15xf32>) -> tensor<13x15xf32> {
  %0 = iree_linalg_ext.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<2x8x8x2xf32> tensor<13x15xf32>) -> tensor<13x15xf32>
  return %0 : tensor<13x15xf32>
}
// CHECK:      func.func @unpack_undo_padding
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9_]+]]
// CHECK:        %[[UNPACK:.+]] = iree_linalg_ext.unpack
// CHECK-SAME:     %[[INPUT]]
// CHECK-SAME      dim_pos = [0, 1]
// CHECK-SAME      inner_pos = [32, 16]
// CHECK-SAME:     into %[[OUTPUT]]
// CHECK:        return %[[UNPACK]]

// -----

func.func @unpack(%arg0: memref<3x3xf32>, %arg1: memref<3x3x1x1xf32>) {
  iree_linalg_ext.unpack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %arg0 : (memref<3x3x1x1xf32> memref<3x3xf32>)
  return
}

// CHECK: func.func @unpack(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<3x3xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<3x3x1x1xf32>) {
// CHECK: iree_linalg_ext.unpack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %[[ARG0]] : (memref<3x3x1x1xf32> memref<3x3xf32>)

// -----

func.func @pack(%arg0: memref<128x256xf32>, %arg1: memref<32x4x32x8xf32>) {
  iree_linalg_ext.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<32x4x32x8xf32>)
  return
}

// CHECK: func.func @pack
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<32x4x32x8xf32>) {
// CHECK: iree_linalg_ext.pack %[[ARG0]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %[[ARG1]] : (memref<128x256xf32> memref<32x4x32x8xf32>)

// -----

func.func @pack(%arg0: memref<128x256xf32>, %arg1: memref<4x32x32x8xf32>) {
  iree_linalg_ext.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<4x32x32x8xf32>)
  return
}

// CHECK: func.func @pack
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<4x32x32x8xf32>) {
// CHECK: iree_linalg_ext.pack %[[ARG0]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %[[ARG1]] : (memref<128x256xf32> memref<4x32x32x8xf32>)

// -----

func.func @unpack(%arg0: memref<128x256xf32>, %arg1: memref<4x32x32x8xf32>) {
  iree_linalg_ext.unpack %arg1 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg0 : (memref<4x32x32x8xf32> memref<128x256xf32>)
  return
}

// CHECK: func.func @unpack
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<4x32x32x8xf32>) {
// CHECK: iree_linalg_ext.unpack %[[ARG1]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %[[ARG0]] : (memref<4x32x32x8xf32> memref<128x256xf32>)

// -----

func.func @unpack(%arg0: memref<128x256xf32>, %arg1: memref<32x4x32x8xf32>) {
  iree_linalg_ext.unpack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg0 : (memref<32x4x32x8xf32> memref<128x256xf32>)
  return
}

// CHECK: func.func @unpack
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<32x4x32x8xf32>) {
// CHECK: iree_linalg_ext.unpack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %[[ARG0]] : (memref<32x4x32x8xf32> memref<128x256xf32>)

// -----

// CHECK: @set_encoding_ops(%[[ARG0:.+]]: tensor<?x?xf32>)
func.func @set_encoding_ops(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>> {
  // CHECK: iree_linalg_ext.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
  return %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
}

// -----

// CHECK: @set_encoding_ops_mixed_dynamic_static(%[[ARG0:.+]]: tensor<?x10xf32>)
func.func @set_encoding_ops_mixed_dynamic_static(%arg0: tensor<?x10xf32>) -> tensor<20x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>> {
  // CHECK: iree_linalg_ext.set_encoding %[[ARG0]] : tensor<?x10xf32> -> tensor<20x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x10xf32> -> tensor<20x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
  return %0 : tensor<20x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
}

// -----

// CHECK: @set_encoding_with_batch_matmul_user(%[[ARG0:.+]]: tensor<?x?x?xf32>)
func.func @set_encoding_with_batch_matmul_user(%arg0: tensor<?x?x?xf32>) {
  // CHECK: iree_linalg_ext.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
  iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
  // CHECK: iree_linalg_ext.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32]>>
  iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32]>>
  // CHECK: iree_linalg_ext.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32]>>
  iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32]>>
  // CHECK: iree_linalg_ext.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16]>>
  iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16]>>
  // CHECK: iree_linalg_ext.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32]>>
  iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32]>>
  // CHECK: iree_linalg_ext.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16]>>
  iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16]>>
  return
}

// -----

// CHECK: @unset_encoding_ops(%[[ARG0:.+]]: tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>>)
func.func @unset_encoding_ops(%arg0: tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>>) -> tensor<?x?xf32> {
  // CHECK: iree_linalg_ext.unset_encoding %[[ARG0]] : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  %0 = iree_linalg_ext.unset_encoding %arg0 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: @unset_encoding_ops_mixed_dynamic_static(%[[ARG0:.+]]: tensor<10x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>>)
func.func @unset_encoding_ops_mixed_dynamic_static(%arg0: tensor<10x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>>) -> tensor<?x20xf32> {
  // CHECK: iree_linalg_ext.unset_encoding %[[ARG0]] : tensor<10x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>>
  %0 = iree_linalg_ext.unset_encoding %arg0 : tensor<10x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>> -> tensor<?x20xf32>
  return %0 : tensor<?x20xf32>
}

// -----

func.func @encoding_tensors_with_ops(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>>
  %2 = iree_linalg_ext.set_encoding %arg2 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>>)
      outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @encoding_tensors_with_ops
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[ARG0]]
//  CHECK-SAME:       tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32]>>
//       CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[ARG1]]
//  CHECK-SAME:       tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32]>>
//       CHECK:   %[[OUT:.+]] = iree_linalg_ext.set_encoding %[[ARG2]]
//  CHECK-SAME:       tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32]>>
//       CHECK:   %[[GEMM:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUT]] :
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.unset_encoding %[[GEMM]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @winograd_input_transform(%arg0: tensor<1x10x10x1280xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<1x10x10x1280xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}
// CHECK:      func.func @winograd_input_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>) ->
// CHECK-SAME:   tensor<8x8x1x2x2x1280xf32> {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:        %[[D1:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<1x10x10x1280xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
// CHECK:        return %[[D1]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:      }

// -----

func.func @winograd_input_transform_dynamic(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<8x8x?x?x?x?xf32>) -> tensor<8x8x?x?x?x?xf32> {
  %1 = iree_linalg_ext.winograd.input_transform
    output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<?x?x?x?xf32>) outs(%arg1 : tensor<8x8x?x?x?x?xf32>) -> tensor<8x8x?x?x?x?xf32>
  return %1 : tensor<8x8x?x?x?x?xf32>
}
// CHECK:      func.func @winograd_input_transform_dynamic(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<8x8x?x?x?x?xf32>) -> tensor<8x8x?x?x?x?xf32> {
// CHECK:        %[[D0:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<?x?x?x?xf32>) outs(%[[ARG1]] :
// CHECK-SAME:     tensor<8x8x?x?x?x?xf32>) -> tensor<8x8x?x?x?x?xf32>
// CHECK:        return %[[D0]] : tensor<8x8x?x?x?x?xf32>
// CHECK:      }

// -----

func.func @winograd_output_transform(%arg0: tensor<8x8x1x2x2x1280xf32>) -> tensor<1x12x12x1280xf32> {
  %0 = tensor.empty() : tensor<1x12x12x1280xf32>
  %1 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<8x8x1x2x2x1280xf32>) outs(%0 : tensor<1x12x12x1280xf32>) -> tensor<1x12x12x1280xf32>
  return %1 : tensor<1x12x12x1280xf32>
}
// CHECK:      func.func @winograd_output_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x1280xf32>) ->
// CHECK-SAME:   tensor<1x12x12x1280xf32> {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<1x12x12x1280xf32>
// CHECK:        %[[D1:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<8x8x1x2x2x1280xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<1x12x12x1280xf32>) -> tensor<1x12x12x1280xf32>
// CHECK:        return %[[D1]] : tensor<1x12x12x1280xf32>
// CHECK:      }

// -----

func.func @winograd_output_transform(%arg0: tensor<8x8x?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %1 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<8x8x?x?x?x?xf32>) outs(%arg1 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
// CHECK:      func.func @winograd_output_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x?x?x?x?xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:        %[[D0:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([1, 2]) ins(%[[ARG0]] : tensor<8x8x?x?x?x?xf32>) outs(%[[ARG1]] :
// CHECK-SAME:     tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:        return %[[D0]] : tensor<?x?x?x?xf32>
// CHECK:      }

// -----

func.func @winograd_input_transform_nchw(%arg0: tensor<1x1280x10x10xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
    ins(%arg0 : tensor<1x1280x10x10xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}
// CHECK:      func.func @winograd_input_transform_nchw(%[[ARG0]]: tensor<1x1280x10x10xf32>) ->
// CHECK-SAME:   tensor<8x8x1x2x2x1280xf32> {
// CHECK:        %[[D0]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:        %[[D1]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([2, 3]) ins(%[[ARG0]] : tensor<1x1280x10x10xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
// CHECK:        return %[[D1]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:      }
// CHECK:    }

// -----

func.func @winograd_output_transform_nchw(%arg0: tensor<8x8x1x2x2x1280xf32>) -> tensor<1x1280x12x12xf32> {
  %0 = tensor.empty() : tensor<1x1280x12x12xf32>
  %1 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
    ins(%arg0 : tensor<8x8x1x2x2x1280xf32>) outs(%0 : tensor<1x1280x12x12xf32>) -> tensor<1x1280x12x12xf32>
  return %1 : tensor<1x1280x12x12xf32>
}
// CHECK:      func.func @winograd_output_transform_nchw(%[[ARG0]]: tensor<8x8x1x2x2x1280xf32>) ->
// CHECK-SAME:   tensor<1x1280x12x12xf32> {
// CHECK:        %[[D0]] = tensor.empty() : tensor<1x1280x12x12xf32>
// CHECK:        %[[D1]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3)
// CHECK-SAME:     image_dimensions([2, 3]) ins(%[[ARG0]] : tensor<8x8x1x2x2x1280xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<1x1280x12x12xf32>) -> tensor<1x1280x12x12xf32>
// CHECK:        return %[[D1]] : tensor<1x1280x12x12xf32>
// CHECK:      }
// CHECK:    }

// -----

func.func @attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
// CHECK:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = iree_linalg_ext.attention ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] :
// CHECK-SAME:     tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }

// -----

func.func @cross_attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x2048x64xf32>, %value: tensor<192x2048x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<192x1024x64xf32>, tensor<192x2048x64xf32>, tensor<192x2048x64xf32>) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
// CHECK:      func.func @cross_attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x2048x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x2048x64xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = iree_linalg_ext.attention ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] :
// CHECK-SAME:     tensor<192x1024x64xf32>, tensor<192x2048x64xf32>, tensor<192x2048x64xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }

// -----

func.func @cross_attention_transposev(%query: tensor<192x1024x64xf32>, %key: tensor<192x2048x64xf32>, %value: tensor<192x64x2048xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %1 = iree_linalg_ext.attention {transpose_v = true} ins(%query, %key, %value : tensor<192x1024x64xf32>, tensor<192x2048x64xf32>, tensor<192x64x2048xf32>) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}
// CHECK:      func.func @cross_attention_transposev(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x2048x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x64x2048xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = iree_linalg_ext.attention {transpose_v = true} ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] :
// CHECK-SAME:     tensor<192x1024x64xf32>, tensor<192x2048x64xf32>, tensor<192x64x2048xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }

// -----

func.func @cross_attention_transposev_dyn(%query: tensor<?x?x?xf32>, %key: tensor<?x?x?xf32>, %value: tensor<?x?x?xf32>, %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %1 = iree_linalg_ext.attention {transpose_v = true} ins(%query, %key, %value : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK:      func.func @cross_attention_transposev_dyn(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<?x?x?xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>, %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-SAME:   {
// CHECK:        %[[D1:.+]] = iree_linalg_ext.attention {transpose_v = true} ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] :
// CHECK-SAME:     tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%[[ARG3]] :
// CHECK-SAME:     tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:        return %[[D1]] : tensor<?x?x?xf32>
// CHECK:      }
