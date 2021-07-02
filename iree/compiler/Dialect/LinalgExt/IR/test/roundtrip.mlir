// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: func @sort_tensor
// CHECK:         linalg_ext.sort
// CHECK-SAME:      outs({{.*}})
// CHECK:           linalg_ext.yield
func @sort_tensor(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  %0 = linalg_ext.sort
    outs(%arg0 : tensor<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %1 = cmpi sgt, %arg1, %arg2 : i32
    linalg_ext.yield %1 : i1
  } -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// -----

// CHECK-LABEL: func @sort_memref
// CHECK:         linalg_ext.sort
// CHECK-SAME:      outs({{.*}})
// CHECK:           linalg_ext.yield
func @sort_memref(%arg0: memref<128xi32>) {
  linalg_ext.sort dimension(0)
    outs(%arg0 : memref<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %0 = cmpi sgt, %arg1, %arg2 : i32
    linalg_ext.yield %0 : i1
  }
  return
}

// -----

func @scatter_tensor_dynamic(
    %original: tensor<?x?xf32>, %indices: tensor<?xi32>,
    %update: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xf32>, tensor<?xi32>)
    outs(%original: tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @scatter_tensor_dynamic(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = linalg_ext.scatter
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     linalg_ext.yield %{{.+}} : f32
//       CHECK:   return %[[RESULT]]

// -----

func @scatter_tensor_static(
    %original: tensor<128x3xf32>, %indices: tensor<48xi32>,
    %update: tensor<48x3xf32>) -> tensor<128x3xf32> {
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<48x3xf32>, tensor<48xi32>)
    outs(%original: tensor<128x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<128x3xf32>
  return %0 : tensor<128x3xf32>
}
// CHECK-LABEL: func @scatter_tensor_static(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: tensor<128x3xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: tensor<48xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: tensor<48x3xf32>
//       CHECK:   %[[RESULT:.+]] = linalg_ext.scatter
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     linalg_ext.yield %{{.+}} : f32
//       CHECK:   return %[[RESULT]]

// -----

func @scatter_memref_dynamic(
    %original: memref<?x?xf32>, %indices: memref<?xi32>,
    %update: memref<?x?xf32>) {
  linalg_ext.scatter
    ins(%update, %indices : memref<?x?xf32>, memref<?xi32>)
    outs(%original: memref<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    }
  return
}
// CHECK-LABEL: func @scatter_memref_dynamic(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: memref<?xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: memref<?x?xf32>
//       CHECK:   linalg_ext.scatter
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     linalg_ext.yield %{{.+}} : f32
//       CHECK:   return

// -----

func @scatter_memref_static(
    %original: memref<128x3xf32>, %indices: memref<48xi32>,
    %update: memref<48x3xf32>) {
  linalg_ext.scatter
    ins(%update, %indices : memref<48x3xf32>, memref<48xi32>)
    outs(%original: memref<128x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    }
  return
}
// CHECK-LABEL: func @scatter_memref_static(
//  CHECK-SAME:   %[[ORIGINAL:[a-zA-Z0-9_]+]]: memref<128x3xf32>
//  CHECK-SAME:   %[[INDICES:[a-zA-Z0-9_]+]]: memref<48xi32>
//  CHECK-SAME:   %[[UPDATE:[a-zA-Z0-9_]+]]: memref<48x3xf32>
//       CHECK:   linalg_ext.scatter
//  CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
//  CHECK-SAME:     outs(%[[ORIGINAL]]
//       CHECK:     linalg_ext.yield %{{.+}} : f32
//       CHECK:   return
