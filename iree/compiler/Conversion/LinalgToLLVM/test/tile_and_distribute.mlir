// RUN: iree-opt --iree-codegen-llvm-linalg-tile-and-distribute=tile-sizes=2,4,1 -cse -split-input-file %s | IreeFileCheck %s

func @dynamic_matmul(%lhs: memref<?x?xf32>, %rhs: memref<?x?xf32>, %result: memref<?x?xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>) outs(%result : memref<?x?xf32>)
  return
}
// CHECK: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (2, s1 - s0 * 2)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK: #[[MAP4:.+]] = affine_map<()[s0, s1] -> (4, s1 - s0 * 4)>
// CHECK: func @dynamic_matmul(%[[LHS:.+]]: memref<?x?xf32>, %[[RHS:.+]]: memref<?x?xf32>, %[[RESULT:.+]]: memref<?x?xf32>)
// CHECK: %[[CONST_0:.+]] = constant 0 : index
// CHECK: %[[CONST_1:.+]] = constant 1 : index
// CHECK: %[[DIM_K:.+]] = dim %[[LHS]], %[[CONST_1]]
// CHECK: %[[THREAD_X_ID:.+]] = iree.workgroup_id  {dimension = "x"} : index
// CHECK: %[[THREAD_Y_ID:.+]] = iree.workgroup_id  {dimension = "y"} : index
// CHECK:  scf.for %[[K:.+]] = %[[CONST_0]] to %[[DIM_K]]
// CHECK:     %[[I:.+]] = affine.apply #[[MAP0]]()[%[[THREAD_Y_ID]]]
// CHECK:     %[[DIM_I:.+]] = dim %[[LHS]], %[[CONST_0]]
// CHECK:     %[[I_OFFSET:.+]] = affine.min #[[MAP1]]()[%[[THREAD_Y_ID]], %[[DIM_I]]]
// CHECK:     %[[LHS_SUBVIEW:.+]] = subview %[[LHS]][%[[I]], %[[K]]] [%[[I_OFFSET]], 1] [1, 1] 
// CHECK:     %[[J:.+]] = affine.apply #[[MAP3]]()[%[[THREAD_X_ID]]]
// CHECK:     %[[DIM_J:.+]] = dim %[[RHS]], %[[CONST_1]] 
// CHECK:     %[[J_OFFSET:.+]] = affine.min #[[MAP4]]()[%[[THREAD_X_ID]], %[[DIM_J]]]
// CHECK:     %[[RHS_SUBVIEW:.+]] = subview %[[RHS]][%[[K]], %[[J]]] [1, %[[J_OFFSET]]] [1, 1]  
// CHECK:     %[[DIM_I:.+]] = dim %[[RESULT]], %[[CONST_0]]
// CHECK:     %[[DIM_I_OFFSET:.+]] = affine.min #[[MAP1]]()[%[[THREAD_Y_ID]], %[[DIM_I]]]
// CHECK:     %[[DIM_J:.+]] = dim %[[RESULT]], %[[CONST_1]]
// CHECK:     %[[DIM_J_OFFSET:.+]] = affine.min #[[MAP4]]()[%[[THREAD_X_ID]], %[[DIM_J]]]
// CHECK:     %[[RESULT_SUBVIEW:.+]] = subview %[[RESULT]][%[[I]], %[[J]]] [%[[DIM_I_OFFSET]], %[[DIM_J_OFFSET]]] [1, 1]
// CHECK:      linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]] : memref<?x1xf32, #[[MAP2]]>, memref<1x?xf32, #[[MAP2]]>) outs(%[[RESULT_SUBVIEW]] : memref<?x?xf32, #[[MAP2]]>)

// -----

func @static_matmul(%lhs: memref<16x4xf32>, %rhs: memref<4x8xf32>, %result: memref<16x8xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<16x4xf32>, memref<4x8xf32>) outs(%result : memref<16x8xf32>)
  return
}
// CHECK: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>
// CHECK: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>
// CHECK: func @static_matmul(%[[LHS:.+]]: memref<16x4xf32>, %[[RHS:.+]]: memref<4x8xf32>, %[[RESULT:.+]]: memref<16x8xf32>)
// CHECK: %[[CONST_0:.+]] = constant 0 : index
// CHECK: %[[CONST_4:.+]] = constant 4 : index
// CHECK: %[[CONST_1:.+]] = constant 1 : index
// CHECK: %[[THREAD_X_ID:.+]] = iree.workgroup_id  {dimension = "x"} : index
// CHECK: %[[THREAD_Y_ID:.+]] = iree.workgroup_id  {dimension = "y"} : index
// CHECK:  scf.for %[[K:.+]] = %[[CONST_0]] to %[[CONST_4]] step %[[CONST_1]] 
// CHECK:    %[[I:.+]] = affine.apply #[[MAP0]]()[%[[THREAD_Y_ID]]]
// CHECK:    %[[LHS_SUBVIEW:.+]] = subview %[[LHS]][%[[I]], %[[K]]] [2, 1] [1, 1]  : memref<16x4xf32> to memref<2x1xf32, #[[MAP1]]>
// CHECK:    %[[J:.+]] = affine.apply #[[MAP2]]()[%[[THREAD_X_ID]]]
// CHECK:    %[[RHS_SUBVIEW:.+]] = subview %[[RHS]][%[[K]], %[[J]]] [1, 4] [1, 1]  : memref<4x8xf32> to memref<1x4xf32, #[[MAP3]]>
// CHECK:    %[[RESULT_SUBVIEW:.+]] = subview %[[RESULT]][%[[I]], %[[J]]] [2, 4] [1, 1]  : memref<16x8xf32> to memref<2x4xf32, #[[MAP3]]>
// CHECK:    linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]] : memref<2x1xf32, #[[MAP1]]>, memref<1x4xf32, #[[MAP3]]>) outs(%6 : memref<2x4xf32, #[[MAP3]]>)
