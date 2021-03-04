// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-llvm-linalg-tile-and-distribute))" -iree-llvm-tile-size=2,4,1 -cse -canonicalize -split-input-file %s | IreeFileCheck %s

// // TODO(GH-4901): Enable the dynamic shape tests when linalg on tensors becomes default.
// hal.executable @dynamic_matmul attributes {sym_visibility = "private"} {
//   hal.interface @legacy_io {
//     hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//     hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
//     hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
//   }
//   hal.executable.target @llvm_aot, filter="dylib*" {
//     hal.executable.entry_point @dynamic_matmul attributes {
//       interface = @legacy_io, ordinal = 0 : i32,
//       signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
//         !flow.dispatch.output<?x?xf32>) -> ()}
//     module {
//       func @dynamic_matmul(%lhs: memref<?x?xf32>, %rhs: memref<?x?xf32>, %result: memref<?x?xf32>) {
//         linalg.matmul ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>) outs(%result : memref<?x?xf32>)
//         return
//       }
//     }
//   }
// }
// // NOCHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 2)>
// // NOCHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (2, s0 * -2 + s1)>
// // NOCHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// // NOCHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 4)>
// // NOCHECK-DAG: #[[MAP4:.+]] = affine_map<()[s0, s1] -> (4, s0 * -4 + s1)>
// //     NOCHECK: func @dynamic_matmul(%[[LHS:.+]]: memref<?x?xf32>, %[[RHS:.+]]: memref<?x?xf32>, %[[RESULT:.+]]: memref<?x?xf32>)
// // NOCHECK-DAG: %[[CONST_0:.+]] = constant 0 : index
// // NOCHECK-DAG: %[[CONST_1:.+]] = constant 1 : index
// // NOCHECK-DAG: %[[DIM_K:.+]] = dim %[[LHS]], %[[CONST_1]]
// // NOCHECK-DAG: %[[THREAD_X_ID:.+]] = hal.interface.workgroup.id[0] : index
// // NOCHECK-DAG: %[[THREAD_Y_ID:.+]] = hal.interface.workgroup.id[1] : index
// //     NOCHECK:  scf.for %[[K:.+]] = %[[CONST_0]] to %[[DIM_K]]
// //     NOCHECK:     %[[I:.+]] = affine.apply #[[MAP0]]()[%[[THREAD_Y_ID]]]
// //     NOCHECK:     %[[DIM_I:.+]] = dim %[[LHS]], %[[CONST_0]]
// //     NOCHECK:     %[[I_OFFSET:.+]] = affine.min #[[MAP1]]()[%[[THREAD_Y_ID]], %[[DIM_I]]]
// //     NOCHECK:     %[[LHS_SUBVIEW:.+]] = subview %[[LHS]][%[[I]], %[[K]]] [%[[I_OFFSET]], 1] [1, 1]
// //     NOCHECK:     %[[J:.+]] = affine.apply #[[MAP3]]()[%[[THREAD_X_ID]]]
// //     NOCHECK:     %[[DIM_J:.+]] = dim %[[RHS]], %[[CONST_1]]
// //     NOCHECK:     %[[J_OFFSET:.+]] = affine.min #[[MAP4]]()[%[[THREAD_X_ID]], %[[DIM_J]]]
// //     NOCHECK:     %[[RHS_SUBVIEW:.+]] = subview %[[RHS]][%[[K]], %[[J]]] [1, %[[J_OFFSET]]] [1, 1]
// //     NOCHECK:     %[[DIM_I:.+]] = dim %[[RESULT]], %[[CONST_0]]
// //     NOCHECK:     %[[DIM_I_OFFSET:.+]] = affine.min #[[MAP1]]()[%[[THREAD_Y_ID]], %[[DIM_I]]]
// //     NOCHECK:     %[[DIM_J:.+]] = dim %[[RESULT]], %[[CONST_1]]
// //     NOCHECK:     %[[DIM_J_OFFSET:.+]] = affine.min #[[MAP4]]()[%[[THREAD_X_ID]], %[[DIM_J]]]
// //     NOCHECK:     %[[RESULT_SUBVIEW:.+]] = subview %[[RESULT]][%[[I]], %[[J]]] [%[[DIM_I_OFFSET]], %[[DIM_J_OFFSET]]] [1, 1]
// //     NOCHECK:      linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]] : memref<?x1xf32, #[[MAP2]]>, memref<1x?xf32, #[[MAP2]]>) outs(%[[RESULT_SUBVIEW]] : memref<?x?xf32, #[[MAP2]]>)

// -----

hal.executable @static_matmul attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @static_matmul attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<16x4xf32>, !flow.dispatch.input<4x8xf32>,
        !flow.dispatch.output<16x8xf32>) -> ()}
    module {
      func @static_matmul(%lhs: memref<16x4xf32>, %rhs: memref<4x8xf32>, %result: memref<16x8xf32>) {
        linalg.matmul ins(%lhs, %rhs : memref<16x4xf32>, memref<4x8xf32>) outs(%result : memref<16x8xf32>)
        return
      }
    }
  }
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>
//     CHECK: func @static_matmul(%[[LHS:.+]]: memref<16x4xf32>, %[[RHS:.+]]: memref<4x8xf32>, %[[RESULT:.+]]: memref<16x8xf32>)
// CHECK-DAG: %[[CONST_0:.+]] = constant 0 : index
// CHECK-DAG: %[[CONST_4:.+]] = constant 4 : index
// CHECK-DAG: %[[CONST_1:.+]] = constant 1 : index
// CHECK-DAG: %[[THREAD_X_ID:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG: %[[THREAD_Y_ID:.+]] = hal.interface.workgroup.id[1] : index
//     CHECK:  scf.for %[[K:.+]] = %[[CONST_0]] to %[[CONST_4]] step %[[CONST_1]]
//     CHECK:    %[[I:.+]] = affine.apply #[[MAP0]]()[%[[THREAD_Y_ID]]]
//     CHECK:    %[[LHS_SUBVIEW:.+]] = subview %[[LHS]][%[[I]], %[[K]]] [2, 1] [1, 1]  : memref<16x4xf32> to memref<2x1xf32, #[[MAP1]]>
//     CHECK:    %[[J:.+]] = affine.apply #[[MAP2]]()[%[[THREAD_X_ID]]]
//     CHECK:    %[[RHS_SUBVIEW:.+]] = subview %[[RHS]][%[[K]], %[[J]]] [1, 4] [1, 1]  : memref<4x8xf32> to memref<1x4xf32, #[[MAP3]]>
//     CHECK:    %[[RESULT_SUBVIEW:.+]] = subview %[[RESULT]][%[[I]], %[[J]]] [2, 4] [1, 1]  : memref<16x8xf32> to memref<2x4xf32, #[[MAP3]]>
//     CHECK:    linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]] : memref<2x1xf32, #[[MAP1]]>, memref<1x4xf32, #[[MAP3]]>) outs(%4 : memref<2x4xf32, #[[MAP3]]>)
