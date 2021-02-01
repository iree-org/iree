// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-llvm-linalg-tile-and-distribute))" -iree-llvm-tile-size=2,4,1 -cse -canonicalize -split-input-file %s | IreeFileCheck %s

hal.executable @dynamic_matmul attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @dynamic_matmul attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
    module {
      func @dynamic_matmul(%lhs: memref<?x?xf32>, %rhs: memref<?x?xf32>, %result: memref<?x?xf32>) {
        linalg.matmul ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>) outs(%result : memref<?x?xf32>)
        return
      }
    }
  }
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (2, s0 * -2 + s1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<()[s0, s1] -> (4, s0 * -4 + s1)>
//     CHECK: func @dynamic_matmul(%[[LHS:.+]]: memref<?x?xf32>, %[[RHS:.+]]: memref<?x?xf32>, %[[RESULT:.+]]: memref<?x?xf32>)
// CHECK-DAG: %[[CONST_0:.+]] = constant 0 : index
// CHECK-DAG: %[[CONST_1:.+]] = constant 1 : index
// CHECK-DAG: %[[DIM_K:.+]] = dim %[[LHS]], %[[CONST_1]]
// CHECK-DAG: %[[THREAD_X_ID:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG: %[[THREAD_Y_ID:.+]] = hal.interface.workgroup.id[1] : index
//     CHECK:  scf.for %[[K:.+]] = %[[CONST_0]] to %[[DIM_K]]
//     CHECK:     %[[I:.+]] = affine.apply #[[MAP0]]()[%[[THREAD_Y_ID]]]
//     CHECK:     %[[DIM_I:.+]] = dim %[[LHS]], %[[CONST_0]]
//     CHECK:     %[[I_OFFSET:.+]] = affine.min #[[MAP1]]()[%[[THREAD_Y_ID]], %[[DIM_I]]]
//     CHECK:     %[[LHS_SUBVIEW:.+]] = subview %[[LHS]][%[[I]], %[[K]]] [%[[I_OFFSET]], 1] [1, 1]
//     CHECK:     %[[J:.+]] = affine.apply #[[MAP3]]()[%[[THREAD_X_ID]]]
//     CHECK:     %[[DIM_J:.+]] = dim %[[RHS]], %[[CONST_1]]
//     CHECK:     %[[J_OFFSET:.+]] = affine.min #[[MAP4]]()[%[[THREAD_X_ID]], %[[DIM_J]]]
//     CHECK:     %[[RHS_SUBVIEW:.+]] = subview %[[RHS]][%[[K]], %[[J]]] [1, %[[J_OFFSET]]] [1, 1]
//     CHECK:     %[[DIM_I:.+]] = dim %[[RESULT]], %[[CONST_0]]
//     CHECK:     %[[DIM_I_OFFSET:.+]] = affine.min #[[MAP1]]()[%[[THREAD_Y_ID]], %[[DIM_I]]]
//     CHECK:     %[[DIM_J:.+]] = dim %[[RESULT]], %[[CONST_1]]
//     CHECK:     %[[DIM_J_OFFSET:.+]] = affine.min #[[MAP4]]()[%[[THREAD_X_ID]], %[[DIM_J]]]
//     CHECK:     %[[RESULT_SUBVIEW:.+]] = subview %[[RESULT]][%[[I]], %[[J]]] [%[[DIM_I_OFFSET]], %[[DIM_J_OFFSET]]] [1, 1]
//     CHECK:      linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]] : memref<?x1xf32, #[[MAP2]]>, memref<1x?xf32, #[[MAP2]]>) outs(%[[RESULT_SUBVIEW]] : memref<?x?xf32, #[[MAP2]]>)

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

// -----

hal.executable @matmul_tensors attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @matmul_tensors attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<?x?xf32>, !flow.dispatch.input<?x?xf32>,
        !flow.dispatch.output<?x?xf32>) -> ()}
    module {
      func @matmul_tensors() {
        %c2 = constant 2 : index
        %c4 = constant 4 : index
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<?xi8>
        %1 = std.view %0[%c0][] : memref<?xi8> to memref<2x3xf32>
        %2 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<?xi8>
        %3 = std.view %2[%c0][] : memref<?xi8> to memref<3x4xf32>
        %4 = hal.interface.binding.subspan @legacy_io::@arg2[%c0] : memref<?xi8>
        %5 = std.view %4[%c0][] : memref<?xi8> to memref<2x4xf32>
        %6 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<?xi8>
        %7 = std.view %6[%c0][] : memref<?xi8> to memref<2x4xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %8 = muli %workgroup_size_y, %workgroup_id_y : index
        %9 = muli %workgroup_size_y, %workgroup_count_y : index
        scf.for %arg0 = %8 to %c2 step %9 {
          %10 = muli %workgroup_size_x, %workgroup_id_x : index
          %11 = muli %workgroup_size_x, %workgroup_count_x : index
          scf.for %arg1 = %10 to %c4 step %11 {
            %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 2)>(%arg0)[%workgroup_size_y]
            %13 = subview %1[%arg0, 0] [%12, 3] [1, 1] : memref<2x3xf32> to memref<?x3xf32, affine_map<(d0, d1)[s0] -> (d0 * 3 + s0 + d1)>>
            %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg1)[%workgroup_size_x]
            %15 = subview %3[0, %arg1] [3, %14] [1, 1] : memref<3x4xf32> to memref<3x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
            %16 = subview %5[%arg0, %arg1] [%12, %14] [1, 1] : memref<2x4xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
            %17 = alloc(%12, %14) : memref<?x?xf32>
            linalg.copy(%16, %17) : memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>, memref<?x?xf32>
            linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%13, %15 : memref<?x3xf32, affine_map<(d0, d1)[s0] -> (d0 * 3 + s0 + d1)>>, memref<3x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>) outs(%17 : memref<?x?xf32>)
            %18 = subview %7[%arg0, %arg1] [%12, %14] [1, 1] : memref<2x4xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
            linalg.copy(%17, %18) : memref<?x?xf32>, memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
          }
        }
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable @matmul_tensors
//       CHECK: hal.executable.entry_point @matmul_tensors
//  CHECK-NEXT:   ^{{[a-zA-Z0-9_]+}}(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:     %[[C3:.+]] = constant 3 : index
//   CHECK-DAG:     %[[C4:.+]] = constant 4 : index
//       CHECK:     %[[T0:.+]] = addi %[[ARG0]], %[[C3]]
//       CHECK:     %[[T1:.+]] = divi_signed %[[T0]], %[[C4]]
//       CHECK:     %[[T2:.+]] = addi %[[ARG1]], %[[C1]]
//       CHECK:     %[[T3:.+]] = divi_signed %[[T2]], %[[C2]]
//       CHECK:     hal.return %[[T1]], %[[T3]], %[[C1]]
//   CHECK-NOT:   hal.interface.workgroup.size
//   CHECK-DAG:   %[[C2:.+]] = constant 2
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[WGID_X:.+]] = hal.interface.workgroup.id[0]
//   CHECK-DAG:   %[[WGID_Y:.+]] = hal.interface.workgroup.id[1]
//   CHECK-DAG:   %[[WGCOUNT_X:.+]] = hal.interface.workgroup.count[0]
//   CHECK-DAG:   %[[WGCOUNT_Y:.+]] = hal.interface.workgroup.count[1]
//       CHECK:   %[[OFFSET_Y:.+]] = muli %[[WGID_Y]], %[[C2]]
//       CHECK:   %[[STEP_Y:.+]] = muli %[[WGCOUNT_Y]], %[[C2]]
//       CHECK:   scf.for %{{.+}} = %[[OFFSET_Y]] to %[[C2]] step %[[STEP_Y]]
//       CHECK:     %[[OFFSET_X:.+]] = muli %[[WGID_X]], %[[C4]]
//       CHECK:     %[[STEP_X:.+]] = muli %[[WGCOUNT_X]], %[[C4]]
//       CHECK:     scf.for %{{.+}} = %[[OFFSET_X]] to %[[C4]] step %[[STEP_X]]
