// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-llvm-materialize-launch-configuration))" -iree-llvm-tile-size=2,4,1 -cse -canonicalize -split-input-file %s | IreeFileCheck %s

hal.executable @matmul_tensors attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @matmul_tensors attributes {
      interface = @io, ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?x?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    module {
      func @matmul_tensors() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?xf32>
        %4 = hal.interface.binding.subspan @io::@arg2[%c0] : memref<?x?xf32>
        %6 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>
        %M = memref.dim %0, %c0 : memref<?x?xf32>
        %N = memref.dim %2, %c1 : memref<?x?xf32>
        %K = memref.dim %0, %c1 : memref<?x?xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %8 = muli %workgroup_size_y, %workgroup_id_y : index
        %9 = muli %workgroup_size_y, %workgroup_count_y : index
        scf.for %arg0 = %8 to %M step %9 {
          %10 = muli %workgroup_size_x, %workgroup_id_x : index
          %11 = muli %workgroup_size_x, %workgroup_count_x : index
          scf.for %arg1 = %10 to %N step %11 {
            %12 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %N]
            %13 = memref.subview %0[%arg0, 0] [%12, %K] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %14 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %M]
            %15 = memref.subview %2[0, %arg1] [%K, %14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %16 = memref.subview %4[%arg0, %arg1] [%12, %14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %17 = memref.alloc(%12, %14) : memref<?x?xf32>
            linalg.copy(%16, %17) : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?x?xf32>
            linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%13, %15 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>) outs(%17 : memref<?x?xf32>)
            %18 = memref.subview %6[%arg0, %arg1] [%12, %14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            linalg.copy(%17, %18) : memref<?x?xf32>, memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
          }
        }
        return
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//       CHECK: hal.executable @matmul_tensors
//       CHECK: hal.executable.entry_point @matmul_tensors
//  CHECK-NEXT:   ^{{[a-zA-Z0-9_]+}}(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[WGX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//   CHECK-DAG:     %[[WGY:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//       CHECK:     hal.return %[[WGX]], %[[WGY]], %[[C1]]
//   CHECK-NOT:   hal.interface.workgroup.size
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[C2:.+]] = constant 2
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[INIT:.+]] = hal.interface.binding.subspan @io::@arg2
//   CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@ret0
//   CHECK-DAG:   %[[M:.+]] = memref.dim %[[LHS]], %[[C0]]
//   CHECK-DAG:   %[[N:.+]] = memref.dim %[[RHS]], %[[C1]]
//   CHECK-DAG:   %[[K:.+]] = memref.dim %[[LHS]], %[[C1]]
//   CHECK-DAG:   %[[WGID_X:.+]] = hal.interface.workgroup.id[0]
//   CHECK-DAG:   %[[WGID_Y:.+]] = hal.interface.workgroup.id[1]
//   CHECK-DAG:   %[[WGCOUNT_X:.+]] = hal.interface.workgroup.count[0]
//   CHECK-DAG:   %[[WGCOUNT_Y:.+]] = hal.interface.workgroup.count[1]
//       CHECK:   %[[OFFSET_Y:.+]] = muli %[[WGID_Y]], %[[C2]]
//       CHECK:   %[[STEP_Y:.+]] = muli %[[WGCOUNT_Y]], %[[C2]]
//       CHECK:   scf.for %{{.+}} = %[[OFFSET_Y]] to %[[M]] step %[[STEP_Y]]
//       CHECK:     %[[OFFSET_X:.+]] = muli %[[WGID_X]], %[[C4]]
//       CHECK:     %[[STEP_X:.+]] = muli %[[WGCOUNT_X]], %[[C4]]
//       CHECK:     scf.for %{{.+}} = %[[OFFSET_X]] to %[[N]] step %[[STEP_X]]

// -----

hal.executable @add attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @add attributes {
      interface = @io, ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    module  {
      func @add() {
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>
        linalg.generic {__internal_linalg_transform__ = "workgroup"} {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%0, %1 : memref<?x?xf32>, memref<?xf32>) outs(%2 : memref<?x?xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
            %3 = addf %arg0, %arg1 : f32
            linalg.yield %3 : f32
          }
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//       CHECK: hal.executable @add
//       CHECK: hal.executable.entry_point @add
//  CHECK-NEXT:   ^{{[a-zA-Z0-9_]+}}(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[WGX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//   CHECK-DAG:     %[[WGY:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//       CHECK:     hal.return %[[WGX]], %[[WGY]], %[[C1]]
