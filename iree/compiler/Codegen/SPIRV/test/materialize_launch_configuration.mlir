// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-concretize-workgroup-tiles))" -canonicalize -cse -split-input-file %s | IreeFileCheck %s

hal.executable @matmul_tensors attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @matmul_tensors attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>} {
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
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
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
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C16:.+]] = constant 16 : index
//   CHECK-DAG:   %[[C8:.+]] = constant 8 : index
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
//       CHECK:   %[[OFFSET_Y:.+]] = muli %[[WGID_Y]], %[[C8]]
//       CHECK:   %[[STEP_Y:.+]] = muli %[[WGCOUNT_Y]], %[[C8]]
//       CHECK:   scf.for %{{.+}} = %[[OFFSET_Y]] to %[[M]] step %[[STEP_Y]]
//       CHECK:     %[[OFFSET_X:.+]] = muli %[[WGID_X]], %[[C16]]
//       CHECK:     %[[STEP_X:.+]] = muli %[[WGCOUNT_X]], %[[C16]]
//       CHECK:     scf.for %{{.+}} = %[[OFFSET_X]] to %[[N]] step %[[STEP_X]]
