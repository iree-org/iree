// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-convert-to-gpu))" -canonicalize -cse -split-input-file %s | IreeFileCheck %s

hal.executable @add attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @add attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module  attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>} {
      func @add() {
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>
        linalg.generic {
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
//       CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (((s0 * s1) * s2) ceildiv 32)>
//       CHECK: hal.executable @add
//       CHECK: hal.executable.entry_point @add
//  CHECK-NEXT:   ^{{[a-zA-Z0-9_]+}}(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[WGCOUNTX:.+]] = affine.apply #[[MAP]]()[%[[ARG0]], %[[ARG1]], %[[ARG2]]]
//       CHECK:     hal.return %[[WGCOUNTX]], %[[C1]], %[[C1]]
//       CHECK: func @add()
//  CHECK-SAME:   spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@ret0
//   CHECK-DAG:   %[[M:.+]] = memref.dim %[[LHS]], %[[C0]]
//   CHECK-DAG:   %[[N:.+]] = memref.dim %[[LHS]], %[[C1]]
//       CHECK:   %[[UB:.+]] = muli %[[N]], %[[M]]
//   CHECK-DAG:   %[[BID:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BDIM:.+]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[TID:.+]] = "gpu.thread_id"() {dimension = "x"}
//       CHECK:   %[[BOFFSET:.+]] = muli %[[BID]], %[[BDIM]]
//       CHECK:   %[[IV:.+]] = addi %[[BOFFSET]], %[[TID]]
//       CHECK:   %[[COND:.+]] = cmpi slt, %[[IV]], %[[UB]]
//       CHECK:   scf.if %[[COND]] {
//       CHECK:     %[[IV0:.+]] = divi_signed %[[IV]], %[[N]]
//       CHECK:     %[[IV1:.+]] = remi_signed %[[IV]], %[[N]]
//   CHECK-DAG:     %[[V1:.+]] = memref.load %[[LHS]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:     %[[V2:.+]] = memref.load %[[RHS]][%[[IV1]]]
//   CHECK-DAG:     %[[STORE:.+]] = addf %[[V1]], %[[V2]]
//       CHECK:     store %[[STORE]], %[[RESULT]][%[[IV0]], %[[IV1]]]
