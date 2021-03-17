// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.target(iree-codegen-convert-to-gpu))' -canonicalize -cse %s | IreeFileCheck %s

// TODO(GH-4901): Enable this test when linalg on tensors becomes default.
// #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// hal.executable @parallel_4D attributes {sym_visibility = "private"} {
//   hal.interface @legacy_io {
//     hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//     hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
//     hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
//   }
//   hal.executable.target @vulkan, filter="vulkan*" {
//     hal.executable.entry_point @parallel_4D attributes {
//       interface = @legacy_io, ordinal = 0 : i32,
//       signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?x?xf32>,
//         !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
//     module attributes {
//       spv.target_env =
//         #spv.target_env<#spv.vce<v1.3,
//         [Shader], [SPV_KHR_storage_buffer_storage_class]>,
//         {max_compute_workgroup_invocations = 128 : i32,
//          max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
//       func @parallel_4D() {
//         %arg0 = iree.placeholder for "interace buffer"
//           {binding = @legacy_io::@arg0, operand_result_index = 4 : i32} : memref<?x?x?x?xf32>
//         %arg1 = iree.placeholder for "interace buffer"
//           {binding = @legacy_io::@arg1, operand_result_index = 9 : i32} : memref<?x?x?x?xf32>
//         %arg2 = iree.placeholder for "interace buffer"
//           {binding = @legacy_io::@ret0, operand_result_index = 10 : i32} : memref<?x?x?x?xf32>
//         linalg.generic {
//            indexing_maps = [#map0, #map0, #map0],
//            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
//           ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
//          outs(%arg2 : memref<?x?x?x?xf32>) {
//         ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
//           %0 = addf %arg3, %arg4 : f32
//           linalg.yield %0 : f32
//         }
//         return
//       }
//       func private @parallel_4D__num_workgroups__
//         (!shapex.ranked_shape<[?,?,?,?]>, !shapex.ranked_shape<[?,?,?,?]>,
//          !shapex.ranked_shape<[?,?,?,?]>) -> (index, index, index)
//       hal.interface @legacy_io attributes {sym_visibility = "private"} {
//         hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//         hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
//         hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
//       }
//     }
//   }
// }
// // NOCHECK-LABEL: func @parallel_4D
// //  NOCHECK-SAME:   local_size = dense<[32, 1, 1]>
// //   NOCHECK-DAG:     %[[C0:.+]] = constant 0 : index
// //   NOCHECK-DAG:     %[[C1:.+]] = constant 1 : index
// //   NOCHECK-DAG:     %[[C2:.+]] = constant 2 : index
// //   NOCHECK-DAG:     %[[C3:.+]] = constant 3 : index
// //   NOCHECK-DAG:     %[[UB0:.+]] = memref.dim %{{.+}}, %[[C0]]
// //   NOCHECK-DAG:     %[[UB1:.+]] = memref.dim %{{.+}}, %[[C1]]
// //   NOCHECK-DAG:     %[[UB2:.+]] = memref.dim %{{.+}}, %[[C2]]
// //   NOCHECK-DAG:     %[[UB3:.+]] = memref.dim %{{.+}}, %[[C3]]
// //       NOCHECK:     %[[T4:.+]] = muli %[[UB3]], %[[UB2]]
// //       NOCHECK:     %[[T5:.+]] = muli %[[T4]], %[[UB1]]
// //       NOCHECK:     %[[UB:.+]] = muli %[[T5]], %[[UB0]]
// //   NOCHECK-DAG:     %[[BID:.+]] = "gpu.block_id"() {dimension = "x"}
// //   NOCHECK-DAG:     %[[BDIM:.+]] = "gpu.block_dim"() {dimension = "x"}
// //   NOCHECK-DAG:     %[[TID:.+]] = "gpu.thread_id"() {dimension = "x"}
// //       NOCHECK:     %[[BOFFSET:.+]] = muli %[[BID]], %[[BDIM]]
// //       NOCHECK:     %[[IV:.+]] = addi %[[BOFFSET]], %[[TID]]
// //       NOCHECK:     %[[COND:.+]] = cmpi slt, %[[IV]], %[[UB]]
// //       NOCHECK:     scf.if %[[COND]]
// //       NOCHECK:       %[[IV0:.+]] = divi_signed %[[IV]], %[[T5]]
// //       NOCHECK:       %[[T14:.+]] = remi_signed %[[IV]], %[[T5]]
// //       NOCHECK:       %[[IV1:.+]] = divi_signed %[[T14]], %[[T4]]
// //       NOCHECK:       %[[T16:.+]] = remi_signed %[[T14]], %[[T4]]
// //       NOCHECK:       %[[IV2:.+]] = divi_signed %[[T16]], %[[UB3]]
// //       NOCHECK:       %[[IV3:.+]] = remi_signed %[[T16]], %[[UB3]]
// //       NOCHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// //       NOCHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// //       NOCHECK:       store %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
hal.executable @parallel_4D_static attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @parallel_4D_static attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?x?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3,
        [Shader], [SPV_KHR_storage_buffer_storage_class]>,
        {max_compute_workgroup_invocations = 128 : i32,
         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @parallel_4D_static() {
        %arg0 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<3x4x5x6xf32>
        %arg1 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<3x4x5x6xf32>
        %arg2 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<3x4x5x6xf32>
        linalg.generic {
           indexing_maps = [#map0, #map0, #map0],
           iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
          ins(%arg0, %arg1 : memref<3x4x5x6xf32>, memref<3x4x5x6xf32>)
         outs(%arg2 : memref<3x4x5x6xf32>) {
        ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
          %0 = addf %arg3, %arg4 : f32
          linalg.yield %0 : f32
        }
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//       CHECK: hal.executable.entry_point @parallel_4D_static
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[C12:.+]] = constant 12 : index
//       CHECK:   hal.return %[[C12]], %[[C1]], %[[C1]]
//       CHECK: func @parallel_4D_static()
//  CHECK-SAME:   local_size = dense<[32, 1, 1]>
//   CHECK-DAG:     %[[C360:.+]] = constant 360 : index
//   CHECK-DAG:     %[[C120:.+]] = constant 120 : index
//   CHECK-DAG:     %[[C30:.+]] = constant 30 : index
//   CHECK-DAG:     %[[C6:.+]] = constant 6 : index
//   CHECK-DAG:     %[[BID:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:     %[[BDIM:.+]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:     %[[TID:.+]] = "gpu.thread_id"() {dimension = "x"}
//       CHECK:     %[[BOFFSET:.+]] = muli %[[BID]], %[[BDIM]]
//       CHECK:     %[[IV:.+]] = addi %[[BOFFSET]], %[[TID]]
//       CHECK:     %[[COND:.+]] = cmpi slt, %[[IV]], %[[C360]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       %[[IV0:.+]] = divi_signed %[[IV]], %[[C120]]
//       CHECK:       %[[T14:.+]] = remi_signed %[[IV]], %[[C120]]
//       CHECK:       %[[IV1:.+]] = divi_signed %[[T14]], %[[C30]]
//       CHECK:       %[[T16:.+]] = remi_signed %[[T14]], %[[C30]]
//       CHECK:       %[[IV2:.+]] = divi_signed %[[T16]], %[[C6]]
//       CHECK:       %[[IV3:.+]] = remi_signed %[[T16]], %[[C6]]
//       CHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       store %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]

// -----

#map0 = affine_map<() -> ()>
#accesses = [#map0, #map0, #map0]
#trait = {
  indexing_maps = #accesses,
  iterator_types = []
}

hal.executable @scalar_add attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @scalar_add attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.tensor<readonly:f32>, !flow.dispatch.tensor<readonly:f32>,
        !flow.dispatch.tensor<writeonly:f32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3,
        [Shader], [SPV_KHR_storage_buffer_storage_class]>,
        {max_compute_workgroup_invocations = 128 : i32,
         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @scalar_add() attributes {hal.num_workgroups_fn = @scalar_add__num_workgroups__} {
        %arg0 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<f32>
        %arg1 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<f32>
        %arg2 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<f32>
        linalg.generic #trait
          ins(%arg0, %arg1 : memref<f32>, memref<f32>)
         outs(%arg2 : memref<f32>) {
        ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
          %0 = addf %arg3, %arg4 : f32
          linalg.yield %0 : f32
         }
         return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//       CHECK: hal.executable.entry_point @scalar_add
//       CHECK:   %[[C1:.+]] = constant 1
//       CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]]
// CHECK-LABEL: func @scalar_add()
//       CHECK:     load
//  CHECK-NEXT:     load
//  CHECK-NEXT:     addf
//  CHECK-NEXT:     store
//  CHECK-NEXT:     return

// -----

// TODO(GH-4901): Convert these tests back to use dynamic shapes when linalg on tensors becomes default.
hal.executable @reduce_sum attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @reduce_sum attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.tensor<readonly:40x50x75xf32>, !flow.dispatch.tensor<readonly:f32>,
        !flow.dispatch.tensor<writeonly:40xf32>) -> ()}
    module {
      func @reduce_sum() {
        %arg0 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<40x50x75xf32>
        %arg1 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<f32>
        %arg2 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<40xf32>
        linalg.indexed_generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                           affine_map<(d0, d1, d2) -> ()>,
                           affine_map<(d0, d1, d2) -> (d0)>],
          iterator_types = ["parallel", "reduction", "reduction"]}
          ins(%arg0, %arg1 : memref<40x50x75xf32>, memref<f32>)
          outs(%arg2 : memref<40xf32>) {
        ^bb0(%arg3: index, %arg4: index, %arg5: index,
          %arg6: f32, %arg7: f32, %arg8: f32):   // no predecessors
          %c0 = constant 0 : index
          %0 = cmpi eq, %arg5, %c0 : index
          %1 = cmpi eq, %arg4, %c0 : index
          %2 = and %0, %1 : i1
          %3 = select %2, %arg7, %arg8 : f32
          %4 = addf %arg6, %3 : f32
          linalg.yield %4 : f32
        }
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//       CHECK: hal.executable.entry_point @reduce_sum
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//       CHECK:   hal.return %[[C2]], %[[C1]], %[[C1]]
//       CHECK: func @reduce_sum
//  CHECK-SAME:   local_size = dense<[32, 1, 1]> : vector<3xi32>
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C40:.+]] = constant 40 : index
//   CHECK-DAG:     %[[C50:.+]] = constant 50 : index
//   CHECK-DAG:     %[[C75:.+]] = constant 75 : index
//       CHECK:     %[[COND:.+]] = cmpi slt, %{{.+}}, %[[C40]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       scf.for %[[IV0:.+]] = %{{.+}} to %[[C50]]
//       CHECK:         scf.for %[[IV1:.+]] = %{{.+}} to %[[C75]]
//   CHECK-DAG:           %[[ISZERO0:.+]] = cmpi eq, %[[IV0]], %[[C0]]
//   CHECK-DAG:           %[[ISZERO1:.+]] = cmpi eq, %[[IV1]], %[[C0]]

// -----

#map0 = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0, s1] -> (8, s1 - s0 * 8)>
#map2 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map3 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>

hal.executable @matmul attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @matmul attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?x?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
                        {max_compute_workgroup_invocations = 128 : i32,
                         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @matmul() {
        %arg0 = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0} : memref<?x?xf32>
        %arg1 = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1} : memref<?x?xf32>
        %arg2 = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0} : memref<?x?xf32>
        %c4 = constant 4 : index
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %0 = memref.dim %arg0, %c1 : memref<?x?xf32>
        %1 = "gpu.block_id"() {dimension = "x"} : () -> index
        %2 = "gpu.block_id"() {dimension = "y"} : () -> index
        scf.for %arg3 = %c0 to %0 step %c4 {
          %3 = affine.apply #map0()[%2]
          %4 = memref.dim %arg0, %c0 : memref<?x?xf32>
          %5 = affine.min #map1()[%2, %4]
          %6 = affine.min #map2(%arg3)[%0]
          %7 = memref.subview %arg0[%3, %arg3] [%5, %6] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map3>
          %8 = memref.dim %arg1, %c0 : memref<?x?xf32>
          %9 = affine.min #map2(%arg3)[%8]
          %10 = affine.apply #map0()[%1]
          %11 = memref.dim %arg1, %c1 : memref<?x?xf32>
          %12 = affine.min #map1()[%1, %11]
          %13 = memref.subview %arg1[%arg3, %10] [%9, %12] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map3>
          %14 = memref.dim %arg2, %c0 : memref<?x?xf32>
          %15 = affine.min #map1()[%2, %14]
          %16 = memref.dim %arg2, %c1 : memref<?x?xf32>
          %17 = affine.min #map1()[%1, %16]
          %18 = memref.subview %arg2[%3, %10] [%15, %17] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map3>
          linalg.matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%7, %13 : memref<?x?xf32, #map3>, memref<?x?xf32, #map3>)
           outs(%18 : memref<?x?xf32, #map3>)
        }
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
// CHECK-LABEL: func @matmul
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//       CHECK:   scf.for
//   CHECK-DAG:     %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:     %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//       CHECK:     %[[INBOUNDY:.+]] = cmpi slt, %[[TIDY]], %{{.*}}
//       CHECK:     %[[INBOUNDX:.+]] = cmpi slt, %[[TIDX]], %{{.*}}
//       CHECK:     %[[COND:.+]] = and %[[INBOUNDY]], %[[INBOUNDX]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       scf.for %{{.+}} = %[[C0]] to %{{.*}} step %[[C1]]
//   CHECK-NOT:         linalg.matmul

// -----


hal.executable @conv_1d attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @conv_1d attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<3x8x1xf32>, tensor<3x1x1xf32>) -> tensor<3x6x1xf32>}
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>}  {
      func @conv_1d() attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
        %cst = constant 0.000000e+00 : f32
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<3x6x1xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<3x8x1xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x1x1xf32>
        %3 = "gpu.block_id"() {dimension = "x"} : () -> index
        %4 = "gpu.block_id"() {dimension = "y"} : () -> index
        %5 = "gpu.block_id"() {dimension = "z"} : () -> index
        %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%4]
        %7 = affine.min affine_map<()[s0] -> (6, s0 * -4 + 8)>()[%4]
        %8 = memref.subview %1[%5, %6, 0] [1, %7, 1] [1, 1, 1] : memref<3x8x1xf32> to memref<1x?x1xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 8 + s0 + d1 + d2)>>
        %9 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%3]
        %10 = affine.min affine_map<()[s0] -> (32, s0 * -32 + 1)>()[%3]
        %11 = memref.subview %2[0, 0, %9] [3, 1, %10] [1, 1, 1] : memref<3x1x1xf32> to memref<3x1x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 + s0 + d1 + d2)>>
        %12 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%4]
        %13 = affine.min affine_map<()[s0] -> (4, s0 * -4 + 6)>()[%4]
        %14 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%3]
        %15 = affine.min affine_map<()[s0] -> (32, s0 * -32 + 1)>()[%3]
        %16 = memref.subview %0[%5, %12, %14] [1, %13, %15] [1, 1, 1] : memref<3x6x1xf32> to memref<1x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 6 + s0 + d1 + d2)>>
        %17 = memref.subview %0[%5, %12, %9] [1, %13, %10] [1, 1, 1] : memref<3x6x1xf32> to memref<1x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 6 + s0 + d1 + d2)>>
        linalg.conv_1d_input_nwc_filter_wcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%8, %11 : memref<1x?x1xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 8 + s0 + d1 + d2)>>, memref<3x1x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 + s0 + d1 + d2)>>) outs(%16 : memref<1x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 6 + s0 + d1 + d2)>>)
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//       CHECK: func @conv_1d
//       CHECK: scf.if
//  CHECK-NEXT:   scf.for
//   CHECK-NOT:     linalg.conv_1d_input_nwc_filter_wcf

// -----

#map0 = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0)[s0] -> (1, -d0 + s0)>
#map3 = affine_map<(d0)[s0, s1] -> (s0 + 4, -d0 + s1)>
#map4 = affine_map<(d0)[s0, s1] -> (s0 + 32, -d0 + s1)>
#map5 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
#map6 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map7 = affine_map<(d0)[s0] -> (32, -d0 + s0)>


hal.executable @conv_no_padding attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="vulkan*" {
    hal.executable.entry_point @conv_no_padding attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?x?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
                        {max_compute_workgroup_invocations = 128 : i32,
                         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @conv_no_padding() {
        %arg0 = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0} : memref<?x?x?x?xf32>
        %arg1 = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1} : memref<?x?x?x?xf32>
        %arg2 = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0} : memref<?x?x?x?xf32>
        %c2 = constant 2 : index
        %c0 = constant 0 : index
        %c3 = constant 3 : index
        %c1 = constant 1 : index
        %0 = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
        %1 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
        %2 = memref.dim %arg1, %c0 : memref<?x?x?x?xf32>
        %3 = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
        %4 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
        %5 = "gpu.block_id"() {dimension = "x"} : () -> index
        %6 = "gpu.grid_dim"() {dimension = "x"} : () -> index
        %7 = "gpu.block_id"() {dimension = "y"} : () -> index
        %8 = "gpu.grid_dim"() {dimension = "y"} : () -> index
        %9 = "gpu.block_id"() {dimension = "z"} : () -> index
        %10 = "gpu.grid_dim"() {dimension = "z"} : () -> index
        %11 = affine.apply #map0()[%7]
        %12 = affine.apply #map0()[%8]
        %13 = affine.apply #map1()[%5]
        %14 = affine.apply #map1()[%6]
        scf.parallel (%arg3, %arg4, %arg5) = (%9, %11, %13) to (%2, %3, %4) step (%10, %12, %14) {
          %15 = affine.min #map2(%arg3)[%2]
          %16 = memref.dim %arg1, %c1 : memref<?x?x?x?xf32>
          %17 = affine.min #map3(%arg4)[%0, %16]
          %18 = memref.dim %arg1, %c2 : memref<?x?x?x?xf32>
          %19 = affine.min #map4(%arg5)[%1, %18]
          %20 = memref.dim %arg1, %c3 : memref<?x?x?x?xf32>
          %21 = memref.subview %arg1[%arg3, %arg4, %arg5, 0] [%15, %17, %19, %20] [1, 1, 1, 1]
                  : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map5>
          %22 = memref.dim %arg2, %c0 : memref<?x?x?x?xf32>
          %23 = affine.min #map2(%arg3)[%22]
          %24 = affine.min #map6(%arg4)[%3]
          %25 = affine.min #map7(%arg5)[%4]
          %26 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>
          %27 = memref.subview %arg2[%arg3, %arg4, %arg5, 0] [%23, %24, %25, %26] [1, 1, 1, 1]
                  : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map5>
          linalg.conv_2d_input_nhwc_filter_hwcf {
            __internal_linalg_transform__ = "workgroup",
            dilations = dense<1> : tensor<2xi64>,
            strides = dense<2> : tensor<2xi64>}
             ins(%21, %arg0 : memref<?x?x?x?xf32, #map5>, memref<?x?x?x?xf32>)
            outs(%27 : memref<?x?x?x?xf32, #map5>)
          scf.yield
        }
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @conv_no_padding
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0}
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1}
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0}
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[C2:.+]] = constant 2
//   CHECK-DAG:   %[[N:.+]] = memref.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[P:.+]] = memref.dim %[[RET0]], %[[C1]]
//   CHECK-DAG:   %[[Q:.+]] = memref.dim %[[RET0]], %[[C2]]
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[NBLOCKSX:.+]] = "gpu.grid_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[NBLOCKSY:.+]] = "gpu.grid_dim"() {dimension = "y"}
//   CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//   CHECK-DAG:   %[[NBLOCKSZ:.+]] = "gpu.grid_dim"() {dimension = "z"}
//       CHECK:   %[[BOFFSETY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[BSTEPY:.+]] = affine.apply #[[MAP0]]()[%[[NBLOCKSY]]]
//       CHECK:   %[[BOFFSETX:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//       CHECK:   %[[BSTEPX:.+]] = affine.apply #[[MAP1]]()[%[[NBLOCKSX]]]
//       CHECK:   scf.for %[[IV3:.+]] = %[[BIDZ]] to %[[N]] step %[[NBLOCKSZ]]
//       CHECK:     scf.for %[[IV4:.+]] = %[[BOFFSETY]] to %[[P]] step %[[BSTEPY]]
//       CHECK:       scf.for %[[IV5:.+]] = %[[BOFFSETX]] to %[[Q]] step %[[BSTEPX]]
//       CHECK:         %[[SV1:.+]] = memref.subview %[[ARG1]][%[[IV3]], %[[IV4]], %[[IV5]], 0]
//       CHECK:         %[[SV2:.+]] = memref.subview %[[RET0]][%[[IV3]], %[[IV4]], %[[IV5]], 0]
//   CHECK-DAG:         %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:         %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//   CHECK-DAG:         %[[TIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//       CHECK:         %[[C1:.+]] = cmpi slt, %[[TIDZ]], %{{.*}}
//       CHECK:         %[[C2:.+]] = cmpi slt, %[[TIDY]], %{{.*}}
//       CHECK:         %[[C3:.+]] = and %[[C1]], %[[C2]]
//       CHECK:         %[[C4:.+]] = cmpi slt, %[[TIDX]], %{{.*}}
//       CHECK:         %[[COND:.+]] = and %[[C3]], %[[C4]]
//       CHECK:         scf.if %[[COND]]
//       CHECK:           scf.for
//       CHECK:             scf.for
//       CHECK:               scf.for
//       CHECK:                 scf.for
//   CHECK-NOT:                   linalg.conv_2d_input_nhwc_filter_hwcf

// -----

hal.executable @conv_3d attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @conv_3d attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<2x8x8x8x3xf32>, tensor<2x2x2x3x2xf32>) -> tensor<2x7x7x7x2xf32>}
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>}  {
      func @conv_3d() attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
        %cst = constant 0.000000e+00 : f32
        %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x7x7x7x2xf32>
        %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x8x8x8x3xf32>
        %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<2x2x2x3x2xf32>
        %3 = "gpu.block_id"() {dimension = "x"} : () -> index
        %4 = "gpu.block_id"() {dimension = "y"} : () -> index
        %5 = "gpu.block_id"() {dimension = "z"} : () -> index
        %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%4]
        %7 = affine.min affine_map<()[s0] -> (5, s0 * -4 + 8)>()[%4]
        %8 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%3]
        %9 = affine.min affine_map<()[s0] -> (33, s0 * -32 + 8)>()[%3]
        %10 = memref.subview %1[%5, %6, %8, 0, 0] [1, %7, %9, 8, 3] [1, 1, 1, 1, 1] : memref<2x8x8x8x3xf32> to memref<1x?x?x8x3xf32, affine_map<(d0, d1, d2, d3, d4)[s0] -> (d0 * 1536 + s0 + d1 * 192 + d2 * 24 + d3 * 3 + d4)>>
        %11 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%4]
        %12 = affine.min affine_map<()[s0] -> (4, s0 * -4 + 7)>()[%4]
        %13 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%3]
        %14 = affine.min affine_map<()[s0] -> (32, s0 * -32 + 7)>()[%3]
        %15 = memref.subview %0[%5, %11, %13, 0, 0] [1, %12, %14, 7, 2] [1, 1, 1, 1, 1] : memref<2x7x7x7x2xf32> to memref<1x?x?x7x2xf32, affine_map<(d0, d1, d2, d3, d4)[s0] -> (d0 * 686 + s0 + d1 * 98 + d2 * 14 + d3 * 2 + d4)>>
        %16 = memref.subview %0[%5, %11, %13, 0, 0] [1, %12, %14, 7, 2] [1, 1, 1, 1, 1] : memref<2x7x7x7x2xf32> to memref<1x?x?x7x2xf32, affine_map<(d0, d1, d2, d3, d4)[s0] -> (d0 * 686 + s0 + d1 * 98 + d2 * 14 + d3 * 2 + d4)>>
        linalg.conv_3d_input_ndhwc_filter_dhwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%10, %2 : memref<1x?x?x8x3xf32, affine_map<(d0, d1, d2, d3, d4)[s0] -> (d0 * 1536 + s0 + d1 * 192 + d2 * 24 + d3 * 3 + d4)>>, memref<2x2x2x3x2xf32>) outs(%15 : memref<1x?x?x7x2xf32, affine_map<(d0, d1, d2, d3, d4)[s0] -> (d0 * 686 + s0 + d1 * 98 + d2 * 14 + d3 * 2 + d4)>>)
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//       CHECK: func @conv_3d
//       CHECK: scf.if
//  CHECK-NEXT:   scf.for
//  CHECK-NEXT:     scf.for
//  CHECK-NEXT:       scf.for
//  CHECK-NEXT:         scf.for
//  CHECK-NEXT:           scf.for
//  CHECK-NEXT:             scf.for
//   CHECK-NOT:               linalg.conv_3d_input_ndhwc_filter_dhwcf

// -----

#map0 = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (6, s0 * -4 + 16)>
#map2 = affine_map<()[s0] -> (s0 * 32)>
#map3 = affine_map<()[s0] -> (35, s0 * -32 + 16)>
#map4 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1536 + s0 + d1 * 96 + d2 * 6 + d3)>
#map5 = affine_map<()[s0] -> (4, s0 * -4 + 14)>
#map6 = affine_map<()[s0] -> (32, s0 * -32 + 13)>
#map7 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 1092 + s0 + d1 * 78 + d2 * 6 + d3)>
module  {
  hal.executable @pooling_nhwc_max attributes {sym_visibility = "private"} {
    hal.interface @legacy_io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vulkan, filter="vulkan*" {
      hal.executable.entry_point @pooling_nhwc_max attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (!flow.dispatch.tensor<readonly:2x16x16x6xf32>, !flow.dispatch.tensor<readonly:1x3x4x2xf32>, !flow.dispatch.tensor<writeonly:2x14x13x5xf32>) -> ()} {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):  // no predecessors
        %c4 = constant 4 : index
        %c1 = constant 1 : index
        hal.return %c1, %c4, %c1 : index, index, index
      }
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>}  {
        func @pooling_nhwc_max() attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
          %0 = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<2x16x16x6xf32>
          %1 = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<3x4xf32>
          %2 = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<2x14x13x6xf32>
          %3 = "gpu.block_id"() {dimension = "x"} : () -> index
          %4 = "gpu.block_id"() {dimension = "y"} : () -> index
          %5 = affine.apply #map0()[%4]
          %6 = affine.min #map1()[%4]
          %7 = affine.apply #map2()[%3]
          %8 = affine.min #map3()[%3]
          %9 = memref.subview %0[0, %5, %7, 0] [2, %6, %8, 6] [1, 1, 1, 1] : memref<2x16x16x6xf32> to memref<2x?x?x6xf32, #map4>
          %10 = affine.min #map5()[%4]
          %11 = affine.min #map6()[%3]
          %12 = memref.subview %2[0, %5, %7, 0] [2, %10, %11, 6] [1, 1, 1, 1] : memref<2x14x13x6xf32> to memref<2x?x?x6xf32, #map7>
          linalg.pooling_nhwc_max {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%9, %1 : memref<2x?x?x6xf32, #map4>, memref<3x4xf32>) outs(%12 : memref<2x?x?x6xf32, #map7>)
          return
        }
        hal.interface @legacy_io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 32)>
//       CHECK: func @pooling_nhwc_max
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1, operand_result_index = 1 : i32}
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0, operand_result_index = 2 : i32}
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//       CHECK:   %[[IV1:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//       CHECK:   %[[IV2:.+]] = affine.apply #[[MAP2]]()[%[[BIDX]]]
//       CHECK:   %[[SV1:.+]] = memref.subview %[[ARG0]][0, %[[IV1]], %[[IV2]], 0]
//       CHECK:   %[[SV2:.+]] = memref.subview %[[RET0]][0, %[[IV1]], %[[IV2]], 0]
//   CHECK-DAG:   %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:   %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//   CHECK-DAG:   %[[TIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//       CHECK:   %[[C1:.+]] = cmpi slt, %[[TIDZ]], %{{.*}}
//       CHECK:   %[[C2:.+]] = cmpi slt, %[[TIDY]], %{{.*}}
//       CHECK:   %[[C3:.+]] = and %[[C1]], %[[C2]] : i1
//       CHECK:   %[[C4:.+]] = cmpi slt, %[[TIDX]], %{{.*}}
//       CHECK:   %[[COND:.+]] = and %[[C3]], %[[C4]]
//       CHECK:   scf.if %[[COND]]
//       CHECK:     scf.for
//       CHECK:       scf.for
//       CHECK:         scf.for
//   CHECK-NOT:           linalg.pooling_nhwc_max
