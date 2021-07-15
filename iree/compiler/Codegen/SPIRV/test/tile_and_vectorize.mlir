// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-tile-and-vectorize,canonicalize,cse))" %s | IreeFileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0, s1] -> (8, s1 - s0 * 8)>
#map2 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map3 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>

hal.executable @matmul attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @matmul attributes {interface = @io, ordinal = 0 : index}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
                        {max_compute_workgroup_invocations = 128 : i32,
                         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @matmul() {
        %c0 = constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?xf32>
        %arg2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>
        %c4 = constant 4 : index
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
      hal.interface @io attributes {sym_visibility = "private"} {
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
//   CHECK-DAG:     %[[BDIMX:.+]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:     %[[BDIMY:.+]] = "gpu.block_dim"() {dimension = "y"}
//       CHECK:     scf.for %{{.+}} = %[[TIDY]] to %{{.*}} step %[[BDIMY]]
//       CHECK:       scf.for %{{.+}} = %[[TIDX]] to %{{.*}} step %[[BDIMX]]
//       CHECK:         scf.for %{{.+}} = %[[C0]] to %{{.*}} step %[[C1]]
//   CHECK-NOT:           linalg.matmul

// -----

hal.executable @conv_1d attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @conv_1d attributes {interface = @io, ordinal = 0 : index}
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>}  {
      func @conv_1d() attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
        %cst = constant 0.000000e+00 : f32
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<3x6x1xf32>
        %1 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<3x8x1xf32>
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<3x1x1xf32>
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
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// CHECK-LABEL: func @conv_1d
//   CHECK-DAG: %[[C0:.+]] = constant 0 : index
//   CHECK-DAG: %[[C1:.+]] = constant 1 : index
//   CHECK-DAG: %[[C3:.+]] = constant 3 : index
//       CHECK: %[[RET:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK: %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//       CHECK: %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//       CHECK: %[[ARG0SV1:.+]] = memref.subview %[[ARG0]]
//       CHECK: %[[ARG1SV1:.+]] = memref.subview %[[ARG1]]
//       CHECK: %[[RETSV1:.+]] = memref.subview %[[RET]]
//       CHECK: %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//       CHECK: %[[BDIMX:.+]] = "gpu.block_dim"() {dimension = "x"}
//       CHECK: %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//       CHECK: %[[BDIMY:.+]] = "gpu.block_dim"() {dimension = "y"}
//       CHECK: %[[TIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//       CHECK: scf.for %[[IV0:.+]] = %[[TIDY]] to %{{.*}} step %[[BDIMY]]
//       CHECK:   scf.for %[[IV1:.+]] = %[[TIDX]] to %{{.*}} step %[[BDIMX]]
//       CHECK:     %[[ARG0SV2:.+]] = memref.subview %[[ARG0SV1]][%[[TIDZ]], %[[IV0]], 0] [1, %{{.+}}, 1]
//       CHECK:     %[[ARG1SV2:.+]] = memref.subview %[[ARG1SV1]][0, 0, %[[IV1]]] [3, 1, 1]
//       CHECK:     %[[RETSV2:.+]] = memref.subview %[[RETSV1]][%[[TIDZ]], %[[IV0]], %[[IV1]]] [1, 1, 1]
//       CHECK:     scf.for %[[IV2:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
//       CHECK:       memref.load %[[ARG0SV2]][%[[C0]], %[[IV2]], %[[C0]]]
//       CHECK:       memref.load %[[ARG1SV2]][%[[IV2]], %[[C0]], %[[C0]]]
//       CHECK:       memref.load %[[RETSV2]][%[[C0]], %[[C0]], %[[C0]]]
//       CHECK:       memref.store %{{.+}}, %[[RETSV2]][%[[C0]], %[[C0]], %[[C0]]]


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
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @conv_no_padding attributes {interface = @io, ordinal = 0 : index}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
                        {max_compute_workgroup_invocations = 128 : i32,
                         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @conv_no_padding() {
        %c0 = constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?x?x?xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?x?x?xf32>
        %arg2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?x?x?xf32>
        %c2 = constant 2 : index
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
        scf.for %arg3 = %9 to %2 step %10 {
          scf.for %arg4 = %11 to %3 step %12 {
            scf.for %arg5 = %13 to %4 step %14 {
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
            }
          }
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
//     CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//     CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//         CHECK: func @conv_no_padding
//     CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//     CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//     CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//     CHECK-DAG:   %[[C1:.+]] = constant 1
//     CHECK-DAG:   %[[C2:.+]] = constant 2
//     CHECK-DAG:   %[[N:.+]] = memref.dim %[[ARG1]], %[[C0]]
//     CHECK-DAG:   %[[P:.+]] = memref.dim %[[RET0]], %[[C1]]
//     CHECK-DAG:   %[[Q:.+]] = memref.dim %[[RET0]], %[[C2]]
//     CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//     CHECK-DAG:   %[[NBLOCKSX:.+]] = "gpu.grid_dim"() {dimension = "x"}
//     CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//     CHECK-DAG:   %[[NBLOCKSY:.+]] = "gpu.grid_dim"() {dimension = "y"}
//     CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//     CHECK-DAG:   %[[NBLOCKSZ:.+]] = "gpu.grid_dim"() {dimension = "z"}
//         CHECK:   %[[BOFFSETY:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//         CHECK:   %[[BSTEPY:.+]] = affine.apply #[[MAP0]]()[%[[NBLOCKSY]]]
//         CHECK:   %[[BOFFSETX:.+]] = affine.apply #[[MAP1]]()[%[[BIDX]]]
//         CHECK:   %[[BSTEPX:.+]] = affine.apply #[[MAP1]]()[%[[NBLOCKSX]]]
//         CHECK:   scf.for %[[IV3:.+]] = %[[BIDZ]] to %[[N]] step %[[NBLOCKSZ]]
//         CHECK:     scf.for %[[IV4:.+]] = %[[BOFFSETY]] to %[[P]] step %[[BSTEPY]]
//         CHECK:       scf.for %[[IV5:.+]] = %[[BOFFSETX]] to %[[Q]] step %[[BSTEPX]]
//         CHECK:         %[[SV1:.+]] = memref.subview %[[ARG1]][%[[IV3]], %[[IV4]], %[[IV5]], 0]
//         CHECK:         %[[SV2:.+]] = memref.subview %[[RET0]][%[[IV3]], %[[IV4]], %[[IV5]], 0]
//     CHECK-DAG:         %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//     CHECK-DAG:         %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//     CHECK-DAG:         %[[TIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//     CHECK-DAG:         %[[BDIMX:.+]] = "gpu.block_dim"() {dimension = "x"}
//     CHECK-DAG:         %[[BDIMY:.+]] = "gpu.block_dim"() {dimension = "y"}
//     CHECK-DAG:         %[[BDIMZ:.+]] = "gpu.block_dim"() {dimension = "z"}
//         CHECK:         scf.for %{{.+}} = %[[TIDZ]] to %{{.*}} step %[[BDIMZ]]
//         CHECK:           scf.for %{{.+}} = %[[TIDY]] to %{{.*}} step %[[BDIMY]]
//         CHECK:             scf.for %{{.+}} = %[[TIDX]] to %{{.*}} step %[[BDIMX]]
// CHECK-COUNT-3:               scf.for
//     CHECK-NOT:               linalg.conv_2d_input_nhwc_filter_hwcf

// -----

hal.executable @conv_3d attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @conv_3d attributes {interface = @io, ordinal = 0 : index}
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, SwiftShader:CPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 16384 : i32, max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>, subgroup_size = 4 : i32}>}  {
      func @conv_3d() attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
        %cst = constant 0.000000e+00 : f32
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<2x7x7x7x2xf32>
        %1 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<2x8x8x8x3xf32>
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<2x2x2x3x2xf32>
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
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//   CHECK-LABEL: func @conv_3d
//     CHECK-DAG:         %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//     CHECK-DAG:         %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//     CHECK-DAG:         %[[TIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//     CHECK-DAG:         %[[BDIMX:.+]] = "gpu.block_dim"() {dimension = "x"}
//     CHECK-DAG:         %[[BDIMY:.+]] = "gpu.block_dim"() {dimension = "y"}
//     CHECK-DAG:         %[[BDIMZ:.+]] = "gpu.block_dim"() {dimension = "z"}
//         CHECK:         scf.for %{{.+}} = %[[TIDZ]] to %{{.*}} step %[[BDIMZ]]
//         CHECK:           scf.for %{{.+}} = %[[TIDY]] to %{{.*}} step %[[BDIMY]]
//         CHECK:             scf.for %{{.+}} = %[[TIDX]] to %{{.*}} step %[[BDIMX]]
// CHECK-COUNT-5:               scf.for
//     CHECK-NOT:               linalg.conv_3d_input_ndhwc_filter_dhwcf

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
    hal.interface @io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.variant @vulkan, target="vulkan" {
      hal.executable.entry_point @pooling_nhwc_max attributes {interface = @io, ordinal = 0 : index} {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):  // no predecessors
        %c4 = constant 4 : index
        %c1 = constant 1 : index
        hal.return %c1, %c4, %c1 : index, index, index
      }
      module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>}  {
        func @pooling_nhwc_max() attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
          %c0 = constant 0 : index
          %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<2x16x16x6xf32>
          %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<3x4xf32>
          %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<2x14x13x6xf32>
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
        hal.interface @io attributes {sym_visibility = "private"} {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
}

//     CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
//     CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 32)>
//         CHECK: func @pooling_nhwc_max
//     CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//     CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//     CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//     CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//     CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//         CHECK:   %[[IV1:.+]] = affine.apply #[[MAP0]]()[%[[BIDY]]]
//         CHECK:   %[[IV2:.+]] = affine.apply #[[MAP2]]()[%[[BIDX]]]
//         CHECK:   %[[SV1:.+]] = memref.subview %[[ARG0]][0, %[[IV1]], %[[IV2]], 0]
//         CHECK:   %[[SV2:.+]] = memref.subview %[[RET0]][0, %[[IV1]], %[[IV2]], 0]
//     CHECK-DAG:   %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//     CHECK-DAG:   %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//     CHECK-DAG:   %[[TIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//     CHECK-DAG:   %[[BDIMX:.+]] = "gpu.block_dim"() {dimension = "x"}
//     CHECK-DAG:   %[[BDIMY:.+]] = "gpu.block_dim"() {dimension = "y"}
//     CHECK-DAG:   %[[BDIMZ:.+]] = "gpu.block_dim"() {dimension = "z"}
//         CHECK:   scf.for %{{.+}} = %[[TIDZ]] to %{{.*}} step %[[BDIMZ]]
//         CHECK:     scf.for %{{.+}} = %[[TIDY]] to %{{.*}} step %[[BDIMY]]
//         CHECK:       scf.for %{{.+}} = %[[TIDX]] to %{{.*}} step %[[BDIMX]]
// CHECK-COUNT-3:         scf.for
//     CHECK-NOT:           linalg.pooling_nhwc_max
