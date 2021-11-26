// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s
hal.executable private @static_1d_sort  {
  hal.interface @io {
    hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @static_1d_sort attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      builtin.func @static_1d_sort() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_rw_external[%c0] : !flow.dispatch.tensor<readwrite:1000xi32>
        %1 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:1000xi32> -> tensor<1000xi32>
        %2 = iree_linalg_ext.sort dimension(0) {__internal_linalg_transform__ = "workgroup"} outs(%1 : tensor<1000xi32>)  {
        ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
          %3 = arith.cmpi slt, %arg0, %arg1 : i32
          iree_linalg_ext.yield %3 : i1
        } -> tensor<1000xi32>
        flow.dispatch.tensor.store %2, %0, offsets = [], sizes = [], strides = [] : tensor<1000xi32> -> !flow.dispatch.tensor<readwrite:1000xi32>
        return
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
      }
    }
  }
}

// Check that the workgroup count and size are (1, 1, 1) for serializing the computation.

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = []>
//       CHECK: hal.executable.entry_point public @static_1d_sort
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [1 : index, 1 : index, 1 : index]
//  CHECK-NEXT: ^{{.+}}(%{{.+}}: index, %{{.+}}: index, %{{.+}}: index):
//  CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
//  CHECK-NEXT:   hal.return %[[ONE]], %[[ONE]], %[[ONE]]

//       CHECK: func @static_1d_sort()
//       CHECK:   iree_linalg_ext.sort
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

hal.executable private @static_3d_sort  {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @static_3d_sort attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      builtin.func @static_3d_sort() {
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<64x32x128xi32>
        %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : memref<64x32x128xi32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %2 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
        %3 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
        scf.for %arg0 = %2 to %c64 step %3 {
          %4 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 64)>(%arg0)[%workgroup_size_y]
          %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_id_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_count_x]
          scf.for %arg1 = %5 to %c128 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg1)[%workgroup_size_x]
            %8 = memref.subview %0[%arg0, 0, %arg1] [%4, 32, %7] [1, 1, 1] : memref<64x32x128xi32> to memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            %9 = memref.cast %8 : memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>> to memref<?x?x?xi32>
            %10 = memref.subview %1[%arg0, 0, %arg1] [%4, 32, %7] [1, 1, 1] : memref<64x32x128xi32> to memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            linalg.copy(%9, %10) : memref<?x?x?xi32>, memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>
            iree_linalg_ext.sort dimension(1) {__internal_linalg_transform__ = "workgroup"} outs(%10 : memref<?x32x?xi32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 128 + d2)>>)  {
            ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
              %11 = arith.cmpi slt, %arg2, %arg3 : i32
              iree_linalg_ext.yield %11 : i1
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 0, 16], [1, 0, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [16, 1]>
//      CHECK: hal.executable.entry_point public @static_3d_sort
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %{{.+}}: index):
// CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[DIV:.+]] = affine.apply #[[MAP]]()[%[[X]]]
// CHECK-NEXT:   hal.return %[[DIV]], %[[Y]], %[[ONE]]

//      CHECK: func @static_3d_sort()
//      CHECK:   iree_linalg_ext.sort
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

hal.executable private @static_1d_fft_stage2  {
  hal.interface @io {
    hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
    hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirvfb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @static_1d_fft_stage2 attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      builtin.func @static_1d_fft_stage2() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan @io::@s0b0_rw_external[%c0] : !flow.dispatch.tensor<readwrite:32xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_rw_external[%c0] : !flow.dispatch.tensor<readwrite:32xf32>
        %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [], sizes = [], strides = [] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        flow.dispatch.tensor.store %4#1, %1, offsets = [], sizes = [], strides = [] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[4]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [4]>
//       CHECK: hal.executable.entry_point public @static_1d_fft_stage2
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//  CHECK-NEXT: ^{{.+}}(%[[ARG0:.+]]: index, %{{.+}}: index, %{{.+}}: index):
//  CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[T:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//  CHECK-NEXT:   hal.return %[[T]], %[[ONE]], %[[ONE]]

//       CHECK: func @static_1d_fft_stage2()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----


hal.executable private @static_3d_fft_stage3  {
  hal.interface @io {
    hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
    hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirvfb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @static_3d_fft_stage3 attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      builtin.func @static_3d_fft_stage3() {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
        %0 = bufferization.to_memref %cst_0 : memref<4xf32>
        %1 = bufferization.to_memref %cst : memref<4xf32>
        %2 = hal.interface.binding.subspan @io::@s0b0_rw_external[%c0] : memref<64x128x32xf32>
        %3 = hal.interface.binding.subspan @io::@s0b1_rw_external[%c0] : memref<64x128x32xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c64 step %workgroup_count_z {
          scf.for %arg1 = %workgroup_id_y to %c128 step %workgroup_count_y {
            %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_x]
            %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_x]
            scf.for %arg2 = %4 to %c32 step %5 {
              %6 = memref.subview %2[%arg0, %arg1, %arg2] [1, 1, 4] [1, 1, 1] : memref<64x128x32xf32> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %7 = memref.cast %6 : memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %8 = memref.subview %3[%arg0, %arg1, %arg2] [1, 1, 4] [1, 1, 1] : memref<64x128x32xf32> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %9 = memref.cast %8 : memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"}
                ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>)
                outs(%7, %9 : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>, memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>)
            }
          }
        }
        return
      }
    }
  }
}


//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 1, 8]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [8, 1, 1]>
//       CHECK: hal.executable.entry_point public @static_3d_fft_stage3
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//  CHECK-NEXT: ^{{.+}}(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index):
//  CHECK-NEXT:   %[[T:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//  CHECK-NEXT:   hal.return %[[T]], %[[ARG1]], %[[ARG2]]

//       CHECK: func @static_3d_fft_stage3()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]
