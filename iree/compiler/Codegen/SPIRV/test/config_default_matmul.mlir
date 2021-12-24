// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

// Odd K that forbids vectorization.

hal.executable @batch_matmul_1x3x32 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 16384 : i32,
        max_compute_workgroup_invocations = 128 : i32,
        max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @batch_matmul_1x3x32 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @batch_matmul_1x3x32() {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c3 = arith.constant 3 : index
        %c1 = arith.constant 1 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:1x3x3xf32>
        %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:1x3x32xf32>
        %2 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:1x3x32xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c1 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c3 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c32 step %8 {
              %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_z]
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 3)>(%arg1)[%workgroup_size_y]
              %11 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1, 0], sizes = [%9, %10, 3], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:1x3x3xf32> -> tensor<?x?x3xf32>
              %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_z]
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %14 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, %arg2], sizes = [%12, 3, %13], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:1x3x32xf32> -> tensor<?x3x?xf32>
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_z]
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 3)>(%arg1)[%workgroup_size_y]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %18 = affine.min affine_map<(d0)[s0] -> (-d0 + 1, s0)>(%arg0)[%workgroup_size_z]
              %19 = affine.min affine_map<(d0)[s0] -> (-d0 + 3, s0)>(%arg1)[%workgroup_size_y]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %21 = linalg.init_tensor [%18, %19, %20] : tensor<?x?x?xf32>
              %22 = linalg.fill(%cst, %21) : f32, tensor<?x?x?xf32> -> tensor<?x?x?xf32>
              %23 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"} ins(%11, %14 : tensor<?x?x3xf32>, tensor<?x3x?xf32>) outs(%22 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
              flow.dispatch.tensor.store %23, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%15, %16, %17], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x3x32xf32>
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 1, 32], [1, 1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [32, 1, 1]>
//      CHECK: hal.executable.entry_point public @batch_matmul_1x3x32
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 1 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index):
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP]]()[%[[X]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y]], %[[Z]]

//      CHECK: func @batch_matmul_1x3x32()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Non-16 / non-32 bit types cannot be vectorized right now.

hal.executable private @matmul_64x16 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 16384 : i32,
        max_compute_workgroup_invocations = 128 : i32,
        max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>,
        subgroup_size = 64 : i32}>
  }> {
    hal.executable.entry_point public @matmul_64x16 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @matmul_64x16() {
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:64x32xi8>
        %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:32x16xi8>
        %2 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:64x16xi32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c64 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c16 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 64)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:64x32xi8> -> tensor<?x32xi8>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [32, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:32x16xi8> -> tensor<32x?xi8>
            %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 64)>(%arg0)[%workgroup_size_y]
            %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg1)[%workgroup_size_x]
            %13 = affine.min affine_map<(d0)[s0] -> (-d0 + 64, s0)>(%arg0)[%workgroup_size_y]
            %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 16, s0)>(%arg1)[%workgroup_size_x]
            %15 = linalg.init_tensor [%13, %14] : tensor<?x?xi32>
            %16 = linalg.fill(%c0_i32, %15) : i32, tensor<?x?xi32> -> tensor<?x?xi32>
            %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%8, %10 : tensor<?x32xi8>, tensor<32x?xi8>) outs(%16 : tensor<?x?xi32>) -> tensor<?x?xi32>
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:64x16xi32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[4, 16], [1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[MAP_Y:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [16, 4]>
//      CHECK: hal.executable.entry_point public @matmul_64x16
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [16 : index, 4 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %{{.+}}: index):
// CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_Y]]()[%[[Y]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[ONE]]

//      CHECK: func @matmul_64x16()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Odd N that forbids vectorization.

hal.executable @matmul_400x273 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 16384 : i32,
        max_compute_workgroup_invocations = 128 : i32,
        max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>,
        subgroup_size = 64 : i32}>
    }> {
    hal.executable.entry_point public @matmul_400x273 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @matmul_400x273() {
        %c0 = arith.constant 0 : index
        %c11775744 = arith.constant 11775744 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c400 = arith.constant 400 : index
        %c273 = arith.constant 273 : index
        %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) offset(%c11775744) : !flow.dispatch.tensor<readonly:273xf32>
        %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:400x576xf32>
        %2 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:576x273xf32>
        %3 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:400x273xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %4 to %c400 step %5 {
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %6 to %c273 step %7 {
            %8 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 273)>(%arg1)[%workgroup_size_x]
            %9 = flow.dispatch.tensor.load %0, offsets = [%arg1], sizes = [%8], strides = [1] : !flow.dispatch.tensor<readonly:273xf32> -> tensor<?xf32>
            %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 400)>(%arg0)[%workgroup_size_y]
            %11 = linalg.init_tensor [%10, %8] : tensor<?x?xf32>
            %12 = affine.min affine_map<(d0)[s0] -> (-d0 + 400, s0)>(%arg0)[%workgroup_size_y]
            %13 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0], sizes = [%12, 576], strides = [1, 1] : !flow.dispatch.tensor<readonly:400x576xf32> -> tensor<?x576xf32>
            %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 273, s0)>(%arg1)[%workgroup_size_x]
            %15 = flow.dispatch.tensor.load %2, offsets = [0, %arg1], sizes = [576, %14], strides = [1, 1] : !flow.dispatch.tensor<readonly:576x273xf32> -> tensor<576x?xf32>
            %16 = linalg.init_tensor [%12, %14] : tensor<?x?xf32>
            %17 = linalg.fill(%cst, %16) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %18 = linalg.matmul ins(%13, %15 : tensor<?x576xf32>, tensor<576x?xf32>) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %18 : tensor<?xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?xf32>) {
            ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
              %20 = arith.addf %arg2, %arg3 : f32
              linalg.yield %20 : f32
            } -> tensor<?x?xf32>
            flow.dispatch.tensor.store %19, %3, offsets = [%arg0, %arg1], sizes = [%10, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:400x273xf32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[2, 32], [1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[MAP_Y:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [32, 2]>

//      CHECK: hal.executable.entry_point public @matmul_400x273
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 2 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %{{.+}}: index):
// CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_Y]]()[%[[Y]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[ONE]]

//      CHECK: func @matmul_400x273()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Odd M and non-4-multiplier N

hal.executable @matmul_25x546 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 16384 : i32,
        max_compute_workgroup_invocations = 128 : i32,
        max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>,
        subgroup_size = 64 : i32}>
  }> {
    hal.executable.entry_point public @matmul_25x546 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @matmul_25x546() {
        %c0 = arith.constant 0 : index
        %c15842560 = arith.constant 15842560 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c25 = arith.constant 25 : index
        %c546 = arith.constant 546 : index
        %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) offset(%c15842560) : !flow.dispatch.tensor<readonly:546xf32>
        %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:25x512xf32>
        %2 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:512x546xf32>
        %3 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:25x546xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %4 to %c25 step %5 {
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %6 to %c546 step %7 {
            %8 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 546)>(%arg1)[%workgroup_size_x]
            %9 = flow.dispatch.tensor.load %0, offsets = [%arg1], sizes = [%8], strides = [1] : !flow.dispatch.tensor<readonly:546xf32> -> tensor<?xf32>
            %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 25)>(%arg0)[%workgroup_size_y]
            %11 = linalg.init_tensor [%10, %8] : tensor<?x?xf32>
            %12 = affine.min affine_map<(d0)[s0] -> (-d0 + 25, s0)>(%arg0)[%workgroup_size_y]
            %13 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0], sizes = [%12, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:25x512xf32> -> tensor<?x512xf32>
            %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 546, s0)>(%arg1)[%workgroup_size_x]
            %15 = flow.dispatch.tensor.load %2, offsets = [0, %arg1], sizes = [512, %14], strides = [1, 1] : !flow.dispatch.tensor<readonly:512x546xf32> -> tensor<512x?xf32>
            %16 = linalg.init_tensor [%12, %14] : tensor<?x?xf32>
            %17 = linalg.fill(%cst, %16) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %18 = linalg.matmul ins(%13, %15 : tensor<?x512xf32>, tensor<512x?xf32>) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %18 : tensor<?xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
            ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
              %20 = arith.addf %arg2, %arg3 : f32
              linalg.yield %20 : f32
            } -> tensor<?x?xf32>
            flow.dispatch.tensor.store %19, %3, offsets = [%arg0, %arg1], sizes = [%10, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:25x546xf32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[32, 2], [1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK-DAG: #[[MAP_Y:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [2, 32]>
//      CHECK: hal.executable.entry_point public @matmul_25x546
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 32 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %{{.+}}: index):
// CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_Y]]()[%[[Y]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[ONE]]

//      CHECK: func @matmul_25x546()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Matmul with consumer pointwise ops

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 256)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map3 = affine_map<(d0)[s0] -> (-d0 + 256, s0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>

hal.executable private @matmul_pointwise_256x1024 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_ro_external, set=0, binding=2, type="StorageBuffer"
    hal.interface.binding public @s0b3_ro_external, set=0, binding=3, type="StorageBuffer"
    hal.interface.binding public @s0b4_xw_external, set=0, binding=4, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 16384 : i32,
        max_compute_workgroup_invocations = 128 : i32,
        max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @matmul_pointwise_256x1024 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @matmul_pointwise_256x1024() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %c256 = arith.constant 256 : index
        %c1024 = arith.constant 1024 : index
        %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:256x1024xf16>
        %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:256x1024xf16>
        %2 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(2) : !flow.dispatch.tensor<readonly:256x128xf16>
        %3 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(3) : !flow.dispatch.tensor<readonly:128x1024xf16>
        %4 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(4) : !flow.dispatch.tensor<writeonly:256x1024xf16>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %5 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
        %6 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %5 to %c256 step %6 {
          %7 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
          %8 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %7 to %c1024 step %8 {
            %9 = affine.min #map1(%arg0)[%workgroup_size_y]
            %10 = affine.min #map2(%arg1)[%workgroup_size_x]
            %11 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%9, %10], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x1024xf16> -> tensor<?x?xf16>
            %12 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%9, %10], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x1024xf16> -> tensor<?x?xf16>
            %13 = linalg.init_tensor [%9, %10] : tensor<?x?xf16>
            %14 = affine.min #map3(%arg0)[%workgroup_size_y]
            %15 = flow.dispatch.tensor.load %2, offsets = [%arg0, 0], sizes = [%14, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x128xf16> -> tensor<?x128xf16>
            %16 = affine.min #map4(%arg1)[%workgroup_size_x]
            %17 = flow.dispatch.tensor.load %3, offsets = [0, %arg1], sizes = [128, %16], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x1024xf16> -> tensor<128x?xf16>
            %18 = linalg.init_tensor [%14, %16] : tensor<?x?xf16>
            %19 = linalg.fill(%cst, %18) : f16, tensor<?x?xf16> -> tensor<?x?xf16>
            %20 = linalg.matmul ins(%15, %17 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%19 : tensor<?x?xf16>) -> tensor<?x?xf16>
            %21 = linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%20, %11, %12 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%13 : tensor<?x?xf16>) {
            ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):  // no predecessors
              %22 = arith.divf %arg2, %arg3 : f16
              %23 = arith.subf %22, %arg4 : f16
              linalg.yield %23 : f16
            } -> tensor<?x?xf16>
            flow.dispatch.tensor.store %21, %4, offsets = [%arg0, %arg1], sizes = [%9, %10], strides = [1, 1] : tensor<?x?xf16> -> !flow.dispatch.tensor<writeonly:256x1024xf16>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[16, 256], [8, 8], [0, 0, 4]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 256)>
//  CHECK-DAG: #[[MAP_Y:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [256, 16]>
//      CHECK: hal.executable.entry_point public @matmul_pointwise_256x1024
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 2 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %{{.+}}: index):
// CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_Y]]()[%[[Y]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[ONE]]

//      CHECK: func @matmul_pointwise_256x1024()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering.config = #[[CONFIG]]
