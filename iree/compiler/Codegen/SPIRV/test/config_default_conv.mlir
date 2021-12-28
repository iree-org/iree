// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

// Convolution with consumer pointwise ops

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 112)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 32)>
#map3 = affine_map<(d0) -> (d0 * 2)>
#map4 = affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>
#map5 = affine_map<(d0)[s0] -> (-d0 + 32, s0)>
#map6 = affine_map<(d0)[s0] -> (-d0 + 112, s0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @conv_pointwise_112x112x32 {
  hal.executable.variant public @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 16384 : i32,
        max_compute_workgroup_invocations = 128 : i32,
        max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @conv_pointwise_112x112x32 layout(#executable_layout)
    builtin.module {
      func @conv_pointwise_112x112x32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c112 = arith.constant 112 : index
        %c32 = arith.constant 32 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x112x112x32xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x225x225x3xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:3x3x3x32xf32>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %4 = affine.apply #map0()[%workgroup_id_z, %workgroup_size_z]
        %5 = affine.apply #map0()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %4 to %c112 step %5 {
          %6 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
          %7 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %6 to %c112 step %7 {
            %8 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
            %9 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %8 to %c32 step %9 {
              %10 = affine.min #map1(%arg0)[%workgroup_size_z]
              %11 = affine.min #map1(%arg1)[%workgroup_size_y]
              %12 = affine.min #map2(%arg2)[%workgroup_size_x]
              %13 = flow.dispatch.tensor.load %0, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %10, %11, %12], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x112x112x32xf32> -> tensor<1x?x?x?xf32>
              %14 = linalg.init_tensor [1, %10, %11, %12] : tensor<1x?x?x?xf32>
              %15 = affine.apply #map3(%arg0)
              %16 = affine.min #map4(%10, %arg0)
              %17 = affine.apply #map3(%arg1)
              %18 = affine.min #map4(%11, %arg1)
              %19 = flow.dispatch.tensor.load %1, offsets = [0, %15, %17, 0], sizes = [1, %16, %18, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x225x225x3xf32> -> tensor<1x?x?x3xf32>
              %20 = affine.min #map5(%arg2)[%workgroup_size_x]
              %21 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 3, %20], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x3x32xf32> -> tensor<3x3x3x?xf32>
              %22 = affine.min #map6(%arg0)[%workgroup_size_z]
              %23 = affine.min #map6(%arg1)[%workgroup_size_y]
              %24 = linalg.init_tensor [1, %22, %23, %20] : tensor<1x?x?x?xf32>
              %25 = linalg.fill(%cst, %24) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %26 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%19, %21 : tensor<1x?x?x3xf32>, tensor<3x3x3x?xf32>) outs(%25 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              %27 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %13 : tensor<1x?x?x?xf32>, tensor<1x?x?x?xf32>) outs(%14 : tensor<1x?x?x?xf32>) {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
                %28 = arith.subf %arg3, %arg4 : f32
                linalg.yield %28 : f32
              } -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %27, %3, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %10, %11, %12], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[0, 4, 4, 32], [0, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [32, 4, 4]>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)
//  CHECK-DAG: #[[MAP_YZ:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: hal.executable.entry_point public @conv_pointwise_112x112x32
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [8 : index, 2 : index, 2 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index):
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_YZ]]()[%[[Y]]]
// CHECK-NEXT:   %[[Z_COUNT:.+]] = affine.apply #[[MAP_YZ]]()[%[[Z]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[Z_COUNT]]

//      CHECK: func @conv_pointwise_112x112x32()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering.config = #[[CONFIG]]
