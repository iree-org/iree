// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-create-fast-slow-path))" --mlir-print-local-scope %s | FileCheck %s

func.func @padded_conv() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c112 = arith.constant 112 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x224x224x3xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
  %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
  %8 = flow.dispatch.tensor.load %2, offsets = [0, %workgroup_id_z, %4, %6], sizes = [1, 1, 4, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>> -> tensor<1x1x4x32xf32>
  %9 = tensor.empty() : tensor<1x1x4x32xf32>
  %10 = affine.apply affine_map<(d0) -> (d0 * 2)>(%workgroup_id_z)
  %11 = affine.min affine_map<(d0) -> (d0 * 2 + 3, 224)>(%workgroup_id_z)
  %12 = affine.apply affine_map<(d0, d1) -> (d0 - d1 * 2)>(%11, %workgroup_id_z)
  %13 = affine.apply affine_map<(d0, d1) -> (-d0 + d1 * 2 + 3)>(%11, %workgroup_id_z)
  %14 = affine.apply affine_map<(d0) -> (d0 * 2)>(%4)
  %15 = affine.min affine_map<(d0) -> (d0 * 2 + 9, 224)>(%4)
  %16 = affine.apply affine_map<(d0, d1) -> (d0 - d1 * 2)>(%15, %4)
  %17 = affine.apply affine_map<(d0, d1) -> (-d0 + d1 * 2 + 9)>(%15, %4)
  %18 = flow.dispatch.tensor.load %0, offsets = [0, %10, %14, 0], sizes = [1, %12, %16, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x224x224x3xf32>> -> tensor<1x?x?x3xf32>
  %19 = tensor.pad %18 low[0, 0, 0, 0] high[0, %13, %17, 0] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst : f32
  } : tensor<1x?x?x3xf32> to tensor<1x?x?x3xf32>
  %20 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %6], sizes = [3, 3, 3, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>> -> tensor<3x3x3x32xf32>
  %21 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1, 4]]>} ins(%cst : f32) outs(%9 : tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32>
  %22 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1, 4]]>, strides = dense<2> : tensor<2xi64>} ins(%19, %20 : tensor<1x?x?x3xf32>, tensor<3x3x3x32xf32>) outs(%21 : tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32>
  %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %8 : tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>) outs(%9 : tensor<1x1x4x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1, 4]]>} {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %24 = arith.subf %arg3, %arg4 : f32
    linalg.yield %24 : f32
  } -> tensor<1x1x4x32xf32>
  flow.dispatch.tensor.store %23, %3, offsets = [0, %workgroup_id_z, %4, %6], sizes = [1, 1, 4, 32], strides = [1, 1, 1, 1] : tensor<1x1x4x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
  return
}

// CHECK-LABEL: func.func @padded_conv

//       CHECK: %[[C0:.+]] = arith.constant 0 : index

//       CHECK:      %[[MIN0:.+]] = affine.min affine_map<(d0) -> (d0 * 2 + 3, 224)>(%{{.*}})
//       CHECK:      %[[SIZE0:.+]] = affine.apply affine_map<(d0, d1) -> (-d0 + d1 * 2 + 3)>(%[[MIN0]], %{{.*}})
//       CHECK:      %[[MIN1:.+]] = affine.min affine_map<(d0) -> (d0 * 2 + 9, 224)>(%{{.*}})
//       CHECK:      %[[SIZE1:.+]] = affine.apply affine_map<(d0, d1) -> (-d0 + d1 * 2 + 9)>(%[[MIN1]], %{{.*}})
//       CHECK:      %[[EQ0:.+]] = arith.cmpi eq, %[[SIZE0]], %[[C0]] : index
//       CHECK:      %[[EQ1:.+]] = arith.cmpi eq, %[[SIZE1]], %[[C0]] : index
//       CHECK:      %[[COND:.+]] = arith.andi %[[EQ0]], %[[EQ1]] : i1

//       CHECK:      scf.if %[[COND]] {

//       CHECK:        flow.dispatch.tensor.load
//       CHECK:        %[[INPUT:.+]] = flow.dispatch.tensor.load
//       CHECK:        %[[FILTER:.+]] = flow.dispatch.tensor.load
//       CHECK:        %[[FILL:.+]] = linalg.fill
//       CHECK:        %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:          ins(%[[INPUT]], %[[FILTER]]
//       CHECK:        %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:          ins(%[[CONV]]
//       CHECK:        flow.dispatch.tensor.store %[[GENERIC]]

//       CHECK:      } else {

//       CHECK:        flow.dispatch.tensor.load
//       CHECK:        %[[INPUT:.+]] = flow.dispatch.tensor.load
//       CHECK:        %[[PAD:.+]] = tensor.pad %[[INPUT]]
//       CHECK:        %[[FILTER:.+]] = flow.dispatch.tensor.load
//       CHECK:        %[[FILL:.+]] = linalg.fill
//       CHECK:        %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:          ins(%[[PAD]], %[[FILTER]]
//       CHECK:        %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:          ins(%[[CONV]]
//       CHECK:        flow.dispatch.tensor.store %[[GENERIC]]

//       CHECK:      }
