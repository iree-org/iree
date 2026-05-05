// RUN: iree-opt --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-rocdl-lowering-pipeline='include-llvm-lowering=false' \
// RUN:     %s | FileCheck %s

// Test that the vector distribute pipeline correctly handles the case where
// dynamic shapes are involved, and as consequence padding is required.

// Check the pipeline didn't fail.
// CHECK-LABEL: func.func @kernel
// CHECK: vector.transfer_read
// CHECK: iree_codegen.dispatch_config @kernel workgroup_size = [512, 1, 1] subgroup_size = 64
func.func @kernel() attributes {
    translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [512, 1, 1] subgroup_size = 64, {iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}>} {
  %cst = arith.constant 0.0721687824 : f32
  %c32 = arith.constant 32 : index
  %cst_0 = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8:4 = util.assume.int
      %4<umin = 1982976, umax = 13880832>,
      %5<umin = 1589760, umax = 11128320>,
      %6<umin = 2376192, umax = 16633344>,
      %7<umin = 32, umax = 224, udiv = 32>
    : index, index, index, index
  %9 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4xi64, #hal.descriptor_type<storage_buffer>>
  %10 = amdgpu.fat_raw_buffer_cast %9 resetOffset : memref<4xi64, #hal.descriptor_type<storage_buffer>> to memref<4xi64, #amdgpu.address_space<fat_raw_buffer>>
  %11 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<f32, #hal.descriptor_type<storage_buffer>>
  %12 = amdgpu.fat_raw_buffer_cast %11 resetOffset : memref<f32, #hal.descriptor_type<storage_buffer>> to memref<f32, #amdgpu.address_space<fat_raw_buffer>>
  %13 = iree_tensor_ext.dispatch.workload.ordinal %8#3, 0 : index
  %14 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%8#0) flags("ReadOnly|Indirect") : memref<192x4x?x4xf32, strided<[?, ?, 4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%13}
  %15 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%8#1) flags("ReadOnly|Indirect") : memref<192x4x?x4xf32, strided<[?, ?, 4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%13}
  %16 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%8#2) flags("ReadOnly|Indirect") : memref<4x?x4x192xf32, strided<[?, 768, 192, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>{%13}
  %17 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<4x4x?x192xf32, #hal.descriptor_type<storage_buffer>>{%13}
  %18 = iree_codegen.load_from_buffer %10 : memref<4xi64, #amdgpu.address_space<fat_raw_buffer>> -> tensor<4xi64>
  %19 = iree_codegen.load_from_buffer %12 : memref<f32, #amdgpu.address_space<fat_raw_buffer>> -> tensor<f32>
  %20 = affine.apply affine_map<()[s0] -> (s0 floordiv 32)>()[%13]
  %21 = tensor.empty(%20, %20) : tensor<4x?x32x?x32xf32>
  %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%18, %19 : tensor<4xi64>, tensor<f32>) outs(%21 : tensor<4x?x32x?x32xf32>) {
  ^bb0(%in: i64, %in_4: f32, %out: f32):
    %28 = linalg.index 4 : index
    %29 = linalg.index 3 : index
    %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 32)>()[%28, %29]
    %31 = arith.index_cast %30 : index to i64
    %32 = arith.cmpi sge, %31, %in : i64
    %33 = linalg.index 2 : index
    %34 = linalg.index 1 : index
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 32)>()[%33, %34]
    %36 = arith.index_cast %35 : index to i64
    %37 = arith.cmpi sgt, %31, %36 : i64
    %38 = arith.ori %37, %32 : i1
    %39 = arith.select %38, %cst_0, %in_4 : f32
    linalg.yield %39 : f32
  } -> tensor<4x?x32x?x32xf32>
  %nval = arith.divsi %13, %c32 : index
  %expand_shape = memref.expand_shape %17 [[0], [1], [2, 3], [4]] output_shape [4, 4, %20, 32, 192] : memref<4x4x?x192xf32, #hal.descriptor_type<storage_buffer>> into memref<4x4x?x32x192xf32, #hal.descriptor_type<storage_buffer>>
  %expand_shape_1 = memref.expand_shape %14 [[0], [1], [2, 3], [4]] output_shape [192, 4, %20, 32, 4] : memref<192x4x?x4xf32, strided<[?, ?, 4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> into memref<192x4x?x32x4xf32, strided<[?, ?, 128, 4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %expand_shape_2 = memref.expand_shape %16 [[0], [1, 2], [3], [4]] output_shape [4, %20, 32, 4, 192] : memref<4x?x4x192xf32, strided<[?, 768, 192, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> into memref<4x?x32x4x192xf32, strided<[?, 24576, 768, 192, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %expand_shape_3 = memref.expand_shape %15 [[0], [1], [2, 3], [4]] output_shape [192, 4, %20, 32, 4] : memref<192x4x?x4xf32, strided<[?, ?, 4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> into memref<192x4x?x32x4xf32, strided<[?, ?, 128, 4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %23 = iree_codegen.load_from_buffer %expand_shape_1 : memref<192x4x?x32x4xf32, strided<[?, ?, 128, 4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> -> tensor<192x4x?x32x4xf32>
  %24 = iree_codegen.load_from_buffer %expand_shape_3 : memref<192x4x?x32x4xf32, strided<[?, ?, 128, 4, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> -> tensor<192x4x?x32x4xf32>
  %25 = iree_codegen.load_from_buffer %expand_shape_2 : memref<4x?x32x4x192xf32, strided<[?, 24576, 768, 192, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> -> tensor<4x?x32x4x192xf32>
  %26 = tensor.empty(%20) : tensor<4x4x?x32x192xf32>
  %27 = tensor.empty(%20) : tensor<4x4x?x32xf32>
  %28:3 = iree_linalg_ext.online_attention {decomposition_config = {pv_attrs = {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 1, 1, 1, 1, 8, 1, 8], [2, 3, 0, 1, 5, 6, 7]], subgroup_basis = [[1, 1, 1, 1, 1, 1, 2, 4], [0, 1, 2, 3, 5, 6, 7]], thread = [0, 0, 0, 0, 4, 0, 0]}>}, qk_attrs = {lowering_config = #iree_gpu.lowering_config<{lane_basis = [[1, 1, 1, 1, 1, 8, 1, 8], [2, 3, 0, 1, 5, 6, 7]], subgroup_basis = [[1, 1, 1, 1, 1, 1, 2, 4], [0, 1, 2, 3, 4, 6, 7]], thread = [0, 0, 0, 0, 4, 0, 0]}>}}, indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d0, d6, d7, d1)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d6, d7, d1, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d2, d3, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>], lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 0, 0, 0, 0, 2, 32], workgroup = [1, 1, 1, 1, 64, 0, 0, 0]}>} ins(%23, %24, %25, %cst, %22 : tensor<192x4x?x32x4xf32>, tensor<192x4x?x32x4xf32>, tensor<4x?x32x4x192xf32>, f32, tensor<4x?x32x?x32xf32>) outs(%26, %27, %27 : tensor<4x4x?x32x192xf32>, tensor<4x4x?x32xf32>, tensor<4x4x?x32xf32>) {
  ^bb0(%arg0: f32):
    iree_linalg_ext.yield %arg0 : f32
  } -> tensor<4x4x?x32x192xf32>, tensor<4x4x?x32xf32>, tensor<4x4x?x32xf32>
  iree_codegen.store_to_buffer %28#0, %expand_shape : tensor<4x4x?x32x192xf32> into memref<4x4x?x32x192xf32, #hal.descriptor_type<storage_buffer>>
  return
}
