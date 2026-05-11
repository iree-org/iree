// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-rocdl-lowering-pipeline='include-llvm-lowering=false' %s | FileCheck %s

// Small 3-D reduction over dim 0 (non-last-dim). Reduction size 2 is smaller
// than subgroup size, so setReductionConfig rejects and falls through to
// setArgCompareConfig (TileAndFuse).

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#config = #iree_gpu.lowering_config<{thread = [0, 1, 1], workgroup = [0, 1, 4]}>
#translation = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_small_3d_dim0()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x4xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x4xi64>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 3, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x4xf32>> -> tensor<2x3x4xf32>
  %4 = tensor.empty() : tensor<3x4xf32>
  %5 = tensor.empty() : tensor<3x4xi64>
  %6:2 = iree_linalg_ext.arg_compare {lowering_config = #config} dimension(0) ins(%3 : tensor<2x3x4xf32>) outs(%4, %5 : tensor<3x4xf32>, tensor<3x4xi64>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x4xf32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x4xi64>>
  return
}

// Non-last-dim reduction over dim 0: the 3-D input is read as vector<2x1x1xf32>
// and shape_cast to vector<2xf32> before the scf.for reduction loop.
// CHECK-LABEL: func.func @argcompare_small_3d_dim0
//     CHECK:     vector.shape_cast %{{.*}} : vector<2x1x1xf32> to vector<2xf32>
//     CHECK:     scf.for
//     CHECK:       arith.cmpf ogt
//     CHECK:       arith.select
//     CHECK:     vector.transfer_write
// CHECK-NOT:     iree_vector_ext.arg_compare
// CHECK-NOT:     iree_linalg_ext.arg_compare

// -----

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#config = #iree_gpu.lowering_config<{thread = [1, 0], workgroup = [4, 0]}>
#translation = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_last_dim_reduction()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x128xf32>> -> tensor<4x128xf32>
  %4 = tensor.empty() : tensor<4xf32>
  %5 = tensor.empty() : tensor<4xi32>
  %6:2 = iree_linalg_ext.arg_compare {lowering_config = #config} dimension(1) ins(%3 : tensor<4x128xf32>) outs(%4, %5 : tensor<4xf32>, tensor<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>
  return
}

// Last-dim reduction: no transpose/shape_cast needed; input read directly as
// vector<128xf32> and reduced via scf.for.
// CHECK-LABEL: func.func @argcompare_last_dim_reduction
// CHECK-NOT:     vector.transpose
//     CHECK:     vector.transfer_read {{.*}} vector<128xf32>
//     CHECK:     scf.for
//     CHECK:       arith.cmpf ogt
//     CHECK:       arith.select
//     CHECK:     vector.transfer_write
// CHECK-NOT:     iree_vector_ext.arg_compare
// CHECK-NOT:     iree_linalg_ext.arg_compare

// -----

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#config = #iree_gpu.lowering_config<{thread = [1, 0], workgroup = [8, 0]}>
#translation = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_unaligned_reduction()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x100xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 100], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x100xf32>> -> tensor<8x100xf32>
  %4 = tensor.empty() : tensor<8xf32>
  %5 = tensor.empty() : tensor<8xi32>
  %6:2 = iree_linalg_ext.arg_compare {lowering_config = #config} dimension(1) ins(%3 : tensor<8x100xf32>) outs(%4, %5 : tensor<8xf32>, tensor<8xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<8xf32>, tensor<8xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [8], strides = [1] : tensor<8xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi32>>
  return
}

// Unaligned reduction: input read as vector<100xf32>, reduced via scf.for.
// CHECK-LABEL: func.func @argcompare_unaligned_reduction
//     CHECK:     vector.transfer_read {{.*}} vector<100xf32>
//     CHECK:     scf.for
//     CHECK:       arith.cmpf ogt
//     CHECK:       arith.select
// CHECK-NOT:     iree_vector_ext.arg_compare
// CHECK-NOT:     iree_linalg_ext.arg_compare

// -----

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#config = #iree_gpu.lowering_config<{thread = [1, 0], workgroup = [8, 0]}>
#translation = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_f64_fallback()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x256xf64>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf64>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x256xf64>> -> tensor<8x256xf64>
  %4 = tensor.empty() : tensor<8xf64>
  %5 = tensor.empty() : tensor<8xi32>
  %6:2 = iree_linalg_ext.arg_compare {lowering_config = #config} dimension(1) ins(%3 : tensor<8x256xf64>) outs(%4, %5 : tensor<8xf64>, tensor<8xi32>) {
    ^bb0(%a: f64, %b: f64):
      %cmp = arith.cmpf ogt, %a, %b : f64
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<8xf64>, tensor<8xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf64>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [8], strides = [1] : tensor<8xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xi32>>
  return
}

// f64 fallback: input read as vector<256xf64>, reduced via scf.for with f64 compare.
// CHECK-LABEL: func.func @argcompare_f64_fallback
//     CHECK:     vector.transfer_read {{.*}} vector<256xf64>
//     CHECK:     scf.for
//     CHECK:       arith.cmpf ogt, %{{.*}}, %{{.*}} : f64
//     CHECK:       arith.select
// CHECK-NOT:     iree_vector_ext.arg_compare
// CHECK-NOT:     iree_linalg_ext.arg_compare
