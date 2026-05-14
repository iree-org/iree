// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-rocdl-lowering-pipeline='include-llvm-lowering=false' %s | FileCheck %s

// Test ArgCompareOp with VectorDistribute pipeline.

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
// Use a simple config with only lane-level reduction (64 threads, 64 elements per thread)
// to avoid needing subgroup-level reduction across multiple subgroups.
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 0],
  partial_reduction = [0, 256],
  thread = [0, 4],
  subgroup_basis = [[1, 1], [0, 1]],
  lane_basis = [[1, 64], [0, 1]]
}>

#translation = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<VectorDistribute>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64, {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      no_reduce_shared_memory_bank_conflicts = false,
      use_igemm_convolution = false>
  }>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_argmax() attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xf32>> -> tensor<1x256xf32>
  %4 = tensor.empty() : tensor<1xf32>
  %5 = tensor.empty() : tensor<1xi32>
  %6:2 = iree_linalg_ext.arg_compare {lowering_config = #config} dimension(1) ins(%3 : tensor<1x256xf32>) outs(%4, %5 : tensor<1xf32>, tensor<1xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<1xf32>, tensor<1xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xf32>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [1], strides = [1] : tensor<1xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xi32>>
  return
}

// CHECK-LABEL: func.func @argcompare_argmax
// Initial vectorized compare+select over the element tile (4 elements).
//     CHECK: %[[CMP:.+]] = arith.cmpf ogt, %{{.*}}, %{{.*}} : vector<1x1x1x1x1x4xf32>
//     CHECK: %[[SEL_VAL:.+]] = arith.select %[[CMP]], %{{.*}}, %{{.*}} : vector<1x1x1x1x1x4xi1>, vector<1x1x1x1x1x4xf32>
//     CHECK: %[[SEL_IDX:.+]] = arith.select %[[CMP]], %{{.*}}, %{{.*}} : vector<1x1x1x1x1x4xi1>, vector<1x1x1x1x1x4xi32>
// Inline scalar reduction across elements within each thread.
//     CHECK: %[[ELEM0:.+]] = vector.extract %[[SEL_VAL]][0, 0, 0, 0, 0, 0] : f32
//     CHECK: %[[SCALAR_CMP:.+]] = arith.cmpf ogt, %[[ELEM0]], %{{.*}} : f32
//     CHECK: arith.select %[[SCALAR_CMP]], %[[ELEM0]], %{{.*}} : f32
// Thread reduction: find global max across all 64 lanes.
//     CHECK: %[[REDUCED:.+]] = gpu.subgroup_reduce maxnumf %{{.*}}
// Ballot to find winning lane.
//     CHECK: %[[IS_WINNER:.+]] = arith.cmpf oeq, %{{.*}}, %[[REDUCED]] : f32
//     CHECK: %[[BALLOT:.+]] = gpu.ballot %[[IS_WINNER]]
//     CHECK: %[[WINNER_I64:.+]] = math.cttz %[[BALLOT]]
//     CHECK: %[[WINNER:.+]] = arith.trunci %[[WINNER_I64]] : i64 to i32
// Broadcast index from winning lane.
//     CHECK: %[[RESULT_IDX:.+]], %{{.*}} = gpu.shuffle idx %{{.*}}, %[[WINNER]]
// Write results back.
//     CHECK: %[[RESULT_VAL:.+]] = vector.broadcast %[[REDUCED]] : f32 to vector<1xf32>
//     CHECK: %[[RESULT_IDX_VEC:.+]] = vector.broadcast %[[RESULT_IDX]] : i32 to vector<1xi32>
//     CHECK: vector.transfer_write %[[RESULT_VAL]]
//     CHECK: vector.transfer_write %[[RESULT_IDX_VEC]]

// -----

// i64 arg_compare on gfx942: commutative comparator picks the ballot path,
// emitting `gpu.subgroup_reduce maxsi : (i64) -> i64` + `gpu.shuffle idx`.

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#config_i64 = #iree_gpu.lowering_config<{
  workgroup = [1, 0],
  partial_reduction = [0, 256],
  thread = [0, 4],
  subgroup_basis = [[1, 1], [0, 1]],
  lane_basis = [[1, 64], [0, 1]]
}>

#translation_i64 = #iree_codegen.translation_info<
  pipeline = #iree_gpu.pipeline<VectorDistribute>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64, {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      no_reduce_shared_memory_bank_conflicts = false,
      use_igemm_convolution = false>
  }>

#pipeline_layout_i64 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @argcompare_argmax_i64() attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation_i64} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_i64) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xi64>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_i64) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xi64>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_i64) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xi64>> -> tensor<1x256xi64>
  %4 = tensor.empty() : tensor<1xi64>
  %5 = tensor.empty() : tensor<1xi32>
  %6:2 = iree_linalg_ext.arg_compare {lowering_config = #config_i64} dimension(1) ins(%3 : tensor<1x256xi64>) outs(%4, %5 : tensor<1xi64>, tensor<1xi32>) {
    ^bb0(%a: i64, %b: i64):
      %cmp = arith.cmpi sgt, %a, %b : i64
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<1xi64>, tensor<1xi32>
  iree_tensor_ext.dispatch.tensor.store %6#0, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xi64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xi64>>
  iree_tensor_ext.dispatch.tensor.store %6#1, %2, offsets = [0], sizes = [1], strides = [1] : tensor<1xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1xi32>>
  return
}

// CHECK-LABEL: func.func @argcompare_argmax_i64
// Vectorized compare+select over the element tile (4 elements).
// CHECK:         arith.cmpi sgt, %{{.*}}, %{{.*}} : vector<1x1x1x1x1x4xi64>
// Thread reduction: subgroup_reduce on i64 (decomposed downstream by ROCDL).
// CHECK:         gpu.subgroup_reduce maxsi {{.*}} : (i64) -> i64
// Ballot to find winning lane.
// CHECK:         gpu.ballot
// Index broadcast via gpu.shuffle idx.
// CHECK:         gpu.shuffle idx
