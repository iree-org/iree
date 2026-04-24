// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

// Test ArgCompareOp with VectorDistribute pipeline.

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

hal.executable private @argcompare_pipeline_test {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @argcompare_argmax ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @argcompare_argmax() attributes {translation_info = #translation} {
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
    }
  }
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
