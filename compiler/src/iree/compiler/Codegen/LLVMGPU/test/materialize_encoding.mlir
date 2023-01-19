// RUN: iree-opt --iree-llvmgpu-materialize-encoding --canonicalize --cse --split-input-file %s | FileCheck %s

func.func @matmul_dispatch_0() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<128x1536xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<128x1536xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1536xf32>> -> tensor<128x1536xf32>
  %3 = iree_linalg_ext.set_encoding %2 : tensor<128x1536xf32> -> tensor<128x1536xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : tensor<128x1536xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> !flow.dispatch.tensor<writeonly:tensor<128x1536xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
  return
}

//       CHECK: func @matmul_dispatch_0()
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[I0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<readonly:tensor<128x1536xf32>>
//       CHECK:   %[[I1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<128x1536xf32>>
//       CHECK:   %[[I2:.+]] = flow.dispatch.tensor.load %[[I0]], offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1536xf32>> -> tensor<128x1536xf32>
//       CHECK: flow.dispatch.tensor.store %[[I2]], %[[I1]], offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : tensor<128x1536xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1536xf32>>

// -----


func.func @matmul_dispatch_1() {
  %c786432 = arith.constant 786432 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c786432) alignment(64) : !flow.dispatch.tensor<readonly:tensor<1536x386xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c786432) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<1536x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS_TRANSPOSE>>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1536, 386], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1536x386xf32>> -> tensor<1536x386xf32>
  %padded = tensor.pad %2 low[0, 0] high[0, 14] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<1536x386xf32> to tensor<1536x400xf32>
  %3 = iree_linalg_ext.set_encoding %padded : tensor<1536x400xf32> -> tensor<1536x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS_TRANSPOSE>>    
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1536, 400], strides = [1, 1] : tensor<1536x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS_TRANSPOSE>> -> !flow.dispatch.tensor<writeonly:tensor<1536x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS_TRANSPOSE>>>
  return
}

//       CHECK: func @matmul_dispatch_1()
//       CHECK:   %[[C0:.+]] = arith.constant 786432 : index
//       CHECK:   %[[I0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<readonly:tensor<1536x386xf32>>
//       CHECK:   %[[I1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<400x1536xf32>>
//       CHECK:   %[[I2:.+]] = flow.dispatch.tensor.load %[[I0]], offsets = [0, 0], sizes = [1536, 386], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1536x386xf32>> -> tensor<1536x386xf32>
//       CHECK:   %[[I3:.+]] = tensor.pad %[[I2]] low[0, 0] high[0, 14]
//       CHECK:   %[[I4:.+]] = tensor.empty() : tensor<400x1536xf32>
//       CHECK:   %[[I5:.+]] = linalg.transpose ins(%[[I3]] : tensor<1536x400xf32>) outs(%[[I4]] : tensor<400x1536xf32>) permutation = [1, 0] 
//       CHECK: flow.dispatch.tensor.store %[[I5]], %[[I1]], offsets = [0, 0], sizes = [1536, 400], strides = [1, 1] : tensor<400x1536xf32> -> !flow.dispatch.tensor<writeonly:tensor<400x1536xf32>>

// -----

func.func @matmul_dispatch_2_matmul_128x400x1536() {
  %c0 = arith.constant 0 : index
  %c786432 = arith.constant 786432 : index
  %c3244032 = arith.constant 3244032 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<128x1536xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c786432) alignment(64) : !flow.dispatch.tensor<readonly:tensor<1536x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS_TRANSPOSE>>>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c3244032) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1536xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>> -> tensor<128x1536xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1536, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1536x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS_TRANSPOSE>>> -> tensor<1536x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS_TRANSPOSE>>
  %5 = tensor.empty() : tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>) -> tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %7 = linalg.matmul ins(%3, %4 : tensor<128x1536xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>, tensor<1536x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS_TRANSPOSE>>) outs(%6 : tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>) -> tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 400], strides = [1, 1] : tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>> -> !flow.dispatch.tensor<writeonly:tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>
  return
}


//       CHECK: func @matmul_dispatch_2_matmul_128x400x1536()
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[C1:.+]] = arith.constant 786432 : index
//       CHECK:   %[[C2:.+]] = arith.constant 3244032 : index
//       CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[I0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<readonly:tensor<128x1536xf32>>
//       CHECK:   %[[I1:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C1]]) alignment(64) : !flow.dispatch.tensor<readonly:tensor<400x1536xf32>> 
//       CHECK:   %[[I2:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C2]]) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<128x400xf32>>
//       CHECK:   %[[I3:.+]] = flow.dispatch.tensor.load %[[I0]], offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1536xf32>> -> tensor<128x1536xf32>
//       CHECK:   %[[I4:.+]] = flow.dispatch.tensor.load %[[I1]], offsets = [0, 0], sizes = [1536, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<400x1536xf32>> -> tensor<400x1536xf32>
//       CHECK:   %[[I5:.+]] = tensor.empty() : tensor<128x400xf32>
//       CHECK:   %[[I6:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[I5]] : tensor<128x400xf32>) -> tensor<128x400xf32>
//       CHECK:   %[[I7:.+]] = linalg.matmul_transpose_b ins(%[[I3]], %[[I4]] : tensor<128x1536xf32>, tensor<400x1536xf32>) outs(%[[I6]] : tensor<128x400xf32>) -> tensor<128x400xf32>
//       CHECK:   flow.dispatch.tensor.store %[[I7]], %[[I2]], offsets = [0, 0], sizes = [128, 400], strides = [1, 1] : tensor<128x400xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x400xf32>>

// -----

func.func @matmul_dispatch_3() {
  %c3244032 = arith.constant 3244032 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c3244032) alignment(64) : !flow.dispatch.tensor<readonly:tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<128x386xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>> -> tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %3 = iree_linalg_ext.unset_encoding %2 : tensor<128x400xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>> -> tensor<128x400xf32>
  %extracted_slice = tensor.extract_slice %3[0, 0] [128, 386] [1, 1] : tensor<128x400xf32> to tensor<128x386xf32>
  flow.dispatch.tensor.store %extracted_slice, %1, offsets = [0, 0], sizes = [128, 386], strides = [1, 1] : tensor<128x386xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x386xf32>>
  return
}

//       CHECK: func @matmul_dispatch_3()
//       CHECK:   %[[C1:.+]] = arith.constant 3244032 : index
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[I0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C1]]) alignment(64) : !flow.dispatch.tensor<readonly:tensor<128x400xf32>>
//       CHECK:   %[[I1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<128x386xf32>>
//       CHECK:   %[[I2:.+]] = flow.dispatch.tensor.load %[[I0]], offsets = [0, 0], sizes = [128, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x400xf32>> -> tensor<128x400xf32>
//       CHECK:   %[[I3:.+]] = tensor.extract_slice %[[I2]][0, 0] [128, 386] [1, 1] : tensor<128x400xf32> to tensor<128x386xf32>
//       CHECK:   flow.dispatch.tensor.store %extracted_slice, %[[I1]], offsets = [0, 0], sizes = [128, 386], strides = [1, 1] : tensor<128x386xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x386xf32>>

