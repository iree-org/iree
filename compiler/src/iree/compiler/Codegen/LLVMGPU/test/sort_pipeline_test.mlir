// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:     --pass-pipeline="builtin.module(func.func(iree-llvmgpu-lower-executable-target))" %s | \
// RUN: FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [1, 1, 1] subgroup_size = 32>
#lowering_config = #iree_gpu.lowering_config<{thread = [0], workgroup = [0]}>
module {
  func.func @sort1D() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi32>>
    %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi32>> -> tensor<4xi32>
    %2 = iree_linalg_ext.sort {lowering_config = #lowering_config} dimension(0) outs(%1 : tensor<4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %3 = arith.cmpi slt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %3 : i1
    } -> tensor<4xi32>
    iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi32>>
    return
  }
}

//   CHECK-LABEL:  func.func @sort1D
//         CHECK:      amdgpu.fat_raw_buffer_cast

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
#lowering_config = #iree_gpu.lowering_config<{thread = [1], workgroup = [1]}>
module {
  func.func @sort2D_static_shape() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2000x30000xi32>>
    %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2000x30000xi32>> -> tensor<2000x30000xi32>
    %2 = iree_linalg_ext.sort {lowering_config = #lowering_config} dimension(1) outs(%1 : tensor<2000x30000xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %3 = arith.cmpi slt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %3 : i1
    } -> tensor<2000x30000xi32>
    iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [2000, 30000], strides = [1, 1] : tensor<2000x30000xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2000x30000xi32>>
    return
  }
}

//   CHECK-LABEL:  func.func @sort2D_static_shape
//         CHECK:      amdgpu.fat_raw_buffer_cast
//         CHECK:      scf.forall
//         CHECK:        memref.subview
//         CHECK:        scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
#lowering_config = #iree_gpu.lowering_config<{thread = [1, 1], workgroup = [1, 2]}>
module {
  func.func @sort3D_dynamic_shape() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
    %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
    %2 = arith.index_castui %0 : i32 to index
    %3 = arith.index_castui %1 : i32 to index
    %4 = iree_tensor_ext.dispatch.workload.ordinal %3, 0 : index
    %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%2) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%4}
    %6 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [%4, 2, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%4} -> tensor<?x2x4xi32>
    %7 = iree_linalg_ext.sort {lowering_config = #lowering_config} dimension(2) outs(%6 : tensor<?x2x4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %8 = arith.cmpi slt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %8 : i1
    } -> tensor<?x2x4xi32>
    iree_tensor_ext.dispatch.tensor.store %7, %5, offsets = [0, 0, 0], sizes = [%4, 2, 4], strides = [1, 1, 1] : tensor<?x2x4xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%4}
    return
  }
}

//   CHECK-LABEL:  func.func @sort3D_dynamic_shape
//         CHECK:      amdgpu.fat_raw_buffer_cast
//         CHECK:      scf.forall
//         CHECK:       scf.for
//         CHECK:         memref.subview
