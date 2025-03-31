// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:     --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target))" %s | \
// RUN: FileCheck %s

func.func @sort1D() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readwrite:tensor<4xi32>>
  %1 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<4xi32>> -> tensor<4xi32>
  %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<4xi32>
  flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<readwrite:tensor<4xi32>>
  return
}

//   CHECK-LABEL:  func.func @sort1D
//         CHECK:      amdgpu.fat_raw_buffer_cast
//         CHECK:      memref.assume_alignment

// -----

func.func @sort2D_static_shape() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readwrite:tensor<2000x30000xi32>>
  %1 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 4], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<2000x30000xi32>> -> tensor<2000x30000xi32>
  %2 = iree_linalg_ext.sort dimension(1) outs(%1 : tensor<2000x30000xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<2000x30000xi32>
  flow.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [2000, 30000], strides = [1, 1] : tensor<2000x30000xi32> -> !flow.dispatch.tensor<readwrite:tensor<2000x30000xi32>>
  return
}

//   CHECK-LABEL:  func.func @sort2D_static_shape
//         CHECK:      amdgpu.fat_raw_buffer_cast
//         CHECK:      memref.assume_alignment
//         CHECK:      scf.forall
//         CHECK:        memref.subview
//         CHECK:        scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
func.func @sort3D_dynamic_shape() {
  %c0 = arith.constant 0 : index
  %i0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %i1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %idx0 = arith.index_castui %i0 : i32 to index
  %idx1 = arith.index_castui %i1 : i32 to index
  %ord1 = flow.dispatch.workload.ordinal %idx1, 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%idx0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%ord1}
  %1 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [%ord1, 2, 4], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%ord1} -> tensor<?x2x4xi32>
  %2 = iree_linalg_ext.sort dimension(2) outs(%1 : tensor<?x2x4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<?x2x4xi32>
  flow.dispatch.tensor.store %2, %0, offsets = [0, 0, 0], sizes = [%ord1, 2, 4], strides = [1, 1, 1] : tensor<?x2x4xi32> -> !flow.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%ord1}
  return
}

//   CHECK-LABEL:  func.func @sort3D_dynamic_shape
//         CHECK:      amdgpu.fat_raw_buffer_cast
//         CHECK:      memref.assume_alignment
//         CHECK:      scf.forall
//         CHECK:       scf.for
//         CHECK:         memref.subview
