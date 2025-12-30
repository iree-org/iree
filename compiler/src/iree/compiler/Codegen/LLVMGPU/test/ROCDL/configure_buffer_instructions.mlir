// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN: --pass-pipeline="builtin.module(func.func(iree-rocdl-configure-buffer-instructions))" %s \
// RUN:  | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_80 \
// RUN: -pass-pipeline="builtin.module(func.func(iree-rocdl-configure-buffer-instructions))" %s \
// RUN:  | FileCheck --check-prefix=CUDA %s

// CUDA-NOT: iree_gpu.use_rocdl_buffer_instructions

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK-LABEL: @const_size_no_offset
// CHECK: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @const_size_no_offset() {
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: @const_size_zero_offset
// CHECK: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @const_size_zero_offset() {
  %c0 = arith.constant 0 : index
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64) offset(%c0)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: @const_size_too_big
// CHECK-NOT: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @const_size_too_big() {
  %c0 = arith.constant 0 : index
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64) offset(%c0)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024x1024xf16>>
  return
}

// CHECK-LABEL: @const_size_const_offset
// CHECK: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @const_size_const_offset() {
  %c8192 = arith.constant 8192 : index
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64) offset(%c8192)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: @const_size_i32_offset
// CHECK: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @const_size_i32_offset() {
  %off.low = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %off = arith.index_castui %off.low : i32 to index
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64) offset(%off)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: @const_size_i64_offset
// CHECK: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @const_size_i64_offset() {
  %c32_i64 = arith.constant 32 : i64
  %off.low = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %off.high = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %off.low.i64 = arith.extui %off.low : i32 to i64
  %off.high.i64 = arith.extui %off.high : i32 to i64
  %off.high.shl = arith.shli %off.high.i64, %c32_i64 : i64
  %off.i64 = arith.ori %off.low.i64, %off.high.shl : i64
  %off = arith.index_castui %off.i64 : i64 to index
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64) offset(%off)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: @const_size_nonuniform_offset_workgroup_id
// CHECK-NOT: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @const_size_nonuniform_offset_workgroup_id() {
  %c8192 = arith.constant 8192 : index
  %wgid = hal.interface.workgroup.id[0] : index
  %off = arith.muli %wgid, %c8192 : index
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64) offset(%off)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xf32>>
  return
}

// CHECK-LABEL: @any_dyn_size
// CHECK-NOT: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @any_dyn_size() {
  %c0 = arith.constant 0 : index
  %m.i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %m = arith.index_castui %m.i32 : i32 to index
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64) offset(%c0)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x256xf32>>{%m}
  return
}

// CHECK-LABEL: @assume_dyn_size
// CHECK: iree_gpu.use_rocdl_buffer_instructions
// CHECK: return
func.func @assume_dyn_size() {
  %c0 = arith.constant 0 : index
  %m.i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %m = arith.index_castui %m.i32 : i32 to index
  %m.assume = util.assume.int %m[<umin = 0, umax = 0>, <umin = 4, umax = 2048, udiv = 4>] : index
  %bind = hal.interface.binding.subspan layout(#pipeline_layout)
    binding(0) alignment(64) offset(%c0)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x256xf32>>{%m.assume}
  return
}
