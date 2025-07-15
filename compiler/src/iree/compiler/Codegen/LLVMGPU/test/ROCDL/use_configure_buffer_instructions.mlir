// RUN: iree-opt --iree-gpu-test-target=gfx942 \
// RUN: --pass-pipeline="builtin.module(func.func(iree-rocdl-use-buffer-instructions))" %s \
// RUN:  | FileCheck %s
// RUN: iree-opt --iree-gpu-test-target=sm_80 \
// RUN: -pass-pipeline="builtin.module(func.func(iree-rocdl-use-buffer-instructions))" %s \
// RUN:  | FileCheck --check-prefix=CUDA %s

// CUDA-NOT: iree_gpu.use_rocdl_buffer_instructions

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK-LABEL: @small_dyn_buffer_small_slice
// CHECK: iree_gpu.buffer_resource_cast
func.func @small_dyn_buffer_small_slice() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 4095> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = tensor.extract_slice %3[0, 0] [32, 32] [1, 1] : tensor<32x?xi64> to tensor<32x32xi64>
  return
}

// CHECK-LABEL: @big_buffer_small_slice
// CHECK: iree_gpu.buffer_resource_cast
func.func @big_buffer_small_slice() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 8589934592> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = tensor.extract_slice %3[0, 0] [32, 32] [1, 1] : tensor<32x?xi64> to tensor<32x32xi64>
  return
}

// CHECK-LABEL: @big_buffer_small_dyn_slice
// CHECK: iree_gpu.buffer_resource_cast
func.func @big_buffer_small_dyn_slice() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 8589934592> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %5 = util.assume.int %1<umin = 1, umax = 8192> : index
  %6 = tensor.extract_slice %3[0, 0] [32, %5] [1, 1] : tensor<32x?xi64> to tensor<32x?xi64>
  return
}

// CHECK-LABEL: @big_buffer_big_dyn_slice
// CHECK-NOT: iree_gpu.buffer_resource_cast
func.func @big_buffer_big_dyn_slice() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 8589934592> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %5 = util.assume.int %1<umin = 1, umax = 8589934592> : index
  %6 = tensor.extract_slice %3[0, 0] [32, %5] [1, 1] : tensor<32x?xi64> to tensor<32x?xi64>
  return
}

// CHECK-LABEL: @dependent_offsets
// CHECK-NOT: iree_gpu.buffer_resource_cast
func.func @dependent_offsets() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 8192> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = gpu.thread_id x
  %5 = tensor.extract_slice %3[0, %4] [32, 32] [1, 1] : tensor<32x?xi64> to tensor<32x32xi64>
  return
}

// CHECK-LABEL: @uniform_loop
// CHECK: iree_gpu.buffer_resource_cast
func.func @uniform_loop() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 8589934592> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %5 = util.assume.int %1<umin = 1, umax = 8192> : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %5 step %c1 {
    %6 = tensor.extract_slice %3[%iv, 0] [32, %5] [1, 1] : tensor<32x?xi64> to tensor<32x?xi64>
    scf.yield
  }
  return
}

// CHECK-LABEL: @non_uniform_loop
// CHECK-NOT: iree_gpu.buffer_resource_cast
func.func @non_uniform_loop() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 8589934592> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %5 = util.assume.int %1<umin = 1, umax = 8192> : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %6 = gpu.thread_id x
  scf.for %iv = %c0 to %6 step %c1 {
    %7 = tensor.extract_slice %3[%iv, 0] [32, %5] [1, 1] : tensor<32x?xi64> to tensor<32x?xi64>
    scf.yield
  }
  return
}

// CHECK-LABEL: @uniform_forall
// CHECK: iree_gpu.buffer_resource_cast
func.func @uniform_forall() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 8589934592> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %5 = util.assume.int %1<umin = 1, umax = 8192> : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %thread_id_x = gpu.thread_id  x
  scf.forall (%arg0) in (32) {
    %extracted_slice = tensor.extract_slice %3[%arg0, 0] [1, %5] [1, 1] : tensor<32x?xi64> to tensor<1x?xi64>
  }
  return
}

// CHECK-LABEL: @non_uniform_forall
// CHECK-NOT: iree_gpu.buffer_resource_cast
func.func @non_uniform_forall() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %2 = util.assume.int %1<umin = 1, umax = 8589934592> : index
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?xi64>>{%1} -> tensor<32x?xi64>
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %5 = util.assume.int %1<umin = 1, umax = 8192> : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %thread_id_x = gpu.thread_id  x
  scf.forall (%arg0) in (%thread_id_x) {
    %extracted_slice = tensor.extract_slice %3[%arg0, 0] [1, %5] [1, 1] : tensor<32x?xi64> to tensor<1x?xi64>
  }
  return
}
