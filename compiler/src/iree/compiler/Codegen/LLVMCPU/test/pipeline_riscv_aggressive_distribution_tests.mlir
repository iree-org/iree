// RUN: iree-opt --iree-llvmcpu-riscv-aggressive-distribution=true --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+zvl1024b,+v", data_layout = "e-m:e-p:64:64-i64:64-i256:256-n32:64-S256", native_vector_size = 256 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
builtin.module {
  func.func @f32_rvv_matmul() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
    %cst = arith.constant 0.0 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x256xf32>>
    %lhs = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
    %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>> -> tensor<512x256xf32>
    %init = tensor.empty() : tensor<384x256xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x256xf32>) -> tensor<384x256xf32>
    %res = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x256xf32>) outs(%fill : tensor<384x256xf32>) -> tensor<384x256xf32>
    iree_tensor_ext.dispatch.tensor.store %res, %2, offsets = [0, 0], sizes = [384, 256], strides = [1, 1] : tensor<384x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x256xf32>>
    return
  }
}
// CHECK-LABEL: func.func @f32_rvv_matmul(
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c7:.+]] = arith.constant 7 : index
// CHECK-DAG:     %[[c128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[c256:.+]] = arith.constant 256 : index
// CHECK-DAG:     %[[c512:.+]] = arith.constant 512 : index
// CHECK:       scf.for {{.*}} step %[[c7]]
// CHECK:         scf.for {{.*}} step %[[c128]]
// CHECK:           scf.for {{.*}} step %[[c1]]
// CHECK-COUNT-7:     vector.fma
// CHECK-COUNT-7:   vector.store
// CHECK:       scf.for {{.*}} step %[[c128]]
// CHECK:           scf.for {{.*}} step %[[c1]]
// CHECK-COUNT-4:     vector.fma
// CHECK-COUNT-4:   vector.store
