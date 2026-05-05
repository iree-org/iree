// RUN: iree-opt --iree-codegen-llvmcpu-configuration-pipeline --iree-codegen-llvmcpu-lowering-pipeline='include-llvm-lowering=false' --iree-llvmcpu-mlir-opt-level=O2 --iree-llvmcpu-experimental-vectorize-to-transfer-gather --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+zvl256b,+v,+zvfh", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 64 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf"}>
func.func @non_projected_perm_generic_vectorized() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x642x642xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x3x3x320x320xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [3, 642, 642], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x642x642xf32>> -> tensor<3x642x642xf32>
  %3 = tensor.empty() : tensor<3x3x3x320x320xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d3 * 2 + d1, d4 * 2 + d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<3x642x642xf32>) outs(%3 : tensor<3x3x3x320x320xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x3x3x320x320xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0, 0], sizes = [3, 3, 3, 320, 320], strides = [1, 1, 1, 1, 1] : tensor<3x3x3x320x320xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x3x3x320x320xf32>>
  return
}

// CHECK-LABEL: func.func @non_projected_perm_generic_vectorized
//       CHECK:   vector.gather

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+zvl256b,+v,+zvfh", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 64 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf"}>
func.func @full_gather_read_vectorized() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128xi64>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8xi64>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x8xf16>>
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8xf16>> -> tensor<8192x8xf16>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [128], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128xi64>> -> tensor<128xi64>
  %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [8], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8xi64>> -> tensor<8xi64>
  %7 = tensor.empty() : tensor<128x8xf16>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : tensor<128xi64>, tensor<8xi64>) outs(%7 : tensor<128x8xf16>) {
  ^bb0(%in: i64, %in_0: i64, %out: f16):
    %9 = arith.index_cast %in : i64 to index
    %10 = arith.index_cast %in_0 : i64 to index
    %extracted = tensor.extract %4[%9, %10] : tensor<8192x8xf16>
    linalg.yield %extracted : f16
  } -> tensor<128x8xf16>
  iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [128, 8], strides = [1, 1] : tensor<128x8xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x8xf16>>
  return
}

// CHECK-LABEL: func.func @full_gather_read_vectorized
//       CHECK:   vector.gather
