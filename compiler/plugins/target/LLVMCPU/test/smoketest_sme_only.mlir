// Regression test for a target with +sme but not +sve (e.g. Apple Silicon):
// compiling a scalable-vectorized matmul with SME tiling disabled used to
// crash LLVM instruction selection ("Cannot select: vscale") because the
// dispatch was left non-streaming despite containing scalable vector code
// with no non-streaming SVE to fall back on. See KernelDispatch.cpp
// (getMatmulVectorSizesUsingFillRegisterFileHeuristic,
// getVectorPreProcStrategy) and Passes.cpp (addLowerToLLVMPasses) /
// LLVMCPUTarget.cpp (requiresArmStreamingForScalableVectors).
//
// RUN: iree-opt --split-input-file --iree-stream-transformation-pipeline --iree-hal-transformation-pipeline --iree-llvmcpu-enable-scalable-vectorization=true --iree-llvmcpu-disable-arm-sme-tiling %s | FileCheck %s

module attributes {
  hal.device.targets = [
    #hal.device.target<"local", [
      #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
        cpu_features = "+sme,+sme2",
        data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
        native_vector_size = 16 : index,
        target_triple = "aarch64-none-elf"
      }>
    ]> : !hal.device
  ]
} {

stream.executable public @matmul_dispatch_0 {
  stream.executable.export @matmul_dispatch_0 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_dispatch_0(%lhs_binding: !stream.binding, %rhs_binding: !stream.binding, %out_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.0 : f32
      %lhs = stream.binding.subspan %lhs_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>>
      %rhs = stream.binding.subspan %rhs_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>>
      %out = stream.binding.subspan %out_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x64xf32>>
      %l = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<64x64xf32>
      %r = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<64x64xf32>
      %init = tensor.empty() : tensor<64x64xf32>
      %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
      %m = linalg.matmul ins(%l, %r : tensor<64x64xf32>, tensor<64x64xf32>) outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
      iree_tensor_ext.dispatch.tensor.store %m, %out, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : tensor<64x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x64xf32>>
      return
    }
  }
}

}

// CHECK:       hal.executable.binary public @embedded_elf_arm_64
// CHECK-SAME:     format = "embedded-elf-arm_64"
