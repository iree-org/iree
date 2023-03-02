// Baseline IREE
// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN: --iree-codegen-llvmgpu-enable-microkernels=false \
// RUN: --iree-hal-cuda-llvm-target-arch=sm_80 

// uCUDA kernels compiled by NVCC linked as PTX by ptxas compiler
// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN: --iree-codegen-llvmgpu-enable-microkernels=true \
// RUN: --iree-hal-cuda-llvm-target-arch=sm_80 
    
// uCUDA kernels compiled by CLANG as linked as BITCODE by llvm compiler
// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN: --iree-codegen-llvmgpu-enable-microkernels=true \
// RUN: --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN: --iree-hal-cuda-link-uk-bitcode=true

!tlhs = tensor<1792x512xf32>
!rlhs = tensor<512x2048xf32>
!tres = tensor<1792x2048xf32>

func.func @matmul_fill(%lhs : !tlhs,   %rhs : !rlhs,   %accum : !tres) -> !tres {
  %cst = arith.constant 40.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%accum : !tres) -> !tres  
  %matmul = linalg.matmul 
                ins(%lhs, %rhs : !tlhs, !rlhs) 
                outs(%out : !tres) -> !tres
  return %matmul: !tres
}

func.func @matmul_no_fill(
  %lhs : !tlhs, %rhs : !rlhs, %accum : !tres) -> !tres {
  %matmul = linalg.matmul 
                ins(%lhs, %rhs : !tlhs, !rlhs) 
                outs(%accum : !tres) -> !tres
  return %matmul: !tres
}
