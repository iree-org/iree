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

func.func @matmul_fill(
  %lhs : !tlhs, 
  %rhs : !rlhs, 
  %accum : !tres) -> !tres 
{
  // Producer: Fill tensor with constant
  %cst = arith.constant 3.4 : f32
  %out = linalg.fill ins(%cst : f32) outs(%accum : !tres) -> !tres  
  
  // Matmul operation
  %matmul = linalg.matmul
                ins(%lhs, %rhs : !tlhs, !rlhs) 
                outs(%out : !tres) -> !tres

  // Consumer: use result of the matmul
  %empty = tensor.empty() : !tres  
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                       affine_map<(d0, d1) -> (d0, d1)>, 
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
  ins(%matmul, %accum : !tres,!tres) outs(%empty : !tres) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %res = arith.divf %arg1, %arg2 : f32
      linalg.yield %res : f32
    } -> !tres

  return %result: !tres
}


func.func @matmul_nofill(
  %lhs : !tlhs, 
  %rhs : !rlhs, 
  %accum : !tres) -> !tres 
{
  // Matmul operation
  %matmul = linalg.matmul
                ins(%lhs, %rhs : !tlhs, !rlhs) 
                outs(%accum : !tres) -> !tres

  // Consumer: use result of the matmul
  %empty = tensor.empty() : !tres  
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                       affine_map<(d0, d1) -> (d0, d1)>, 
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
  ins(%matmul, %accum : !tres,!tres) outs(%empty : !tres) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %res = arith.divf %arg1, %arg2 : f32
      linalg.yield %res : f32
    } -> !tres

  return %result: !tres
}


