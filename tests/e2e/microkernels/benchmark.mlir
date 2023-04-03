// uCUDA kernels compiled by CLANG as linked as BITCODE
// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN: --iree-codegen-llvmgpu-enable-microkernels=true \
// RUN: --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN: --iree-hal-cuda-link-uk-bitcode=true \
// RUN: --debug-only=iree-codegen-llvmgpu-kernelconfig

// uCUDA kernels compiled by NVCC linked as PTX
// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN: --iree-codegen-llvmgpu-enable-microkernels=true \
// RUN: --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN: --debug-only=iree-codegen-llvmgpu-kernelconfig
    

#s1688gemm_128x128_32x5_nn_align4 = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[128, 128, 32]]>,
  translation_info = <LLVMGPUMicroKernel
  pipeline_depth = 5>,
  workgroup_size = [128 : index, 1 : index, 1 : index]
>

#s1688gemm_128x128_16x5_nn_align4 = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[128, 128, 16]]>,
  translation_info = <LLVMGPUMicroKernel
  pipeline_depth = 5>,
  workgroup_size = [128 : index, 1 : index, 1 : index]
>

#s1688gemm_128x256_32x3_nn_align4 = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[128, 256, 32]]>,
  translation_info = <LLVMGPUMicroKernel
  pipeline_depth = 3>,
  workgroup_size = [256 : index, 1 : index, 1 : index]
>

#s1688gemm_256x128_32x3_nn_align4 = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[256, 128, 32]]>,
  translation_info = <LLVMGPUMicroKernel
  pipeline_depth = 3>,
  workgroup_size = [256 : index, 1 : index, 1 : index]
>

func.func @matmul_fill_1024(  %lhs: tensor<1024x1024xf32>,  %rhs: tensor<1024x1024xf32>,
  %init: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_128x128_32x5_nn_align4} 
                     ins(%lhs, %rhs: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%out: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %result : tensor<1024x1024xf32>
}

func.func @matmul_fill_2048(
  %lhs: tensor<2048x2048xf32>,  %rhs: tensor<2048x2048xf32>,
  %init: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_128x128_32x5_nn_align4} 
                     ins(%lhs, %rhs: tensor<2048x2048xf32>, tensor<2048x2048xf32>)
                     outs(%out: tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  return %result : tensor<2048x2048xf32>
}

func.func @matmul_fill_3456(
  %lhs: tensor<3456x3456xf32>,
  %rhs: tensor<3456x3456xf32>,
  %init: tensor<3456x3456xf32>) -> tensor<3456x3456xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<3456x3456xf32>) -> tensor<3456x3456xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_128x128_16x5_nn_align4} 
                     ins(%lhs, %rhs: tensor<3456x3456xf32>, tensor<3456x3456xf32>)
                     outs(%out: tensor<3456x3456xf32>) -> tensor<3456x3456xf32>
  return %result : tensor<3456x3456xf32>
}

func.func @matmul_fill_4096(
  %lhs: tensor<4096x4096xf32>,
  %rhs: tensor<4096x4096xf32>,
  %init: tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_128x256_32x3_nn_align4} 
                     ins(%lhs, %rhs: tensor<4096x4096xf32>, tensor<4096x4096xf32>)
                     outs(%out: tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %result : tensor<4096x4096xf32>
}

func.func @matmul_fill_6912(
  %lhs: tensor<6912x6912xf32>,
  %rhs: tensor<6912x6912xf32>,
  %init: tensor<6912x6912xf32>) -> tensor<6912x6912xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<6912x6912xf32>) -> tensor<6912x6912xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_128x256_32x3_nn_align4} 
                     ins(%lhs, %rhs: tensor<6912x6912xf32>, tensor<6912x6912xf32>)
                     outs(%out: tensor<6912x6912xf32>) -> tensor<6912x6912xf32>
  return %result : tensor<6912x6912xf32>
}

func.func @matmul_fill_8192(
  %lhs: tensor<8192x8192xf32>,
  %rhs: tensor<8192x8192xf32>,
  %init: tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_128x256_32x3_nn_align4} 
                     ins(%lhs, %rhs: tensor<8192x8192xf32>, tensor<8192x8192xf32>)
                     outs(%out: tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
  return %result : tensor<8192x8192xf32>
}


func.func @matmul_fill_10496(
  %lhs: tensor<10496x10496xf32>,
  %rhs: tensor<10496x10496xf32>,
  %init: tensor<10496x10496xf32>) -> tensor<10496x10496xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<10496x10496xf32>) -> tensor<10496x10496xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_256x128_32x3_nn_align4} 
                     ins(%lhs, %rhs: tensor<10496x10496xf32>, tensor<10496x10496xf32>)
                     outs(%out: tensor<10496x10496xf32>) -> tensor<10496x10496xf32>
  return %result : tensor<10496x10496xf32>
}

func.func @matmul_fill_16384(
  %lhs: tensor<16384x16384xf32>,
  %rhs: tensor<16384x16384xf32>,
  %init: tensor<16384x16384xf32>) -> tensor<16384x16384xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<16384x16384xf32>) -> tensor<16384x16384xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_128x256_32x3_nn_align4} 
                     ins(%lhs, %rhs: tensor<16384x16384xf32>, tensor<16384x16384xf32>)
                     outs(%out: tensor<16384x16384xf32>) -> tensor<16384x16384xf32>
  return %result : tensor<16384x16384xf32>
}


func.func @matmul_fill_27648(
  %lhs: tensor<27648x27648xf32>,
  %rhs: tensor<27648x27648xf32>,
  %init: tensor<27648x27648xf32>) -> tensor<27648x27648xf32>
{
  %cst = arith.constant 0.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<27648x27648xf32>) -> tensor<27648x27648xf32>
  %result = linalg.matmul {compilation_info = #s1688gemm_128x256_32x3_nn_align4} 
                     ins(%lhs, %rhs: tensor<27648x27648xf32>, tensor<27648x27648xf32>)
                     outs(%out: tensor<27648x27648xf32>) -> tensor<27648x27648xf32>
  return %result : tensor<27648x27648xf32>
}