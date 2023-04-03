// uCUDA kernels compiled by CLANG as linked as BITCODE by llvm compiler
// iree-compile matmul.mlir --iree-hal-target-backends=cuda \
// --iree-codegen-llvmgpu-enable-microkernels=true \
// --iree-hal-cuda-llvm-target-arch=sm_80 \
// -o matmul.vmfb

// Run the program `fill_matmul` function
// iree-run-module --module=matmul.vmfb --device=cuda \
//    --function=fill_matmul \
//    --input=2048x2048xf32=3 \
//    --input=2048x2048xf32=5 \
//    --input=2048x2048xf32=1 

// Run the program `fill_matmul_generic` function
// iree-run-module --module=matmul.vmfb --device=cuda \
//    --function=fill_matmul_generic \
//    --input=2048x2048xf32=3 \
//    --input=2048x2048xf32=5 \
//    --input=2048x2048xf32=1 

!tlhs = tensor<2048x2048xf32>
!rlhs = tensor<2048x2048xf32>
!tres = tensor<2048x2048xf32>

func.func @matmul(
  %lhs : !tlhs, %rhs : !rlhs, %accum : !tres) -> !tres {
  %matmul = linalg.matmul 
                ins(%lhs, %rhs : !tlhs, !rlhs) 
                outs(%accum : !tres) -> !tres
  return %matmul: !tres
}

func.func @fill_matmul(%lhs : !tlhs,   %rhs : !rlhs,   %accum : !tres) -> !tres {
  %cst = arith.constant 40.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%accum : !tres) -> !tres  
  %matmul = linalg.matmul 
                ins(%lhs, %rhs : !tlhs, !rlhs) 
                outs(%out : !tres) -> !tres
  return %matmul: !tres
}

func.func @fill_matmul_generic(
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


func.func @matmul_generic(
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
