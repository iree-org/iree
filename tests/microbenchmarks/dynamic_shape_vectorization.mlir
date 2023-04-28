// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu-features=host --Xcompiler,iree-codegen-llvm-generic-ops-workgroup-size=2048 %s

//===----------------------------------------------------------------------===//
// Dynamic shape micro-benchmarks.
// Naming convention: '_'.join(
//   [dynamic,
//    {op-kind])
//
//===----------------------------------------------------------------------===//

func.func @dynamic_matmul() -> tensor<?x?xf32> {
  %c0 = arith.constant 0.000000e+00 : f32
  %dim0 = util.unfoldable_constant 257 : index
  %dim1 = util.unfoldable_constant 513 : index
  %dim2 = util.unfoldable_constant 385 : index

  %A = flow.tensor.constant  dense<1.0> : tensor<513x257xf32> -> tensor<?x?xf32>
  %B = flow.tensor.constant  dense<2.0> : tensor<257x385xf32> -> tensor<?x?xf32>
  %C = flow.tensor.constant  dense<0.0> : tensor<513x385xf32> -> tensor<?x?xf32>

  %gemm = linalg.matmul
      ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %gemm : tensor<?x?xf32>
}

func.func @dynamic_elw() -> tensor<?x?xf32> {
  %c0 = arith.constant 0.000000e+00 : f32
  %A = flow.tensor.constant  dense<1.0> : tensor<513x1025xf32> -> tensor<?x?xf32>
  %B = flow.tensor.constant  dense<2.0> : tensor<513x1025xf32> -> tensor<?x?xf32>
  %C = flow.tensor.constant  dense<0.0> : tensor<513x1025xf32> -> tensor<?x?xf32>

  %gen = linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)> ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%C : tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %add0 = arith.addf %a, %b : f32
    %mul0 = arith.mulf %add0, %add0 : f32
    %div_b0 = arith.divf %mul0, %b : f32
    %div_a0 = arith.divf %mul0, %a : f32
    %sub0 = arith.subf %div_b0, %div_a0 : f32
    %res0 = arith.mulf %sub0, %sub0 : f32
    %add1 = arith.addf %res0, %b : f32
    %mul1 = arith.mulf %add1, %add1 : f32
    %div_b1 = arith.divf %mul1, %b : f32
    %div_a1 = arith.divf %mul1, %a : f32
    %sub1 = arith.subf %div_b1, %div_a1 : f32
    %res1 = arith.mulf %sub1, %sub1 : f32
    %add2 = arith.addf %res1, %b : f32
    %mul2 = arith.mulf %add2, %add2 : f32
    %div_b2 = arith.divf %mul2, %b : f32
    %div_a2 = arith.divf %mul2, %a : f32
    %sub2 = arith.subf %div_b2, %div_a2 : f32
    %res2 = arith.mulf %sub2, %sub2 : f32
    %add3 = arith.addf %res2, %b : f32
    %mul3 = arith.mulf %add3, %add3 : f32
    %div_b3 = arith.divf %mul3, %b : f32
    %div_a3 = arith.divf %mul3, %a : f32
    %sub3 = arith.subf %div_b3, %div_a3 : f32
    %res3 = arith.mulf %sub3, %sub3 : f32
    %add4 = arith.addf %res3, %b : f32
    %mul4 = arith.mulf %add4, %add4 : f32
    %div_b4 = arith.divf %mul4, %b : f32
    %div_a4 = arith.divf %mul4, %a : f32
    %sub4 = arith.subf %div_b4, %div_a4 : f32
    %res4 = arith.mulf %sub4, %sub4 : f32
    linalg.yield %res4 : f32
  } -> tensor<?x?xf32>
  return %gen : tensor<?x?xf32>
}
