// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op, iree-codegen-gpu-apply-tiling-level, canonicalize, cse))" %s | FileCheck %s

#config = #iree_gpu.lowering_config<
  {
    workgroup = [1, 32, 0, 0],
    reduction = [0, 0, 0, 32]
  }
>

#QK_trait = {
  indexing_maps = [
    affine_map<(B, M, K2, K1) -> (B, M, K1)>,
    affine_map<(B, M, K2, K1) -> (B, K2, K1)>,
    affine_map<(B, M, K2, K1) -> (B, M, K2)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
}

#S_elementwise = {
  indexing_maps = [
    affine_map<(B, M, K2) -> (B, M, K2)>,
    affine_map<(B, M, K2) -> (B, M, K2)>
  ],
  iterator_types = ["parallel", "parallel", "parallel"]
}

#PV_sum_div = {
  indexing_maps = [
    affine_map<(B, M, N) -> (B, M, N)>,
    affine_map<(B, M, N) -> (B, M, N)>,
    affine_map<(B, M, N) -> (B, M, N)>
  ],
  iterator_types = ["parallel", "parallel", "parallel"]
}

func.func @main(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) {
  %cst0 = arith.constant 0.0 : f32
  %cst = arith.constant 1.250000e-01 : f32
  %c0 = arith.constant 0 : index
  %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>

  %S_empty = tensor.empty() : tensor<20x4096x4096xf32>
  %S_fill  = linalg.fill ins(%cst0 : f32)
                         outs(%S_empty : tensor<20x4096x4096xf32>)
                         -> tensor<20x4096x4096xf32>

  %S = linalg.generic #QK_trait
                      ins(%4, %5 : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>)
                      outs(%S_empty : tensor<20x4096x4096xf32>) {
  ^bb0(%q : f16, %k : f16, %s : f32):
    %q_32 = arith.extf %q : f16 to f32
    %k_32 = arith.extf %k : f16 to f32
    %mul  = arith.mulf %q_32, %k_32 : f32
    %sum  = arith.addf %mul, %s : f32
    linalg.yield %sum : f32
  } -> tensor<20x4096x4096xf32>

  %scaled_S = linalg.generic #S_elementwise
                             ins(%S : tensor<20x4096x4096xf32>)
                             outs(%S_empty : tensor<20x4096x4096xf32>) {
  ^bb0(%a : f32, %b : f32):
    %out = arith.mulf %a, %cst : f32
    linalg.yield %out : f32
  } -> tensor<20x4096x4096xf32>

  %red_empty = tensor.empty() : tensor<20x4096x64xf32>

  %max_el = arith.constant -3.40282347E+38 : f32
  %max_init = linalg.fill ins(%max_el : f32)
                          outs(%red_empty : tensor<20x4096x64xf32>)
                          -> tensor<20x4096x64xf32>

  %sum_el = arith.constant 0.000000e+00 : f32
  %sum_init = linalg.fill ins(%sum_el : f32)
                          outs(%red_empty : tensor<20x4096x64xf32>)
                          -> tensor<20x4096x64xf32>
  %acc_init = linalg.fill ins(%sum_el : f32)
                          outs(%red_empty : tensor<20x4096x64xf32>)
                          -> tensor<20x4096x64xf32>

  %MAX, %SUM, %PV = iree_linalg_ext.exp_reduction {
    indexing_maps = [
      affine_map<(B, M, N, K2) -> (B, M, K2)>,
      affine_map<(B, M, N, K2) -> (B, K2, N)>,
      affine_map<(B, M, N, K2) -> (B, M, N)>,
      affine_map<(B, M, N, K2) -> (B, M, N)>,
      affine_map<(B, M, N, K2) -> (B, M, N)>
    ],
    iterator_types = [
    #iree_linalg_ext.iterator_type<parallel>,
    #iree_linalg_ext.iterator_type<parallel>,
    #iree_linalg_ext.iterator_type<parallel>,
    #iree_linalg_ext.iterator_type<reduction>
    ]
  } attributes {
    lowering_config = #config
  } ins(%scaled_S, %6 : tensor<20x4096x4096xf32>, tensor<20x4096x64xf16>)
  outs(%max_init, %sum_init, %acc_init : tensor<20x4096x64xf32>, tensor<20x4096x64xf32>, tensor<20x4096x64xf32>) {
  ^bb0(%ex : f32, %v : f16, %m : f32, %sum : f32, %acc : f32):
    %nsum = arith.addf %ex, %sum : f32
    %v_32 = arith.extf %v : f16 to f32
    %mul  = arith.mulf %ex, %v_32 : f32
    %nacc = arith.addf %mul, %acc : f32
    iree_linalg_ext.yield %m, %nsum, %nacc : f32, f32, f32
  } -> tensor<20x4096x64xf32>, tensor<20x4096x64xf32>, tensor<20x4096x64xf32>

  %result_empty = tensor.empty() : tensor<20x4096x64xf16>

  %result = linalg.generic #PV_sum_div
                           ins(%PV, %SUM : tensor<20x4096x64xf32>, tensor<20x4096x64xf32>)
                           outs(%result_empty : tensor<20x4096x64xf16>) {
  ^bb0(%pv : f32, %sum : f32, %res : f16):
    %out = arith.divf %pv, %sum : f32
    %trunc = arith.truncf %out : f32 to f16
    linalg.yield %trunc : f16
  } -> tensor<20x4096x64xf16>

  flow.dispatch.tensor.store %result, %3, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
  return
}
