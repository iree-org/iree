// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-fuse-dequantization-matmul{enable-quantized-matmul-reassociation=true},canonicalize))" %s | FileCheck %s

module {
  func.func @grouped_quantized_matmul_reassociate(%arg0: tensor<11008x32x128xi4>, %arg1: tensor<32x128xf32>, %arg2: tensor<11008x32xf32>, %arg3: tensor<11008x32xf32>) -> tensor<11008xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<11008xf32>
    %1 = tensor.empty() : tensor<11008x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<11008xf32>) -> tensor<11008xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg2, %arg3 : tensor<11008x32x128xi4>, tensor<11008x32xf32>, tensor<11008x32xf32>) outs(%1 : tensor<11008x32x128xf32>) {
    ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i4 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<11008x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0)>],
        iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%arg1, %3 : tensor<32x128xf32>, tensor<11008x32x128xf32>) outs(%2 : tensor<11008xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<11008xf32>
    return %4 : tensor<11008xf32>
  }
}
//   CHECK-DAG: #[[MAP0:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d0)>
//   CHECK-DAG: #[[MAP2:[a-zA-Z0-9]+]] = affine_map<(d0) -> (d0)>
//   CHECK-DAG: #[[MAP3:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   CHECK-DAG: #[[MAP4:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   CHECK-DAG: #[[MAP5:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//   CHECK-DAG: #[[MAP6:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d1)>
//       CHECK: func.func @grouped_quantized_matmul_reassociate(
//  CHECK-SAME:   %[[QUANT:[a-zA-Z0-9_]+]]: tensor<11008x32x128xi4>
//  CHECK-SAME:   %[[UNQUANT:[a-zA-Z0-9_]+]]: tensor<32x128xf32>
//  CHECK-SAME:   %[[SCALES:[a-zA-Z0-9_]+]]: tensor<11008x32xf32>
//  CHECK-SAME:   %[[ZPS:[a-zA-Z0-9_]+]]: tensor<11008x32xf32>
//       CHECK:   %[[C0I32:.+]] = arith.constant 0 : i32
//       CHECK:   %[[RANGE:.+]] = arith.constant 3.276700e+04 : f32
//       CHECK:   %[[C0F32:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INITOUT:.+]] = tensor.empty() : tensor<11008xf32>
//       CHECK:   %[[FILLOUT:.+]] = linalg.fill ins(%[[C0F32]]
//  CHECK-SAME:       outs(%[[INITOUT]] :
//       CHECK:   %[[INITMAX:.+]] = tensor.empty() : tensor<32xf32>
//       CHECK:   %[[FILLMAX:.+]] = linalg.fill ins(%[[C0F32]]
//  CHECK-SAME:       outs(%[[INITMAX]] :
//       CHECK:   %[[GENMAX:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:       ins(%[[UNQUANT]] :
//  CHECK-SAME:       outs(%[[FILLMAX]] :
//       CHECK:   ^bb0(%[[MAXIN0:.+]]: f32, %[[MAXOUT0:.+]]: f32):
//       CHECK:   %[[MAXABSF:.+]] = math.absf %[[MAXIN0]] : f32
//       CHECK:   %[[MAXMAXIMUMF:.+]] = arith.maximumf %[[MAXABSF]], %[[MAXOUT0]] : f32
//       CHECK:   linalg.yield %[[MAXMAXIMUMF]] : f32
//       CHECK:   %[[INITSCALES:.+]] = tensor.empty() : tensor<32xf32>
//       CHECK:   %[[GENSCALES:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP2]], #[[MAP2]]]
//  CHECK-SAME:       iterator_types = ["parallel"]
//  CHECK-SAME:       ins(%[[GENMAX]] :
//  CHECK-SAME:       outs(%[[INITSCALES]] :
//       CHECK:   ^bb0(%[[SCALESIN0:.+]]: f32, %[[SCALESOUT0:.+]]: f32):
//       CHECK:   %[[SCALESDIVF:.+]] = arith.divf %[[SCALESIN0]], %[[RANGE]] : f32
//       CHECK:   linalg.yield %[[SCALESDIVF]] : f32
//       CHECK:   %[[INITSUM:.+]] = tensor.empty() : tensor<32xf32>
//       CHECK:   %[[FILLSUM:.+]] = linalg.fill ins(%[[C0F32]]
//  CHECK-SAME:       outs(%[[INITSUM]] :
//       CHECK:   %[[GENSUM:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:       ins(%[[UNQUANT]] :
//  CHECK-SAME:       outs(%[[FILLSUM]] :
//       CHECK:   ^bb0(%[[SUMIN0:.+]]: f32, %[[SUMOUT0:.+]]: f32):
//       CHECK:   %[[SUMADDF:.+]] = arith.addf %[[SUMIN0]], %[[SUMOUT0]] : f32
//       CHECK:   linalg.yield %[[SUMADDF]] : f32
//       CHECK:   %[[INITQUANT:.+]] = tensor.empty() : tensor<32x128xi16>
//       CHECK:   %[[GENQUANT:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP0]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:       ins(%[[UNQUANT]], %[[GENSCALES]] :
//  CHECK-SAME:       outs(%[[INITQUANT]] :
//       CHECK:   ^bb0(%[[QUANTIN0:.+]]: f32, %[[QUANTIN1:.+]]: f32, %[[QUANTOUT0:.+]]: i16):
//       CHECK:   %[[QUANTDIVF:.+]] = arith.divf %[[QUANTIN0]], %[[QUANTIN1]] : f32
//       CHECK:   %[[QUANTFPTOSI:.+]] = arith.fptosi %[[QUANTDIVF]] : f32 to i16
//       CHECK:   linalg.yield %[[QUANTFPTOSI]] : i16
//       CHECK:   %[[INITMATMUL:.+]] = tensor.empty() : tensor<11008x32xi32>
//       CHECK:   %[[FILLMATMUL:.+]] = linalg.fill ins(%[[C0I32]]
//  CHECK-SAME:       outs(%[[INITMATMUL]] :
//       CHECK:   %[[GENMATMUL:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[GENQUANT]], %[[QUANT]] :
//  CHECK-SAME:       outs(%[[FILLMATMUL]] :
//       CHECK:   ^bb0(%[[MATMULIN0:.+]]: i16, %[[MATMULIN1:.+]]: i4, %[[MATMULOUT0:.+]]: i32):
//   CHECK-DAG:   %[[MATMULEXTSI:.+]] = arith.extsi %[[MATMULIN0]] : i16 to i32
//   CHECK-DAG:   %[[MATMULEXTUI:.+]] = arith.extui %[[MATMULIN1]] : i4 to i32
//       CHECK:   %[[MATMULMULI:.+]] = arith.muli %[[MATMULEXTSI]], %[[MATMULEXTUI]] : i32
//       CHECK:   %[[MATMULADDI:.+]] = arith.addi %[[MATMULMULI]], %[[MATMULOUT0]] : i32
//       CHECK:   linalg.yield %[[MATMULADDI]] : i32
//       CHECK:   %[[GENREASSOCIATE:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP6]], #[[MAP6]], #[[MAP0]], #[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:       ins(%[[GENMATMUL]], %[[GENSCALES]], %[[GENSUM]], %[[SCALES]], %[[ZPS]] :
//  CHECK-SAME:       outs(%[[FILLOUT]] :
//       CHECK:   ^bb0(%[[REIN0:.+]]: i32, %[[REIN1:.+]]: f32, %[[REIN2:.+]]: f32, %[[REIN3:.+]]: f32, %[[REIN4:.+]]: f32, %[[REOUT0:.+]]: f32):
//   CHECK-DAG:   %[[RESITOFP:.+]] = arith.sitofp %[[REIN0]] : i32 to f32
//   CHECK-DAG:   %[[REMULF0:.+]] = arith.mulf %[[RESITOFP]], %[[REIN1]] : f32
//   CHECK-DAG:   %[[REMULF1:.+]] = arith.mulf %[[REMULF0]], %[[REIN3]] : f32
//   CHECK-DAG:   %[[REMULF2:.+]] = arith.mulf %[[REIN4]], %[[REIN3]] : f32
//   CHECK-DAG:   %[[REMULF3:.+]] = arith.mulf %[[REMULF2]], %[[REIN2]] : f32
//       CHECK:   %[[RESUBF:.+]] = arith.subf %[[REMULF1]], %[[REMULF3]] : f32
//       CHECK:   %[[READDF:.+]] = arith.addf %[[RESUBF]], %[[REOUT0]] : f32
//       CHECK:   linalg.yield %[[READDF]] : f32
//       CHECK:   return %[[GENREASSOCIATE]]

// -----

module {
  func.func @grouped_quantized_matmul_reassociate_f16(%arg0: tensor<11008x32x128xi4>, %arg1: tensor<32x128xf16>, %arg2: tensor<11008x32xf16>, %arg3: tensor<11008x32xf16>) -> tensor<11008xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<11008xf16>
    %1 = tensor.empty() : tensor<11008x32x128xf16>
    %2 = linalg.fill ins(%cst : f16) outs(%0 : tensor<11008xf16>) -> tensor<11008xf16>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg2, %arg3 : tensor<11008x32x128xi4>, tensor<11008x32xf16>, tensor<11008x32xf16>) outs(%1 : tensor<11008x32x128xf16>) {
    ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
      %5 = arith.extui %in : i4 to i32
      %6 = arith.uitofp %5 : i32 to f16
      %7 = arith.subf %6, %in_1 : f16
      %8 = arith.mulf %7, %in_0 : f16
      linalg.yield %8 : f16
    } -> tensor<11008x32x128xf16>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0)>],
        iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%arg1, %3 : tensor<32x128xf16>, tensor<11008x32x128xf16>) outs(%2 : tensor<11008xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %5 = arith.mulf %in, %in_0 : f16
      %6 = arith.addf %5, %out : f16
      linalg.yield %6 : f16
    } -> tensor<11008xf16>
    return %4 : tensor<11008xf16>
  }
}

//   CHECK-DAG: #[[MAP0:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d0)>
//   CHECK-DAG: #[[MAP2:[a-zA-Z0-9]+]] = affine_map<(d0) -> (d0)>
//   CHECK-DAG: #[[MAP3:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   CHECK-DAG: #[[MAP4:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   CHECK-DAG: #[[MAP5:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//   CHECK-DAG: #[[MAP6:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d1)>
//       CHECK: func.func @grouped_quantized_matmul_reassociate_f16(
//  CHECK-SAME:   %[[QUANT:[a-zA-Z0-9_]+]]: tensor<11008x32x128xi4>
//  CHECK-SAME:   %[[UNQUANT:[a-zA-Z0-9_]+]]: tensor<32x128xf16>
//  CHECK-SAME:   %[[SCALES:[a-zA-Z0-9_]+]]: tensor<11008x32xf16>
//  CHECK-SAME:   %[[ZPS:[a-zA-Z0-9_]+]]: tensor<11008x32xf16>
//       CHECK:   %[[C0I32:.+]] = arith.constant 0 : i32
//       CHECK:   %[[RANGE:.+]] = arith.constant 3.276800e+04 : f16
//       CHECK:   %[[C0f16:.+]] = arith.constant 0.000000e+00 : f16
//       CHECK:   %[[INITOUT:.+]] = tensor.empty() : tensor<11008xf16>
//       CHECK:   %[[FILLOUT:.+]] = linalg.fill ins(%[[C0f16]]
//  CHECK-SAME:       outs(%[[INITOUT]] :
//       CHECK:   %[[INITMAX:.+]] = tensor.empty() : tensor<32xf16>
//       CHECK:   %[[FILLMAX:.+]] = linalg.fill ins(%[[C0f16]]
//  CHECK-SAME:       outs(%[[INITMAX]] :
//       CHECK:   %[[GENMAX:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:       ins(%[[UNQUANT]] :
//  CHECK-SAME:       outs(%[[FILLMAX]] :
//       CHECK:   ^bb0(%[[MAXIN0:.+]]: f16, %[[MAXOUT0:.+]]: f16):
//       CHECK:   %[[MAXABSF:.+]] = math.absf %[[MAXIN0]] : f16
//       CHECK:   %[[MAXMAXIMUMF:.+]] = arith.maximumf %[[MAXABSF]], %[[MAXOUT0]] : f16
//       CHECK:   linalg.yield %[[MAXMAXIMUMF]] : f16
//       CHECK:   %[[INITSCALES:.+]] = tensor.empty() : tensor<32xf16>
//       CHECK:   %[[GENSCALES:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP2]], #[[MAP2]]]
//  CHECK-SAME:       iterator_types = ["parallel"]
//  CHECK-SAME:       ins(%[[GENMAX]] :
//  CHECK-SAME:       outs(%[[INITSCALES]] :
//       CHECK:   ^bb0(%[[SCALESIN0:.+]]: f16, %[[SCALESOUT0:.+]]: f16):
//       CHECK:   %[[SCALESDIVF:.+]] = arith.divf %[[SCALESIN0]], %[[RANGE]] : f16
//       CHECK:   linalg.yield %[[SCALESDIVF]] : f16
//       CHECK:   %[[INITSUM:.+]] = tensor.empty() : tensor<32xf16>
//       CHECK:   %[[FILLSUM:.+]] = linalg.fill ins(%[[C0f16]]
//  CHECK-SAME:       outs(%[[INITSUM]] :
//       CHECK:   %[[GENSUM:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:       ins(%[[UNQUANT]] :
//  CHECK-SAME:       outs(%[[FILLSUM]] :
//       CHECK:   ^bb0(%[[SUMIN0:.+]]: f16, %[[SUMOUT0:.+]]: f16):
//       CHECK:   %[[SUMADDF:.+]] = arith.addf %[[SUMIN0]], %[[SUMOUT0]] : f16
//       CHECK:   linalg.yield %[[SUMADDF]] : f16
//       CHECK:   %[[INITQUANT:.+]] = tensor.empty() : tensor<32x128xi16>
//       CHECK:   %[[GENQUANT:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP0]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:       ins(%[[UNQUANT]], %[[GENSCALES]] :
//  CHECK-SAME:       outs(%[[INITQUANT]] :
//       CHECK:   ^bb0(%[[QUANTIN0:.+]]: f16, %[[QUANTIN1:.+]]: f16, %[[QUANTOUT0:.+]]: i16):
//       CHECK:   %[[QUANTDIVF:.+]] = arith.divf %[[QUANTIN0]], %[[QUANTIN1]] : f16
//       CHECK:   %[[QUANTFPTOSI:.+]] = arith.fptosi %[[QUANTDIVF]] : f16 to i16
//       CHECK:   linalg.yield %[[QUANTFPTOSI]] : i16
//       CHECK:   %[[INITMATMUL:.+]] = tensor.empty() : tensor<11008x32xi32>
//       CHECK:   %[[FILLMATMUL:.+]] = linalg.fill ins(%[[C0I32]]
//  CHECK-SAME:       outs(%[[INITMATMUL]] :
//       CHECK:   %[[GENMATMUL:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[GENQUANT]], %[[QUANT]] :
//  CHECK-SAME:       outs(%[[FILLMATMUL]] :
//       CHECK:   ^bb0(%[[MATMULIN0:.+]]: i16, %[[MATMULIN1:.+]]: i4, %[[MATMULOUT0:.+]]: i32):
//   CHECK-DAG:   %[[MATMULEXTSI:.+]] = arith.extsi %[[MATMULIN0]] : i16 to i32
//   CHECK-DAG:   %[[MATMULEXTUI:.+]] = arith.extui %[[MATMULIN1]] : i4 to i32
//       CHECK:   %[[MATMULMULI:.+]] = arith.muli %[[MATMULEXTSI]], %[[MATMULEXTUI]] : i32
//       CHECK:   %[[MATMULADDI:.+]] = arith.addi %[[MATMULMULI]], %[[MATMULOUT0]] : i32
//       CHECK:   linalg.yield %[[MATMULADDI]] : i32
//       CHECK:   %[[GENREASSOCIATE:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP6]], #[[MAP6]], #[[MAP0]], #[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:       ins(%[[GENMATMUL]], %[[GENSCALES]], %[[GENSUM]], %[[SCALES]], %[[ZPS]] :
//  CHECK-SAME:       outs(%[[FILLOUT]] :
//       CHECK:   ^bb0(%[[REIN0:.+]]: i32, %[[REIN1:.+]]: f16, %[[REIN2:.+]]: f16, %[[REIN3:.+]]: f16, %[[REIN4:.+]]: f16, %[[REOUT0:.+]]: f16):
//   CHECK-DAG:   %[[RESITOFP:.+]] = arith.sitofp %[[REIN0]] : i32 to f16
//   CHECK-DAG:   %[[REMULF0:.+]] = arith.mulf %[[RESITOFP]], %[[REIN1]] : f16
//   CHECK-DAG:   %[[REMULF1:.+]] = arith.mulf %[[REMULF0]], %[[REIN3]] : f16
//   CHECK-DAG:   %[[REMULF2:.+]] = arith.mulf %[[REIN4]], %[[REIN3]] : f16
//   CHECK-DAG:   %[[REMULF3:.+]] = arith.mulf %[[REMULF2]], %[[REIN2]] : f16
//       CHECK:   %[[RESUBF:.+]] = arith.subf %[[REMULF1]], %[[REMULF3]] : f16
//       CHECK:   %[[READDF:.+]] = arith.addf %[[RESUBF]], %[[REOUT0]] : f16
//       CHECK:   linalg.yield %[[READDF]] : f16
//       CHECK:   return %[[GENREASSOCIATE]]
