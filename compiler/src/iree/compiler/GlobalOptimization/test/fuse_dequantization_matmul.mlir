// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-fuse-dequantization-matmul{enable-quantized-matmul-reassociation=true},canonicalize))" %s | FileCheck %s --check-prefix=REASSOCIATE-CHECK
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-fuse-dequantization-matmul,canonicalize))" %s | FileCheck %s --check-prefix=FUSE-CHECK

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
//   REASSOCIATE-CHECK-DAG: #[[MAP0:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d0, d1)>
//   REASSOCIATE-CHECK-DAG: #[[MAP1:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d0)>
//   REASSOCIATE-CHECK-DAG: #[[MAP2:[a-zA-Z0-9]+]] = affine_map<(d0) -> (d0)>
//   REASSOCIATE-CHECK-DAG: #[[MAP3:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   REASSOCIATE-CHECK-DAG: #[[MAP4:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   REASSOCIATE-CHECK-DAG: #[[MAP5:[a-zA-Z0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//   REASSOCIATE-CHECK-DAG: #[[MAP6:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d1)>
//       REASSOCIATE-CHECK: func.func @grouped_quantized_matmul_reassociate(
//  REASSOCIATE-CHECK-SAME:   %[[QUANT:[a-zA-Z0-9_]+]]: tensor<11008x32x128xi4>
//  REASSOCIATE-CHECK-SAME:   %[[UNQUANT:[a-zA-Z0-9_]+]]: tensor<32x128xf32>
//  REASSOCIATE-CHECK-SAME:   %[[SCALES:[a-zA-Z0-9_]+]]: tensor<11008x32xf32>
//  REASSOCIATE-CHECK-SAME:   %[[ZPS:[a-zA-Z0-9_]+]]: tensor<11008x32xf32>
//       REASSOCIATE-CHECK:   %[[C0I32:.+]] = arith.constant 0 : i32
//       REASSOCIATE-CHECK:   %[[RANGE:.+]] = arith.constant 3.276700e+04 : f32
//       REASSOCIATE-CHECK:   %[[C0F32:.+]] = arith.constant 0.000000e+00 : f32
//       REASSOCIATE-CHECK:   %[[INITOUT:.+]] = tensor.empty() : tensor<11008xf32>
//       REASSOCIATE-CHECK:   %[[FILLOUT:.+]] = linalg.fill ins(%[[C0F32]]
//  REASSOCIATE-CHECK-SAME:       outs(%[[INITOUT]] :
//       REASSOCIATE-CHECK:   %[[INITMAX:.+]] = tensor.empty() : tensor<32xf32>
//       REASSOCIATE-CHECK:   %[[FILLMAX:.+]] = linalg.fill ins(%[[C0F32]]
//  REASSOCIATE-CHECK-SAME:       outs(%[[INITMAX]] :
//       REASSOCIATE-CHECK:   %[[GENMAX:.+]] = linalg.generic
//  REASSOCIATE-CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  REASSOCIATE-CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  REASSOCIATE-CHECK-SAME:       ins(%[[UNQUANT]] :
//  REASSOCIATE-CHECK-SAME:       outs(%[[FILLMAX]] :
//       REASSOCIATE-CHECK:   ^bb0(%[[MAXIN0:.+]]: f32, %[[MAXOUT0:.+]]: f32):
//       REASSOCIATE-CHECK:   %[[MAXABSF:.+]] = math.absf %[[MAXIN0]] : f32
//       REASSOCIATE-CHECK:   %[[MAXMAXIMUMF:.+]] = arith.maximumf %[[MAXABSF]], %[[MAXOUT0]] : f32
//       REASSOCIATE-CHECK:   linalg.yield %[[MAXMAXIMUMF]] : f32
//       REASSOCIATE-CHECK:   %[[INITSCALES:.+]] = tensor.empty() : tensor<32xf32>
//       REASSOCIATE-CHECK:   %[[GENSCALES:.+]] = linalg.generic
//  REASSOCIATE-CHECK-SAME:       indexing_maps = [#[[MAP2]], #[[MAP2]]]
//  REASSOCIATE-CHECK-SAME:       iterator_types = ["parallel"]
//  REASSOCIATE-CHECK-SAME:       ins(%[[GENMAX]] :
//  REASSOCIATE-CHECK-SAME:       outs(%[[INITSCALES]] :
//       REASSOCIATE-CHECK:   ^bb0(%[[SCALESIN0:.+]]: f32, %[[SCALESOUT0:.+]]: f32):
//       REASSOCIATE-CHECK:   %[[SCALESDIVF:.+]] = arith.divf %[[SCALESIN0]], %[[RANGE]] : f32
//       REASSOCIATE-CHECK:   linalg.yield %[[SCALESDIVF]] : f32
//       REASSOCIATE-CHECK:   %[[INITSUM:.+]] = tensor.empty() : tensor<32xf32>
//       REASSOCIATE-CHECK:   %[[FILLSUM:.+]] = linalg.fill ins(%[[C0F32]]
//  REASSOCIATE-CHECK-SAME:       outs(%[[INITSUM]] :
//       REASSOCIATE-CHECK:   %[[GENSUM:.+]] = linalg.generic
//  REASSOCIATE-CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  REASSOCIATE-CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  REASSOCIATE-CHECK-SAME:       ins(%[[UNQUANT]] :
//  REASSOCIATE-CHECK-SAME:       outs(%[[FILLSUM]] :
//       REASSOCIATE-CHECK:   ^bb0(%[[SUMIN0:.+]]: f32, %[[SUMOUT0:.+]]: f32):
//       REASSOCIATE-CHECK:   %[[SUMADDF:.+]] = arith.addf %[[SUMIN0]], %[[SUMOUT0]] : f32
//       REASSOCIATE-CHECK:   linalg.yield %[[SUMADDF]] : f32
//       REASSOCIATE-CHECK:   %[[INITQUANT:.+]] = tensor.empty() : tensor<32x128xi16>
//       REASSOCIATE-CHECK:   %[[GENQUANT:.+]] = linalg.generic
//  REASSOCIATE-CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP0]]]
//  REASSOCIATE-CHECK-SAME:       iterator_types = ["parallel", "parallel"]
//  REASSOCIATE-CHECK-SAME:       ins(%[[UNQUANT]], %[[GENSCALES]] :
//  REASSOCIATE-CHECK-SAME:       outs(%[[INITQUANT]] :
//       REASSOCIATE-CHECK:   ^bb0(%[[QUANTIN0:.+]]: f32, %[[QUANTIN1:.+]]: f32, %[[QUANTOUT0:.+]]: i16):
//       REASSOCIATE-CHECK:   %[[QUANTDIVF:.+]] = arith.divf %[[QUANTIN0]], %[[QUANTIN1]] : f32
//       REASSOCIATE-CHECK:   %[[QUANTFPTOSI:.+]] = arith.fptosi %[[QUANTDIVF]] : f32 to i16
//       REASSOCIATE-CHECK:   linalg.yield %[[QUANTFPTOSI]] : i16
//       REASSOCIATE-CHECK:   %[[INITMATMUL:.+]] = tensor.empty() : tensor<11008x32xi32>
//       REASSOCIATE-CHECK:   %[[FILLMATMUL:.+]] = linalg.fill ins(%[[C0I32]]
//  REASSOCIATE-CHECK-SAME:       outs(%[[INITMATMUL]] :
//       REASSOCIATE-CHECK:   %[[GENMATMUL:.+]] = linalg.generic
//  REASSOCIATE-CHECK-SAME:       indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]]
//  REASSOCIATE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  REASSOCIATE-CHECK-SAME:       ins(%[[GENQUANT]], %[[QUANT]] :
//  REASSOCIATE-CHECK-SAME:       outs(%[[FILLMATMUL]] :
//       REASSOCIATE-CHECK:   ^bb0(%[[MATMULIN0:.+]]: i16, %[[MATMULIN1:.+]]: i4, %[[MATMULOUT0:.+]]: i32):
//   REASSOCIATE-CHECK-DAG:   %[[MATMULEXTSI:.+]] = arith.extsi %[[MATMULIN0]] : i16 to i32
//   REASSOCIATE-CHECK-DAG:   %[[MATMULEXTUI:.+]] = arith.extui %[[MATMULIN1]] : i4 to i32
//       REASSOCIATE-CHECK:   %[[MATMULMULI:.+]] = arith.muli %[[MATMULEXTSI]], %[[MATMULEXTUI]] : i32
//       REASSOCIATE-CHECK:   %[[MATMULADDI:.+]] = arith.addi %[[MATMULMULI]], %[[MATMULOUT0]] : i32
//       REASSOCIATE-CHECK:   linalg.yield %[[MATMULADDI]] : i32
//       REASSOCIATE-CHECK:   %[[GENREASSOCIATE:.+]] = linalg.generic
//  REASSOCIATE-CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP6]], #[[MAP6]], #[[MAP0]], #[[MAP0]], #[[MAP1]]]
//  REASSOCIATE-CHECK-SAME:       iterator_types = ["parallel", "reduction"]
//  REASSOCIATE-CHECK-SAME:       ins(%[[GENMATMUL]], %[[GENSCALES]], %[[GENSUM]], %[[SCALES]], %[[ZPS]] :
//  REASSOCIATE-CHECK-SAME:       outs(%[[FILLOUT]] :
//       REASSOCIATE-CHECK:   ^bb0(%[[REIN0:.+]]: i32, %[[REIN1:.+]]: f32, %[[REIN2:.+]]: f32, %[[REIN3:.+]]: f32, %[[REIN4:.+]]: f32, %[[REOUT0:.+]]: f32):
//   REASSOCIATE-CHECK-DAG:   %[[RESITOFP:.+]] = arith.sitofp %[[REIN0]] : i32 to f32
//   REASSOCIATE-CHECK-DAG:   %[[REMULF0:.+]] = arith.mulf %[[RESITOFP]], %[[REIN1]] : f32
//   REASSOCIATE-CHECK-DAG:   %[[REMULF1:.+]] = arith.mulf %[[REMULF0]], %[[REIN3]] : f32
//   REASSOCIATE-CHECK-DAG:   %[[REMULF2:.+]] = arith.mulf %[[REIN4]], %[[REIN3]] : f32
//   REASSOCIATE-CHECK-DAG:   %[[REMULF3:.+]] = arith.mulf %[[REMULF2]], %[[REIN2]] : f32
//       REASSOCIATE-CHECK:   %[[RESUBF:.+]] = arith.subf %[[REMULF1]], %[[REMULF3]] : f32
//       REASSOCIATE-CHECK:   %[[READDF:.+]] = arith.addf %[[RESUBF]], %[[REOUT0]] : f32
//       REASSOCIATE-CHECK:   linalg.yield %[[READDF]] : f32
//       REASSOCIATE-CHECK:   return %[[GENREASSOCIATE]]

// -----

module {
  func.func @grouped_quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} 
        ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    return %4 : tensor<1x1x4096xf32>
  }
}
//       FUSE-CHECK: func.func @grouped_quantized_matmul(
//  FUSE-CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  FUSE-CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  FUSE-CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  FUSE-CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//       FUSE-CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       FUSE-CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       FUSE-CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       FUSE-CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       FUSE-CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  FUSE-CHECK-SAME:       outs(%[[INIT1]] :
//       FUSE-CHECK:   %[[GEN0:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  FUSE-CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  FUSE-CHECK-SAME:       outs(%[[INIT0]] :
//       FUSE-CHECK:   %[[GEN1:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  FUSE-CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  FUSE-CHECK-SAME:       outs(%[[FILL]] :
//       FUSE-CHECK:   flow.return %[[GEN1]] :
//       FUSE-CHECK:   return %[[DISP]]

// -----

module {
  func.func @nofill_grouped_quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>, %arg4: tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} 
        ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%arg4 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    return %4 : tensor<1x1x4096xf32>
  }
}
//       FUSE-CHECK: func.func @nofill_grouped_quantized_matmul(
//  FUSE-CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  FUSE-CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  FUSE-CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  FUSE-CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  FUSE-CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<1x1x4096xf32>
//       FUSE-CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       FUSE-CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       FUSE-CHECK:   %[[GEN0:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  FUSE-CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  FUSE-CHECK-SAME:       outs(%[[INIT0]] :
//       FUSE-CHECK:   %[[GEN1:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  FUSE-CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  FUSE-CHECK-SAME:       outs(%[[ARG4]] :
//       FUSE-CHECK:   flow.return %[[GEN1]] :
//       FUSE-CHECK:   return %[[DISP]]

// -----

module {
  func.func @grouped_quantized_matvec(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<32x128xf32>, %arg2: tensor<4096x32xf32>, %arg3: tensor<4096x32xf32>) -> tensor<4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0)>], 
        iterator_types = ["parallel", "reduction", "reduction"]} 
        ins(%arg1, %3 : tensor<32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<4096xf32>
    return %4 : tensor<4096xf32>
  }
}
//       FUSE-CHECK: func.func @grouped_quantized_matvec(
//  FUSE-CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  FUSE-CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32x128xf32>
//  FUSE-CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//  FUSE-CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//       FUSE-CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       FUSE-CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<4096xf32>
//       FUSE-CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       FUSE-CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<4096xf32>)
//       FUSE-CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  FUSE-CHECK-SAME:       outs(%[[INIT1]] :
//       FUSE-CHECK:   %[[GEN0:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  FUSE-CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  FUSE-CHECK-SAME:       outs(%[[INIT0]] :
//       FUSE-CHECK:   %[[GEN1:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "reduction", "reduction"]
//  FUSE-CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  FUSE-CHECK-SAME:       outs(%[[FILL]] :
//       FUSE-CHECK:   flow.return %[[GEN1]] :
//       FUSE-CHECK:   return %[[DISP]]

// -----

module {
  func.func @ungrouped_quantized_matmul(%arg0: tensor<4096x32xi8>, %arg1: tensor<1x1x32xf32>, %arg2: tensor<4096x32xf32>, %arg3: tensor<4096x32xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                         affine_map<(d0, d1) -> (d0, d1)>, 
                         affine_map<(d0, d1) -> (d0, d1)>, 
                         affine_map<(d0, d1) -> (d0, d1)>], 
        iterator_types = ["parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%1 : tensor<4096x32xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, 
                         affine_map<(d0, d1, d2, d3) -> (d2, d3)>, 
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]} 
        ins(%arg1, %3 : tensor<1x1x32xf32>, tensor<4096x32xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    return %4 : tensor<1x1x4096xf32>
  }
}

//       FUSE-CHECK: func.func @ungrouped_quantized_matmul(
//  FUSE-CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32xi8>
//  FUSE-CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32xf32>
//  FUSE-CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//  FUSE-CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32xf32>
//       FUSE-CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       FUSE-CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       FUSE-CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32xf32>
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  FUSE-CHECK-SAME:       outs(%[[INIT1]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[GEN0:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel"]
//  FUSE-CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  FUSE-CHECK-SAME:       outs(%[[INIT0]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[GEN1:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  FUSE-CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  FUSE-CHECK-SAME:       outs(%[[FILL]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   return %[[GEN1]]

// -----

module {
  func.func @non_dequantization(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      linalg.yield %6 : f32
    } -> tensor<4096x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} 
        ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    return %4 : tensor<1x1x4096xf32>
  }
}
//       FUSE-CHECK: func.func @non_dequantization(
//  FUSE-CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  FUSE-CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  FUSE-CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  FUSE-CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//       FUSE-CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       FUSE-CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       FUSE-CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  FUSE-CHECK-SAME:       outs(%[[INIT1]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[GEN0:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  FUSE-CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  FUSE-CHECK-SAME:       outs(%[[INIT0]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[GEN1:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  FUSE-CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  FUSE-CHECK-SAME:       outs(%[[FILL]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   return %[[GEN1]]

// -----

module {
  func.func @non_dequantization(%arg0: tensor<4096x32x128xi32>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi32>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i32, %in_0: f32, %in_1: f32, %out: f32):
      %6 = arith.uitofp %in : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} 
        ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    return %4 : tensor<1x1x4096xf32>
  }
}
//       FUSE-CHECK: func.func @non_dequantization(
//  FUSE-CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi32>
//  FUSE-CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  FUSE-CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  FUSE-CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//       FUSE-CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       FUSE-CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       FUSE-CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  FUSE-CHECK-SAME:       outs(%[[INIT1]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[GEN0:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  FUSE-CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  FUSE-CHECK-SAME:       outs(%[[INIT0]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   %[[GEN1:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  FUSE-CHECK-SAME:       ins(%[[ARG1]], %[[GEN0]] :
//  FUSE-CHECK-SAME:       outs(%[[FILL]] :
//   FUSE-CHECK-NOT:   flow.dispatch.region
//   FUSE-CHECK-NOT:   flow.return
//       FUSE-CHECK:   return %[[GEN1]]

// -----

module {
  func.func @clone_grouped_quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32x128xf32>
    %barrier = util.optimization_barrier %3 : tensor<4096x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} 
        ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    return %4 : tensor<1x1x4096xf32>
  }
}
//       FUSE-CHECK: func.func @clone_grouped_quantized_matmul(
//  FUSE-CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  FUSE-CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  FUSE-CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  FUSE-CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//       FUSE-CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       FUSE-CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       FUSE-CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       FUSE-CHECK:   %[[GEN0:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  FUSE-CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  FUSE-CHECK-SAME:       outs(%[[INIT0]] :
//       FUSE-CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       FUSE-CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  FUSE-CHECK-SAME:       outs(%[[INIT1]] :
//       FUSE-CHECK:   %[[CLONE:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  FUSE-CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  FUSE-CHECK-SAME:       outs(%[[INIT0]] :
//       FUSE-CHECK:   %[[GEN1:.+]] = linalg.generic
//  FUSE-CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  FUSE-CHECK-SAME:       ins(%[[ARG1]], %[[CLONE]] :
//  FUSE-CHECK-SAME:       outs(%[[FILL]] :
//       FUSE-CHECK:   flow.return %[[GEN1]] :
//       FUSE-CHECK:   return %[[DISP]]
