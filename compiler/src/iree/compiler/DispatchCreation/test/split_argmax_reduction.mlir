// RUN: iree-opt --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-split-reduction-ops,cse, canonicalize))' --iree-dispatch-creation-split-argmax-reduction=4 %s | FileCheck %s

util.func public @argmax(%arg0: tensor<?x131072xbf16>, %arg1: index) -> tensor<?xi64> {
  %cst = arith.constant 0xFF80 : bf16
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty(%arg1) : tensor<?xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<?xbf16>) -> tensor<?xbf16>
  %2 = tensor.empty(%arg1) : tensor<?xi64>
  %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<?xi64>) -> tensor<?xi64>
  %4:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x131072xbf16>) outs(%1, %3 : tensor<?xbf16>, tensor<?xi64>) {
  ^bb0(%in: bf16, %out: bf16, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : bf16
    %8 = arith.cmpf ogt, %in, %out : bf16
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : bf16, i64
  } -> (tensor<?xbf16>, tensor<?xi64>)
  util.return %4#1 : tensor<?xi64>
}

// CHECK-LABEL:   util.func public @argmax

// Check identity value preparation
// CHECK: %[[CST:.+]] = arith.constant 0xFF80 : bf16
// CHECK: %[[ZERO:.+]] = arith.constant 0 : i64
// CHECK: %[[FINALVAL_EMPTY:.+]] = tensor.empty(%[[ARG1:.+]]) : tensor<?xbf16>
// CHECK: %[[FINALVAL:.+]] = linalg.fill ins(%[[CST]] : bf16) outs(%[[FINALVAL_EMPTY]] : tensor<?xbf16>) -> tensor<?xbf16>
// CHECK: %[[FINALIDX_EMPTY:.+]] = tensor.empty(%[[ARG1]]) : tensor<?xi64>
// CHECK: %[[FINALIDX:.+]] = linalg.fill ins(%[[ZERO]] : i64) outs(%[[FINALIDX_EMPTY]] : tensor<?xi64>) -> tensor<?xi64>

// Check partial reduction
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %arg0 {{\[}}[0], [1, 2]] output_shape [%{{.+}}, 4, 32768] : tensor<?x131072xbf16> into tensor<?x4x32768xbf16>
// CHECK: %[[INITVAL:.+]] = tensor.empty(%{{.+}}) : tensor<?x4xbf16>
// CHECK: %[[FILLVAL:.+]] = linalg.fill ins(%{{.+}} : bf16) outs(%[[INITVAL]] : tensor<?x4xbf16>) -> tensor<?x4xbf16>
// CHECK: %[[INITIDX:.+]] = tensor.empty(%{{.+}}) : tensor<?x4xi64>
// CHECK: %[[FILLIDX:.+]] = linalg.fill ins(%{{.+}} : i64) outs(%[[INITIDX]] : tensor<?x4xi64>) -> tensor<?x4xi64>

// CHECK: %[[PARTIAL:.+]]:2 = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[EXPAND]] : tensor<?x4x32768xbf16>)
// CHECK-SAME: outs(%[[FILLVAL]], %[[FILLIDX]] : tensor<?x4xbf16>, tensor<?x4xi64>)
// CHECK: ^bb0(%[[VAL:.+]]: bf16, %[[ACC:.+]]: bf16, %[[IDX:.+]]: i64)
// CHECK: %[[OUTER:.+]] = linalg.index 1 : index
// CHECK: %[[INNER:.+]] = linalg.index 2 : index
// CHECK: %[[OFFSET:.+]] = arith.muli %[[OUTER]], %{{.+}} : index
// CHECK: %[[GIDX:.+]] = arith.addi %[[OFFSET]], %[[INNER]] : index
// CHECK: %[[CAST:.+]] = arith.index_cast %[[GIDX]] : index to i64
// CHECK: %[[MAX:.+]] = arith.maximumf %[[VAL]], %[[ACC]] : bf16
// CHECK: %[[CMP:.+]] = arith.cmpf ogt, %[[VAL]], %[[ACC]] : bf16
// CHECK: %[[SEL:.+]] = arith.select %[[CMP]], %[[CAST]], %[[IDX]] : i64
// CHECK: linalg.yield %[[MAX]], %[[SEL]] : bf16, i64

// Final reduction
// CHECK: %[[FINAL:.+]]:2 = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%[[PARTIAL]]#0, %[[PARTIAL]]#1 : tensor<?x4xbf16>, tensor<?x4xi64>)
// CHECK-SAME: outs(%[[FINALVAL]], %[[FINALIDX]] : tensor<?xbf16>, tensor<?xi64>)
// CHECK: ^bb0(%[[V1:.+]]: bf16, %[[I1:.+]]: i64, %[[V2:.+]]: bf16, %[[I2:.+]]: i64)
// CHECK: %[[MAX2:.+]] = arith.maximumf %[[V1]], %[[V2]] : bf16
// CHECK: %[[CMP2:.+]] = arith.cmpf ogt, %[[V1]], %[[V2]] : bf16
// CHECK: %[[SEL2:.+]] = arith.select %[[CMP2]], %[[I1]], %[[I2]] : i64
// CHECK: linalg.yield %[[MAX2]], %[[SEL2]] : bf16, i64

// Final return
// CHECK: util.return %[[FINAL]]#1 : tensor<?xi64>
