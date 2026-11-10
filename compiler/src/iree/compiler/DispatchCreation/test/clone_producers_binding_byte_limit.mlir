// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-clone-producers-into-dispatch-regions))" %s | FileCheck %s --check-prefix=CLONE
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-clone-producers-into-dispatch-regions{max-binding-bytes=100000}))" %s | FileCheck %s --check-prefix=CLONE-LIMIT

// Each global below is 512x64xf16, 65536 bytes, so a dispatch reading two of
// them binds more than the 100000 byte limit.

// Cloning the slice into the dispatch would make it bind the whole table
// instead of the 1 KiB slice, next to the table the dispatch already reads.

#map = affine_map<(d0, d1) -> (d0, d1)>
util.global private @table0 = dense<1.0> : tensor<512x64xf16>
util.global private @table1 = dense<2.0> : tensor<512x64xf16>
util.func public @clone_slice() -> tensor<8x64xf16> {
  %table0 = util.global.load @table0 : tensor<512x64xf16>
  %table1 = util.global.load @table1 : tensor<512x64xf16>
  %slice = tensor.extract_slice %table0[0, 0] [8, 64] [1, 1] : tensor<512x64xf16> to tensor<8x64xf16>
  %empty = tensor.empty() : tensor<8x64xf16>
  %0 = flow.dispatch.region -> (tensor<8x64xf16>) {
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%slice : tensor<8x64xf16>) outs(%empty : tensor<8x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %v = tensor.extract %table1[%i, %j] : tensor<512x64xf16>
      %add = arith.addf %in, %v : f16
      linalg.yield %add : f16
    } -> tensor<8x64xf16>
    flow.return %1 : tensor<8x64xf16>
  }
  util.return %0 : tensor<8x64xf16>
}

// CLONE-LABEL: util.func public @clone_slice
//       CLONE:   flow.dispatch.region
//       CLONE:     tensor.empty
//       CLONE:     %[[SLICE:.+]] = tensor.extract_slice
//       CLONE:     linalg.generic
//  CLONE-SAME:         ins(%[[SLICE]] :

// CLONE-LIMIT-LABEL: util.func public @clone_slice
//       CLONE-LIMIT:   %[[SLICE:.+]] = tensor.extract_slice
//       CLONE-LIMIT:   flow.dispatch.region
//       CLONE-LIMIT:     tensor.empty
//   CLONE-LIMIT-NOT:     tensor.extract_slice
//       CLONE-LIMIT:     linalg.generic
//  CLONE-LIMIT-SAME:         ins(%[[SLICE]] :
