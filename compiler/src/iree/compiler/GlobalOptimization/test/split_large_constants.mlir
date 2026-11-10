// RUN: iree-opt --split-input-file --verify-diagnostics --iree-global-opt-split-large-constants %s | FileCheck %s

// 6x4xf16 is 48 bytes; with a 16 byte limit it becomes three 2x4 shards. An
// embedding lookup becomes a chain of ops, one per shard.

#id = affine_map<(d0, d1) -> (d0)>
#out = affine_map<(d0, d1) -> (d0, d1)>
module @gather attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  util.global private @table = dense<[
    [0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0],
    [12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]
  ]> : tensor<6x4xf16>
  util.func public @main(%ids: tensor<8xi64>) -> tensor<8x4xf16> {
    %table = util.global.load immutable @table : tensor<6x4xf16>
    %empty = tensor.empty() : tensor<8x4xf16>
    %0 = linalg.generic {
        indexing_maps = [#id, #out], iterator_types = ["parallel", "parallel"]}
        ins(%ids : tensor<8xi64>) outs(%empty : tensor<8x4xf16>) {
    ^bb0(%in: i64, %out: f16):
      %i = arith.index_cast %in : i64 to index
      %j = linalg.index 1 : index
      %v = tensor.extract %table[%i, %j] : tensor<6x4xf16>
      linalg.yield %v : f16
    } -> tensor<8x4xf16>
    util.return %0 : tensor<8x4xf16>
  }
}

// CHECK-LABEL: module @gather
//   CHECK-NOT:   util.global private @table =
//       CHECK:   util.global private @table_shard0 = dense<{{\[\[}}0.000000e+00, {{.+}}, 7.000000e+00]]> : tensor<2x4xf16>
//       CHECK:   util.global private @table_shard1 = dense<{{\[\[}}8.000000e+00, {{.+}}, 1.500000e+01]]> : tensor<2x4xf16>
//       CHECK:   util.global private @table_shard2 = dense<{{\[\[}}1.600000e+01, {{.+}}, 2.300000e+01]]> : tensor<2x4xf16>
//       CHECK:   util.func public @main(%[[IDS:.+]]: tensor<8xi64>)
//   CHECK-DAG:     %[[S0:.+]] = util.global.load immutable @table_shard0 : tensor<2x4xf16>
//   CHECK-DAG:     %[[S1:.+]] = util.global.load immutable @table_shard1 : tensor<2x4xf16>
//   CHECK-DAG:     %[[S2:.+]] = util.global.load immutable @table_shard2 : tensor<2x4xf16>
//       CHECK:     %[[G0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[IDS]] : tensor<8xi64>)
//       CHECK:       %[[I0:.+]] = arith.index_cast
//       CHECK:       %[[LAST0:.+]] = arith.constant 1 : index
//       CHECK:       %[[CLAMP0:.+]] = arith.minui %[[I0]], %[[LAST0]]
//       CHECK:       %[[E0:.+]] = tensor.extract %[[S0]][%[[CLAMP0]], %{{.+}}]
//       CHECK:       linalg.yield %[[E0]]
//       CHECK:     %[[G1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[IDS]], %[[G0]] : tensor<8xi64>, tensor<8x4xf16>)
//       CHECK:     ^bb0(%{{.+}}: i64, %[[PREV1:.+]]: f16, %{{.+}}: f16):
//       CHECK:       %[[I1:.+]] = arith.index_cast
//       CHECK:       %[[OFF1:.+]] = arith.constant 2 : index
//       CHECK:       %[[REL1:.+]] = arith.subi %[[I1]], %[[OFF1]]
//       CHECK:       %[[CLAMP1:.+]] = arith.minui %[[REL1]]
//       CHECK:       %[[IN1:.+]] = arith.cmpi ult, %[[REL1]]
//       CHECK:       %[[E1:.+]] = tensor.extract %[[S1]][%[[CLAMP1]], %{{.+}}]
//       CHECK:       %[[SEL1:.+]] = arith.select %[[IN1]], %[[E1]], %[[PREV1]]
//       CHECK:       linalg.yield %[[SEL1]]
//       CHECK:     %[[G2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[IDS]], %[[G1]] : tensor<8xi64>, tensor<8x4xf16>)
//       CHECK:       arith.constant 4 : index
//       CHECK:       tensor.extract %[[S2]]
//       CHECK:       arith.select
//       CHECK:     util.return %[[G2]]

// -----

// A tied LM head: the oversized weight is N x K, so the split dimension is
// the parallel N loop. Each shard's matmul fills its own columns.

module @matmul_parallel attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  util.global private @weight = dense<1.0> : tensor<6x4xf16>
  util.func public @main(%lhs: tensor<3x4xf16>) -> tensor<3x6xf32> {
    %cst = arith.constant 0.0 : f32
    %weight = util.global.load immutable @weight : tensor<6x4xf16>
    %empty = tensor.empty() : tensor<3x6xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<3x6xf32>) -> tensor<3x6xf32>
    %0 = linalg.matmul
        indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                         affine_map<(m, n, k) -> (n, k)>,
                         affine_map<(m, n, k) -> (m, n)>]
        ins(%lhs, %weight : tensor<3x4xf16>, tensor<6x4xf16>)
        outs(%fill : tensor<3x6xf32>) -> tensor<3x6xf32>
    util.return %0 : tensor<3x6xf32>
  }
}

// CHECK-LABEL: module @matmul_parallel
//       CHECK:   util.global private @weight_shard0 = dense<1.000000e+00> : tensor<2x4xf16>
//       CHECK:   util.global private @weight_shard1 = dense<1.000000e+00> : tensor<2x4xf16>
//       CHECK:   util.global private @weight_shard2 = dense<1.000000e+00> : tensor<2x4xf16>
//       CHECK:   util.func public @main
//   CHECK-DAG:     %[[S0:.+]] = util.global.load immutable @weight_shard0
//   CHECK-DAG:     %[[S1:.+]] = util.global.load immutable @weight_shard1
//   CHECK-DAG:     %[[S2:.+]] = util.global.load immutable @weight_shard2
//       CHECK:     %[[FILL:.+]] = linalg.fill
//       CHECK:     %[[M0:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%{{.+}}, %[[S0]] : tensor<3x4xf16>, tensor<2x4xf16>)
//  CHECK-SAME:         -> tensor<3x2xf32>
//       CHECK:     %[[R0:.+]] = tensor.insert_slice %[[M0]] into %[[FILL]][0, 0] [3, 2] [1, 1]
//       CHECK:     %[[M1:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%{{.+}}, %[[S1]] : tensor<3x4xf16>, tensor<2x4xf16>)
//       CHECK:     %[[R1:.+]] = tensor.insert_slice %[[M1]] into %[[R0]][0, 2] [3, 2] [1, 1]
//       CHECK:     %[[M2:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%{{.+}}, %[[S2]] : tensor<3x4xf16>, tensor<2x4xf16>)
//       CHECK:     %[[R2:.+]] = tensor.insert_slice %[[M2]] into %[[R1]][0, 4] [3, 2] [1, 1]
//       CHECK:     util.return %[[R2]]

// -----

// The oversized weight is K x N, so the split dimension is the reduction K
// loop. Each shard's matmul accumulates into the previous one's result.

module @matmul_reduction attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  util.global private @weight = dense<1.0> : tensor<6x4xf16>
  util.func public @main(%lhs: tensor<3x6xf16>) -> tensor<3x4xf32> {
    %cst = arith.constant 0.0 : f32
    %weight = util.global.load immutable @weight : tensor<6x4xf16>
    %empty = tensor.empty() : tensor<3x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<3x4xf32>) -> tensor<3x4xf32>
    %0 = linalg.matmul ins(%lhs, %weight : tensor<3x6xf16>, tensor<6x4xf16>)
        outs(%fill : tensor<3x4xf32>) -> tensor<3x4xf32>
    util.return %0 : tensor<3x4xf32>
  }
}

// CHECK-LABEL: module @matmul_reduction
//       CHECK:   util.func public @main(%[[LHS:.+]]: tensor<3x6xf16>)
//   CHECK-DAG:     %[[S0:.+]] = util.global.load immutable @weight_shard0
//   CHECK-DAG:     %[[S1:.+]] = util.global.load immutable @weight_shard1
//   CHECK-DAG:     %[[S2:.+]] = util.global.load immutable @weight_shard2
//       CHECK:     %[[L0:.+]] = tensor.extract_slice %[[LHS]][0, 0] [3, 2] [1, 1]
//       CHECK:     %[[M0:.+]] = linalg.matmul ins(%[[L0]], %[[S0]] : tensor<3x2xf16>, tensor<2x4xf16>)
//       CHECK:     %[[L1:.+]] = tensor.extract_slice %[[LHS]][0, 2] [3, 2] [1, 1]
//       CHECK:     %[[M1:.+]] = linalg.matmul ins(%[[L1]], %[[S1]] : tensor<3x2xf16>, tensor<2x4xf16>) outs(%[[M0]] : tensor<3x4xf32>)
//       CHECK:     %[[L2:.+]] = tensor.extract_slice %[[LHS]][0, 4] [3, 2] [1, 1]
//       CHECK:     %[[M2:.+]] = linalg.matmul ins(%[[L2]], %[[S2]] : tensor<3x2xf16>, tensor<2x4xf16>) outs(%[[M1]] : tensor<3x4xf32>)
//       CHECK:     util.return %[[M2]]

// -----

// A leading unit dimension is skipped: 1x3x4xi16 (24 bytes, 8 per index of
// dimension 1) is split along dimension 1. The shard bytes are checked in
// split_large_constants_resource.mlir, as --split-input-file does not print
// resources.

module @resource attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  util.global private @table = dense_resource<blob> : tensor<1x3x4xi16>
  util.func public @main(%lhs: tensor<5x4xi16>) -> tensor<5x1x3xi16> {
    %table = util.global.load immutable @table : tensor<1x3x4xi16>
    %empty = tensor.empty() : tensor<5x1x3xi16>
    %0 = linalg.generic {
        indexing_maps = [affine_map<(m, b, n, k) -> (m, k)>,
                         affine_map<(m, b, n, k) -> (b, n, k)>,
                         affine_map<(m, b, n, k) -> (m, b, n)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %table : tensor<5x4xi16>, tensor<1x3x4xi16>)
        outs(%empty : tensor<5x1x3xi16>) {
    ^bb0(%a: i16, %b: i16, %acc: i16):
      %mul = arith.muli %a, %b : i16
      %add = arith.addi %acc, %mul : i16
      linalg.yield %add : i16
    } -> tensor<5x1x3xi16>
    util.return %0 : tensor<5x1x3xi16>
  }
}


// CHECK-LABEL: module @resource
//       CHECK:   util.global private @table_shard0 = dense_resource<table_shard0> : tensor<1x2x4xi16>
//       CHECK:   util.global private @table_shard1 = dense_resource<table_shard1> : tensor<1x1x4xi16>
//       CHECK:     %[[G0:.+]] = linalg.generic
//  CHECK-SAME:         tensor<5x4xi16>, tensor<1x2x4xi16>
//       CHECK:     %[[R0:.+]] = tensor.insert_slice %[[G0]] into %{{.+}}[0, 0, 0] [5, 1, 2] [1, 1, 1]
//       CHECK:     %[[G1:.+]] = linalg.generic
//  CHECK-SAME:         tensor<5x4xi16>, tensor<1x1x4xi16>
//       CHECK:     tensor.insert_slice %[[G1]] into %[[R0]][0, 0, 2] [5, 1, 1] [1, 1, 1]

{-#
  dialect_resources: {
    builtin: {
      blob: "0x02000000000001000200030004000500060007000800090010001100"
    }
  }
#-}

// -----

// A named parameter is split without reading it: each shard refers to its byte
// range of the same parameter. 6x4xf16 (48 bytes, 8 per row) becomes three 2x4
// shards of 16 bytes at offsets 0, 16 and 32.

module @parameter attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  util.global private @weight = #flow.parameter.named<"model"::"lm_head.weight"> : tensor<6x4xf16>
  util.func public @main(%lhs: tensor<3x4xf16>) -> tensor<3x6xf32> {
    %cst = arith.constant 0.0 : f32
    %weight = util.global.load immutable @weight : tensor<6x4xf16>
    %empty = tensor.empty() : tensor<3x6xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<3x6xf32>) -> tensor<3x6xf32>
    %0 = linalg.matmul
        indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                         affine_map<(m, n, k) -> (n, k)>,
                         affine_map<(m, n, k) -> (m, n)>]
        ins(%lhs, %weight : tensor<3x4xf16>, tensor<6x4xf16>)
        outs(%fill : tensor<3x6xf32>) -> tensor<3x6xf32>
    util.return %0 : tensor<3x6xf32>
  }
}

// CHECK-LABEL: module @parameter
//   CHECK-NOT:   util.global private @weight =
//       CHECK:   util.global private @weight_shard0 = #flow.parameter.named<"model"::"lm_head.weight", {length = 16 : i64, offset = 0 : i64}> : tensor<2x4xf16>
//       CHECK:   util.global private @weight_shard1 = #flow.parameter.named<"model"::"lm_head.weight", {length = 16 : i64, offset = 16 : i64}> : tensor<2x4xf16>
//       CHECK:   util.global private @weight_shard2 = #flow.parameter.named<"model"::"lm_head.weight", {length = 16 : i64, offset = 32 : i64}> : tensor<2x4xf16>
//       CHECK:   util.func public @main
//       CHECK:     linalg.matmul
//  CHECK-SAME:         tensor<3x4xf16>, tensor<2x4xf16>
//       CHECK:     linalg.matmul
//  CHECK-SAME:         tensor<3x4xf16>, tensor<2x4xf16>
//       CHECK:     linalg.matmul
//  CHECK-SAME:         tensor<3x4xf16>, tensor<2x4xf16>

// -----

// A parameter that already is a range of a larger entry: the shard offsets are
// relative to its offset of 1024, and other config entries are kept. 5x4xf16
// (40 bytes) becomes shards of 2, 2 and 1 rows, so the last one has 8 bytes.

#id = affine_map<(d0, d1) -> (d0)>
#out = affine_map<(d0, d1) -> (d0, d1)>
module @parameter_range attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  util.global private @table = #flow.parameter.named<"model"::"embed", {offset = 1024 : i64, length = 40 : i64, note = "kept"}> : tensor<5x4xf16>
  util.func public @main(%ids: tensor<8xi64>) -> tensor<8x4xf16> {
    %table = util.global.load immutable @table : tensor<5x4xf16>
    %empty = tensor.empty() : tensor<8x4xf16>
    %0 = linalg.generic {
        indexing_maps = [#id, #out], iterator_types = ["parallel", "parallel"]}
        ins(%ids : tensor<8xi64>) outs(%empty : tensor<8x4xf16>) {
    ^bb0(%in: i64, %out: f16):
      %i = arith.index_cast %in : i64 to index
      %j = linalg.index 1 : index
      %v = tensor.extract %table[%i, %j] : tensor<5x4xf16>
      linalg.yield %v : f16
    } -> tensor<8x4xf16>
    util.return %0 : tensor<8x4xf16>
  }
}

// CHECK-LABEL: module @parameter_range
//   CHECK-NOT:   util.global private @table =
//       CHECK:   util.global private @table_shard0 = #flow.parameter.named<"model"::"embed", {length = 16 : i64, note = "kept", offset = 1024 : i64}> : tensor<2x4xf16>
//       CHECK:   util.global private @table_shard1 = #flow.parameter.named<"model"::"embed", {length = 16 : i64, note = "kept", offset = 1040 : i64}> : tensor<2x4xf16>
//       CHECK:   util.global private @table_shard2 = #flow.parameter.named<"model"::"embed", {length = 8 : i64, note = "kept", offset = 1056 : i64}> : tensor<1x4xf16>
//       CHECK:   util.func public @main
//       CHECK:     tensor.extract %{{.+}} : tensor<2x4xf16>
//       CHECK:     tensor.extract %{{.+}} : tensor<2x4xf16>
//       CHECK:     tensor.extract %{{.+}} : tensor<1x4xf16>

// -----

// A constant within the limit is left alone.

module @fits attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 64}>
} {
  util.global private @table = dense<1.0> : tensor<6x4xf16>
  util.func public @main() -> tensor<6x4xf16> {
    %table = util.global.load immutable @table : tensor<6x4xf16>
    util.return %table : tensor<6x4xf16>
  }
}

// CHECK-LABEL: module @fits
//       CHECK:   util.global private @table = dense<1.000000e+00> : tensor<6x4xf16>
//   CHECK-NOT:   _shard

// -----

// An oversized constant with a use that cannot be split is left alone.

module @unsupported attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  // expected-note @below {{constant declared here}}
  util.global private @table = dense<1.0> : tensor<6x4xf16>
  util.func public @main() -> tensor<6x4xf16> {
    %table = util.global.load immutable @table : tensor<6x4xf16>
    // expected-warning @below {{cannot split the use of oversized constant @table: unsupported user}}
    util.return %table : tensor<6x4xf16>
  }
}

// CHECK-LABEL: module @unsupported
//       CHECK:   util.global private @table = dense<1.000000e+00> : tensor<6x4xf16>
//   CHECK-NOT:   _shard

// -----

// An op that uses two loads of the same constant can be rewritten for only one
// of them, so the constant is left alone.

module @two_loads_one_op attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 32}>
} {
  // expected-note @below {{constant declared here}}
  util.global private @big = dense<1.0> : tensor<8x8xf16>
  util.func public @main(%init: tensor<8x8xf16>) -> tensor<8x8xf16> {
    %a = util.global.load immutable @big : tensor<8x8xf16>
    %b = util.global.load immutable @big : tensor<8x8xf16>
    // expected-warning @below {{cannot split the use of oversized constant @big: used by more than one operand}}
    %r = linalg.matmul ins(%a, %b : tensor<8x8xf16>, tensor<8x8xf16>)
                       outs(%init : tensor<8x8xf16>) -> tensor<8x8xf16>
    util.return %r : tensor<8x8xf16>
  }
}

// CHECK-LABEL: module @two_loads_one_op
//       CHECK:   util.global private @big = dense<1.000000e+00> : tensor<8x8xf16>
//   CHECK-NOT:   _shard

// -----

// The same for an op that reads two loads of the same constant in its body.

#id = affine_map<(d0, d1) -> (d0)>
#out = affine_map<(d0, d1) -> (d0, d1)>
module @two_loads_one_gather attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  // expected-note @below {{constant declared here}}
  util.global private @table = dense<1.0> : tensor<6x4xf16>
  util.func public @main(%ids: tensor<8xi64>) -> tensor<8x4xf16> {
    %a = util.global.load immutable @table : tensor<6x4xf16>
    %b = util.global.load immutable @table : tensor<6x4xf16>
    %empty = tensor.empty() : tensor<8x4xf16>
    %0 = linalg.generic {
        indexing_maps = [#id, #out], iterator_types = ["parallel", "parallel"]}
        ins(%ids : tensor<8xi64>) outs(%empty : tensor<8x4xf16>) {
    ^bb0(%in: i64, %out: f16):
      %i = arith.index_cast %in : i64 to index
      %j = linalg.index 1 : index
      %va = tensor.extract %a[%i, %j] : tensor<6x4xf16>
      // expected-warning @below {{cannot split the use of oversized constant @table: read more than once in the same op}}
      %vb = tensor.extract %b[%i, %j] : tensor<6x4xf16>
      %v = arith.addf %va, %vb : f16
      linalg.yield %v : f16
    } -> tensor<8x4xf16>
    util.return %0 : tensor<8x4xf16>
  }
}

// CHECK-LABEL: module @two_loads_one_gather
//       CHECK:   util.global private @table = dense<1.000000e+00> : tensor<6x4xf16>
//   CHECK-NOT:   _shard

// -----

// A public constant is not split.

module @public_global attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  // expected-warning @below {{constant of 48 bytes exceeds the max allocation size of 16 bytes and is not split: only private immutable globals are split}}
  util.global public @table = dense<1.0> : tensor<6x4xf16>
}

// -----

// A parameter whose length does not match its type is not split: the shard
// ranges would not cover the bytes the type describes.

module @parameter_length_mismatch attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  // expected-warning @below {{constant of 48 bytes exceeds the max allocation size of 16 bytes and is not split: the parameter has 32 bytes, expected 48}}
  util.global private @table = #flow.parameter.named<"model"::"table", {length = 32 : i64}> : tensor<6x4xf16>
}

// CHECK-LABEL: module @parameter_length_mismatch
//   CHECK-NOT:   _shard

// -----

// A parameter offset that is not an integer cannot be added to, so the
// parameter is not split.

module @parameter_offset_not_integer attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 16}>
} {
  // expected-warning @below {{constant of 48 bytes exceeds the max allocation size of 16 bytes and is not split: the parameter offset is not an integer}}
  util.global private @table = #flow.parameter.named<"model"::"table", {offset = "16"}> : tensor<6x4xf16>
}

// CHECK-LABEL: module @parameter_offset_not_integer
//   CHECK-NOT:   _shard
