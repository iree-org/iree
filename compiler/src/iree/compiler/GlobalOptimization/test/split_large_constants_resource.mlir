// RUN: iree-opt --iree-global-opt-split-large-constants %s | FileCheck %s

// Each shard of a resource is a view of a contiguous byte range of the
// original: 1x3x4xi16 holding 0..9, 16, 17 is split along dimension 1 into rows
// {0, 1} and {2}. The original is 32 byte aligned; a shard keeps that alignment
// only as far as its byte offset allows, so the shard at offset 16 is 16 byte
// aligned.

module attributes {
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

{-#
  dialect_resources: {
    builtin: {
      blob: "0x20000000000001000200030004000500060007000800090010001100"
    }
  }
#-}

// CHECK: util.global private @table_shard0 = dense_resource<table_shard0> : tensor<1x2x4xi16>
// CHECK: util.global private @table_shard1 = dense_resource<table_shard1> : tensor<1x1x4xi16>
// CHECK: table_shard0: "0x2000000000000100020003000400050006000700"
// CHECK: table_shard1: "0x100000000800090010001100"
