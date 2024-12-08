// RUN: iree-opt --split-input-file --iree-global-opt-decompose-low-parallelism-attention --cse %s | FileCheck %s

// CHECK-LABEL: @attention_small_m
// CHECK-NOT: iree_linalg_ext.attention
util.func @attention_small_m(%q : tensor<128x1x32x64xf16>,
                             %k : tensor<128x1024x32x64xf16>,
                             %v : tensor<128x1024x32x64xf16>)
                             -> tensor<128x32x1x64xf16> {
  %cst = arith.constant 1.250000e-01 : f16
  %empty = tensor.empty() : tensor<128x32x1x64xf16>
  %out = iree_linalg_ext.attention
          {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d4)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d4)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d3)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]}
          ins(%q, %k, %v, %cst : tensor<128x1x32x64xf16>, tensor<128x1024x32x64xf16>, tensor<128x1024x32x64xf16>, f16)
          outs(%empty : tensor<128x32x1x64xf16>) {
            ^bb0(%score : f32):
              iree_linalg_ext.yield %score : f32
          }-> tensor<128x32x1x64xf16>
  util.return %out : tensor<128x32x1x64xf16>
}

// CHECK-LABEL: @attention_large_k1
// CHECK-NOT: iree_linalg_ext.attention
util.func @attention_large_k1(%q : tensor<128x1024x32x512xf16>,
                             %k : tensor<128x1024x32x512xf16>,
                             %v : tensor<128x1024x32x64xf16>)
                             -> tensor<128x32x1024x64xf16> {
  %cst = arith.constant 1.250000e-01 : f16
  %empty = tensor.empty() : tensor<128x32x1024x64xf16>
  %out = iree_linalg_ext.attention
          {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d4)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d4)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d3)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]}
          ins(%q, %k, %v, %cst : tensor<128x1024x32x512xf16>, tensor<128x1024x32x512xf16>, tensor<128x1024x32x64xf16>, f16)
          outs(%empty : tensor<128x32x1024x64xf16>) {
            ^bb0(%score : f32):
              iree_linalg_ext.yield %score : f32
          }-> tensor<128x32x1024x64xf16>
  util.return %out : tensor<128x32x1024x64xf16>
}

// CHECK-LABEL: @attention_normal
// CHECK: iree_linalg_ext.attention
util.func @attention_normal(%q : tensor<128x1024x32x64xf16>,
                             %k : tensor<128x1024x32x64xf16>,
                             %v : tensor<128x1024x32x64xf16>)
                             -> tensor<128x32x1024x64xf16> {
  %cst = arith.constant 1.250000e-01 : f16
  %empty = tensor.empty() : tensor<128x32x1024x64xf16>
  %out = iree_linalg_ext.attention
          {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d4)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d4)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d3)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]}
          ins(%q, %k, %v, %cst : tensor<128x1024x32x64xf16>, tensor<128x1024x32x64xf16>, tensor<128x1024x32x64xf16>, f16)
          outs(%empty : tensor<128x32x1024x64xf16>) {
            ^bb0(%score : f32):
              iree_linalg_ext.yield %score : f32
          }-> tensor<128x32x1024x64xf16>
  util.return %out : tensor<128x32x1024x64xf16>
}
