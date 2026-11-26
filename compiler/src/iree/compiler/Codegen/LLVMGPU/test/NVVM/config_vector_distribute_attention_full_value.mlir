// RUN: iree-opt --split-input-file --iree-gpu-test-target=rtx4090@cuda \
// RUN:   --iree-llvmgpu-experimental-nvvm-full-value-attention-tile \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefixes=CHECK,ON
// RUN: iree-opt --split-input-file --iree-gpu-test-target=rtx4090@cuda \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefixes=CHECK,OFF

// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_120 \
// RUN:   --iree-llvmgpu-experimental-nvvm-full-value-attention-tile \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefixes=CHECK,OFF

// CHECK-LABEL: func.func @attention_1024x64
// CHECK: iree_linalg_ext.online_attention
// ON-SAME: workgroup = [1, 32, 0, 0, 64]
// OFF-SAME: workgroup = [1, 64, 0, 0, 32]
#q = affine_map<(b,m,k1,k2,n) -> (b,m,k1)>
#k = affine_map<(b,m,k1,k2,n) -> (b,k2,k1)>
#v = affine_map<(b,m,k1,k2,n) -> (b,k2,n)>
#s = affine_map<(b,m,k1,k2,n) -> ()>
#o = affine_map<(b,m,k1,k2,n) -> (b,m,n)>
#r = affine_map<(b,m,k1,k2,n) -> (b,m)>

func.func @attention_1024x64(%q: tensor<12x1024x64xbf16>, %k: tensor<12x1024x64xbf16>,
    %v: tensor<12x1024x64xbf16>, %scale: bf16, %out: tensor<12x1024x64xf32>,
    %max: tensor<12x1024xf32>, %sum: tensor<12x1024xf32>)
    -> (tensor<12x1024x64xf32>, tensor<12x1024xf32>, tensor<12x1024xf32>) {
  %r:3 = iree_linalg_ext.online_attention <indexing_maps = [#q, #k, #v, #s, #o, #r, #r]>
      ins(%q, %k, %v, %scale : tensor<12x1024x64xbf16>, tensor<12x1024x64xbf16>, tensor<12x1024x64xbf16>, bf16)
      outs(%out, %max, %sum : tensor<12x1024x64xf32>, tensor<12x1024xf32>, tensor<12x1024xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<12x1024x64xf32>, tensor<12x1024xf32>, tensor<12x1024xf32>
  return %r#0, %r#1, %r#2 : tensor<12x1024x64xf32>, tensor<12x1024xf32>, tensor<12x1024xf32>
}

// -----

// Keep the original schedule below the key-length profitability guard.
// CHECK-LABEL: func.func @attention_256x64
// CHECK: iree_linalg_ext.online_attention
// CHECK-SAME: workgroup = [1, 64, 0, 0, 32]
#q = affine_map<(b,m,k1,k2,n) -> (b,m,k1)>
#k = affine_map<(b,m,k1,k2,n) -> (b,k2,k1)>
#v = affine_map<(b,m,k1,k2,n) -> (b,k2,n)>
#s = affine_map<(b,m,k1,k2,n) -> ()>
#o = affine_map<(b,m,k1,k2,n) -> (b,m,n)>
#r = affine_map<(b,m,k1,k2,n) -> (b,m)>

func.func @attention_256x64(%q: tensor<12x256x64xbf16>, %k: tensor<12x256x64xbf16>,
    %v: tensor<12x256x64xbf16>, %scale: bf16, %out: tensor<12x256x64xf32>,
    %max: tensor<12x256xf32>, %sum: tensor<12x256xf32>)
    -> (tensor<12x256x64xf32>, tensor<12x256xf32>, tensor<12x256xf32>) {
  %r:3 = iree_linalg_ext.online_attention <indexing_maps = [#q, #k, #v, #s, #o, #r, #r]>
      ins(%q, %k, %v, %scale : tensor<12x256x64xbf16>, tensor<12x256x64xbf16>, tensor<12x256x64xbf16>, bf16)
      outs(%out, %max, %sum : tensor<12x256x64xf32>, tensor<12x256xf32>, tensor<12x256xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<12x256x64xf32>, tensor<12x256xf32>, tensor<12x256xf32>
  return %r#0, %r#1, %r#2 : tensor<12x256x64xf32>, tensor<12x256xf32>, tensor<12x256xf32>
}

// -----

// CHECK-LABEL: func.func @attention_wide_128
// CHECK: iree_linalg_ext.online_attention
// ON-SAME: workgroup = [1, 64, 0, 0, 128]
// OFF-SAME: workgroup = [1, 64, 0, 0, 32]
#q = affine_map<(b,m,k1,k2,n) -> (b,m,k1)>
#k = affine_map<(b,m,k1,k2,n) -> (b,k2,k1)>
#v = affine_map<(b,m,k1,k2,n) -> (b,k2,n)>
#s = affine_map<(b,m,k1,k2,n) -> ()>
#o = affine_map<(b,m,k1,k2,n) -> (b,m,n)>
#r = affine_map<(b,m,k1,k2,n) -> (b,m)>

func.func @attention_wide_128(
    %q: tensor<32x1280x128xbf16>, %k: tensor<32x1280x128xbf16>,
    %v: tensor<32x1280x128xbf16>, %scale: bf16,
    %out: tensor<32x1280x128xf32>, %max: tensor<32x1280xf32>, %sum: tensor<32x1280xf32>)
    -> (tensor<32x1280x128xf32>, tensor<32x1280xf32>, tensor<32x1280xf32>) {
  %r:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#q, #k, #v, #s, #o, #r, #r]>
      ins(%q, %k, %v, %scale : tensor<32x1280x128xbf16>, tensor<32x1280x128xbf16>, tensor<32x1280x128xbf16>, bf16)
      outs(%out, %max, %sum : tensor<32x1280x128xf32>, tensor<32x1280xf32>, tensor<32x1280xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<32x1280x128xf32>, tensor<32x1280xf32>, tensor<32x1280xf32>
  return %r#0, %r#1, %r#2 : tensor<32x1280x128xf32>, tensor<32x1280xf32>, tensor<32x1280xf32>
}

// -----

// CHECK-LABEL: func.func @attention_incompatible_three_subgroups
// CHECK: iree_linalg_ext.online_attention
// A three-warp value tile would require an incompatible K copy.
// CHECK-SAME: workgroup = [1, 128, 0, 0, 16]
#q = affine_map<(b,m,k1,k2,n) -> (b,m,k1)>
#k = affine_map<(b,m,k1,k2,n) -> (b,k2,k1)>
#v = affine_map<(b,m,k1,k2,n) -> (b,k2,n)>
#s = affine_map<(b,m,k1,k2,n) -> ()>
#o = affine_map<(b,m,k1,k2,n) -> (b,m,n)>
#r = affine_map<(b,m,k1,k2,n) -> (b,m)>

func.func @attention_incompatible_three_subgroups(
    %q: tensor<16x1152x64xbf16>, %k: tensor<16x1152x64xbf16>,
    %v: tensor<16x1152x48xbf16>, %scale: bf16,
    %out: tensor<16x1152x48xf32>, %max: tensor<16x1152xf32>, %sum: tensor<16x1152xf32>)
    -> (tensor<16x1152x48xf32>, tensor<16x1152xf32>, tensor<16x1152xf32>) {
  %r:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#q, #k, #v, #s, #o, #r, #r]>
      ins(%q, %k, %v, %scale : tensor<16x1152x64xbf16>, tensor<16x1152x64xbf16>, tensor<16x1152x48xbf16>, bf16)
      outs(%out, %max, %sum : tensor<16x1152x48xf32>, tensor<16x1152xf32>, tensor<16x1152xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<16x1152x48xf32>, tensor<16x1152xf32>, tensor<16x1152xf32>
  return %r#0, %r#1, %r#2 : tensor<16x1152x48xf32>, tensor<16x1152xf32>, tensor<16x1152xf32>
}

// -----

// CHECK-LABEL: func.func @attention_grouped_query
// CHECK: iree_linalg_ext.online_attention
// ON-SAME: workgroup = [1, 1, 128, 0, 0, 64]
// OFF-SAME: workgroup = [1, 1, 64, 0, 0, 32]
#q = affine_map<(b,g,m,k1,k2,n) -> (b,g,m,k1)>
#k = affine_map<(b,g,m,k1,k2,n) -> (b,k2,k1)>
#v = affine_map<(b,g,m,k1,k2,n) -> (b,k2,n)>
#s = affine_map<(b,g,m,k1,k2,n) -> ()>
#o = affine_map<(b,g,m,k1,k2,n) -> (b,g,m,n)>
#r = affine_map<(b,g,m,k1,k2,n) -> (b,g,m)>

func.func @attention_grouped_query(
    %q: tensor<8x4x2048x64xbf16>, %k: tensor<8x2048x64xbf16>,
    %v: tensor<8x2048x64xbf16>, %scale: bf16,
    %out: tensor<8x4x2048x64xf32>, %max: tensor<8x4x2048xf32>,
    %sum: tensor<8x4x2048xf32>)
    -> (tensor<8x4x2048x64xf32>, tensor<8x4x2048xf32>, tensor<8x4x2048xf32>) {
  %r:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#q, #k, #v, #s, #o, #r, #r]>
      ins(%q, %k, %v, %scale : tensor<8x4x2048x64xbf16>,
          tensor<8x2048x64xbf16>, tensor<8x2048x64xbf16>, bf16)
      outs(%out, %max, %sum : tensor<8x4x2048x64xf32>,
          tensor<8x4x2048xf32>, tensor<8x4x2048xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<8x4x2048x64xf32>, tensor<8x4x2048xf32>, tensor<8x4x2048xf32>
  return %r#0, %r#1, %r#2 : tensor<8x4x2048x64xf32>,
      tensor<8x4x2048xf32>, tensor<8x4x2048xf32>
}

// -----

// Full-width output would leave fewer than two workgroups per SM.
// CHECK-LABEL: func.func @attention_sparse_query_grid
// CHECK: iree_linalg_ext.online_attention
// CHECK-SAME: workgroup = [1, 64, 0, 0, 32]
#q = affine_map<(b,m,k1,k2,n) -> (b,m,k1)>
#k = affine_map<(b,m,k1,k2,n) -> (b,k2,k1)>
#v = affine_map<(b,m,k1,k2,n) -> (b,k2,n)>
#s = affine_map<(b,m,k1,k2,n) -> ()>
#o = affine_map<(b,m,k1,k2,n) -> (b,m,n)>
#r = affine_map<(b,m,k1,k2,n) -> (b,m)>

func.func @attention_sparse_query_grid(%q: tensor<4x1408x64xbf16>, %k: tensor<4x1408x64xbf16>,
    %v: tensor<4x1408x64xbf16>, %scale: bf16, %out: tensor<4x1408x64xf32>,
    %max: tensor<4x1408xf32>, %sum: tensor<4x1408xf32>)
    -> (tensor<4x1408x64xf32>, tensor<4x1408xf32>, tensor<4x1408xf32>) {
  %r:3 = iree_linalg_ext.online_attention <indexing_maps = [#q, #k, #v, #s, #o, #r, #r]>
      ins(%q, %k, %v, %scale : tensor<4x1408x64xbf16>, tensor<4x1408x64xbf16>, tensor<4x1408x64xbf16>, bf16)
      outs(%out, %max, %sum : tensor<4x1408x64xf32>, tensor<4x1408xf32>, tensor<4x1408xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<4x1408x64xf32>, tensor<4x1408xf32>, tensor<4x1408xf32>
  return %r#0, %r#1, %r#2 : tensor<4x1408x64xf32>, tensor<4x1408xf32>, tensor<4x1408xf32>
}
