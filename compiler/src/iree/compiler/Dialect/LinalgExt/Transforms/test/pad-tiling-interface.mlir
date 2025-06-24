// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --verify-diagnostics -canonicalize -cse %s | FileCheck  %s

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//     CHECK-LABEL: online_attention_tile_then_pad
func.func @online_attention_tile_then_pad(%query: tensor<192x1024x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  //      CHECK: iree_linalg_ext.online_attention {{.*}} ins(%{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}
  // CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x128x64xf32>, tensor<192x128x64xf32>, f32)
  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %online_attention = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op

    // Tile then pad should give us a static shape.
    // TODO: this currently does not work, FIXME.
    %tiled_online_attention, %loops_l1 = transform.structured.tile_using_for %online_attention tile_sizes [0, 0, 0, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %padded, %pad = transform.structured.pad_tiling_interface %tiled_online_attention to padding_sizes [0, 0, 128] pad_to_multiple_of {
      padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.affine.simplify_min_max_affine_ops %func : !transform.any_op

    transform.yield
  }
}

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//     CHECK-LABEL: online_attention_pad_then_tile
func.func @online_attention_pad_then_tile(%query: tensor<192x1024x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  //      CHECK: iree_linalg_ext.online_attention {{.*}} ins(%{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}
  // CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x128x64xf32>, tensor<192x128x64xf32>, f32)
  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %online_attention = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op

    // Pad then tile should give us a static shape.
    %padded, %pad = transform.structured.pad_tiling_interface %online_attention to padding_sizes [0, 0, 128] pad_to_multiple_of {
      padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_online_attention, %loops_l1 = transform.structured.tile_using_for %padded tile_sizes [0, 0, 0, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.affine.simplify_min_max_affine_ops %func : !transform.any_op

    transform.yield
  }
}
