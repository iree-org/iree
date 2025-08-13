// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level),canonicalize,cse,func.func(iree-codegen-gpu-apply-padding-level),canonicalize,cse)" --split-input-file --verify-diagnostics %s | FileCheck  %s

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>


//                                                        batch, m, k1, k2
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32]}>

// CHECK-LABEL: func.func @online_attention_fail_to_pad_no_mask
func.func @online_attention_fail_to_pad_no_mask(%query: tensor<192x1024x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  //  CHECK-NOT: tensor.pad
  //      CHECK: iree_linalg_ext.online_attention {{.*}} ins(%{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}
  // CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32)
  // expected-remark@+1{{failed to pad op: requires a mask operand to pad to the proper value. Consider materializing the mask operand explicitly.}}
  %out:3 = iree_linalg_ext.online_attention
        {
          indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR],
          lowering_config = #lowering_config
        }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//                                                        batch, m, k1, k2
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32]}>

// CHECK-LABEL: func.func @online_attention_tile_then_pad
func.func @online_attention_tile_then_pad(%query: tensor<192x1024x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>, %mask: tensor<192x1024x?xf32>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  //         CHECK: arith.constant 0xFF800000 : f32
  // CHECK-COUNT-3: tensor.pad
  //         CHECK: iree_linalg_ext.online_attention {{.*}} ins(%{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}
  //    CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x32x64xf32>, tensor<192x32x64xf32>, f32, tensor<192x1024x32xf32>
  %out:3 = iree_linalg_ext.online_attention
        {
          indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR],
          lowering_config = #lowering_config
        }
        ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32, tensor<192x1024x?xf32>)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>)
        {
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score: f32
        }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//                                                        batch, m, k1, k2
#lowering_config = #iree_gpu.lowering_config<{reduction = [   4, 8,  0, 32]}>

// CHECK-LABEL: func.func @online_attention_tile_then_pad_7
func.func @online_attention_tile_then_pad_7(%n_batches: index, %query: tensor<?x1021x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>, %mask: tensor<?x1021xf32>) -> tensor<?x1021x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty(%n_batches) : tensor<?x1021x64xf32>
  %row_red_empty = tensor.empty(%n_batches) : tensor<?x1021xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<?x1021x64xf32>) -> tensor<?x1021x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<?x1021xf32>) -> tensor<?x1021xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<?x1021xf32>) -> tensor<?x1021xf32>

  //         CHECK: arith.constant 0xFF800000 : f32
  // CHECK-COUNT-7: tensor.pad
  //         CHECK: iree_linalg_ext.online_attention {{.*}} ins(%{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}
  //    CHECK-SAME:   : tensor<4x8x64xf32>, tensor<4x32x64xf32>, tensor<4x32x64xf32>, f32, tensor<4x8xf32>)
  //    CHECK-SAME:     outs(%{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}
  //    CHECK-SAME:   : tensor<4x8x64xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  %out:3 = iree_linalg_ext.online_attention
        {
          indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR],
          lowering_config = #lowering_config
        }
        ins(%query, %key, %value, %scale, %mask : tensor<?x1021x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32, tensor<?x1021xf32>)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<?x1021x64xf32>, tensor<?x1021xf32>, tensor<?x1021xf32>)
        {
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score: f32
        }
        -> tensor<?x1021x64xf32>, tensor<?x1021xf32>, tensor<?x1021xf32>

  return %out#0 : tensor<?x1021x64xf32>
}
