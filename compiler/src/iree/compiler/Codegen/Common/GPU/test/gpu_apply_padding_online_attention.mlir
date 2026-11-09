// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level),canonicalize,cse,func.func(iree-codegen-gpu-apply-padding-level),canonicalize,cse)" --split-input-file --verify-diagnostics %s | FileCheck  %s

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>


//                                                        batch, m, k1, k2, n
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32, 0]}>

// Without a mask operand the region hides the padded keys: past the key bound
// a score is -inf.
// CHECK-LABEL: func.func @online_attention_tile_then_pad_no_mask
func.func @online_attention_tile_then_pad_no_mask(%query: tensor<6x40x24xf32>, %key: tensor<6x?x24xf32>, %value: tensor<6x?x48xf32>) -> tensor<6x40x48xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<6x40x48xf32>
  %row_red_empty = tensor.empty() : tensor<6x40xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<6x40x48xf32>) -> tensor<6x40x48xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<6x40xf32>) -> tensor<6x40xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<6x40xf32>) -> tensor<6x40xf32>

  //     CHECK-DAG: %[[NEG_INF:.+]] = arith.constant 0xFF800000 : f32
  // CHECK-COUNT-2: tensor.pad
  //         CHECK: iree_linalg_ext.online_attention
  //    CHECK-SAME:   : tensor<6x40x24xf32>, tensor<6x32x24xf32>, tensor<6x32x48xf32>, f32)
  //         CHECK:   ^bb0(%[[SCORE:.+]]: f32):
  //         CHECK:     %[[KEY:.+]] = iree_linalg_ext.index 2 : index
  //         CHECK:     %[[IN_BOUNDS:.+]] = arith.cmpi ult, %[[KEY]], %{{.+}} : index
  //         CHECK:     %[[MASKED:.+]] = arith.select %[[IN_BOUNDS]], %[[SCORE]], %[[NEG_INF]] : f32
  //         CHECK:     iree_linalg_ext.yield %[[MASKED]] : f32
  %out:3 = iree_linalg_ext.online_attention
        <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR]> {lowering_config = #lowering_config}
        ins(%query, %key, %value, %scale : tensor<6x40x24xf32>, tensor<6x?x24xf32>, tensor<6x?x48xf32>, f32)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<6x40x48xf32>, tensor<6x40xf32>, tensor<6x40xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<6x40x48xf32>, tensor<6x40xf32>, tensor<6x40xf32>

  return %out#0 : tensor<6x40x48xf32>
}

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//                                                        batch, m, k1, k2, n
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32, 0]}>

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
  //         CHECK: iree_linalg_ext.online_attention {{.*}} ins(%{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}
  //    CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x32x64xf32>, tensor<192x32x64xf32>, f32, tensor<192x1024x32xf32>)
  %out:3 = iree_linalg_ext.online_attention
        <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]> {lowering_config = #lowering_config}
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
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//                                                        batch, m, k1, k2, n
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32, 0]}>

// CHECK-LABEL: func.func @online_attention_tile_then_pad_bool_mask
func.func @online_attention_tile_then_pad_bool_mask(%query: tensor<192x1024x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>, %mask: tensor<192x1024x?xi1>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  //         CHECK: arith.constant false
  // CHECK-COUNT-3: tensor.pad
  //         CHECK: iree_linalg_ext.online_attention
  //    CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x32x64xf32>, tensor<192x32x64xf32>, f32, tensor<192x1024x32xi1>)
  %out:3 = iree_linalg_ext.online_attention
        <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]> {lowering_config = #lowering_config}
        ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32, tensor<192x1024x?xi1>)
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
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//                                                        batch, m, k1, k2, n
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32, 0]}>

// CHECK-LABEL: func.func @online_attention_tile_then_pad_i8_mask
func.func @online_attention_tile_then_pad_i8_mask(%query: tensor<192x1024x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>, %mask: tensor<192x1024x?xi8>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  //         CHECK: arith.constant 0 : i8
  // CHECK-COUNT-3: tensor.pad
  //         CHECK: iree_linalg_ext.online_attention
  //    CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x32x64xf32>, tensor<192x32x64xf32>, f32, tensor<192x1024x32xi8>)
  %out:3 = iree_linalg_ext.online_attention
        <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]> {lowering_config = #lowering_config}
        ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32, tensor<192x1024x?xi8>)
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

//                                                        batch, m, k1, k2, n
#lowering_config = #iree_gpu.lowering_config<{reduction = [   4, 8,  0, 32, 0]}>

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

  // CHECK-COUNT-7: tensor.pad
  //         CHECK: iree_linalg_ext.online_attention {{.*}} ins(%{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}
  //    CHECK-SAME:   : tensor<4x8x64xf32>, tensor<4x32x64xf32>, tensor<4x32x64xf32>, f32, tensor<4x8xf32>)
  //    CHECK-SAME:     outs(%{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}, %{{[0-9a-z_]*}}
  //    CHECK-SAME:   : tensor<4x8x64xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  %out:3 = iree_linalg_ext.online_attention
        <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]> {lowering_config = #lowering_config}
        ins(%query, %key, %value, %scale, %mask : tensor<?x1021x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32, tensor<?x1021xf32>)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<?x1021x64xf32>, tensor<?x1021xf32>, tensor<?x1021xf32>)
        {
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score: f32
        }
        -> tensor<?x1021x64xf32>, tensor<?x1021xf32>, tensor<?x1021xf32>

  return %out#0 : tensor<?x1021x64xf32>
}

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//                                                        batch, m, k1, k2, n
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32, 0]}>

// CHECK-LABEL: func.func @online_attention_no_mask_aligned_dims
func.func @online_attention_no_mask_aligned_dims(%query: tensor<192x1024x64xf32>, %key: tensor<192x64x64xf32>, %value: tensor<192x64x64xf32>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  //        CHECK: iree_linalg_ext.online_attention {{.*}} ins(%{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}
  //   CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x32x64xf32>, tensor<192x32x64xf32>, f32)
  //   CHECK-SAME:     outs(%{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}, %{{[A-Za-z0-9_]+}}
  //   CHECK-SAME:   : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>)
  %out:3 = iree_linalg_ext.online_attention
        <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR]> {lowering_config = #lowering_config}
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x64x64xf32>, tensor<192x64x64xf32>, f32)
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
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//                                                        batch, m, k1, k2, n
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32, 0]}>

func.func @online_attention_additive_float_padding(%query: tensor<192x1024x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>, %mask: tensor<192x1024x?xf32>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  %out:3 = iree_linalg_ext.online_attention
        <additive_mask, indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]> {lowering_config = #lowering_config}
        ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32, tensor<192x1024x?xf32>)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>)
        {
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score: f32
        }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}

// CHECK-LABEL: func.func @online_attention_additive_float_padding
// CHECK-NOT: ub.poison
// CHECK: %[[NEG_INF:.+]] = arith.constant 0xFF800000 : f32
// CHECK: iree_linalg_ext.online_attention
// CHECK-SAME: additive_mask
// CHECK: ^bb0(%[[SCORE:.+]]: f32):
// CHECK: %[[KEY_INDEX:.+]] = iree_linalg_ext.index 2 : index
// CHECK: %[[IN_BOUNDS:.+]] = arith.cmpi ult, %[[KEY_INDEX]], %{{.+}} : index
// CHECK: %[[GUARDED:.+]] = arith.select %[[IN_BOUNDS]], %[[SCORE]], %[[NEG_INF]] : f32
// CHECK: iree_linalg_ext.yield %[[GUARDED]] : f32
// CHECK-NOT: ub.poison
// CHECK: return

// -----


#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

//                                                        batch, m, k1, k2, n
#lowering_config = #iree_gpu.lowering_config<{reduction = [   0, 0,  0, 32, 0]}>

func.func @online_attention_additive_bool_padding(%query: tensor<192x1024x64xf32>, %key: tensor<192x?x64xf32>, %value: tensor<192x?x64xf32>, %mask: tensor<192x1024x?xi1>) -> tensor<192x1024x64xf32> {
  %scale = arith.constant 1.0 : f32

  %output_empty = tensor.empty() : tensor<192x1024x64xf32>
  %row_red_empty = tensor.empty() : tensor<192x1024xf32>

  %sum_ident = arith.constant 0.000000e+00 : f32
  %max_ident = arith.constant -3.40282347E+38 : f32

  %output_fill = linalg.fill ins(%sum_ident : f32) outs(%output_empty : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  %acc_fill = linalg.fill ins(%max_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>
  %sum_fill = linalg.fill ins(%sum_ident : f32) outs(%row_red_empty : tensor<192x1024xf32>) -> tensor<192x1024xf32>

  %out:3 = iree_linalg_ext.online_attention
        <additive_mask, indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]> {lowering_config = #lowering_config}
        ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf32>, tensor<192x?x64xf32>, tensor<192x?x64xf32>, f32, tensor<192x1024x?xi1>)
        outs(%output_fill, %acc_fill, %sum_fill : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>)
        {
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score: f32
        }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0 : tensor<192x1024x64xf32>
}

// CHECK-LABEL: func.func @online_attention_additive_bool_padding
// CHECK-NOT: ub.poison
// CHECK: %[[NEG_INF:.+]] = arith.constant 0xFF800000 : f32
// CHECK: iree_linalg_ext.online_attention
// CHECK-SAME: additive_mask
// CHECK: ^bb0(%[[SCORE:.+]]: f32):
// CHECK: %[[KEY_INDEX:.+]] = iree_linalg_ext.index 2 : index
// CHECK: %[[IN_BOUNDS:.+]] = arith.cmpi ult, %[[KEY_INDEX]], %{{.+}} : index
// CHECK: %[[GUARDED:.+]] = arith.select %[[IN_BOUNDS]], %[[SCORE]], %[[NEG_INF]] : f32
// CHECK: iree_linalg_ext.yield %[[GUARDED]] : f32
// CHECK-NOT: ub.poison
// CHECK: return

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (b, m, k2)>
#out = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#row = affine_map<(b, m, k1, k2, n) -> (b, m)>
#config = #iree_gpu.lowering_config<{reduction = [0, 0, 0, 32, 0]}>

func.func @low_precision_additive_padding_declines(%q: tensor<1x1x4xf8E5M2>, %k: tensor<1x?x4xf8E5M2>, %v: tensor<1x?x6xf8E5M2>, %mask: tensor<1x1x?xi1>, %acc: tensor<1x1x6xf32>, %max: tensor<1x1xf32>, %sum: tensor<1x1xf32>) -> tensor<1x1x6xf32> {
  %scale = arith.constant 0.0 : f32
  %result:3 = iree_linalg_ext.online_attention <additive_mask, indexing_maps = [#q, #k, #v, #s, #mask, #out, #row, #row]> {lowering_config = #config}
      ins(%q, %k, %v, %scale, %mask : tensor<1x1x4xf8E5M2>, tensor<1x?x4xf8E5M2>, tensor<1x?x6xf8E5M2>, f32, tensor<1x1x?xi1>)
      outs(%acc, %max, %sum : tensor<1x1x6xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x1x6xf32>, tensor<1x1xf32>, tensor<1x1xf32>
  return %result#0 : tensor<1x1x6xf32>
}
// CHECK-LABEL: func.func @low_precision_additive_padding_declines
// CHECK-NOT: tensor.pad
// CHECK: iree_linalg_ext.online_attention
// CHECK-SAME: additive_mask
// CHECK-NOT: tensor.pad
// CHECK: return

// -----


#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#out = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#row = affine_map<(b, m, k1, k2, n) -> (b, m)>
#config = #iree_gpu.lowering_config<{reduction = [0, 0, 0, 32, 0]}>

func.func @low_precision_unmasked_padding_declines(%q: tensor<1x1x4xf8E5M2>, %k: tensor<1x?x4xf8E5M2>, %v: tensor<1x?x6xf8E5M2>, %acc: tensor<1x1x6xf32>, %max: tensor<1x1xf32>, %sum: tensor<1x1xf32>) -> tensor<1x1x6xf32> {
  %scale = arith.constant 0.0 : f32
  %result:3 = iree_linalg_ext.online_attention <indexing_maps = [#q, #k, #v, #s, #out, #row, #row]> {lowering_config = #config}
      ins(%q, %k, %v, %scale : tensor<1x1x4xf8E5M2>, tensor<1x?x4xf8E5M2>, tensor<1x?x6xf8E5M2>, f32)
      outs(%acc, %max, %sum : tensor<1x1x6xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x1x6xf32>, tensor<1x1xf32>, tensor<1x1xf32>
  return %result#0 : tensor<1x1x6xf32>
}
// CHECK-LABEL: func.func @low_precision_unmasked_padding_declines
// CHECK-NOT: tensor.pad
// CHECK: iree_linalg_ext.online_attention
// CHECK-NOT: tensor.pad
// CHECK: return
