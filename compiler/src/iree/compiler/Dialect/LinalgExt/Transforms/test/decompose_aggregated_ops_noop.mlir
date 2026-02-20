// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops), canonicalize, cse)"  --verify-diagnostics %s

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

// expected-warning@+1 {{decompose-aggregated-op op list is empty!}}
func.func @attention_f16(%query: tensor<192x1024x64xf16>,
                         %key: tensor<192x1024x64xf16>,
                         %value: tensor<192x1024x64xf16>,
                         %output: tensor<192x1024x64xf32>)
                         -> (tensor<192x1024x64xf32>) {

  %out = iree_linalg_ext.attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapO] }
        ins(%query, %key, %value : tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, tensor<192x1024x64xf16>)
        outs(%output : tensor<192x1024x64xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>

  return %out : tensor<192x1024x64xf32>
}

// CHECK-LABEL: @attention
// CHECK-SAME: %[[query:[0-9A-Za-z]*]]: tensor<192x1024x64xf16>
// CHECK-SAME: %[[key:[0-9A-Za-z]*]]: tensor<192x1024x64xf16>
// CHECK-SAME: %[[value:[0-9A-Za-z]*]]: tensor<192x1024x64xf16>
// CHECK-SAME: %[[output:[0-9A-Za-z]*]]: tensor<192x1024x64xf16>
// CHECK: %[[OUT:.*]]:3 = iree_linalg_ext.attention
