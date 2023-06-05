// MobileBert encoder model with placeholder weights, for testing.

module {
  util.global private @"__iree_flow_bert/embeddings/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/embeddings/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/embeddings/embedding_transformation/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/embeddings/embedding_transformation/kernel" {noinline} = dense<9.99999974E-5> : tensor<384x512xf32>
  util.global private @"__iree_flow_bert/embeddings/position_embeddings" {noinline} = dense<0.000000e+00> : tensor<512x512xf32>
  util.global private @"__iree_flow_bert/embeddings/token_type_embeddings" {noinline} = dense<0.000000e+00> : tensor<2x512xf32>
  util.global private @"__iree_flow_bert/embeddings/word_embeddings" {noinline} = dense<0.000000e+00> : tensor<30522x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/key/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/key/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/query/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/query/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/value/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/value/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/input/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/input/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/input/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/input/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/intermediate/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/intermediate/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/bottleneck/FakeLayerNorm/beta" = dense<1.000000e+00> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/bottleneck/FakeLayerNorm/gamma" = dense<4.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/bottleneck/dense/bias" = dense<1.000000e-01> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/bottleneck/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/dense/bias" = dense<1.000000e-01> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/dense/kernel" {noinline} = dense<9.99999974E-5> : tensor<512x128xf32>
  util.global private @"__iree_flow_cls/squad/output_bias" = dense<1.000000e-01> : tensor<2xf32>
  util.global private @"__iree_flow_cls/squad/output_weights" = dense<1.000000e+00> : tensor<2x512xf32>
  func.func @serving_default() attributes {iree.module.export} {
    %0 = util.unfoldable_constant dense<0> : tensor<1x384xi32>
    %1 = util.unfoldable_constant dense<0> : tensor<1x384xi32>
    %2 = util.unfoldable_constant dense<0> : tensor<1x384xi32>
    %ptr___iree_flow_bert2Fembeddings2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/embeddings/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fembeddings2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/embeddings/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fembeddings2Fembedding_transformation2Fbias = util.global.address @"__iree_flow_bert/embeddings/embedding_transformation/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fembeddings2Fembedding_transformation2Fkernel = util.global.address @"__iree_flow_bert/embeddings/embedding_transformation/kernel" : !util.ptr<tensor<384x512xf32>>
    %ptr___iree_flow_bert2Fembeddings2Fposition_embeddings = util.global.address @"__iree_flow_bert/embeddings/position_embeddings" : !util.ptr<tensor<512x512xf32>>
    %ptr___iree_flow_bert2Fembeddings2Ftoken_type_embeddings = util.global.address @"__iree_flow_bert/embeddings/token_type_embeddings" : !util.ptr<tensor<2x512xf32>>
    %ptr___iree_flow_bert2Fembeddings2Fword_embeddings = util.global.address @"__iree_flow_bert/embeddings/word_embeddings" : !util.ptr<tensor<30522x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_0/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_0/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_1/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_1/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_10/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_10/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_10/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_10/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_10/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_10/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_11/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_11/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_11/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_11/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_11/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_11/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_12/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_12/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_12/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_12/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_12/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_12/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_13/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_13/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_13/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_13/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_13/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_13/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_14/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_14/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_14/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_14/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_14/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_14/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_15/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_15/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_15/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_15/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_15/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_15/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_16/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_16/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_16/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_16/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_16/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_16/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_17/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_17/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_17/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_17/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_17/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_17/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_18/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_18/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_18/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_18/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_18/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_18/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_19/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_19/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_19/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_19/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_19/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_19/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_2/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_2/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_20/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_20/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_20/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_20/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_20/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_20/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_21/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_21/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_21/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_21/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_21/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_21/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_22/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_22/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_22/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_22/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_22/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_22/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_23/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_23/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_23/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_23/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_23/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_23/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_3/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_3/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_3/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_3/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_3/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_3/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_4/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_4/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_4/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_4/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_4/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_4/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_5/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_5/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_5/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_5/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_5/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_5/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_6/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_6/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_6/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_6/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_6/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_6/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_7/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_7/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_7/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_7/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_7/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_7/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_8/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_8/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_8/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_8/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_8/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_8/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fkey2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fkey2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fquery2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fquery2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fvalue2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fvalue2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Fattention2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Fattention2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Fattention2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Fattention2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Finput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Finput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Finput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Finput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fintermediate2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Fintermediate2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_9/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_9/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fbottleneck2FFakeLayerNorm2Fbeta = util.global.address @"__iree_flow_bert/encoder/layer_9/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fbottleneck2FFakeLayerNorm2Fgamma = util.global.address @"__iree_flow_bert/encoder/layer_9/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fbottleneck2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fbottleneck2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fdense2Fbias = util.global.address @"__iree_flow_bert/encoder/layer_9/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fdense2Fkernel = util.global.address @"__iree_flow_bert/encoder/layer_9/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %ptr___iree_flow_cls2Fsquad2Foutput_bias = util.global.address @"__iree_flow_cls/squad/output_bias" : !util.ptr<tensor<2xf32>>
    %ptr___iree_flow_cls2Fsquad2Foutput_weights = util.global.address @"__iree_flow_cls/squad/output_weights" : !util.ptr<tensor<2x512xf32>>
    %3 = stablehlo.constant dense<-1.000000e+04> : tensor<1x1x384x384xf32>
    %4 = stablehlo.constant dense<0.176776692> : tensor<1x4x384x384xf32>
    %5 = stablehlo.constant dense<1.000000e+04> : tensor<1x1x384x384xf32>
    %6 = stablehlo.constant dense<1.000000e+00> : tensor<1x384x384xf32>
    %7 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.constant dense<0.000000e+00> : tensor<1x384x512xf32>
    %10 = util.global.load.indirect %ptr___iree_flow_bert2Fembeddings2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %11 = util.global.load.indirect %ptr___iree_flow_bert2Fembeddings2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %12 = util.global.load.indirect %ptr___iree_flow_bert2Fembeddings2Fembedding_transformation2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %13 = util.global.load.indirect %ptr___iree_flow_bert2Fembeddings2Fembedding_transformation2Fkernel : !util.ptr<tensor<384x512xf32>> -> tensor<384x512xf32>
    %14 = util.global.load.indirect %ptr___iree_flow_bert2Fembeddings2Fposition_embeddings : !util.ptr<tensor<512x512xf32>> -> tensor<512x512xf32>
    %15 = "stablehlo.slice"(%14) {limit_indices = dense<[384, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x512xf32>) -> tensor<384x512xf32>
    %16 = stablehlo.reshape %15 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %17 = util.global.load.indirect %ptr___iree_flow_bert2Fembeddings2Ftoken_type_embeddings : !util.ptr<tensor<2x512xf32>> -> tensor<2x512xf32>
    %18 = util.global.load.indirect %ptr___iree_flow_bert2Fembeddings2Fword_embeddings : !util.ptr<tensor<30522x128xf32>> -> tensor<30522x128xf32>
    %19 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %20 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %21 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %22 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %23 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %24 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %25 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %26 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %27 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %28 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %29 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %30 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %31 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %32 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %33 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %34 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %35 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %36 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %37 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %38 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %39 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %40 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %41 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %42 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %43 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %44 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %45 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %46 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %47 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %48 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %49 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %50 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %51 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %52 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %53 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %54 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %55 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %56 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %57 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %58 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %59 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %60 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %61 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %62 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %63 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %64 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %65 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %66 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %67 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %68 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %69 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %70 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %71 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %72 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %73 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %74 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %75 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %76 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %77 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %78 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %79 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %80 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %81 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %82 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %83 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %84 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %85 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %86 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %87 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %88 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %89 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %90 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %91 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %92 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %93 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %94 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %95 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %96 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %97 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %98 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %99 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %100 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %101 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %102 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %103 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %104 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %105 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %106 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %107 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %108 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %109 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %110 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %111 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %112 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %113 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %114 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %115 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %116 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %117 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %118 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %119 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %120 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %121 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %122 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %123 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %124 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %125 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %126 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %127 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %128 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %129 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %130 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %131 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %132 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %133 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %134 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %135 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %136 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %137 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %138 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %139 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %140 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %141 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %142 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %143 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %144 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %145 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %146 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %147 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %148 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %149 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %150 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %151 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %152 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %153 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %154 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %155 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %156 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_102Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %157 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %158 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %159 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %160 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %161 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %162 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %163 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %164 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %165 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %166 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %167 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %168 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %169 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %170 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %171 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %172 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %173 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %174 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %175 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %176 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %177 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %178 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %179 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %180 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %181 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %182 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %183 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %184 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %185 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %186 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %187 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %188 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %189 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %190 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %191 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %192 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %193 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %194 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %195 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %196 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %197 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %198 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %199 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %200 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %201 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %202 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_112Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %203 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %204 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %205 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %206 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %207 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %208 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %209 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %210 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %211 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %212 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %213 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %214 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %215 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %216 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %217 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %218 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %219 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %220 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %221 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %222 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %223 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %224 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %225 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %226 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %227 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %228 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %229 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %230 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %231 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %232 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %233 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %234 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %235 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %236 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %237 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %238 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %239 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %240 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %241 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %242 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %243 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %244 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %245 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %246 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %247 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %248 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_122Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %249 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %250 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %251 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %252 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %253 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %254 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %255 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %256 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %257 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %258 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %259 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %260 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %261 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %262 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %263 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %264 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %265 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %266 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %267 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %268 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %269 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %270 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %271 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %272 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %273 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %274 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %275 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %276 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %277 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %278 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %279 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %280 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %281 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %282 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %283 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %284 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %285 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %286 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %287 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %288 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %289 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %290 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %291 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %292 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %293 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %294 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_132Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %295 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %296 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %297 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %298 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %299 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %300 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %301 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %302 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %303 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %304 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %305 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %306 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %307 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %308 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %309 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %310 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %311 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %312 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %313 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %314 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %315 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %316 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %317 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %318 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %319 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %320 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %321 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %322 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %323 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %324 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %325 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %326 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %327 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %328 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %329 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %330 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %331 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %332 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %333 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %334 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %335 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %336 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %337 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %338 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %339 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %340 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_142Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %341 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %342 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %343 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %344 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %345 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %346 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %347 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %348 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %349 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %350 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %351 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %352 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %353 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %354 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %355 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %356 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %357 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %358 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %359 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %360 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %361 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %362 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %363 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %364 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %365 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %366 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %367 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %368 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %369 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %370 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %371 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %372 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %373 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %374 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %375 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %376 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %377 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %378 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %379 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %380 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %381 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %382 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %383 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %384 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %385 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %386 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_152Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %387 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %388 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %389 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %390 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %391 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %392 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %393 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %394 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %395 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %396 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %397 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %398 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %399 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %400 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %401 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %402 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %403 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %404 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %405 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %406 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %407 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %408 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %409 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %410 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %411 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %412 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %413 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %414 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %415 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %416 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %417 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %418 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %419 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %420 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %421 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %422 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %423 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %424 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %425 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %426 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %427 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %428 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %429 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %430 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %431 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %432 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_162Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %433 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %434 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %435 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %436 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %437 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %438 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %439 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %440 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %441 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %442 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %443 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %444 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %445 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %446 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %447 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %448 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %449 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %450 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %451 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %452 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %453 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %454 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %455 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %456 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %457 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %458 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %459 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %460 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %461 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %462 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %463 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %464 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %465 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %466 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %467 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %468 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %469 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %470 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %471 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %472 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %473 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %474 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %475 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %476 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %477 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %478 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_172Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %479 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %480 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %481 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %482 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %483 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %484 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %485 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %486 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %487 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %488 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %489 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %490 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %491 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %492 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %493 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %494 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %495 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %496 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %497 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %498 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %499 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %500 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %501 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %502 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %503 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %504 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %505 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %506 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %507 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %508 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %509 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %510 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %511 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %512 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %513 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %514 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %515 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %516 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %517 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %518 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %519 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %520 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %521 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %522 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %523 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %524 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_182Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %525 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %526 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %527 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %528 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %529 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %530 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %531 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %532 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %533 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %534 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %535 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %536 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %537 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %538 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %539 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %540 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %541 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %542 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %543 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %544 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %545 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %546 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %547 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %548 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %549 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %550 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %551 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %552 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %553 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %554 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %555 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %556 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %557 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %558 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %559 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %560 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %561 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %562 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %563 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %564 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %565 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %566 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %567 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %568 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %569 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %570 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_192Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %571 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %572 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %573 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %574 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %575 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %576 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %577 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %578 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %579 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %580 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %581 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %582 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %583 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %584 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %585 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %586 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %587 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %588 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %589 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %590 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %591 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %592 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %593 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %594 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %595 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %596 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %597 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %598 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %599 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %600 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %601 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %602 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %603 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %604 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %605 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %606 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %607 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %608 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %609 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %610 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %611 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %612 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %613 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %614 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %615 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %616 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %617 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %618 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %619 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %620 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %621 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %622 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %623 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %624 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %625 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %626 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %627 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %628 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %629 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %630 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %631 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %632 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %633 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %634 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %635 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %636 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %637 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %638 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %639 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %640 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %641 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %642 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %643 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %644 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %645 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %646 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %647 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %648 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %649 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %650 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %651 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %652 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %653 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %654 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %655 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %656 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %657 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %658 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %659 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %660 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %661 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %662 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_202Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %663 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %664 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %665 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %666 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %667 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %668 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %669 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %670 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %671 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %672 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %673 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %674 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %675 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %676 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %677 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %678 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %679 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %680 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %681 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %682 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %683 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %684 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %685 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %686 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %687 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %688 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %689 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %690 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %691 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %692 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %693 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %694 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %695 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %696 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %697 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %698 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %699 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %700 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %701 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %702 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %703 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %704 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %705 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %706 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %707 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %708 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_212Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %709 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %710 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %711 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %712 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %713 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %714 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %715 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %716 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %717 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %718 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %719 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %720 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %721 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %722 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %723 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %724 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %725 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %726 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %727 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %728 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %729 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %730 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %731 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %732 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %733 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %734 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %735 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %736 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %737 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %738 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %739 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %740 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %741 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %742 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %743 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %744 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %745 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %746 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %747 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %748 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %749 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %750 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %751 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %752 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %753 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %754 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_222Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %755 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %756 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %757 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %758 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %759 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %760 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %761 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %762 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %763 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %764 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %765 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %766 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %767 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %768 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %769 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %770 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %771 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %772 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %773 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %774 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %775 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %776 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %777 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %778 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %779 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %780 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %781 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %782 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %783 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %784 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %785 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %786 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %787 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %788 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %789 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %790 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %791 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %792 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %793 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %794 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %795 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %796 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %797 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %798 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %799 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %800 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_232Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %801 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %802 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %803 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %804 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %805 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %806 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %807 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %808 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %809 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %810 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %811 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %812 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %813 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %814 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %815 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %816 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %817 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %818 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %819 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %820 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %821 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %822 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %823 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %824 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %825 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %826 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %827 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %828 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %829 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %830 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %831 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %832 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %833 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %834 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %835 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %836 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %837 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %838 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %839 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %840 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %841 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %842 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %843 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %844 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %845 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %846 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_32Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %847 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %848 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %849 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %850 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %851 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %852 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %853 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %854 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %855 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %856 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %857 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %858 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %859 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %860 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %861 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %862 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %863 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %864 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %865 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %866 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %867 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %868 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %869 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %870 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %871 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %872 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %873 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %874 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %875 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %876 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %877 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %878 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %879 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %880 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %881 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %882 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %883 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %884 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %885 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %886 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %887 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %888 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %889 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %890 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %891 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %892 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_42Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %893 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %894 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %895 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %896 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %897 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %898 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %899 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %900 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %901 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %902 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %903 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %904 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %905 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %906 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %907 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %908 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %909 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %910 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %911 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %912 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %913 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %914 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %915 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %916 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %917 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %918 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %919 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %920 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %921 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %922 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %923 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %924 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %925 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %926 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %927 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %928 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %929 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %930 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %931 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %932 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %933 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %934 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %935 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %936 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %937 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %938 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_52Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %939 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %940 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %941 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %942 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %943 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %944 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %945 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %946 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %947 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %948 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %949 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %950 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %951 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %952 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %953 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %954 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %955 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %956 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %957 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %958 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %959 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %960 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %961 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %962 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %963 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %964 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %965 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %966 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %967 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %968 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %969 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %970 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %971 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %972 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %973 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %974 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %975 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %976 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %977 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %978 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %979 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %980 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %981 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %982 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %983 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %984 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_62Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %985 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %986 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %987 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %988 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %989 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %990 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %991 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %992 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %993 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %994 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %995 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %996 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %997 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %998 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %999 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1000 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1001 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1002 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1003 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1004 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1005 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1006 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1007 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1008 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1009 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1010 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1011 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1012 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1013 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1014 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1015 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1016 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1017 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1018 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1019 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1020 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1021 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1022 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1023 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1024 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1025 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1026 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1027 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1028 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1029 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1030 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_72Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1031 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1032 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1033 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1034 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1035 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1036 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1037 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1038 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1039 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1040 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1041 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1042 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1043 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1044 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1045 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1046 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1047 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1048 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1049 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1050 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1051 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1052 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1053 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1054 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1055 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1056 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1057 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1058 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1059 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1060 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1061 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1062 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1063 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1064 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1065 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1066 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1067 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1068 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1069 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1070 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1071 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1072 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1073 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1074 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1075 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1076 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_82Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1077 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1078 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1079 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1080 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Foutput2Fdense2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1081 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fkey2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1082 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fkey2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1083 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fquery2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1084 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fquery2Fkernel : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1085 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fvalue2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1086 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fattention2Fself2Fvalue2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1087 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Fattention2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1088 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Fattention2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1089 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Fattention2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1090 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Fattention2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1091 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Finput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1092 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Finput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1093 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Finput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1094 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fbottleneck2Finput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1095 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1096 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1097 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1098 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1099 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1100 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_02Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1101 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1102 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1103 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1104 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1105 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1106 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_12Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1107 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1108 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1109 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1110 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1111 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1112 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fffn_layer_22Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1113 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fintermediate2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1114 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Fintermediate2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1115 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2FFakeLayerNorm2Fbeta : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1116 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2FFakeLayerNorm2Fgamma : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1117 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fbottleneck2FFakeLayerNorm2Fbeta : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1118 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fbottleneck2FFakeLayerNorm2Fgamma : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1119 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fbottleneck2Fdense2Fbias : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1120 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fbottleneck2Fdense2Fkernel : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1121 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fdense2Fbias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1122 = util.global.load.indirect %ptr___iree_flow_bert2Fencoder2Flayer_92Foutput2Fdense2Fkernel : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1123 = util.global.load.indirect %ptr___iree_flow_cls2Fsquad2Foutput_bias : !util.ptr<tensor<2xf32>> -> tensor<2xf32>
    %1124 = util.global.load.indirect %ptr___iree_flow_cls2Fsquad2Foutput_weights : !util.ptr<tensor<2x512xf32>> -> tensor<2x512xf32>
    %1125 = stablehlo.reshape %1 : (tensor<1x384xi32>) -> tensor<1x384x1xi32>
    %1126 = "stablehlo.torch_index_select"(%18, %1125) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<30522x128xf32>, tensor<1x384x1xi32>) -> tensor<1x384x1x128xf32>
    %1127 = stablehlo.reshape %1126 : (tensor<1x384x1x128xf32>) -> tensor<1x384x128xf32>
    %1128 = "stablehlo.slice"(%1127) {limit_indices = dense<[1, 384, 128]> : tensor<3xi64>, start_indices = dense<[0, 1, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x384x128xf32>) -> tensor<1x383x128xf32>
    %1129 = stablehlo.pad %1128, %8, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<1x383x128xf32>, tensor<f32>) -> tensor<1x384x128xf32>
    %1130 = "stablehlo.slice"(%1127) {limit_indices = dense<[1, 383, 128]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x384x128xf32>) -> tensor<1x383x128xf32>
    %1131 = stablehlo.pad %1130, %8, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x383x128xf32>, tensor<f32>) -> tensor<1x384x128xf32>
    %1132 = stablehlo.concatenate %1129, %1127, %1131, dim = 2 : (tensor<1x384x128xf32>, tensor<1x384x128xf32>, tensor<1x384x128xf32>) -> tensor<1x384x384xf32>
    %1133 = stablehlo.reshape %1132 : (tensor<1x384x384xf32>) -> tensor<384x384xf32>
    %1134 = stablehlo.dot %1133, %13 : (tensor<384x384xf32>, tensor<384x512xf32>) -> tensor<384x512xf32>
    %1135 = stablehlo.broadcast_in_dim %12, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1136 = stablehlo.add %1134, %1135 : tensor<384x512xf32>
    %1137 = stablehlo.reshape %1136 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1138 = stablehlo.convert %0 : (tensor<1x384xi32>) -> tensor<1x384xf32>
    %1139 = stablehlo.reshape %1138 : (tensor<1x384xf32>) -> tensor<1x1x384xf32>
    %1140 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2] : (tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
    %1141 = stablehlo.reshape %1140 : (tensor<1x384x384xf32>) -> tensor<1x1x384x384xf32>
    %1142 = stablehlo.multiply %1141, %5 : tensor<1x1x384x384xf32>
    %1143 = stablehlo.add %1142, %3 : tensor<1x1x384x384xf32>
    %1144 = "stablehlo.torch_index_select"(%17, %2) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<2x512xf32>, tensor<1x384xi32>) -> tensor<1x384x512xf32>
    %1145 = stablehlo.add %1137, %1144 : tensor<1x384x512xf32>
    %1146 = stablehlo.add %1145, %16 : tensor<1x384x512xf32>
    %1147 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1148 = stablehlo.multiply %1146, %1147 : tensor<1x384x512xf32>
    %1149 = stablehlo.broadcast_in_dim %10, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1150 = stablehlo.add %1148, %1149 : tensor<1x384x512xf32>
    %1151 = stablehlo.reshape %1150 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1152 = stablehlo.dot %1151, %28 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1153 = stablehlo.broadcast_in_dim %27, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1154 = stablehlo.add %1152, %1153 : tensor<384x128xf32>
    %1155 = stablehlo.reshape %1154 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1156 = stablehlo.transpose %1155, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1157 = stablehlo.dot %1151, %32 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1158 = stablehlo.reshape %1157 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1159 = stablehlo.broadcast_in_dim %31, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1160 = stablehlo.add %1158, %1159 : tensor<1x384x128xf32>
    %1161 = stablehlo.broadcast_in_dim %30, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1162 = stablehlo.multiply %1160, %1161 : tensor<1x384x128xf32>
    %1163 = stablehlo.broadcast_in_dim %29, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1164 = stablehlo.add %1162, %1163 : tensor<1x384x128xf32>
    %1165 = stablehlo.reshape %1164 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1166 = stablehlo.dot %1165, %24 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1167 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1168 = stablehlo.add %1166, %1167 : tensor<384x128xf32>
    %1169 = stablehlo.reshape %1168 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1170 = stablehlo.transpose %1169, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1171 = stablehlo.dot %1165, %26 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1172 = stablehlo.broadcast_in_dim %25, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1173 = stablehlo.add %1171, %1172 : tensor<384x128xf32>
    %1174 = stablehlo.reshape %1173 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1175 = stablehlo.transpose %1174, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1176 = stablehlo.dot_general %1175, %1170, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %1177 = stablehlo.multiply %1176, %4 : tensor<1x4x384x384xf32>
    %1178 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %1179 = stablehlo.add %1177, %1178 : tensor<1x4x384x384xf32>
    %1180 = stablehlo.reduce(%1179 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1181 = stablehlo.broadcast_in_dim %1180, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1182 = stablehlo.subtract %1179, %1181 : tensor<1x4x384x384xf32>
    %1183 = stablehlo.exponential %1182 : tensor<1x4x384x384xf32>
    %1184 = stablehlo.reduce(%1183 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1185 = stablehlo.broadcast_in_dim %1184, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1186 = stablehlo.divide %1183, %1185 : tensor<1x4x384x384xf32>
    %1187 = stablehlo.dot_general %1186, %1156, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %1188 = stablehlo.transpose %1187, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %1189 = stablehlo.reshape %1188 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %1190 = stablehlo.dot %1189, %22 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1191 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1192 = stablehlo.add %1190, %1191 : tensor<384x128xf32>
    %1193 = stablehlo.reshape %1192 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1194 = stablehlo.dot %1151, %36 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1195 = stablehlo.broadcast_in_dim %35, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1196 = stablehlo.add %1194, %1195 : tensor<384x128xf32>
    %1197 = stablehlo.reshape %1196 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1198 = stablehlo.broadcast_in_dim %34, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1199 = stablehlo.multiply %1197, %1198 : tensor<1x384x128xf32>
    %1200 = stablehlo.broadcast_in_dim %33, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1201 = stablehlo.add %1199, %1200 : tensor<1x384x128xf32>
    %1202 = stablehlo.add %1193, %1201 : tensor<1x384x128xf32>
    %1203 = stablehlo.broadcast_in_dim %20, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1204 = stablehlo.multiply %1202, %1203 : tensor<1x384x128xf32>
    %1205 = stablehlo.broadcast_in_dim %19, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1206 = stablehlo.add %1204, %1205 : tensor<1x384x128xf32>
    %1207 = stablehlo.reshape %1206 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1208 = stablehlo.dot %1207, %38 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1209 = stablehlo.broadcast_in_dim %37, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1210 = stablehlo.add %1208, %1209 : tensor<384x512xf32>
    %1211 = stablehlo.reshape %1210 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1212 = stablehlo.maximum %1211, %9 : tensor<1x384x512xf32>
    %1213 = stablehlo.reshape %1212 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1214 = stablehlo.dot %1213, %42 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1215 = stablehlo.broadcast_in_dim %41, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1216 = stablehlo.add %1214, %1215 : tensor<384x128xf32>
    %1217 = stablehlo.reshape %1216 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1218 = stablehlo.add %1217, %1206 : tensor<1x384x128xf32>
    %1219 = stablehlo.broadcast_in_dim %40, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1220 = stablehlo.multiply %1218, %1219 : tensor<1x384x128xf32>
    %1221 = stablehlo.broadcast_in_dim %39, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1222 = stablehlo.add %1220, %1221 : tensor<1x384x128xf32>
    %1223 = stablehlo.reshape %1222 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1224 = stablehlo.dot %1223, %44 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1225 = stablehlo.broadcast_in_dim %43, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1226 = stablehlo.add %1224, %1225 : tensor<384x512xf32>
    %1227 = stablehlo.reshape %1226 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1228 = stablehlo.maximum %1227, %9 : tensor<1x384x512xf32>
    %1229 = stablehlo.reshape %1228 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1230 = stablehlo.dot %1229, %48 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1231 = stablehlo.broadcast_in_dim %47, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1232 = stablehlo.add %1230, %1231 : tensor<384x128xf32>
    %1233 = stablehlo.reshape %1232 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1234 = stablehlo.add %1233, %1222 : tensor<1x384x128xf32>
    %1235 = stablehlo.broadcast_in_dim %46, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1236 = stablehlo.multiply %1234, %1235 : tensor<1x384x128xf32>
    %1237 = stablehlo.broadcast_in_dim %45, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1238 = stablehlo.add %1236, %1237 : tensor<1x384x128xf32>
    %1239 = stablehlo.reshape %1238 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1240 = stablehlo.dot %1239, %50 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1241 = stablehlo.broadcast_in_dim %49, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1242 = stablehlo.add %1240, %1241 : tensor<384x512xf32>
    %1243 = stablehlo.reshape %1242 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1244 = stablehlo.maximum %1243, %9 : tensor<1x384x512xf32>
    %1245 = stablehlo.reshape %1244 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1246 = stablehlo.dot %1245, %54 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1247 = stablehlo.broadcast_in_dim %53, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1248 = stablehlo.add %1246, %1247 : tensor<384x128xf32>
    %1249 = stablehlo.reshape %1248 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1250 = stablehlo.add %1249, %1238 : tensor<1x384x128xf32>
    %1251 = stablehlo.broadcast_in_dim %52, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1252 = stablehlo.multiply %1250, %1251 : tensor<1x384x128xf32>
    %1253 = stablehlo.broadcast_in_dim %51, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1254 = stablehlo.add %1252, %1253 : tensor<1x384x128xf32>
    %1255 = stablehlo.reshape %1254 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1256 = stablehlo.dot %1255, %56 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1257 = stablehlo.broadcast_in_dim %55, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1258 = stablehlo.add %1256, %1257 : tensor<384x512xf32>
    %1259 = stablehlo.reshape %1258 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1260 = stablehlo.maximum %1259, %9 : tensor<1x384x512xf32>
    %1261 = stablehlo.reshape %1260 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1262 = stablehlo.dot %1261, %64 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1263 = stablehlo.broadcast_in_dim %63, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1264 = stablehlo.add %1262, %1263 : tensor<384x128xf32>
    %1265 = stablehlo.reshape %1264 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1266 = stablehlo.add %1265, %1254 : tensor<1x384x128xf32>
    %1267 = stablehlo.broadcast_in_dim %58, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1268 = stablehlo.multiply %1266, %1267 : tensor<1x384x128xf32>
    %1269 = stablehlo.broadcast_in_dim %57, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1270 = stablehlo.add %1268, %1269 : tensor<1x384x128xf32>
    %1271 = stablehlo.reshape %1270 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1272 = stablehlo.dot %1271, %62 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1273 = stablehlo.broadcast_in_dim %61, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1274 = stablehlo.add %1272, %1273 : tensor<384x512xf32>
    %1275 = stablehlo.reshape %1274 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1276 = stablehlo.add %1275, %1150 : tensor<1x384x512xf32>
    %1277 = stablehlo.broadcast_in_dim %60, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1278 = stablehlo.multiply %1276, %1277 : tensor<1x384x512xf32>
    %1279 = stablehlo.broadcast_in_dim %59, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1280 = stablehlo.add %1278, %1279 : tensor<1x384x512xf32>
    %1281 = stablehlo.reshape %1280 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1282 = stablehlo.dot %1281, %74 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1283 = stablehlo.broadcast_in_dim %73, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1284 = stablehlo.add %1282, %1283 : tensor<384x128xf32>
    %1285 = stablehlo.reshape %1284 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1286 = stablehlo.transpose %1285, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1287 = stablehlo.dot %1281, %78 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1288 = stablehlo.reshape %1287 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1289 = stablehlo.broadcast_in_dim %77, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1290 = stablehlo.add %1288, %1289 : tensor<1x384x128xf32>
    %1291 = stablehlo.broadcast_in_dim %76, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1292 = stablehlo.multiply %1290, %1291 : tensor<1x384x128xf32>
    %1293 = stablehlo.broadcast_in_dim %75, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1294 = stablehlo.add %1292, %1293 : tensor<1x384x128xf32>
    %1295 = stablehlo.reshape %1294 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1296 = stablehlo.dot %1295, %70 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1297 = stablehlo.broadcast_in_dim %69, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1298 = stablehlo.add %1296, %1297 : tensor<384x128xf32>
    %1299 = stablehlo.reshape %1298 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1300 = stablehlo.transpose %1299, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1301 = stablehlo.dot %1295, %72 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1302 = stablehlo.broadcast_in_dim %71, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1303 = stablehlo.add %1301, %1302 : tensor<384x128xf32>
    %1304 = stablehlo.reshape %1303 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1305 = stablehlo.transpose %1304, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1306 = stablehlo.dot_general %1305, %1300, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %1307 = stablehlo.multiply %1306, %4 : tensor<1x4x384x384xf32>
    %1308 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %1309 = stablehlo.add %1307, %1308 : tensor<1x4x384x384xf32>
    %1310 = stablehlo.reduce(%1309 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1311 = stablehlo.broadcast_in_dim %1310, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1312 = stablehlo.subtract %1309, %1311 : tensor<1x4x384x384xf32>
    %1313 = stablehlo.exponential %1312 : tensor<1x4x384x384xf32>
    %1314 = stablehlo.reduce(%1313 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1315 = stablehlo.broadcast_in_dim %1314, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1316 = stablehlo.divide %1313, %1315 : tensor<1x4x384x384xf32>
    %1317 = stablehlo.dot_general %1316, %1286, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %1318 = stablehlo.transpose %1317, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %1319 = stablehlo.reshape %1318 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %1320 = stablehlo.dot %1319, %68 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1321 = stablehlo.broadcast_in_dim %67, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1322 = stablehlo.add %1320, %1321 : tensor<384x128xf32>
    %1323 = stablehlo.reshape %1322 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1324 = stablehlo.dot %1281, %82 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1325 = stablehlo.broadcast_in_dim %81, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1326 = stablehlo.add %1324, %1325 : tensor<384x128xf32>
    %1327 = stablehlo.reshape %1326 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1328 = stablehlo.broadcast_in_dim %80, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1329 = stablehlo.multiply %1327, %1328 : tensor<1x384x128xf32>
    %1330 = stablehlo.broadcast_in_dim %79, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1331 = stablehlo.add %1329, %1330 : tensor<1x384x128xf32>
    %1332 = stablehlo.add %1323, %1331 : tensor<1x384x128xf32>
    %1333 = stablehlo.broadcast_in_dim %66, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1334 = stablehlo.multiply %1332, %1333 : tensor<1x384x128xf32>
    %1335 = stablehlo.broadcast_in_dim %65, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1336 = stablehlo.add %1334, %1335 : tensor<1x384x128xf32>
    %1337 = stablehlo.reshape %1336 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1338 = stablehlo.dot %1337, %84 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1339 = stablehlo.broadcast_in_dim %83, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1340 = stablehlo.add %1338, %1339 : tensor<384x512xf32>
    %1341 = stablehlo.reshape %1340 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1342 = stablehlo.maximum %1341, %9 : tensor<1x384x512xf32>
    %1343 = stablehlo.reshape %1342 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1344 = stablehlo.dot %1343, %88 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1345 = stablehlo.broadcast_in_dim %87, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1346 = stablehlo.add %1344, %1345 : tensor<384x128xf32>
    %1347 = stablehlo.reshape %1346 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1348 = stablehlo.add %1347, %1336 : tensor<1x384x128xf32>
    %1349 = stablehlo.broadcast_in_dim %86, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1350 = stablehlo.multiply %1348, %1349 : tensor<1x384x128xf32>
    %1351 = stablehlo.broadcast_in_dim %85, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1352 = stablehlo.add %1350, %1351 : tensor<1x384x128xf32>
    %1353 = stablehlo.reshape %1352 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1354 = stablehlo.dot %1353, %90 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1355 = stablehlo.broadcast_in_dim %89, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1356 = stablehlo.add %1354, %1355 : tensor<384x512xf32>
    %1357 = stablehlo.reshape %1356 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1358 = stablehlo.maximum %1357, %9 : tensor<1x384x512xf32>
    %1359 = stablehlo.reshape %1358 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1360 = stablehlo.dot %1359, %94 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1361 = stablehlo.broadcast_in_dim %93, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1362 = stablehlo.add %1360, %1361 : tensor<384x128xf32>
    %1363 = stablehlo.reshape %1362 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1364 = stablehlo.add %1363, %1352 : tensor<1x384x128xf32>
    %1365 = stablehlo.broadcast_in_dim %92, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1366 = stablehlo.multiply %1364, %1365 : tensor<1x384x128xf32>
    %1367 = stablehlo.broadcast_in_dim %91, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1368 = stablehlo.add %1366, %1367 : tensor<1x384x128xf32>
    %1369 = stablehlo.reshape %1368 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1370 = stablehlo.dot %1369, %96 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1371 = stablehlo.broadcast_in_dim %95, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1372 = stablehlo.add %1370, %1371 : tensor<384x512xf32>
    %1373 = stablehlo.reshape %1372 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1374 = stablehlo.maximum %1373, %9 : tensor<1x384x512xf32>
    %1375 = stablehlo.reshape %1374 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1376 = stablehlo.dot %1375, %100 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1377 = stablehlo.broadcast_in_dim %99, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1378 = stablehlo.add %1376, %1377 : tensor<384x128xf32>
    %1379 = stablehlo.reshape %1378 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1380 = stablehlo.add %1379, %1368 : tensor<1x384x128xf32>
    %1381 = stablehlo.broadcast_in_dim %98, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1382 = stablehlo.multiply %1380, %1381 : tensor<1x384x128xf32>
    %1383 = stablehlo.broadcast_in_dim %97, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1384 = stablehlo.add %1382, %1383 : tensor<1x384x128xf32>
    %1385 = stablehlo.reshape %1384 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1386 = stablehlo.dot %1385, %102 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1387 = stablehlo.broadcast_in_dim %101, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1388 = stablehlo.add %1386, %1387 : tensor<384x512xf32>
    %1389 = stablehlo.reshape %1388 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1390 = stablehlo.maximum %1389, %9 : tensor<1x384x512xf32>
    %1391 = stablehlo.reshape %1390 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1392 = stablehlo.dot %1391, %110 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1393 = stablehlo.broadcast_in_dim %109, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1394 = stablehlo.add %1392, %1393 : tensor<384x128xf32>
    %1395 = stablehlo.reshape %1394 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1396 = stablehlo.add %1395, %1384 : tensor<1x384x128xf32>
    %1397 = stablehlo.broadcast_in_dim %104, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1398 = stablehlo.multiply %1396, %1397 : tensor<1x384x128xf32>
    %1399 = stablehlo.broadcast_in_dim %103, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1400 = stablehlo.add %1398, %1399 : tensor<1x384x128xf32>
    %1401 = stablehlo.reshape %1400 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1402 = stablehlo.dot %1401, %108 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1403 = stablehlo.broadcast_in_dim %107, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1404 = stablehlo.add %1402, %1403 : tensor<384x512xf32>
    %1405 = stablehlo.reshape %1404 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1406 = stablehlo.add %1405, %1280 : tensor<1x384x512xf32>
    %1407 = stablehlo.broadcast_in_dim %106, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1408 = stablehlo.multiply %1406, %1407 : tensor<1x384x512xf32>
    %1409 = stablehlo.broadcast_in_dim %105, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1410 = stablehlo.add %1408, %1409 : tensor<1x384x512xf32>
    %1411 = stablehlo.reshape %1410 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1412 = stablehlo.dot %1411, %580 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1413 = stablehlo.broadcast_in_dim %579, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1414 = stablehlo.add %1412, %1413 : tensor<384x128xf32>
    %1415 = stablehlo.reshape %1414 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1416 = stablehlo.transpose %1415, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1417 = stablehlo.dot %1411, %584 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1418 = stablehlo.reshape %1417 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1419 = stablehlo.broadcast_in_dim %583, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1420 = stablehlo.add %1418, %1419 : tensor<1x384x128xf32>
    %1421 = stablehlo.broadcast_in_dim %582, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1422 = stablehlo.multiply %1420, %1421 : tensor<1x384x128xf32>
    %1423 = stablehlo.broadcast_in_dim %581, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1424 = stablehlo.add %1422, %1423 : tensor<1x384x128xf32>
    %1425 = stablehlo.reshape %1424 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1426 = stablehlo.dot %1425, %576 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1427 = stablehlo.broadcast_in_dim %575, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1428 = stablehlo.add %1426, %1427 : tensor<384x128xf32>
    %1429 = stablehlo.reshape %1428 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1430 = stablehlo.transpose %1429, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1431 = stablehlo.dot %1425, %578 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1432 = stablehlo.broadcast_in_dim %577, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1433 = stablehlo.add %1431, %1432 : tensor<384x128xf32>
    %1434 = stablehlo.reshape %1433 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1435 = stablehlo.transpose %1434, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1436 = stablehlo.dot_general %1435, %1430, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %1437 = stablehlo.multiply %1436, %4 : tensor<1x4x384x384xf32>
    %1438 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %1439 = stablehlo.add %1437, %1438 : tensor<1x4x384x384xf32>
    %1440 = stablehlo.reduce(%1439 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1441 = stablehlo.broadcast_in_dim %1440, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1442 = stablehlo.subtract %1439, %1441 : tensor<1x4x384x384xf32>
    %1443 = stablehlo.exponential %1442 : tensor<1x4x384x384xf32>
    %1444 = stablehlo.reduce(%1443 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1445 = stablehlo.broadcast_in_dim %1444, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1446 = stablehlo.divide %1443, %1445 : tensor<1x4x384x384xf32>
    %1447 = stablehlo.dot_general %1446, %1416, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %1448 = stablehlo.transpose %1447, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %1449 = stablehlo.reshape %1448 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %1450 = stablehlo.dot %1449, %574 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1451 = stablehlo.broadcast_in_dim %573, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1452 = stablehlo.add %1450, %1451 : tensor<384x128xf32>
    %1453 = stablehlo.reshape %1452 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1454 = stablehlo.dot %1411, %588 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1455 = stablehlo.broadcast_in_dim %587, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1456 = stablehlo.add %1454, %1455 : tensor<384x128xf32>
    %1457 = stablehlo.reshape %1456 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1458 = stablehlo.broadcast_in_dim %586, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1459 = stablehlo.multiply %1457, %1458 : tensor<1x384x128xf32>
    %1460 = stablehlo.broadcast_in_dim %585, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1461 = stablehlo.add %1459, %1460 : tensor<1x384x128xf32>
    %1462 = stablehlo.add %1453, %1461 : tensor<1x384x128xf32>
    %1463 = stablehlo.broadcast_in_dim %572, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1464 = stablehlo.multiply %1462, %1463 : tensor<1x384x128xf32>
    %1465 = stablehlo.broadcast_in_dim %571, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1466 = stablehlo.add %1464, %1465 : tensor<1x384x128xf32>
    %1467 = stablehlo.reshape %1466 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1468 = stablehlo.dot %1467, %590 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1469 = stablehlo.broadcast_in_dim %589, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1470 = stablehlo.add %1468, %1469 : tensor<384x512xf32>
    %1471 = stablehlo.reshape %1470 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1472 = stablehlo.maximum %1471, %9 : tensor<1x384x512xf32>
    %1473 = stablehlo.reshape %1472 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1474 = stablehlo.dot %1473, %594 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1475 = stablehlo.broadcast_in_dim %593, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1476 = stablehlo.add %1474, %1475 : tensor<384x128xf32>
    %1477 = stablehlo.reshape %1476 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1478 = stablehlo.add %1477, %1466 : tensor<1x384x128xf32>
    %1479 = stablehlo.broadcast_in_dim %592, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1480 = stablehlo.multiply %1478, %1479 : tensor<1x384x128xf32>
    %1481 = stablehlo.broadcast_in_dim %591, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1482 = stablehlo.add %1480, %1481 : tensor<1x384x128xf32>
    %1483 = stablehlo.reshape %1482 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1484 = stablehlo.dot %1483, %596 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1485 = stablehlo.broadcast_in_dim %595, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1486 = stablehlo.add %1484, %1485 : tensor<384x512xf32>
    %1487 = stablehlo.reshape %1486 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1488 = stablehlo.maximum %1487, %9 : tensor<1x384x512xf32>
    %1489 = stablehlo.reshape %1488 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1490 = stablehlo.dot %1489, %600 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1491 = stablehlo.broadcast_in_dim %599, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1492 = stablehlo.add %1490, %1491 : tensor<384x128xf32>
    %1493 = stablehlo.reshape %1492 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1494 = stablehlo.add %1493, %1482 : tensor<1x384x128xf32>
    %1495 = stablehlo.broadcast_in_dim %598, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1496 = stablehlo.multiply %1494, %1495 : tensor<1x384x128xf32>
    %1497 = stablehlo.broadcast_in_dim %597, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1498 = stablehlo.add %1496, %1497 : tensor<1x384x128xf32>
    %1499 = stablehlo.reshape %1498 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1500 = stablehlo.dot %1499, %602 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1501 = stablehlo.broadcast_in_dim %601, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1502 = stablehlo.add %1500, %1501 : tensor<384x512xf32>
    %1503 = stablehlo.reshape %1502 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1504 = stablehlo.maximum %1503, %9 : tensor<1x384x512xf32>
    %1505 = stablehlo.reshape %1504 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1506 = stablehlo.dot %1505, %606 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1507 = stablehlo.broadcast_in_dim %605, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1508 = stablehlo.add %1506, %1507 : tensor<384x128xf32>
    %1509 = stablehlo.reshape %1508 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1510 = stablehlo.add %1509, %1498 : tensor<1x384x128xf32>
    %1511 = stablehlo.broadcast_in_dim %604, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1512 = stablehlo.multiply %1510, %1511 : tensor<1x384x128xf32>
    %1513 = stablehlo.broadcast_in_dim %603, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1514 = stablehlo.add %1512, %1513 : tensor<1x384x128xf32>
    %1515 = stablehlo.reshape %1514 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1516 = stablehlo.dot %1515, %608 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1517 = stablehlo.broadcast_in_dim %607, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1518 = stablehlo.add %1516, %1517 : tensor<384x512xf32>
    %1519 = stablehlo.reshape %1518 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1520 = stablehlo.maximum %1519, %9 : tensor<1x384x512xf32>
    %1521 = stablehlo.reshape %1520 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1522 = stablehlo.dot %1521, %616 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1523 = stablehlo.broadcast_in_dim %615, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1524 = stablehlo.add %1522, %1523 : tensor<384x128xf32>
    %1525 = stablehlo.reshape %1524 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1526 = stablehlo.add %1525, %1514 : tensor<1x384x128xf32>
    %1527 = stablehlo.broadcast_in_dim %610, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1528 = stablehlo.multiply %1526, %1527 : tensor<1x384x128xf32>
    %1529 = stablehlo.broadcast_in_dim %609, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1530 = stablehlo.add %1528, %1529 : tensor<1x384x128xf32>
    %1531 = stablehlo.reshape %1530 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1532 = stablehlo.dot %1531, %614 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1533 = stablehlo.broadcast_in_dim %613, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1534 = stablehlo.add %1532, %1533 : tensor<384x512xf32>
    %1535 = stablehlo.reshape %1534 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1536 = stablehlo.add %1535, %1410 : tensor<1x384x512xf32>
    %1537 = stablehlo.broadcast_in_dim %612, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1538 = stablehlo.multiply %1536, %1537 : tensor<1x384x512xf32>
    %1539 = stablehlo.broadcast_in_dim %611, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1540 = stablehlo.add %1538, %1539 : tensor<1x384x512xf32>
    %1541 = stablehlo.reshape %1540 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1542 = stablehlo.dot %1541, %810 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1543 = stablehlo.broadcast_in_dim %809, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1544 = stablehlo.add %1542, %1543 : tensor<384x128xf32>
    %1545 = stablehlo.reshape %1544 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1546 = stablehlo.transpose %1545, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1547 = stablehlo.dot %1541, %814 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1548 = stablehlo.reshape %1547 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1549 = stablehlo.broadcast_in_dim %813, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1550 = stablehlo.add %1548, %1549 : tensor<1x384x128xf32>
    %1551 = stablehlo.broadcast_in_dim %812, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1552 = stablehlo.multiply %1550, %1551 : tensor<1x384x128xf32>
    %1553 = stablehlo.broadcast_in_dim %811, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1554 = stablehlo.add %1552, %1553 : tensor<1x384x128xf32>
    %1555 = stablehlo.reshape %1554 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1556 = stablehlo.dot %1555, %806 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1557 = stablehlo.broadcast_in_dim %805, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1558 = stablehlo.add %1556, %1557 : tensor<384x128xf32>
    %1559 = stablehlo.reshape %1558 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1560 = stablehlo.transpose %1559, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1561 = stablehlo.dot %1555, %808 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1562 = stablehlo.broadcast_in_dim %807, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1563 = stablehlo.add %1561, %1562 : tensor<384x128xf32>
    %1564 = stablehlo.reshape %1563 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1565 = stablehlo.transpose %1564, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1566 = stablehlo.dot_general %1565, %1560, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %1567 = stablehlo.multiply %1566, %4 : tensor<1x4x384x384xf32>
    %1568 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %1569 = stablehlo.add %1567, %1568 : tensor<1x4x384x384xf32>
    %1570 = stablehlo.reduce(%1569 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1571 = stablehlo.broadcast_in_dim %1570, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1572 = stablehlo.subtract %1569, %1571 : tensor<1x4x384x384xf32>
    %1573 = stablehlo.exponential %1572 : tensor<1x4x384x384xf32>
    %1574 = stablehlo.reduce(%1573 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1575 = stablehlo.broadcast_in_dim %1574, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1576 = stablehlo.divide %1573, %1575 : tensor<1x4x384x384xf32>
    %1577 = stablehlo.dot_general %1576, %1546, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %1578 = stablehlo.transpose %1577, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %1579 = stablehlo.reshape %1578 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %1580 = stablehlo.dot %1579, %804 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1581 = stablehlo.broadcast_in_dim %803, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1582 = stablehlo.add %1580, %1581 : tensor<384x128xf32>
    %1583 = stablehlo.reshape %1582 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1584 = stablehlo.dot %1541, %818 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1585 = stablehlo.broadcast_in_dim %817, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1586 = stablehlo.add %1584, %1585 : tensor<384x128xf32>
    %1587 = stablehlo.reshape %1586 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1588 = stablehlo.broadcast_in_dim %816, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1589 = stablehlo.multiply %1587, %1588 : tensor<1x384x128xf32>
    %1590 = stablehlo.broadcast_in_dim %815, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1591 = stablehlo.add %1589, %1590 : tensor<1x384x128xf32>
    %1592 = stablehlo.add %1583, %1591 : tensor<1x384x128xf32>
    %1593 = stablehlo.broadcast_in_dim %802, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1594 = stablehlo.multiply %1592, %1593 : tensor<1x384x128xf32>
    %1595 = stablehlo.broadcast_in_dim %801, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1596 = stablehlo.add %1594, %1595 : tensor<1x384x128xf32>
    %1597 = stablehlo.reshape %1596 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1598 = stablehlo.dot %1597, %820 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1599 = stablehlo.broadcast_in_dim %819, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1600 = stablehlo.add %1598, %1599 : tensor<384x512xf32>
    %1601 = stablehlo.reshape %1600 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1602 = stablehlo.maximum %1601, %9 : tensor<1x384x512xf32>
    %1603 = stablehlo.reshape %1602 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1604 = stablehlo.dot %1603, %824 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1605 = stablehlo.broadcast_in_dim %823, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1606 = stablehlo.add %1604, %1605 : tensor<384x128xf32>
    %1607 = stablehlo.reshape %1606 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1608 = stablehlo.add %1607, %1596 : tensor<1x384x128xf32>
    %1609 = stablehlo.broadcast_in_dim %822, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1610 = stablehlo.multiply %1608, %1609 : tensor<1x384x128xf32>
    %1611 = stablehlo.broadcast_in_dim %821, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1612 = stablehlo.add %1610, %1611 : tensor<1x384x128xf32>
    %1613 = stablehlo.reshape %1612 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1614 = stablehlo.dot %1613, %826 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1615 = stablehlo.broadcast_in_dim %825, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1616 = stablehlo.add %1614, %1615 : tensor<384x512xf32>
    %1617 = stablehlo.reshape %1616 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1618 = stablehlo.maximum %1617, %9 : tensor<1x384x512xf32>
    %1619 = stablehlo.reshape %1618 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1620 = stablehlo.dot %1619, %830 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1621 = stablehlo.broadcast_in_dim %829, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1622 = stablehlo.add %1620, %1621 : tensor<384x128xf32>
    %1623 = stablehlo.reshape %1622 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1624 = stablehlo.add %1623, %1612 : tensor<1x384x128xf32>
    %1625 = stablehlo.broadcast_in_dim %828, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1626 = stablehlo.multiply %1624, %1625 : tensor<1x384x128xf32>
    %1627 = stablehlo.broadcast_in_dim %827, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1628 = stablehlo.add %1626, %1627 : tensor<1x384x128xf32>
    %1629 = stablehlo.reshape %1628 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1630 = stablehlo.dot %1629, %832 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1631 = stablehlo.broadcast_in_dim %831, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1632 = stablehlo.add %1630, %1631 : tensor<384x512xf32>
    %1633 = stablehlo.reshape %1632 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1634 = stablehlo.maximum %1633, %9 : tensor<1x384x512xf32>
    %1635 = stablehlo.reshape %1634 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1636 = stablehlo.dot %1635, %836 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1637 = stablehlo.broadcast_in_dim %835, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1638 = stablehlo.add %1636, %1637 : tensor<384x128xf32>
    %1639 = stablehlo.reshape %1638 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1640 = stablehlo.add %1639, %1628 : tensor<1x384x128xf32>
    %1641 = stablehlo.broadcast_in_dim %834, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1642 = stablehlo.multiply %1640, %1641 : tensor<1x384x128xf32>
    %1643 = stablehlo.broadcast_in_dim %833, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1644 = stablehlo.add %1642, %1643 : tensor<1x384x128xf32>
    %1645 = stablehlo.reshape %1644 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1646 = stablehlo.dot %1645, %838 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1647 = stablehlo.broadcast_in_dim %837, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1648 = stablehlo.add %1646, %1647 : tensor<384x512xf32>
    %1649 = stablehlo.reshape %1648 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1650 = stablehlo.maximum %1649, %9 : tensor<1x384x512xf32>
    %1651 = stablehlo.reshape %1650 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1652 = stablehlo.dot %1651, %846 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1653 = stablehlo.broadcast_in_dim %845, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1654 = stablehlo.add %1652, %1653 : tensor<384x128xf32>
    %1655 = stablehlo.reshape %1654 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1656 = stablehlo.add %1655, %1644 : tensor<1x384x128xf32>
    %1657 = stablehlo.broadcast_in_dim %840, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1658 = stablehlo.multiply %1656, %1657 : tensor<1x384x128xf32>
    %1659 = stablehlo.broadcast_in_dim %839, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1660 = stablehlo.add %1658, %1659 : tensor<1x384x128xf32>
    %1661 = stablehlo.reshape %1660 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1662 = stablehlo.dot %1661, %844 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1663 = stablehlo.broadcast_in_dim %843, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1664 = stablehlo.add %1662, %1663 : tensor<384x512xf32>
    %1665 = stablehlo.reshape %1664 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1666 = stablehlo.add %1665, %1540 : tensor<1x384x512xf32>
    %1667 = stablehlo.broadcast_in_dim %842, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1668 = stablehlo.multiply %1666, %1667 : tensor<1x384x512xf32>
    %1669 = stablehlo.broadcast_in_dim %841, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1670 = stablehlo.add %1668, %1669 : tensor<1x384x512xf32>
    %1671 = stablehlo.reshape %1670 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1672 = stablehlo.dot %1671, %856 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1673 = stablehlo.broadcast_in_dim %855, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1674 = stablehlo.add %1672, %1673 : tensor<384x128xf32>
    %1675 = stablehlo.reshape %1674 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1676 = stablehlo.transpose %1675, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1677 = stablehlo.dot %1671, %860 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1678 = stablehlo.reshape %1677 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1679 = stablehlo.broadcast_in_dim %859, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1680 = stablehlo.add %1678, %1679 : tensor<1x384x128xf32>
    %1681 = stablehlo.broadcast_in_dim %858, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1682 = stablehlo.multiply %1680, %1681 : tensor<1x384x128xf32>
    %1683 = stablehlo.broadcast_in_dim %857, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1684 = stablehlo.add %1682, %1683 : tensor<1x384x128xf32>
    %1685 = stablehlo.reshape %1684 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1686 = stablehlo.dot %1685, %852 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1687 = stablehlo.broadcast_in_dim %851, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1688 = stablehlo.add %1686, %1687 : tensor<384x128xf32>
    %1689 = stablehlo.reshape %1688 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1690 = stablehlo.transpose %1689, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1691 = stablehlo.dot %1685, %854 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1692 = stablehlo.broadcast_in_dim %853, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1693 = stablehlo.add %1691, %1692 : tensor<384x128xf32>
    %1694 = stablehlo.reshape %1693 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1695 = stablehlo.transpose %1694, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1696 = stablehlo.dot_general %1695, %1690, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %1697 = stablehlo.multiply %1696, %4 : tensor<1x4x384x384xf32>
    %1698 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %1699 = stablehlo.add %1697, %1698 : tensor<1x4x384x384xf32>
    %1700 = stablehlo.reduce(%1699 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1701 = stablehlo.broadcast_in_dim %1700, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1702 = stablehlo.subtract %1699, %1701 : tensor<1x4x384x384xf32>
    %1703 = stablehlo.exponential %1702 : tensor<1x4x384x384xf32>
    %1704 = stablehlo.reduce(%1703 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1705 = stablehlo.broadcast_in_dim %1704, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1706 = stablehlo.divide %1703, %1705 : tensor<1x4x384x384xf32>
    %1707 = stablehlo.dot_general %1706, %1676, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %1708 = stablehlo.transpose %1707, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %1709 = stablehlo.reshape %1708 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %1710 = stablehlo.dot %1709, %850 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1711 = stablehlo.broadcast_in_dim %849, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1712 = stablehlo.add %1710, %1711 : tensor<384x128xf32>
    %1713 = stablehlo.reshape %1712 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1714 = stablehlo.dot %1671, %864 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1715 = stablehlo.broadcast_in_dim %863, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1716 = stablehlo.add %1714, %1715 : tensor<384x128xf32>
    %1717 = stablehlo.reshape %1716 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1718 = stablehlo.broadcast_in_dim %862, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1719 = stablehlo.multiply %1717, %1718 : tensor<1x384x128xf32>
    %1720 = stablehlo.broadcast_in_dim %861, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1721 = stablehlo.add %1719, %1720 : tensor<1x384x128xf32>
    %1722 = stablehlo.add %1713, %1721 : tensor<1x384x128xf32>
    %1723 = stablehlo.broadcast_in_dim %848, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1724 = stablehlo.multiply %1722, %1723 : tensor<1x384x128xf32>
    %1725 = stablehlo.broadcast_in_dim %847, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1726 = stablehlo.add %1724, %1725 : tensor<1x384x128xf32>
    %1727 = stablehlo.reshape %1726 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1728 = stablehlo.dot %1727, %866 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1729 = stablehlo.broadcast_in_dim %865, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1730 = stablehlo.add %1728, %1729 : tensor<384x512xf32>
    %1731 = stablehlo.reshape %1730 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1732 = stablehlo.maximum %1731, %9 : tensor<1x384x512xf32>
    %1733 = stablehlo.reshape %1732 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1734 = stablehlo.dot %1733, %870 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1735 = stablehlo.broadcast_in_dim %869, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1736 = stablehlo.add %1734, %1735 : tensor<384x128xf32>
    %1737 = stablehlo.reshape %1736 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1738 = stablehlo.add %1737, %1726 : tensor<1x384x128xf32>
    %1739 = stablehlo.broadcast_in_dim %868, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1740 = stablehlo.multiply %1738, %1739 : tensor<1x384x128xf32>
    %1741 = stablehlo.broadcast_in_dim %867, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1742 = stablehlo.add %1740, %1741 : tensor<1x384x128xf32>
    %1743 = stablehlo.reshape %1742 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1744 = stablehlo.dot %1743, %872 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1745 = stablehlo.broadcast_in_dim %871, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1746 = stablehlo.add %1744, %1745 : tensor<384x512xf32>
    %1747 = stablehlo.reshape %1746 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1748 = stablehlo.maximum %1747, %9 : tensor<1x384x512xf32>
    %1749 = stablehlo.reshape %1748 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1750 = stablehlo.dot %1749, %876 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1751 = stablehlo.broadcast_in_dim %875, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1752 = stablehlo.add %1750, %1751 : tensor<384x128xf32>
    %1753 = stablehlo.reshape %1752 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1754 = stablehlo.add %1753, %1742 : tensor<1x384x128xf32>
    %1755 = stablehlo.broadcast_in_dim %874, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1756 = stablehlo.multiply %1754, %1755 : tensor<1x384x128xf32>
    %1757 = stablehlo.broadcast_in_dim %873, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1758 = stablehlo.add %1756, %1757 : tensor<1x384x128xf32>
    %1759 = stablehlo.reshape %1758 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1760 = stablehlo.dot %1759, %878 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1761 = stablehlo.broadcast_in_dim %877, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1762 = stablehlo.add %1760, %1761 : tensor<384x512xf32>
    %1763 = stablehlo.reshape %1762 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1764 = stablehlo.maximum %1763, %9 : tensor<1x384x512xf32>
    %1765 = stablehlo.reshape %1764 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1766 = stablehlo.dot %1765, %882 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1767 = stablehlo.broadcast_in_dim %881, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1768 = stablehlo.add %1766, %1767 : tensor<384x128xf32>
    %1769 = stablehlo.reshape %1768 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1770 = stablehlo.add %1769, %1758 : tensor<1x384x128xf32>
    %1771 = stablehlo.broadcast_in_dim %880, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1772 = stablehlo.multiply %1770, %1771 : tensor<1x384x128xf32>
    %1773 = stablehlo.broadcast_in_dim %879, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1774 = stablehlo.add %1772, %1773 : tensor<1x384x128xf32>
    %1775 = stablehlo.reshape %1774 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1776 = stablehlo.dot %1775, %884 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1777 = stablehlo.broadcast_in_dim %883, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1778 = stablehlo.add %1776, %1777 : tensor<384x512xf32>
    %1779 = stablehlo.reshape %1778 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1780 = stablehlo.maximum %1779, %9 : tensor<1x384x512xf32>
    %1781 = stablehlo.reshape %1780 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1782 = stablehlo.dot %1781, %892 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1783 = stablehlo.broadcast_in_dim %891, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1784 = stablehlo.add %1782, %1783 : tensor<384x128xf32>
    %1785 = stablehlo.reshape %1784 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1786 = stablehlo.add %1785, %1774 : tensor<1x384x128xf32>
    %1787 = stablehlo.broadcast_in_dim %886, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1788 = stablehlo.multiply %1786, %1787 : tensor<1x384x128xf32>
    %1789 = stablehlo.broadcast_in_dim %885, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1790 = stablehlo.add %1788, %1789 : tensor<1x384x128xf32>
    %1791 = stablehlo.reshape %1790 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1792 = stablehlo.dot %1791, %890 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1793 = stablehlo.broadcast_in_dim %889, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1794 = stablehlo.add %1792, %1793 : tensor<384x512xf32>
    %1795 = stablehlo.reshape %1794 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1796 = stablehlo.add %1795, %1670 : tensor<1x384x512xf32>
    %1797 = stablehlo.broadcast_in_dim %888, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1798 = stablehlo.multiply %1796, %1797 : tensor<1x384x512xf32>
    %1799 = stablehlo.broadcast_in_dim %887, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1800 = stablehlo.add %1798, %1799 : tensor<1x384x512xf32>
    %1801 = stablehlo.reshape %1800 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1802 = stablehlo.dot %1801, %902 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1803 = stablehlo.broadcast_in_dim %901, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1804 = stablehlo.add %1802, %1803 : tensor<384x128xf32>
    %1805 = stablehlo.reshape %1804 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1806 = stablehlo.transpose %1805, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1807 = stablehlo.dot %1801, %906 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1808 = stablehlo.reshape %1807 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1809 = stablehlo.broadcast_in_dim %905, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1810 = stablehlo.add %1808, %1809 : tensor<1x384x128xf32>
    %1811 = stablehlo.broadcast_in_dim %904, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1812 = stablehlo.multiply %1810, %1811 : tensor<1x384x128xf32>
    %1813 = stablehlo.broadcast_in_dim %903, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1814 = stablehlo.add %1812, %1813 : tensor<1x384x128xf32>
    %1815 = stablehlo.reshape %1814 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1816 = stablehlo.dot %1815, %898 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1817 = stablehlo.broadcast_in_dim %897, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1818 = stablehlo.add %1816, %1817 : tensor<384x128xf32>
    %1819 = stablehlo.reshape %1818 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1820 = stablehlo.transpose %1819, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1821 = stablehlo.dot %1815, %900 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1822 = stablehlo.broadcast_in_dim %899, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1823 = stablehlo.add %1821, %1822 : tensor<384x128xf32>
    %1824 = stablehlo.reshape %1823 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1825 = stablehlo.transpose %1824, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1826 = stablehlo.dot_general %1825, %1820, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %1827 = stablehlo.multiply %1826, %4 : tensor<1x4x384x384xf32>
    %1828 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %1829 = stablehlo.add %1827, %1828 : tensor<1x4x384x384xf32>
    %1830 = stablehlo.reduce(%1829 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1831 = stablehlo.broadcast_in_dim %1830, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1832 = stablehlo.subtract %1829, %1831 : tensor<1x4x384x384xf32>
    %1833 = stablehlo.exponential %1832 : tensor<1x4x384x384xf32>
    %1834 = stablehlo.reduce(%1833 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1835 = stablehlo.broadcast_in_dim %1834, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1836 = stablehlo.divide %1833, %1835 : tensor<1x4x384x384xf32>
    %1837 = stablehlo.dot_general %1836, %1806, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %1838 = stablehlo.transpose %1837, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %1839 = stablehlo.reshape %1838 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %1840 = stablehlo.dot %1839, %896 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1841 = stablehlo.broadcast_in_dim %895, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1842 = stablehlo.add %1840, %1841 : tensor<384x128xf32>
    %1843 = stablehlo.reshape %1842 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1844 = stablehlo.dot %1801, %910 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1845 = stablehlo.broadcast_in_dim %909, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1846 = stablehlo.add %1844, %1845 : tensor<384x128xf32>
    %1847 = stablehlo.reshape %1846 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1848 = stablehlo.broadcast_in_dim %908, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1849 = stablehlo.multiply %1847, %1848 : tensor<1x384x128xf32>
    %1850 = stablehlo.broadcast_in_dim %907, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1851 = stablehlo.add %1849, %1850 : tensor<1x384x128xf32>
    %1852 = stablehlo.add %1843, %1851 : tensor<1x384x128xf32>
    %1853 = stablehlo.broadcast_in_dim %894, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1854 = stablehlo.multiply %1852, %1853 : tensor<1x384x128xf32>
    %1855 = stablehlo.broadcast_in_dim %893, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1856 = stablehlo.add %1854, %1855 : tensor<1x384x128xf32>
    %1857 = stablehlo.reshape %1856 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1858 = stablehlo.dot %1857, %912 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1859 = stablehlo.broadcast_in_dim %911, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1860 = stablehlo.add %1858, %1859 : tensor<384x512xf32>
    %1861 = stablehlo.reshape %1860 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1862 = stablehlo.maximum %1861, %9 : tensor<1x384x512xf32>
    %1863 = stablehlo.reshape %1862 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1864 = stablehlo.dot %1863, %916 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1865 = stablehlo.broadcast_in_dim %915, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1866 = stablehlo.add %1864, %1865 : tensor<384x128xf32>
    %1867 = stablehlo.reshape %1866 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1868 = stablehlo.add %1867, %1856 : tensor<1x384x128xf32>
    %1869 = stablehlo.broadcast_in_dim %914, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1870 = stablehlo.multiply %1868, %1869 : tensor<1x384x128xf32>
    %1871 = stablehlo.broadcast_in_dim %913, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1872 = stablehlo.add %1870, %1871 : tensor<1x384x128xf32>
    %1873 = stablehlo.reshape %1872 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1874 = stablehlo.dot %1873, %918 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1875 = stablehlo.broadcast_in_dim %917, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1876 = stablehlo.add %1874, %1875 : tensor<384x512xf32>
    %1877 = stablehlo.reshape %1876 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1878 = stablehlo.maximum %1877, %9 : tensor<1x384x512xf32>
    %1879 = stablehlo.reshape %1878 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1880 = stablehlo.dot %1879, %922 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1881 = stablehlo.broadcast_in_dim %921, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1882 = stablehlo.add %1880, %1881 : tensor<384x128xf32>
    %1883 = stablehlo.reshape %1882 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1884 = stablehlo.add %1883, %1872 : tensor<1x384x128xf32>
    %1885 = stablehlo.broadcast_in_dim %920, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1886 = stablehlo.multiply %1884, %1885 : tensor<1x384x128xf32>
    %1887 = stablehlo.broadcast_in_dim %919, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1888 = stablehlo.add %1886, %1887 : tensor<1x384x128xf32>
    %1889 = stablehlo.reshape %1888 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1890 = stablehlo.dot %1889, %924 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1891 = stablehlo.broadcast_in_dim %923, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1892 = stablehlo.add %1890, %1891 : tensor<384x512xf32>
    %1893 = stablehlo.reshape %1892 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1894 = stablehlo.maximum %1893, %9 : tensor<1x384x512xf32>
    %1895 = stablehlo.reshape %1894 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1896 = stablehlo.dot %1895, %928 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1897 = stablehlo.broadcast_in_dim %927, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1898 = stablehlo.add %1896, %1897 : tensor<384x128xf32>
    %1899 = stablehlo.reshape %1898 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1900 = stablehlo.add %1899, %1888 : tensor<1x384x128xf32>
    %1901 = stablehlo.broadcast_in_dim %926, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1902 = stablehlo.multiply %1900, %1901 : tensor<1x384x128xf32>
    %1903 = stablehlo.broadcast_in_dim %925, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1904 = stablehlo.add %1902, %1903 : tensor<1x384x128xf32>
    %1905 = stablehlo.reshape %1904 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1906 = stablehlo.dot %1905, %930 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1907 = stablehlo.broadcast_in_dim %929, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1908 = stablehlo.add %1906, %1907 : tensor<384x512xf32>
    %1909 = stablehlo.reshape %1908 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1910 = stablehlo.maximum %1909, %9 : tensor<1x384x512xf32>
    %1911 = stablehlo.reshape %1910 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1912 = stablehlo.dot %1911, %938 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1913 = stablehlo.broadcast_in_dim %937, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1914 = stablehlo.add %1912, %1913 : tensor<384x128xf32>
    %1915 = stablehlo.reshape %1914 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1916 = stablehlo.add %1915, %1904 : tensor<1x384x128xf32>
    %1917 = stablehlo.broadcast_in_dim %932, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1918 = stablehlo.multiply %1916, %1917 : tensor<1x384x128xf32>
    %1919 = stablehlo.broadcast_in_dim %931, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1920 = stablehlo.add %1918, %1919 : tensor<1x384x128xf32>
    %1921 = stablehlo.reshape %1920 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1922 = stablehlo.dot %1921, %936 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1923 = stablehlo.broadcast_in_dim %935, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1924 = stablehlo.add %1922, %1923 : tensor<384x512xf32>
    %1925 = stablehlo.reshape %1924 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1926 = stablehlo.add %1925, %1800 : tensor<1x384x512xf32>
    %1927 = stablehlo.broadcast_in_dim %934, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1928 = stablehlo.multiply %1926, %1927 : tensor<1x384x512xf32>
    %1929 = stablehlo.broadcast_in_dim %933, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %1930 = stablehlo.add %1928, %1929 : tensor<1x384x512xf32>
    %1931 = stablehlo.reshape %1930 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1932 = stablehlo.dot %1931, %948 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1933 = stablehlo.broadcast_in_dim %947, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1934 = stablehlo.add %1932, %1933 : tensor<384x128xf32>
    %1935 = stablehlo.reshape %1934 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1936 = stablehlo.transpose %1935, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1937 = stablehlo.dot %1931, %952 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1938 = stablehlo.reshape %1937 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1939 = stablehlo.broadcast_in_dim %951, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1940 = stablehlo.add %1938, %1939 : tensor<1x384x128xf32>
    %1941 = stablehlo.broadcast_in_dim %950, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1942 = stablehlo.multiply %1940, %1941 : tensor<1x384x128xf32>
    %1943 = stablehlo.broadcast_in_dim %949, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1944 = stablehlo.add %1942, %1943 : tensor<1x384x128xf32>
    %1945 = stablehlo.reshape %1944 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1946 = stablehlo.dot %1945, %944 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1947 = stablehlo.broadcast_in_dim %943, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1948 = stablehlo.add %1946, %1947 : tensor<384x128xf32>
    %1949 = stablehlo.reshape %1948 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1950 = stablehlo.transpose %1949, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1951 = stablehlo.dot %1945, %946 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1952 = stablehlo.broadcast_in_dim %945, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1953 = stablehlo.add %1951, %1952 : tensor<384x128xf32>
    %1954 = stablehlo.reshape %1953 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %1955 = stablehlo.transpose %1954, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %1956 = stablehlo.dot_general %1955, %1950, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %1957 = stablehlo.multiply %1956, %4 : tensor<1x4x384x384xf32>
    %1958 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %1959 = stablehlo.add %1957, %1958 : tensor<1x4x384x384xf32>
    %1960 = stablehlo.reduce(%1959 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1961 = stablehlo.broadcast_in_dim %1960, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1962 = stablehlo.subtract %1959, %1961 : tensor<1x4x384x384xf32>
    %1963 = stablehlo.exponential %1962 : tensor<1x4x384x384xf32>
    %1964 = stablehlo.reduce(%1963 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %1965 = stablehlo.broadcast_in_dim %1964, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %1966 = stablehlo.divide %1963, %1965 : tensor<1x4x384x384xf32>
    %1967 = stablehlo.dot_general %1966, %1936, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %1968 = stablehlo.transpose %1967, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %1969 = stablehlo.reshape %1968 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %1970 = stablehlo.dot %1969, %942 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %1971 = stablehlo.broadcast_in_dim %941, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1972 = stablehlo.add %1970, %1971 : tensor<384x128xf32>
    %1973 = stablehlo.reshape %1972 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1974 = stablehlo.dot %1931, %956 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1975 = stablehlo.broadcast_in_dim %955, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1976 = stablehlo.add %1974, %1975 : tensor<384x128xf32>
    %1977 = stablehlo.reshape %1976 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1978 = stablehlo.broadcast_in_dim %954, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1979 = stablehlo.multiply %1977, %1978 : tensor<1x384x128xf32>
    %1980 = stablehlo.broadcast_in_dim %953, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1981 = stablehlo.add %1979, %1980 : tensor<1x384x128xf32>
    %1982 = stablehlo.add %1973, %1981 : tensor<1x384x128xf32>
    %1983 = stablehlo.broadcast_in_dim %940, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1984 = stablehlo.multiply %1982, %1983 : tensor<1x384x128xf32>
    %1985 = stablehlo.broadcast_in_dim %939, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %1986 = stablehlo.add %1984, %1985 : tensor<1x384x128xf32>
    %1987 = stablehlo.reshape %1986 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %1988 = stablehlo.dot %1987, %958 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %1989 = stablehlo.broadcast_in_dim %957, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %1990 = stablehlo.add %1988, %1989 : tensor<384x512xf32>
    %1991 = stablehlo.reshape %1990 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1992 = stablehlo.maximum %1991, %9 : tensor<1x384x512xf32>
    %1993 = stablehlo.reshape %1992 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %1994 = stablehlo.dot %1993, %962 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %1995 = stablehlo.broadcast_in_dim %961, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %1996 = stablehlo.add %1994, %1995 : tensor<384x128xf32>
    %1997 = stablehlo.reshape %1996 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %1998 = stablehlo.add %1997, %1986 : tensor<1x384x128xf32>
    %1999 = stablehlo.broadcast_in_dim %960, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2000 = stablehlo.multiply %1998, %1999 : tensor<1x384x128xf32>
    %2001 = stablehlo.broadcast_in_dim %959, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2002 = stablehlo.add %2000, %2001 : tensor<1x384x128xf32>
    %2003 = stablehlo.reshape %2002 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2004 = stablehlo.dot %2003, %964 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2005 = stablehlo.broadcast_in_dim %963, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2006 = stablehlo.add %2004, %2005 : tensor<384x512xf32>
    %2007 = stablehlo.reshape %2006 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2008 = stablehlo.maximum %2007, %9 : tensor<1x384x512xf32>
    %2009 = stablehlo.reshape %2008 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2010 = stablehlo.dot %2009, %968 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2011 = stablehlo.broadcast_in_dim %967, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2012 = stablehlo.add %2010, %2011 : tensor<384x128xf32>
    %2013 = stablehlo.reshape %2012 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2014 = stablehlo.add %2013, %2002 : tensor<1x384x128xf32>
    %2015 = stablehlo.broadcast_in_dim %966, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2016 = stablehlo.multiply %2014, %2015 : tensor<1x384x128xf32>
    %2017 = stablehlo.broadcast_in_dim %965, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2018 = stablehlo.add %2016, %2017 : tensor<1x384x128xf32>
    %2019 = stablehlo.reshape %2018 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2020 = stablehlo.dot %2019, %970 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2021 = stablehlo.broadcast_in_dim %969, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2022 = stablehlo.add %2020, %2021 : tensor<384x512xf32>
    %2023 = stablehlo.reshape %2022 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2024 = stablehlo.maximum %2023, %9 : tensor<1x384x512xf32>
    %2025 = stablehlo.reshape %2024 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2026 = stablehlo.dot %2025, %974 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2027 = stablehlo.broadcast_in_dim %973, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2028 = stablehlo.add %2026, %2027 : tensor<384x128xf32>
    %2029 = stablehlo.reshape %2028 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2030 = stablehlo.add %2029, %2018 : tensor<1x384x128xf32>
    %2031 = stablehlo.broadcast_in_dim %972, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2032 = stablehlo.multiply %2030, %2031 : tensor<1x384x128xf32>
    %2033 = stablehlo.broadcast_in_dim %971, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2034 = stablehlo.add %2032, %2033 : tensor<1x384x128xf32>
    %2035 = stablehlo.reshape %2034 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2036 = stablehlo.dot %2035, %976 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2037 = stablehlo.broadcast_in_dim %975, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2038 = stablehlo.add %2036, %2037 : tensor<384x512xf32>
    %2039 = stablehlo.reshape %2038 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2040 = stablehlo.maximum %2039, %9 : tensor<1x384x512xf32>
    %2041 = stablehlo.reshape %2040 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2042 = stablehlo.dot %2041, %984 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2043 = stablehlo.broadcast_in_dim %983, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2044 = stablehlo.add %2042, %2043 : tensor<384x128xf32>
    %2045 = stablehlo.reshape %2044 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2046 = stablehlo.add %2045, %2034 : tensor<1x384x128xf32>
    %2047 = stablehlo.broadcast_in_dim %978, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2048 = stablehlo.multiply %2046, %2047 : tensor<1x384x128xf32>
    %2049 = stablehlo.broadcast_in_dim %977, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2050 = stablehlo.add %2048, %2049 : tensor<1x384x128xf32>
    %2051 = stablehlo.reshape %2050 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2052 = stablehlo.dot %2051, %982 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2053 = stablehlo.broadcast_in_dim %981, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2054 = stablehlo.add %2052, %2053 : tensor<384x512xf32>
    %2055 = stablehlo.reshape %2054 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2056 = stablehlo.add %2055, %1930 : tensor<1x384x512xf32>
    %2057 = stablehlo.broadcast_in_dim %980, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2058 = stablehlo.multiply %2056, %2057 : tensor<1x384x512xf32>
    %2059 = stablehlo.broadcast_in_dim %979, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2060 = stablehlo.add %2058, %2059 : tensor<1x384x512xf32>
    %2061 = stablehlo.reshape %2060 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2062 = stablehlo.dot %2061, %994 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2063 = stablehlo.broadcast_in_dim %993, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2064 = stablehlo.add %2062, %2063 : tensor<384x128xf32>
    %2065 = stablehlo.reshape %2064 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2066 = stablehlo.transpose %2065, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2067 = stablehlo.dot %2061, %998 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2068 = stablehlo.reshape %2067 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2069 = stablehlo.broadcast_in_dim %997, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2070 = stablehlo.add %2068, %2069 : tensor<1x384x128xf32>
    %2071 = stablehlo.broadcast_in_dim %996, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2072 = stablehlo.multiply %2070, %2071 : tensor<1x384x128xf32>
    %2073 = stablehlo.broadcast_in_dim %995, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2074 = stablehlo.add %2072, %2073 : tensor<1x384x128xf32>
    %2075 = stablehlo.reshape %2074 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2076 = stablehlo.dot %2075, %990 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2077 = stablehlo.broadcast_in_dim %989, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2078 = stablehlo.add %2076, %2077 : tensor<384x128xf32>
    %2079 = stablehlo.reshape %2078 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2080 = stablehlo.transpose %2079, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2081 = stablehlo.dot %2075, %992 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2082 = stablehlo.broadcast_in_dim %991, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2083 = stablehlo.add %2081, %2082 : tensor<384x128xf32>
    %2084 = stablehlo.reshape %2083 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2085 = stablehlo.transpose %2084, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2086 = stablehlo.dot_general %2085, %2080, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2087 = stablehlo.multiply %2086, %4 : tensor<1x4x384x384xf32>
    %2088 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2089 = stablehlo.add %2087, %2088 : tensor<1x4x384x384xf32>
    %2090 = stablehlo.reduce(%2089 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2091 = stablehlo.broadcast_in_dim %2090, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2092 = stablehlo.subtract %2089, %2091 : tensor<1x4x384x384xf32>
    %2093 = stablehlo.exponential %2092 : tensor<1x4x384x384xf32>
    %2094 = stablehlo.reduce(%2093 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2095 = stablehlo.broadcast_in_dim %2094, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2096 = stablehlo.divide %2093, %2095 : tensor<1x4x384x384xf32>
    %2097 = stablehlo.dot_general %2096, %2066, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2098 = stablehlo.transpose %2097, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2099 = stablehlo.reshape %2098 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2100 = stablehlo.dot %2099, %988 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2101 = stablehlo.broadcast_in_dim %987, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2102 = stablehlo.add %2100, %2101 : tensor<384x128xf32>
    %2103 = stablehlo.reshape %2102 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2104 = stablehlo.dot %2061, %1002 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2105 = stablehlo.broadcast_in_dim %1001, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2106 = stablehlo.add %2104, %2105 : tensor<384x128xf32>
    %2107 = stablehlo.reshape %2106 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2108 = stablehlo.broadcast_in_dim %1000, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2109 = stablehlo.multiply %2107, %2108 : tensor<1x384x128xf32>
    %2110 = stablehlo.broadcast_in_dim %999, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2111 = stablehlo.add %2109, %2110 : tensor<1x384x128xf32>
    %2112 = stablehlo.add %2103, %2111 : tensor<1x384x128xf32>
    %2113 = stablehlo.broadcast_in_dim %986, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2114 = stablehlo.multiply %2112, %2113 : tensor<1x384x128xf32>
    %2115 = stablehlo.broadcast_in_dim %985, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2116 = stablehlo.add %2114, %2115 : tensor<1x384x128xf32>
    %2117 = stablehlo.reshape %2116 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2118 = stablehlo.dot %2117, %1004 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2119 = stablehlo.broadcast_in_dim %1003, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2120 = stablehlo.add %2118, %2119 : tensor<384x512xf32>
    %2121 = stablehlo.reshape %2120 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2122 = stablehlo.maximum %2121, %9 : tensor<1x384x512xf32>
    %2123 = stablehlo.reshape %2122 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2124 = stablehlo.dot %2123, %1008 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2125 = stablehlo.broadcast_in_dim %1007, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2126 = stablehlo.add %2124, %2125 : tensor<384x128xf32>
    %2127 = stablehlo.reshape %2126 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2128 = stablehlo.add %2127, %2116 : tensor<1x384x128xf32>
    %2129 = stablehlo.broadcast_in_dim %1006, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2130 = stablehlo.multiply %2128, %2129 : tensor<1x384x128xf32>
    %2131 = stablehlo.broadcast_in_dim %1005, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2132 = stablehlo.add %2130, %2131 : tensor<1x384x128xf32>
    %2133 = stablehlo.reshape %2132 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2134 = stablehlo.dot %2133, %1010 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2135 = stablehlo.broadcast_in_dim %1009, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2136 = stablehlo.add %2134, %2135 : tensor<384x512xf32>
    %2137 = stablehlo.reshape %2136 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2138 = stablehlo.maximum %2137, %9 : tensor<1x384x512xf32>
    %2139 = stablehlo.reshape %2138 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2140 = stablehlo.dot %2139, %1014 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2141 = stablehlo.broadcast_in_dim %1013, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2142 = stablehlo.add %2140, %2141 : tensor<384x128xf32>
    %2143 = stablehlo.reshape %2142 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2144 = stablehlo.add %2143, %2132 : tensor<1x384x128xf32>
    %2145 = stablehlo.broadcast_in_dim %1012, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2146 = stablehlo.multiply %2144, %2145 : tensor<1x384x128xf32>
    %2147 = stablehlo.broadcast_in_dim %1011, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2148 = stablehlo.add %2146, %2147 : tensor<1x384x128xf32>
    %2149 = stablehlo.reshape %2148 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2150 = stablehlo.dot %2149, %1016 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2151 = stablehlo.broadcast_in_dim %1015, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2152 = stablehlo.add %2150, %2151 : tensor<384x512xf32>
    %2153 = stablehlo.reshape %2152 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2154 = stablehlo.maximum %2153, %9 : tensor<1x384x512xf32>
    %2155 = stablehlo.reshape %2154 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2156 = stablehlo.dot %2155, %1020 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2157 = stablehlo.broadcast_in_dim %1019, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2158 = stablehlo.add %2156, %2157 : tensor<384x128xf32>
    %2159 = stablehlo.reshape %2158 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2160 = stablehlo.add %2159, %2148 : tensor<1x384x128xf32>
    %2161 = stablehlo.broadcast_in_dim %1018, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2162 = stablehlo.multiply %2160, %2161 : tensor<1x384x128xf32>
    %2163 = stablehlo.broadcast_in_dim %1017, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2164 = stablehlo.add %2162, %2163 : tensor<1x384x128xf32>
    %2165 = stablehlo.reshape %2164 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2166 = stablehlo.dot %2165, %1022 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2167 = stablehlo.broadcast_in_dim %1021, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2168 = stablehlo.add %2166, %2167 : tensor<384x512xf32>
    %2169 = stablehlo.reshape %2168 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2170 = stablehlo.maximum %2169, %9 : tensor<1x384x512xf32>
    %2171 = stablehlo.reshape %2170 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2172 = stablehlo.dot %2171, %1030 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2173 = stablehlo.broadcast_in_dim %1029, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2174 = stablehlo.add %2172, %2173 : tensor<384x128xf32>
    %2175 = stablehlo.reshape %2174 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2176 = stablehlo.add %2175, %2164 : tensor<1x384x128xf32>
    %2177 = stablehlo.broadcast_in_dim %1024, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2178 = stablehlo.multiply %2176, %2177 : tensor<1x384x128xf32>
    %2179 = stablehlo.broadcast_in_dim %1023, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2180 = stablehlo.add %2178, %2179 : tensor<1x384x128xf32>
    %2181 = stablehlo.reshape %2180 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2182 = stablehlo.dot %2181, %1028 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2183 = stablehlo.broadcast_in_dim %1027, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2184 = stablehlo.add %2182, %2183 : tensor<384x512xf32>
    %2185 = stablehlo.reshape %2184 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2186 = stablehlo.add %2185, %2060 : tensor<1x384x512xf32>
    %2187 = stablehlo.broadcast_in_dim %1026, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2188 = stablehlo.multiply %2186, %2187 : tensor<1x384x512xf32>
    %2189 = stablehlo.broadcast_in_dim %1025, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2190 = stablehlo.add %2188, %2189 : tensor<1x384x512xf32>
    %2191 = stablehlo.reshape %2190 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2192 = stablehlo.dot %2191, %1040 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2193 = stablehlo.broadcast_in_dim %1039, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2194 = stablehlo.add %2192, %2193 : tensor<384x128xf32>
    %2195 = stablehlo.reshape %2194 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2196 = stablehlo.transpose %2195, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2197 = stablehlo.dot %2191, %1044 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2198 = stablehlo.reshape %2197 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2199 = stablehlo.broadcast_in_dim %1043, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2200 = stablehlo.add %2198, %2199 : tensor<1x384x128xf32>
    %2201 = stablehlo.broadcast_in_dim %1042, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2202 = stablehlo.multiply %2200, %2201 : tensor<1x384x128xf32>
    %2203 = stablehlo.broadcast_in_dim %1041, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2204 = stablehlo.add %2202, %2203 : tensor<1x384x128xf32>
    %2205 = stablehlo.reshape %2204 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2206 = stablehlo.dot %2205, %1036 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2207 = stablehlo.broadcast_in_dim %1035, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2208 = stablehlo.add %2206, %2207 : tensor<384x128xf32>
    %2209 = stablehlo.reshape %2208 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2210 = stablehlo.transpose %2209, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2211 = stablehlo.dot %2205, %1038 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2212 = stablehlo.broadcast_in_dim %1037, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2213 = stablehlo.add %2211, %2212 : tensor<384x128xf32>
    %2214 = stablehlo.reshape %2213 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2215 = stablehlo.transpose %2214, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2216 = stablehlo.dot_general %2215, %2210, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2217 = stablehlo.multiply %2216, %4 : tensor<1x4x384x384xf32>
    %2218 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2219 = stablehlo.add %2217, %2218 : tensor<1x4x384x384xf32>
    %2220 = stablehlo.reduce(%2219 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2221 = stablehlo.broadcast_in_dim %2220, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2222 = stablehlo.subtract %2219, %2221 : tensor<1x4x384x384xf32>
    %2223 = stablehlo.exponential %2222 : tensor<1x4x384x384xf32>
    %2224 = stablehlo.reduce(%2223 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2225 = stablehlo.broadcast_in_dim %2224, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2226 = stablehlo.divide %2223, %2225 : tensor<1x4x384x384xf32>
    %2227 = stablehlo.dot_general %2226, %2196, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2228 = stablehlo.transpose %2227, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2229 = stablehlo.reshape %2228 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2230 = stablehlo.dot %2229, %1034 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2231 = stablehlo.broadcast_in_dim %1033, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2232 = stablehlo.add %2230, %2231 : tensor<384x128xf32>
    %2233 = stablehlo.reshape %2232 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2234 = stablehlo.dot %2191, %1048 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2235 = stablehlo.broadcast_in_dim %1047, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2236 = stablehlo.add %2234, %2235 : tensor<384x128xf32>
    %2237 = stablehlo.reshape %2236 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2238 = stablehlo.broadcast_in_dim %1046, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2239 = stablehlo.multiply %2237, %2238 : tensor<1x384x128xf32>
    %2240 = stablehlo.broadcast_in_dim %1045, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2241 = stablehlo.add %2239, %2240 : tensor<1x384x128xf32>
    %2242 = stablehlo.add %2233, %2241 : tensor<1x384x128xf32>
    %2243 = stablehlo.broadcast_in_dim %1032, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2244 = stablehlo.multiply %2242, %2243 : tensor<1x384x128xf32>
    %2245 = stablehlo.broadcast_in_dim %1031, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2246 = stablehlo.add %2244, %2245 : tensor<1x384x128xf32>
    %2247 = stablehlo.reshape %2246 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2248 = stablehlo.dot %2247, %1050 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2249 = stablehlo.broadcast_in_dim %1049, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2250 = stablehlo.add %2248, %2249 : tensor<384x512xf32>
    %2251 = stablehlo.reshape %2250 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2252 = stablehlo.maximum %2251, %9 : tensor<1x384x512xf32>
    %2253 = stablehlo.reshape %2252 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2254 = stablehlo.dot %2253, %1054 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2255 = stablehlo.broadcast_in_dim %1053, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2256 = stablehlo.add %2254, %2255 : tensor<384x128xf32>
    %2257 = stablehlo.reshape %2256 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2258 = stablehlo.add %2257, %2246 : tensor<1x384x128xf32>
    %2259 = stablehlo.broadcast_in_dim %1052, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2260 = stablehlo.multiply %2258, %2259 : tensor<1x384x128xf32>
    %2261 = stablehlo.broadcast_in_dim %1051, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2262 = stablehlo.add %2260, %2261 : tensor<1x384x128xf32>
    %2263 = stablehlo.reshape %2262 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2264 = stablehlo.dot %2263, %1056 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2265 = stablehlo.broadcast_in_dim %1055, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2266 = stablehlo.add %2264, %2265 : tensor<384x512xf32>
    %2267 = stablehlo.reshape %2266 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2268 = stablehlo.maximum %2267, %9 : tensor<1x384x512xf32>
    %2269 = stablehlo.reshape %2268 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2270 = stablehlo.dot %2269, %1060 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2271 = stablehlo.broadcast_in_dim %1059, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2272 = stablehlo.add %2270, %2271 : tensor<384x128xf32>
    %2273 = stablehlo.reshape %2272 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2274 = stablehlo.add %2273, %2262 : tensor<1x384x128xf32>
    %2275 = stablehlo.broadcast_in_dim %1058, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2276 = stablehlo.multiply %2274, %2275 : tensor<1x384x128xf32>
    %2277 = stablehlo.broadcast_in_dim %1057, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2278 = stablehlo.add %2276, %2277 : tensor<1x384x128xf32>
    %2279 = stablehlo.reshape %2278 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2280 = stablehlo.dot %2279, %1062 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2281 = stablehlo.broadcast_in_dim %1061, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2282 = stablehlo.add %2280, %2281 : tensor<384x512xf32>
    %2283 = stablehlo.reshape %2282 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2284 = stablehlo.maximum %2283, %9 : tensor<1x384x512xf32>
    %2285 = stablehlo.reshape %2284 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2286 = stablehlo.dot %2285, %1066 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2287 = stablehlo.broadcast_in_dim %1065, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2288 = stablehlo.add %2286, %2287 : tensor<384x128xf32>
    %2289 = stablehlo.reshape %2288 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2290 = stablehlo.add %2289, %2278 : tensor<1x384x128xf32>
    %2291 = stablehlo.broadcast_in_dim %1064, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2292 = stablehlo.multiply %2290, %2291 : tensor<1x384x128xf32>
    %2293 = stablehlo.broadcast_in_dim %1063, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2294 = stablehlo.add %2292, %2293 : tensor<1x384x128xf32>
    %2295 = stablehlo.reshape %2294 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2296 = stablehlo.dot %2295, %1068 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2297 = stablehlo.broadcast_in_dim %1067, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2298 = stablehlo.add %2296, %2297 : tensor<384x512xf32>
    %2299 = stablehlo.reshape %2298 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2300 = stablehlo.maximum %2299, %9 : tensor<1x384x512xf32>
    %2301 = stablehlo.reshape %2300 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2302 = stablehlo.dot %2301, %1076 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2303 = stablehlo.broadcast_in_dim %1075, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2304 = stablehlo.add %2302, %2303 : tensor<384x128xf32>
    %2305 = stablehlo.reshape %2304 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2306 = stablehlo.add %2305, %2294 : tensor<1x384x128xf32>
    %2307 = stablehlo.broadcast_in_dim %1070, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2308 = stablehlo.multiply %2306, %2307 : tensor<1x384x128xf32>
    %2309 = stablehlo.broadcast_in_dim %1069, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2310 = stablehlo.add %2308, %2309 : tensor<1x384x128xf32>
    %2311 = stablehlo.reshape %2310 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2312 = stablehlo.dot %2311, %1074 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2313 = stablehlo.broadcast_in_dim %1073, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2314 = stablehlo.add %2312, %2313 : tensor<384x512xf32>
    %2315 = stablehlo.reshape %2314 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2316 = stablehlo.add %2315, %2190 : tensor<1x384x512xf32>
    %2317 = stablehlo.broadcast_in_dim %1072, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2318 = stablehlo.multiply %2316, %2317 : tensor<1x384x512xf32>
    %2319 = stablehlo.broadcast_in_dim %1071, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2320 = stablehlo.add %2318, %2319 : tensor<1x384x512xf32>
    %2321 = stablehlo.reshape %2320 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2322 = stablehlo.dot %2321, %1086 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2323 = stablehlo.broadcast_in_dim %1085, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2324 = stablehlo.add %2322, %2323 : tensor<384x128xf32>
    %2325 = stablehlo.reshape %2324 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2326 = stablehlo.transpose %2325, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2327 = stablehlo.dot %2321, %1090 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2328 = stablehlo.reshape %2327 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2329 = stablehlo.broadcast_in_dim %1089, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2330 = stablehlo.add %2328, %2329 : tensor<1x384x128xf32>
    %2331 = stablehlo.broadcast_in_dim %1088, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2332 = stablehlo.multiply %2330, %2331 : tensor<1x384x128xf32>
    %2333 = stablehlo.broadcast_in_dim %1087, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2334 = stablehlo.add %2332, %2333 : tensor<1x384x128xf32>
    %2335 = stablehlo.reshape %2334 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2336 = stablehlo.dot %2335, %1082 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2337 = stablehlo.broadcast_in_dim %1081, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2338 = stablehlo.add %2336, %2337 : tensor<384x128xf32>
    %2339 = stablehlo.reshape %2338 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2340 = stablehlo.transpose %2339, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2341 = stablehlo.dot %2335, %1084 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2342 = stablehlo.broadcast_in_dim %1083, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2343 = stablehlo.add %2341, %2342 : tensor<384x128xf32>
    %2344 = stablehlo.reshape %2343 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2345 = stablehlo.transpose %2344, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2346 = stablehlo.dot_general %2345, %2340, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2347 = stablehlo.multiply %2346, %4 : tensor<1x4x384x384xf32>
    %2348 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2349 = stablehlo.add %2347, %2348 : tensor<1x4x384x384xf32>
    %2350 = stablehlo.reduce(%2349 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2351 = stablehlo.broadcast_in_dim %2350, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2352 = stablehlo.subtract %2349, %2351 : tensor<1x4x384x384xf32>
    %2353 = stablehlo.exponential %2352 : tensor<1x4x384x384xf32>
    %2354 = stablehlo.reduce(%2353 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2355 = stablehlo.broadcast_in_dim %2354, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2356 = stablehlo.divide %2353, %2355 : tensor<1x4x384x384xf32>
    %2357 = stablehlo.dot_general %2356, %2326, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2358 = stablehlo.transpose %2357, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2359 = stablehlo.reshape %2358 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2360 = stablehlo.dot %2359, %1080 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2361 = stablehlo.broadcast_in_dim %1079, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2362 = stablehlo.add %2360, %2361 : tensor<384x128xf32>
    %2363 = stablehlo.reshape %2362 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2364 = stablehlo.dot %2321, %1094 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2365 = stablehlo.broadcast_in_dim %1093, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2366 = stablehlo.add %2364, %2365 : tensor<384x128xf32>
    %2367 = stablehlo.reshape %2366 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2368 = stablehlo.broadcast_in_dim %1092, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2369 = stablehlo.multiply %2367, %2368 : tensor<1x384x128xf32>
    %2370 = stablehlo.broadcast_in_dim %1091, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2371 = stablehlo.add %2369, %2370 : tensor<1x384x128xf32>
    %2372 = stablehlo.add %2363, %2371 : tensor<1x384x128xf32>
    %2373 = stablehlo.broadcast_in_dim %1078, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2374 = stablehlo.multiply %2372, %2373 : tensor<1x384x128xf32>
    %2375 = stablehlo.broadcast_in_dim %1077, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2376 = stablehlo.add %2374, %2375 : tensor<1x384x128xf32>
    %2377 = stablehlo.reshape %2376 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2378 = stablehlo.dot %2377, %1096 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2379 = stablehlo.broadcast_in_dim %1095, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2380 = stablehlo.add %2378, %2379 : tensor<384x512xf32>
    %2381 = stablehlo.reshape %2380 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2382 = stablehlo.maximum %2381, %9 : tensor<1x384x512xf32>
    %2383 = stablehlo.reshape %2382 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2384 = stablehlo.dot %2383, %1100 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2385 = stablehlo.broadcast_in_dim %1099, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2386 = stablehlo.add %2384, %2385 : tensor<384x128xf32>
    %2387 = stablehlo.reshape %2386 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2388 = stablehlo.add %2387, %2376 : tensor<1x384x128xf32>
    %2389 = stablehlo.broadcast_in_dim %1098, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2390 = stablehlo.multiply %2388, %2389 : tensor<1x384x128xf32>
    %2391 = stablehlo.broadcast_in_dim %1097, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2392 = stablehlo.add %2390, %2391 : tensor<1x384x128xf32>
    %2393 = stablehlo.reshape %2392 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2394 = stablehlo.dot %2393, %1102 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2395 = stablehlo.broadcast_in_dim %1101, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2396 = stablehlo.add %2394, %2395 : tensor<384x512xf32>
    %2397 = stablehlo.reshape %2396 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2398 = stablehlo.maximum %2397, %9 : tensor<1x384x512xf32>
    %2399 = stablehlo.reshape %2398 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2400 = stablehlo.dot %2399, %1106 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2401 = stablehlo.broadcast_in_dim %1105, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2402 = stablehlo.add %2400, %2401 : tensor<384x128xf32>
    %2403 = stablehlo.reshape %2402 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2404 = stablehlo.add %2403, %2392 : tensor<1x384x128xf32>
    %2405 = stablehlo.broadcast_in_dim %1104, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2406 = stablehlo.multiply %2404, %2405 : tensor<1x384x128xf32>
    %2407 = stablehlo.broadcast_in_dim %1103, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2408 = stablehlo.add %2406, %2407 : tensor<1x384x128xf32>
    %2409 = stablehlo.reshape %2408 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2410 = stablehlo.dot %2409, %1108 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2411 = stablehlo.broadcast_in_dim %1107, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2412 = stablehlo.add %2410, %2411 : tensor<384x512xf32>
    %2413 = stablehlo.reshape %2412 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2414 = stablehlo.maximum %2413, %9 : tensor<1x384x512xf32>
    %2415 = stablehlo.reshape %2414 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2416 = stablehlo.dot %2415, %1112 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2417 = stablehlo.broadcast_in_dim %1111, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2418 = stablehlo.add %2416, %2417 : tensor<384x128xf32>
    %2419 = stablehlo.reshape %2418 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2420 = stablehlo.add %2419, %2408 : tensor<1x384x128xf32>
    %2421 = stablehlo.broadcast_in_dim %1110, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2422 = stablehlo.multiply %2420, %2421 : tensor<1x384x128xf32>
    %2423 = stablehlo.broadcast_in_dim %1109, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2424 = stablehlo.add %2422, %2423 : tensor<1x384x128xf32>
    %2425 = stablehlo.reshape %2424 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2426 = stablehlo.dot %2425, %1114 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2427 = stablehlo.broadcast_in_dim %1113, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2428 = stablehlo.add %2426, %2427 : tensor<384x512xf32>
    %2429 = stablehlo.reshape %2428 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2430 = stablehlo.maximum %2429, %9 : tensor<1x384x512xf32>
    %2431 = stablehlo.reshape %2430 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2432 = stablehlo.dot %2431, %1122 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2433 = stablehlo.broadcast_in_dim %1121, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2434 = stablehlo.add %2432, %2433 : tensor<384x128xf32>
    %2435 = stablehlo.reshape %2434 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2436 = stablehlo.add %2435, %2424 : tensor<1x384x128xf32>
    %2437 = stablehlo.broadcast_in_dim %1116, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2438 = stablehlo.multiply %2436, %2437 : tensor<1x384x128xf32>
    %2439 = stablehlo.broadcast_in_dim %1115, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2440 = stablehlo.add %2438, %2439 : tensor<1x384x128xf32>
    %2441 = stablehlo.reshape %2440 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2442 = stablehlo.dot %2441, %1120 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2443 = stablehlo.broadcast_in_dim %1119, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2444 = stablehlo.add %2442, %2443 : tensor<384x512xf32>
    %2445 = stablehlo.reshape %2444 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2446 = stablehlo.add %2445, %2320 : tensor<1x384x512xf32>
    %2447 = stablehlo.broadcast_in_dim %1118, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2448 = stablehlo.multiply %2446, %2447 : tensor<1x384x512xf32>
    %2449 = stablehlo.broadcast_in_dim %1117, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2450 = stablehlo.add %2448, %2449 : tensor<1x384x512xf32>
    %2451 = stablehlo.reshape %2450 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2452 = stablehlo.dot %2451, %120 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2453 = stablehlo.broadcast_in_dim %119, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2454 = stablehlo.add %2452, %2453 : tensor<384x128xf32>
    %2455 = stablehlo.reshape %2454 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2456 = stablehlo.transpose %2455, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2457 = stablehlo.dot %2451, %124 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2458 = stablehlo.reshape %2457 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2459 = stablehlo.broadcast_in_dim %123, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2460 = stablehlo.add %2458, %2459 : tensor<1x384x128xf32>
    %2461 = stablehlo.broadcast_in_dim %122, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2462 = stablehlo.multiply %2460, %2461 : tensor<1x384x128xf32>
    %2463 = stablehlo.broadcast_in_dim %121, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2464 = stablehlo.add %2462, %2463 : tensor<1x384x128xf32>
    %2465 = stablehlo.reshape %2464 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2466 = stablehlo.dot %2465, %116 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2467 = stablehlo.broadcast_in_dim %115, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2468 = stablehlo.add %2466, %2467 : tensor<384x128xf32>
    %2469 = stablehlo.reshape %2468 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2470 = stablehlo.transpose %2469, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2471 = stablehlo.dot %2465, %118 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2472 = stablehlo.broadcast_in_dim %117, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2473 = stablehlo.add %2471, %2472 : tensor<384x128xf32>
    %2474 = stablehlo.reshape %2473 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2475 = stablehlo.transpose %2474, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2476 = stablehlo.dot_general %2475, %2470, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2477 = stablehlo.multiply %2476, %4 : tensor<1x4x384x384xf32>
    %2478 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2479 = stablehlo.add %2477, %2478 : tensor<1x4x384x384xf32>
    %2480 = stablehlo.reduce(%2479 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2481 = stablehlo.broadcast_in_dim %2480, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2482 = stablehlo.subtract %2479, %2481 : tensor<1x4x384x384xf32>
    %2483 = stablehlo.exponential %2482 : tensor<1x4x384x384xf32>
    %2484 = stablehlo.reduce(%2483 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2485 = stablehlo.broadcast_in_dim %2484, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2486 = stablehlo.divide %2483, %2485 : tensor<1x4x384x384xf32>
    %2487 = stablehlo.dot_general %2486, %2456, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2488 = stablehlo.transpose %2487, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2489 = stablehlo.reshape %2488 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2490 = stablehlo.dot %2489, %114 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2491 = stablehlo.broadcast_in_dim %113, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2492 = stablehlo.add %2490, %2491 : tensor<384x128xf32>
    %2493 = stablehlo.reshape %2492 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2494 = stablehlo.dot %2451, %128 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2495 = stablehlo.broadcast_in_dim %127, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2496 = stablehlo.add %2494, %2495 : tensor<384x128xf32>
    %2497 = stablehlo.reshape %2496 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2498 = stablehlo.broadcast_in_dim %126, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2499 = stablehlo.multiply %2497, %2498 : tensor<1x384x128xf32>
    %2500 = stablehlo.broadcast_in_dim %125, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2501 = stablehlo.add %2499, %2500 : tensor<1x384x128xf32>
    %2502 = stablehlo.add %2493, %2501 : tensor<1x384x128xf32>
    %2503 = stablehlo.broadcast_in_dim %112, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2504 = stablehlo.multiply %2502, %2503 : tensor<1x384x128xf32>
    %2505 = stablehlo.broadcast_in_dim %111, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2506 = stablehlo.add %2504, %2505 : tensor<1x384x128xf32>
    %2507 = stablehlo.reshape %2506 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2508 = stablehlo.dot %2507, %130 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2509 = stablehlo.broadcast_in_dim %129, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2510 = stablehlo.add %2508, %2509 : tensor<384x512xf32>
    %2511 = stablehlo.reshape %2510 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2512 = stablehlo.maximum %2511, %9 : tensor<1x384x512xf32>
    %2513 = stablehlo.reshape %2512 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2514 = stablehlo.dot %2513, %134 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2515 = stablehlo.broadcast_in_dim %133, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2516 = stablehlo.add %2514, %2515 : tensor<384x128xf32>
    %2517 = stablehlo.reshape %2516 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2518 = stablehlo.add %2517, %2506 : tensor<1x384x128xf32>
    %2519 = stablehlo.broadcast_in_dim %132, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2520 = stablehlo.multiply %2518, %2519 : tensor<1x384x128xf32>
    %2521 = stablehlo.broadcast_in_dim %131, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2522 = stablehlo.add %2520, %2521 : tensor<1x384x128xf32>
    %2523 = stablehlo.reshape %2522 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2524 = stablehlo.dot %2523, %136 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2525 = stablehlo.broadcast_in_dim %135, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2526 = stablehlo.add %2524, %2525 : tensor<384x512xf32>
    %2527 = stablehlo.reshape %2526 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2528 = stablehlo.maximum %2527, %9 : tensor<1x384x512xf32>
    %2529 = stablehlo.reshape %2528 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2530 = stablehlo.dot %2529, %140 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2531 = stablehlo.broadcast_in_dim %139, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2532 = stablehlo.add %2530, %2531 : tensor<384x128xf32>
    %2533 = stablehlo.reshape %2532 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2534 = stablehlo.add %2533, %2522 : tensor<1x384x128xf32>
    %2535 = stablehlo.broadcast_in_dim %138, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2536 = stablehlo.multiply %2534, %2535 : tensor<1x384x128xf32>
    %2537 = stablehlo.broadcast_in_dim %137, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2538 = stablehlo.add %2536, %2537 : tensor<1x384x128xf32>
    %2539 = stablehlo.reshape %2538 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2540 = stablehlo.dot %2539, %142 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2541 = stablehlo.broadcast_in_dim %141, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2542 = stablehlo.add %2540, %2541 : tensor<384x512xf32>
    %2543 = stablehlo.reshape %2542 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2544 = stablehlo.maximum %2543, %9 : tensor<1x384x512xf32>
    %2545 = stablehlo.reshape %2544 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2546 = stablehlo.dot %2545, %146 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2547 = stablehlo.broadcast_in_dim %145, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2548 = stablehlo.add %2546, %2547 : tensor<384x128xf32>
    %2549 = stablehlo.reshape %2548 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2550 = stablehlo.add %2549, %2538 : tensor<1x384x128xf32>
    %2551 = stablehlo.broadcast_in_dim %144, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2552 = stablehlo.multiply %2550, %2551 : tensor<1x384x128xf32>
    %2553 = stablehlo.broadcast_in_dim %143, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2554 = stablehlo.add %2552, %2553 : tensor<1x384x128xf32>
    %2555 = stablehlo.reshape %2554 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2556 = stablehlo.dot %2555, %148 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2557 = stablehlo.broadcast_in_dim %147, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2558 = stablehlo.add %2556, %2557 : tensor<384x512xf32>
    %2559 = stablehlo.reshape %2558 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2560 = stablehlo.maximum %2559, %9 : tensor<1x384x512xf32>
    %2561 = stablehlo.reshape %2560 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2562 = stablehlo.dot %2561, %156 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2563 = stablehlo.broadcast_in_dim %155, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2564 = stablehlo.add %2562, %2563 : tensor<384x128xf32>
    %2565 = stablehlo.reshape %2564 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2566 = stablehlo.add %2565, %2554 : tensor<1x384x128xf32>
    %2567 = stablehlo.broadcast_in_dim %150, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2568 = stablehlo.multiply %2566, %2567 : tensor<1x384x128xf32>
    %2569 = stablehlo.broadcast_in_dim %149, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2570 = stablehlo.add %2568, %2569 : tensor<1x384x128xf32>
    %2571 = stablehlo.reshape %2570 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2572 = stablehlo.dot %2571, %154 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2573 = stablehlo.broadcast_in_dim %153, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2574 = stablehlo.add %2572, %2573 : tensor<384x512xf32>
    %2575 = stablehlo.reshape %2574 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2576 = stablehlo.add %2575, %2450 : tensor<1x384x512xf32>
    %2577 = stablehlo.broadcast_in_dim %152, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2578 = stablehlo.multiply %2576, %2577 : tensor<1x384x512xf32>
    %2579 = stablehlo.broadcast_in_dim %151, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2580 = stablehlo.add %2578, %2579 : tensor<1x384x512xf32>
    %2581 = stablehlo.reshape %2580 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2582 = stablehlo.dot %2581, %166 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2583 = stablehlo.broadcast_in_dim %165, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2584 = stablehlo.add %2582, %2583 : tensor<384x128xf32>
    %2585 = stablehlo.reshape %2584 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2586 = stablehlo.transpose %2585, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2587 = stablehlo.dot %2581, %170 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2588 = stablehlo.reshape %2587 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2589 = stablehlo.broadcast_in_dim %169, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2590 = stablehlo.add %2588, %2589 : tensor<1x384x128xf32>
    %2591 = stablehlo.broadcast_in_dim %168, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2592 = stablehlo.multiply %2590, %2591 : tensor<1x384x128xf32>
    %2593 = stablehlo.broadcast_in_dim %167, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2594 = stablehlo.add %2592, %2593 : tensor<1x384x128xf32>
    %2595 = stablehlo.reshape %2594 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2596 = stablehlo.dot %2595, %162 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2597 = stablehlo.broadcast_in_dim %161, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2598 = stablehlo.add %2596, %2597 : tensor<384x128xf32>
    %2599 = stablehlo.reshape %2598 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2600 = stablehlo.transpose %2599, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2601 = stablehlo.dot %2595, %164 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2602 = stablehlo.broadcast_in_dim %163, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2603 = stablehlo.add %2601, %2602 : tensor<384x128xf32>
    %2604 = stablehlo.reshape %2603 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2605 = stablehlo.transpose %2604, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2606 = stablehlo.dot_general %2605, %2600, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2607 = stablehlo.multiply %2606, %4 : tensor<1x4x384x384xf32>
    %2608 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2609 = stablehlo.add %2607, %2608 : tensor<1x4x384x384xf32>
    %2610 = stablehlo.reduce(%2609 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2611 = stablehlo.broadcast_in_dim %2610, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2612 = stablehlo.subtract %2609, %2611 : tensor<1x4x384x384xf32>
    %2613 = stablehlo.exponential %2612 : tensor<1x4x384x384xf32>
    %2614 = stablehlo.reduce(%2613 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2615 = stablehlo.broadcast_in_dim %2614, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2616 = stablehlo.divide %2613, %2615 : tensor<1x4x384x384xf32>
    %2617 = stablehlo.dot_general %2616, %2586, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2618 = stablehlo.transpose %2617, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2619 = stablehlo.reshape %2618 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2620 = stablehlo.dot %2619, %160 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2621 = stablehlo.broadcast_in_dim %159, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2622 = stablehlo.add %2620, %2621 : tensor<384x128xf32>
    %2623 = stablehlo.reshape %2622 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2624 = stablehlo.dot %2581, %174 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2625 = stablehlo.broadcast_in_dim %173, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2626 = stablehlo.add %2624, %2625 : tensor<384x128xf32>
    %2627 = stablehlo.reshape %2626 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2628 = stablehlo.broadcast_in_dim %172, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2629 = stablehlo.multiply %2627, %2628 : tensor<1x384x128xf32>
    %2630 = stablehlo.broadcast_in_dim %171, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2631 = stablehlo.add %2629, %2630 : tensor<1x384x128xf32>
    %2632 = stablehlo.add %2623, %2631 : tensor<1x384x128xf32>
    %2633 = stablehlo.broadcast_in_dim %158, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2634 = stablehlo.multiply %2632, %2633 : tensor<1x384x128xf32>
    %2635 = stablehlo.broadcast_in_dim %157, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2636 = stablehlo.add %2634, %2635 : tensor<1x384x128xf32>
    %2637 = stablehlo.reshape %2636 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2638 = stablehlo.dot %2637, %176 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2639 = stablehlo.broadcast_in_dim %175, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2640 = stablehlo.add %2638, %2639 : tensor<384x512xf32>
    %2641 = stablehlo.reshape %2640 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2642 = stablehlo.maximum %2641, %9 : tensor<1x384x512xf32>
    %2643 = stablehlo.reshape %2642 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2644 = stablehlo.dot %2643, %180 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2645 = stablehlo.broadcast_in_dim %179, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2646 = stablehlo.add %2644, %2645 : tensor<384x128xf32>
    %2647 = stablehlo.reshape %2646 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2648 = stablehlo.add %2647, %2636 : tensor<1x384x128xf32>
    %2649 = stablehlo.broadcast_in_dim %178, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2650 = stablehlo.multiply %2648, %2649 : tensor<1x384x128xf32>
    %2651 = stablehlo.broadcast_in_dim %177, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2652 = stablehlo.add %2650, %2651 : tensor<1x384x128xf32>
    %2653 = stablehlo.reshape %2652 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2654 = stablehlo.dot %2653, %182 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2655 = stablehlo.broadcast_in_dim %181, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2656 = stablehlo.add %2654, %2655 : tensor<384x512xf32>
    %2657 = stablehlo.reshape %2656 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2658 = stablehlo.maximum %2657, %9 : tensor<1x384x512xf32>
    %2659 = stablehlo.reshape %2658 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2660 = stablehlo.dot %2659, %186 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2661 = stablehlo.broadcast_in_dim %185, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2662 = stablehlo.add %2660, %2661 : tensor<384x128xf32>
    %2663 = stablehlo.reshape %2662 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2664 = stablehlo.add %2663, %2652 : tensor<1x384x128xf32>
    %2665 = stablehlo.broadcast_in_dim %184, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2666 = stablehlo.multiply %2664, %2665 : tensor<1x384x128xf32>
    %2667 = stablehlo.broadcast_in_dim %183, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2668 = stablehlo.add %2666, %2667 : tensor<1x384x128xf32>
    %2669 = stablehlo.reshape %2668 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2670 = stablehlo.dot %2669, %188 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2671 = stablehlo.broadcast_in_dim %187, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2672 = stablehlo.add %2670, %2671 : tensor<384x512xf32>
    %2673 = stablehlo.reshape %2672 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2674 = stablehlo.maximum %2673, %9 : tensor<1x384x512xf32>
    %2675 = stablehlo.reshape %2674 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2676 = stablehlo.dot %2675, %192 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2677 = stablehlo.broadcast_in_dim %191, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2678 = stablehlo.add %2676, %2677 : tensor<384x128xf32>
    %2679 = stablehlo.reshape %2678 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2680 = stablehlo.add %2679, %2668 : tensor<1x384x128xf32>
    %2681 = stablehlo.broadcast_in_dim %190, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2682 = stablehlo.multiply %2680, %2681 : tensor<1x384x128xf32>
    %2683 = stablehlo.broadcast_in_dim %189, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2684 = stablehlo.add %2682, %2683 : tensor<1x384x128xf32>
    %2685 = stablehlo.reshape %2684 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2686 = stablehlo.dot %2685, %194 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2687 = stablehlo.broadcast_in_dim %193, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2688 = stablehlo.add %2686, %2687 : tensor<384x512xf32>
    %2689 = stablehlo.reshape %2688 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2690 = stablehlo.maximum %2689, %9 : tensor<1x384x512xf32>
    %2691 = stablehlo.reshape %2690 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2692 = stablehlo.dot %2691, %202 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2693 = stablehlo.broadcast_in_dim %201, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2694 = stablehlo.add %2692, %2693 : tensor<384x128xf32>
    %2695 = stablehlo.reshape %2694 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2696 = stablehlo.add %2695, %2684 : tensor<1x384x128xf32>
    %2697 = stablehlo.broadcast_in_dim %196, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2698 = stablehlo.multiply %2696, %2697 : tensor<1x384x128xf32>
    %2699 = stablehlo.broadcast_in_dim %195, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2700 = stablehlo.add %2698, %2699 : tensor<1x384x128xf32>
    %2701 = stablehlo.reshape %2700 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2702 = stablehlo.dot %2701, %200 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2703 = stablehlo.broadcast_in_dim %199, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2704 = stablehlo.add %2702, %2703 : tensor<384x512xf32>
    %2705 = stablehlo.reshape %2704 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2706 = stablehlo.add %2705, %2580 : tensor<1x384x512xf32>
    %2707 = stablehlo.broadcast_in_dim %198, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2708 = stablehlo.multiply %2706, %2707 : tensor<1x384x512xf32>
    %2709 = stablehlo.broadcast_in_dim %197, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2710 = stablehlo.add %2708, %2709 : tensor<1x384x512xf32>
    %2711 = stablehlo.reshape %2710 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2712 = stablehlo.dot %2711, %212 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2713 = stablehlo.broadcast_in_dim %211, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2714 = stablehlo.add %2712, %2713 : tensor<384x128xf32>
    %2715 = stablehlo.reshape %2714 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2716 = stablehlo.transpose %2715, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2717 = stablehlo.dot %2711, %216 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2718 = stablehlo.reshape %2717 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2719 = stablehlo.broadcast_in_dim %215, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2720 = stablehlo.add %2718, %2719 : tensor<1x384x128xf32>
    %2721 = stablehlo.broadcast_in_dim %214, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2722 = stablehlo.multiply %2720, %2721 : tensor<1x384x128xf32>
    %2723 = stablehlo.broadcast_in_dim %213, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2724 = stablehlo.add %2722, %2723 : tensor<1x384x128xf32>
    %2725 = stablehlo.reshape %2724 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2726 = stablehlo.dot %2725, %208 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2727 = stablehlo.broadcast_in_dim %207, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2728 = stablehlo.add %2726, %2727 : tensor<384x128xf32>
    %2729 = stablehlo.reshape %2728 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2730 = stablehlo.transpose %2729, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2731 = stablehlo.dot %2725, %210 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2732 = stablehlo.broadcast_in_dim %209, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2733 = stablehlo.add %2731, %2732 : tensor<384x128xf32>
    %2734 = stablehlo.reshape %2733 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2735 = stablehlo.transpose %2734, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2736 = stablehlo.dot_general %2735, %2730, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2737 = stablehlo.multiply %2736, %4 : tensor<1x4x384x384xf32>
    %2738 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2739 = stablehlo.add %2737, %2738 : tensor<1x4x384x384xf32>
    %2740 = stablehlo.reduce(%2739 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2741 = stablehlo.broadcast_in_dim %2740, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2742 = stablehlo.subtract %2739, %2741 : tensor<1x4x384x384xf32>
    %2743 = stablehlo.exponential %2742 : tensor<1x4x384x384xf32>
    %2744 = stablehlo.reduce(%2743 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2745 = stablehlo.broadcast_in_dim %2744, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2746 = stablehlo.divide %2743, %2745 : tensor<1x4x384x384xf32>
    %2747 = stablehlo.dot_general %2746, %2716, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2748 = stablehlo.transpose %2747, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2749 = stablehlo.reshape %2748 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2750 = stablehlo.dot %2749, %206 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2751 = stablehlo.broadcast_in_dim %205, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2752 = stablehlo.add %2750, %2751 : tensor<384x128xf32>
    %2753 = stablehlo.reshape %2752 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2754 = stablehlo.dot %2711, %220 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2755 = stablehlo.broadcast_in_dim %219, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2756 = stablehlo.add %2754, %2755 : tensor<384x128xf32>
    %2757 = stablehlo.reshape %2756 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2758 = stablehlo.broadcast_in_dim %218, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2759 = stablehlo.multiply %2757, %2758 : tensor<1x384x128xf32>
    %2760 = stablehlo.broadcast_in_dim %217, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2761 = stablehlo.add %2759, %2760 : tensor<1x384x128xf32>
    %2762 = stablehlo.add %2753, %2761 : tensor<1x384x128xf32>
    %2763 = stablehlo.broadcast_in_dim %204, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2764 = stablehlo.multiply %2762, %2763 : tensor<1x384x128xf32>
    %2765 = stablehlo.broadcast_in_dim %203, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2766 = stablehlo.add %2764, %2765 : tensor<1x384x128xf32>
    %2767 = stablehlo.reshape %2766 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2768 = stablehlo.dot %2767, %222 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2769 = stablehlo.broadcast_in_dim %221, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2770 = stablehlo.add %2768, %2769 : tensor<384x512xf32>
    %2771 = stablehlo.reshape %2770 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2772 = stablehlo.maximum %2771, %9 : tensor<1x384x512xf32>
    %2773 = stablehlo.reshape %2772 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2774 = stablehlo.dot %2773, %226 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2775 = stablehlo.broadcast_in_dim %225, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2776 = stablehlo.add %2774, %2775 : tensor<384x128xf32>
    %2777 = stablehlo.reshape %2776 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2778 = stablehlo.add %2777, %2766 : tensor<1x384x128xf32>
    %2779 = stablehlo.broadcast_in_dim %224, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2780 = stablehlo.multiply %2778, %2779 : tensor<1x384x128xf32>
    %2781 = stablehlo.broadcast_in_dim %223, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2782 = stablehlo.add %2780, %2781 : tensor<1x384x128xf32>
    %2783 = stablehlo.reshape %2782 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2784 = stablehlo.dot %2783, %228 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2785 = stablehlo.broadcast_in_dim %227, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2786 = stablehlo.add %2784, %2785 : tensor<384x512xf32>
    %2787 = stablehlo.reshape %2786 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2788 = stablehlo.maximum %2787, %9 : tensor<1x384x512xf32>
    %2789 = stablehlo.reshape %2788 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2790 = stablehlo.dot %2789, %232 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2791 = stablehlo.broadcast_in_dim %231, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2792 = stablehlo.add %2790, %2791 : tensor<384x128xf32>
    %2793 = stablehlo.reshape %2792 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2794 = stablehlo.add %2793, %2782 : tensor<1x384x128xf32>
    %2795 = stablehlo.broadcast_in_dim %230, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2796 = stablehlo.multiply %2794, %2795 : tensor<1x384x128xf32>
    %2797 = stablehlo.broadcast_in_dim %229, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2798 = stablehlo.add %2796, %2797 : tensor<1x384x128xf32>
    %2799 = stablehlo.reshape %2798 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2800 = stablehlo.dot %2799, %234 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2801 = stablehlo.broadcast_in_dim %233, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2802 = stablehlo.add %2800, %2801 : tensor<384x512xf32>
    %2803 = stablehlo.reshape %2802 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2804 = stablehlo.maximum %2803, %9 : tensor<1x384x512xf32>
    %2805 = stablehlo.reshape %2804 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2806 = stablehlo.dot %2805, %238 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2807 = stablehlo.broadcast_in_dim %237, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2808 = stablehlo.add %2806, %2807 : tensor<384x128xf32>
    %2809 = stablehlo.reshape %2808 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2810 = stablehlo.add %2809, %2798 : tensor<1x384x128xf32>
    %2811 = stablehlo.broadcast_in_dim %236, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2812 = stablehlo.multiply %2810, %2811 : tensor<1x384x128xf32>
    %2813 = stablehlo.broadcast_in_dim %235, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2814 = stablehlo.add %2812, %2813 : tensor<1x384x128xf32>
    %2815 = stablehlo.reshape %2814 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2816 = stablehlo.dot %2815, %240 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2817 = stablehlo.broadcast_in_dim %239, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2818 = stablehlo.add %2816, %2817 : tensor<384x512xf32>
    %2819 = stablehlo.reshape %2818 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2820 = stablehlo.maximum %2819, %9 : tensor<1x384x512xf32>
    %2821 = stablehlo.reshape %2820 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2822 = stablehlo.dot %2821, %248 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2823 = stablehlo.broadcast_in_dim %247, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2824 = stablehlo.add %2822, %2823 : tensor<384x128xf32>
    %2825 = stablehlo.reshape %2824 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2826 = stablehlo.add %2825, %2814 : tensor<1x384x128xf32>
    %2827 = stablehlo.broadcast_in_dim %242, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2828 = stablehlo.multiply %2826, %2827 : tensor<1x384x128xf32>
    %2829 = stablehlo.broadcast_in_dim %241, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2830 = stablehlo.add %2828, %2829 : tensor<1x384x128xf32>
    %2831 = stablehlo.reshape %2830 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2832 = stablehlo.dot %2831, %246 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2833 = stablehlo.broadcast_in_dim %245, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2834 = stablehlo.add %2832, %2833 : tensor<384x512xf32>
    %2835 = stablehlo.reshape %2834 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2836 = stablehlo.add %2835, %2710 : tensor<1x384x512xf32>
    %2837 = stablehlo.broadcast_in_dim %244, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2838 = stablehlo.multiply %2836, %2837 : tensor<1x384x512xf32>
    %2839 = stablehlo.broadcast_in_dim %243, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2840 = stablehlo.add %2838, %2839 : tensor<1x384x512xf32>
    %2841 = stablehlo.reshape %2840 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2842 = stablehlo.dot %2841, %258 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2843 = stablehlo.broadcast_in_dim %257, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2844 = stablehlo.add %2842, %2843 : tensor<384x128xf32>
    %2845 = stablehlo.reshape %2844 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2846 = stablehlo.transpose %2845, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2847 = stablehlo.dot %2841, %262 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2848 = stablehlo.reshape %2847 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2849 = stablehlo.broadcast_in_dim %261, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2850 = stablehlo.add %2848, %2849 : tensor<1x384x128xf32>
    %2851 = stablehlo.broadcast_in_dim %260, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2852 = stablehlo.multiply %2850, %2851 : tensor<1x384x128xf32>
    %2853 = stablehlo.broadcast_in_dim %259, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2854 = stablehlo.add %2852, %2853 : tensor<1x384x128xf32>
    %2855 = stablehlo.reshape %2854 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2856 = stablehlo.dot %2855, %254 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2857 = stablehlo.broadcast_in_dim %253, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2858 = stablehlo.add %2856, %2857 : tensor<384x128xf32>
    %2859 = stablehlo.reshape %2858 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2860 = stablehlo.transpose %2859, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2861 = stablehlo.dot %2855, %256 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2862 = stablehlo.broadcast_in_dim %255, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2863 = stablehlo.add %2861, %2862 : tensor<384x128xf32>
    %2864 = stablehlo.reshape %2863 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2865 = stablehlo.transpose %2864, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2866 = stablehlo.dot_general %2865, %2860, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2867 = stablehlo.multiply %2866, %4 : tensor<1x4x384x384xf32>
    %2868 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2869 = stablehlo.add %2867, %2868 : tensor<1x4x384x384xf32>
    %2870 = stablehlo.reduce(%2869 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2871 = stablehlo.broadcast_in_dim %2870, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2872 = stablehlo.subtract %2869, %2871 : tensor<1x4x384x384xf32>
    %2873 = stablehlo.exponential %2872 : tensor<1x4x384x384xf32>
    %2874 = stablehlo.reduce(%2873 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %2875 = stablehlo.broadcast_in_dim %2874, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2876 = stablehlo.divide %2873, %2875 : tensor<1x4x384x384xf32>
    %2877 = stablehlo.dot_general %2876, %2846, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2878 = stablehlo.transpose %2877, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2879 = stablehlo.reshape %2878 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2880 = stablehlo.dot %2879, %252 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2881 = stablehlo.broadcast_in_dim %251, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2882 = stablehlo.add %2880, %2881 : tensor<384x128xf32>
    %2883 = stablehlo.reshape %2882 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2884 = stablehlo.dot %2841, %266 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2885 = stablehlo.broadcast_in_dim %265, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2886 = stablehlo.add %2884, %2885 : tensor<384x128xf32>
    %2887 = stablehlo.reshape %2886 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2888 = stablehlo.broadcast_in_dim %264, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2889 = stablehlo.multiply %2887, %2888 : tensor<1x384x128xf32>
    %2890 = stablehlo.broadcast_in_dim %263, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2891 = stablehlo.add %2889, %2890 : tensor<1x384x128xf32>
    %2892 = stablehlo.add %2883, %2891 : tensor<1x384x128xf32>
    %2893 = stablehlo.broadcast_in_dim %250, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2894 = stablehlo.multiply %2892, %2893 : tensor<1x384x128xf32>
    %2895 = stablehlo.broadcast_in_dim %249, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2896 = stablehlo.add %2894, %2895 : tensor<1x384x128xf32>
    %2897 = stablehlo.reshape %2896 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2898 = stablehlo.dot %2897, %268 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2899 = stablehlo.broadcast_in_dim %267, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2900 = stablehlo.add %2898, %2899 : tensor<384x512xf32>
    %2901 = stablehlo.reshape %2900 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2902 = stablehlo.maximum %2901, %9 : tensor<1x384x512xf32>
    %2903 = stablehlo.reshape %2902 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2904 = stablehlo.dot %2903, %272 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2905 = stablehlo.broadcast_in_dim %271, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2906 = stablehlo.add %2904, %2905 : tensor<384x128xf32>
    %2907 = stablehlo.reshape %2906 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2908 = stablehlo.add %2907, %2896 : tensor<1x384x128xf32>
    %2909 = stablehlo.broadcast_in_dim %270, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2910 = stablehlo.multiply %2908, %2909 : tensor<1x384x128xf32>
    %2911 = stablehlo.broadcast_in_dim %269, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2912 = stablehlo.add %2910, %2911 : tensor<1x384x128xf32>
    %2913 = stablehlo.reshape %2912 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2914 = stablehlo.dot %2913, %274 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2915 = stablehlo.broadcast_in_dim %273, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2916 = stablehlo.add %2914, %2915 : tensor<384x512xf32>
    %2917 = stablehlo.reshape %2916 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2918 = stablehlo.maximum %2917, %9 : tensor<1x384x512xf32>
    %2919 = stablehlo.reshape %2918 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2920 = stablehlo.dot %2919, %278 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2921 = stablehlo.broadcast_in_dim %277, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2922 = stablehlo.add %2920, %2921 : tensor<384x128xf32>
    %2923 = stablehlo.reshape %2922 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2924 = stablehlo.add %2923, %2912 : tensor<1x384x128xf32>
    %2925 = stablehlo.broadcast_in_dim %276, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2926 = stablehlo.multiply %2924, %2925 : tensor<1x384x128xf32>
    %2927 = stablehlo.broadcast_in_dim %275, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2928 = stablehlo.add %2926, %2927 : tensor<1x384x128xf32>
    %2929 = stablehlo.reshape %2928 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2930 = stablehlo.dot %2929, %280 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2931 = stablehlo.broadcast_in_dim %279, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2932 = stablehlo.add %2930, %2931 : tensor<384x512xf32>
    %2933 = stablehlo.reshape %2932 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2934 = stablehlo.maximum %2933, %9 : tensor<1x384x512xf32>
    %2935 = stablehlo.reshape %2934 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2936 = stablehlo.dot %2935, %284 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2937 = stablehlo.broadcast_in_dim %283, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2938 = stablehlo.add %2936, %2937 : tensor<384x128xf32>
    %2939 = stablehlo.reshape %2938 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2940 = stablehlo.add %2939, %2928 : tensor<1x384x128xf32>
    %2941 = stablehlo.broadcast_in_dim %282, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2942 = stablehlo.multiply %2940, %2941 : tensor<1x384x128xf32>
    %2943 = stablehlo.broadcast_in_dim %281, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2944 = stablehlo.add %2942, %2943 : tensor<1x384x128xf32>
    %2945 = stablehlo.reshape %2944 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2946 = stablehlo.dot %2945, %286 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2947 = stablehlo.broadcast_in_dim %285, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2948 = stablehlo.add %2946, %2947 : tensor<384x512xf32>
    %2949 = stablehlo.reshape %2948 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2950 = stablehlo.maximum %2949, %9 : tensor<1x384x512xf32>
    %2951 = stablehlo.reshape %2950 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2952 = stablehlo.dot %2951, %294 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2953 = stablehlo.broadcast_in_dim %293, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2954 = stablehlo.add %2952, %2953 : tensor<384x128xf32>
    %2955 = stablehlo.reshape %2954 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2956 = stablehlo.add %2955, %2944 : tensor<1x384x128xf32>
    %2957 = stablehlo.broadcast_in_dim %288, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2958 = stablehlo.multiply %2956, %2957 : tensor<1x384x128xf32>
    %2959 = stablehlo.broadcast_in_dim %287, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2960 = stablehlo.add %2958, %2959 : tensor<1x384x128xf32>
    %2961 = stablehlo.reshape %2960 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2962 = stablehlo.dot %2961, %292 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2963 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %2964 = stablehlo.add %2962, %2963 : tensor<384x512xf32>
    %2965 = stablehlo.reshape %2964 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2966 = stablehlo.add %2965, %2840 : tensor<1x384x512xf32>
    %2967 = stablehlo.broadcast_in_dim %290, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2968 = stablehlo.multiply %2966, %2967 : tensor<1x384x512xf32>
    %2969 = stablehlo.broadcast_in_dim %289, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2970 = stablehlo.add %2968, %2969 : tensor<1x384x512xf32>
    %2971 = stablehlo.reshape %2970 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2972 = stablehlo.dot %2971, %304 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2973 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2974 = stablehlo.add %2972, %2973 : tensor<384x128xf32>
    %2975 = stablehlo.reshape %2974 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2976 = stablehlo.transpose %2975, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2977 = stablehlo.dot %2971, %308 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2978 = stablehlo.reshape %2977 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2979 = stablehlo.broadcast_in_dim %307, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2980 = stablehlo.add %2978, %2979 : tensor<1x384x128xf32>
    %2981 = stablehlo.broadcast_in_dim %306, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2982 = stablehlo.multiply %2980, %2981 : tensor<1x384x128xf32>
    %2983 = stablehlo.broadcast_in_dim %305, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2984 = stablehlo.add %2982, %2983 : tensor<1x384x128xf32>
    %2985 = stablehlo.reshape %2984 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2986 = stablehlo.dot %2985, %300 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2987 = stablehlo.broadcast_in_dim %299, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2988 = stablehlo.add %2986, %2987 : tensor<384x128xf32>
    %2989 = stablehlo.reshape %2988 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2990 = stablehlo.transpose %2989, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2991 = stablehlo.dot %2985, %302 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2992 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %2993 = stablehlo.add %2991, %2992 : tensor<384x128xf32>
    %2994 = stablehlo.reshape %2993 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2995 = stablehlo.transpose %2994, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2996 = stablehlo.dot_general %2995, %2990, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2997 = stablehlo.multiply %2996, %4 : tensor<1x4x384x384xf32>
    %2998 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2999 = stablehlo.add %2997, %2998 : tensor<1x4x384x384xf32>
    %3000 = stablehlo.reduce(%2999 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3001 = stablehlo.broadcast_in_dim %3000, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3002 = stablehlo.subtract %2999, %3001 : tensor<1x4x384x384xf32>
    %3003 = stablehlo.exponential %3002 : tensor<1x4x384x384xf32>
    %3004 = stablehlo.reduce(%3003 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3005 = stablehlo.broadcast_in_dim %3004, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3006 = stablehlo.divide %3003, %3005 : tensor<1x4x384x384xf32>
    %3007 = stablehlo.dot_general %3006, %2976, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3008 = stablehlo.transpose %3007, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3009 = stablehlo.reshape %3008 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3010 = stablehlo.dot %3009, %298 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3011 = stablehlo.broadcast_in_dim %297, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3012 = stablehlo.add %3010, %3011 : tensor<384x128xf32>
    %3013 = stablehlo.reshape %3012 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3014 = stablehlo.dot %2971, %312 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3015 = stablehlo.broadcast_in_dim %311, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3016 = stablehlo.add %3014, %3015 : tensor<384x128xf32>
    %3017 = stablehlo.reshape %3016 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3018 = stablehlo.broadcast_in_dim %310, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3019 = stablehlo.multiply %3017, %3018 : tensor<1x384x128xf32>
    %3020 = stablehlo.broadcast_in_dim %309, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3021 = stablehlo.add %3019, %3020 : tensor<1x384x128xf32>
    %3022 = stablehlo.add %3013, %3021 : tensor<1x384x128xf32>
    %3023 = stablehlo.broadcast_in_dim %296, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3024 = stablehlo.multiply %3022, %3023 : tensor<1x384x128xf32>
    %3025 = stablehlo.broadcast_in_dim %295, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3026 = stablehlo.add %3024, %3025 : tensor<1x384x128xf32>
    %3027 = stablehlo.reshape %3026 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3028 = stablehlo.dot %3027, %314 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3029 = stablehlo.broadcast_in_dim %313, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3030 = stablehlo.add %3028, %3029 : tensor<384x512xf32>
    %3031 = stablehlo.reshape %3030 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3032 = stablehlo.maximum %3031, %9 : tensor<1x384x512xf32>
    %3033 = stablehlo.reshape %3032 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3034 = stablehlo.dot %3033, %318 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3035 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3036 = stablehlo.add %3034, %3035 : tensor<384x128xf32>
    %3037 = stablehlo.reshape %3036 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3038 = stablehlo.add %3037, %3026 : tensor<1x384x128xf32>
    %3039 = stablehlo.broadcast_in_dim %316, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3040 = stablehlo.multiply %3038, %3039 : tensor<1x384x128xf32>
    %3041 = stablehlo.broadcast_in_dim %315, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3042 = stablehlo.add %3040, %3041 : tensor<1x384x128xf32>
    %3043 = stablehlo.reshape %3042 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3044 = stablehlo.dot %3043, %320 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3045 = stablehlo.broadcast_in_dim %319, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3046 = stablehlo.add %3044, %3045 : tensor<384x512xf32>
    %3047 = stablehlo.reshape %3046 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3048 = stablehlo.maximum %3047, %9 : tensor<1x384x512xf32>
    %3049 = stablehlo.reshape %3048 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3050 = stablehlo.dot %3049, %324 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3051 = stablehlo.broadcast_in_dim %323, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3052 = stablehlo.add %3050, %3051 : tensor<384x128xf32>
    %3053 = stablehlo.reshape %3052 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3054 = stablehlo.add %3053, %3042 : tensor<1x384x128xf32>
    %3055 = stablehlo.broadcast_in_dim %322, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3056 = stablehlo.multiply %3054, %3055 : tensor<1x384x128xf32>
    %3057 = stablehlo.broadcast_in_dim %321, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3058 = stablehlo.add %3056, %3057 : tensor<1x384x128xf32>
    %3059 = stablehlo.reshape %3058 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3060 = stablehlo.dot %3059, %326 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3061 = stablehlo.broadcast_in_dim %325, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3062 = stablehlo.add %3060, %3061 : tensor<384x512xf32>
    %3063 = stablehlo.reshape %3062 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3064 = stablehlo.maximum %3063, %9 : tensor<1x384x512xf32>
    %3065 = stablehlo.reshape %3064 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3066 = stablehlo.dot %3065, %330 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3067 = stablehlo.broadcast_in_dim %329, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3068 = stablehlo.add %3066, %3067 : tensor<384x128xf32>
    %3069 = stablehlo.reshape %3068 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3070 = stablehlo.add %3069, %3058 : tensor<1x384x128xf32>
    %3071 = stablehlo.broadcast_in_dim %328, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3072 = stablehlo.multiply %3070, %3071 : tensor<1x384x128xf32>
    %3073 = stablehlo.broadcast_in_dim %327, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3074 = stablehlo.add %3072, %3073 : tensor<1x384x128xf32>
    %3075 = stablehlo.reshape %3074 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3076 = stablehlo.dot %3075, %332 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3077 = stablehlo.broadcast_in_dim %331, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3078 = stablehlo.add %3076, %3077 : tensor<384x512xf32>
    %3079 = stablehlo.reshape %3078 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3080 = stablehlo.maximum %3079, %9 : tensor<1x384x512xf32>
    %3081 = stablehlo.reshape %3080 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3082 = stablehlo.dot %3081, %340 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3083 = stablehlo.broadcast_in_dim %339, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3084 = stablehlo.add %3082, %3083 : tensor<384x128xf32>
    %3085 = stablehlo.reshape %3084 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3086 = stablehlo.add %3085, %3074 : tensor<1x384x128xf32>
    %3087 = stablehlo.broadcast_in_dim %334, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3088 = stablehlo.multiply %3086, %3087 : tensor<1x384x128xf32>
    %3089 = stablehlo.broadcast_in_dim %333, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3090 = stablehlo.add %3088, %3089 : tensor<1x384x128xf32>
    %3091 = stablehlo.reshape %3090 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3092 = stablehlo.dot %3091, %338 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3093 = stablehlo.broadcast_in_dim %337, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3094 = stablehlo.add %3092, %3093 : tensor<384x512xf32>
    %3095 = stablehlo.reshape %3094 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3096 = stablehlo.add %3095, %2970 : tensor<1x384x512xf32>
    %3097 = stablehlo.broadcast_in_dim %336, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3098 = stablehlo.multiply %3096, %3097 : tensor<1x384x512xf32>
    %3099 = stablehlo.broadcast_in_dim %335, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3100 = stablehlo.add %3098, %3099 : tensor<1x384x512xf32>
    %3101 = stablehlo.reshape %3100 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3102 = stablehlo.dot %3101, %350 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3103 = stablehlo.broadcast_in_dim %349, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3104 = stablehlo.add %3102, %3103 : tensor<384x128xf32>
    %3105 = stablehlo.reshape %3104 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3106 = stablehlo.transpose %3105, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3107 = stablehlo.dot %3101, %354 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3108 = stablehlo.reshape %3107 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3109 = stablehlo.broadcast_in_dim %353, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3110 = stablehlo.add %3108, %3109 : tensor<1x384x128xf32>
    %3111 = stablehlo.broadcast_in_dim %352, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3112 = stablehlo.multiply %3110, %3111 : tensor<1x384x128xf32>
    %3113 = stablehlo.broadcast_in_dim %351, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3114 = stablehlo.add %3112, %3113 : tensor<1x384x128xf32>
    %3115 = stablehlo.reshape %3114 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3116 = stablehlo.dot %3115, %346 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3117 = stablehlo.broadcast_in_dim %345, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3118 = stablehlo.add %3116, %3117 : tensor<384x128xf32>
    %3119 = stablehlo.reshape %3118 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3120 = stablehlo.transpose %3119, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3121 = stablehlo.dot %3115, %348 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3122 = stablehlo.broadcast_in_dim %347, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3123 = stablehlo.add %3121, %3122 : tensor<384x128xf32>
    %3124 = stablehlo.reshape %3123 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3125 = stablehlo.transpose %3124, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3126 = stablehlo.dot_general %3125, %3120, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3127 = stablehlo.multiply %3126, %4 : tensor<1x4x384x384xf32>
    %3128 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3129 = stablehlo.add %3127, %3128 : tensor<1x4x384x384xf32>
    %3130 = stablehlo.reduce(%3129 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3131 = stablehlo.broadcast_in_dim %3130, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3132 = stablehlo.subtract %3129, %3131 : tensor<1x4x384x384xf32>
    %3133 = stablehlo.exponential %3132 : tensor<1x4x384x384xf32>
    %3134 = stablehlo.reduce(%3133 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3135 = stablehlo.broadcast_in_dim %3134, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3136 = stablehlo.divide %3133, %3135 : tensor<1x4x384x384xf32>
    %3137 = stablehlo.dot_general %3136, %3106, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3138 = stablehlo.transpose %3137, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3139 = stablehlo.reshape %3138 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3140 = stablehlo.dot %3139, %344 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3141 = stablehlo.broadcast_in_dim %343, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3142 = stablehlo.add %3140, %3141 : tensor<384x128xf32>
    %3143 = stablehlo.reshape %3142 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3144 = stablehlo.dot %3101, %358 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3145 = stablehlo.broadcast_in_dim %357, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3146 = stablehlo.add %3144, %3145 : tensor<384x128xf32>
    %3147 = stablehlo.reshape %3146 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3148 = stablehlo.broadcast_in_dim %356, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3149 = stablehlo.multiply %3147, %3148 : tensor<1x384x128xf32>
    %3150 = stablehlo.broadcast_in_dim %355, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3151 = stablehlo.add %3149, %3150 : tensor<1x384x128xf32>
    %3152 = stablehlo.add %3143, %3151 : tensor<1x384x128xf32>
    %3153 = stablehlo.broadcast_in_dim %342, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3154 = stablehlo.multiply %3152, %3153 : tensor<1x384x128xf32>
    %3155 = stablehlo.broadcast_in_dim %341, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3156 = stablehlo.add %3154, %3155 : tensor<1x384x128xf32>
    %3157 = stablehlo.reshape %3156 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3158 = stablehlo.dot %3157, %360 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3159 = stablehlo.broadcast_in_dim %359, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3160 = stablehlo.add %3158, %3159 : tensor<384x512xf32>
    %3161 = stablehlo.reshape %3160 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3162 = stablehlo.maximum %3161, %9 : tensor<1x384x512xf32>
    %3163 = stablehlo.reshape %3162 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3164 = stablehlo.dot %3163, %364 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3165 = stablehlo.broadcast_in_dim %363, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3166 = stablehlo.add %3164, %3165 : tensor<384x128xf32>
    %3167 = stablehlo.reshape %3166 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3168 = stablehlo.add %3167, %3156 : tensor<1x384x128xf32>
    %3169 = stablehlo.broadcast_in_dim %362, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3170 = stablehlo.multiply %3168, %3169 : tensor<1x384x128xf32>
    %3171 = stablehlo.broadcast_in_dim %361, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3172 = stablehlo.add %3170, %3171 : tensor<1x384x128xf32>
    %3173 = stablehlo.reshape %3172 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3174 = stablehlo.dot %3173, %366 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3175 = stablehlo.broadcast_in_dim %365, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3176 = stablehlo.add %3174, %3175 : tensor<384x512xf32>
    %3177 = stablehlo.reshape %3176 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3178 = stablehlo.maximum %3177, %9 : tensor<1x384x512xf32>
    %3179 = stablehlo.reshape %3178 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3180 = stablehlo.dot %3179, %370 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3181 = stablehlo.broadcast_in_dim %369, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3182 = stablehlo.add %3180, %3181 : tensor<384x128xf32>
    %3183 = stablehlo.reshape %3182 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3184 = stablehlo.add %3183, %3172 : tensor<1x384x128xf32>
    %3185 = stablehlo.broadcast_in_dim %368, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3186 = stablehlo.multiply %3184, %3185 : tensor<1x384x128xf32>
    %3187 = stablehlo.broadcast_in_dim %367, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3188 = stablehlo.add %3186, %3187 : tensor<1x384x128xf32>
    %3189 = stablehlo.reshape %3188 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3190 = stablehlo.dot %3189, %372 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3191 = stablehlo.broadcast_in_dim %371, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3192 = stablehlo.add %3190, %3191 : tensor<384x512xf32>
    %3193 = stablehlo.reshape %3192 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3194 = stablehlo.maximum %3193, %9 : tensor<1x384x512xf32>
    %3195 = stablehlo.reshape %3194 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3196 = stablehlo.dot %3195, %376 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3197 = stablehlo.broadcast_in_dim %375, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3198 = stablehlo.add %3196, %3197 : tensor<384x128xf32>
    %3199 = stablehlo.reshape %3198 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3200 = stablehlo.add %3199, %3188 : tensor<1x384x128xf32>
    %3201 = stablehlo.broadcast_in_dim %374, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3202 = stablehlo.multiply %3200, %3201 : tensor<1x384x128xf32>
    %3203 = stablehlo.broadcast_in_dim %373, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3204 = stablehlo.add %3202, %3203 : tensor<1x384x128xf32>
    %3205 = stablehlo.reshape %3204 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3206 = stablehlo.dot %3205, %378 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3207 = stablehlo.broadcast_in_dim %377, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3208 = stablehlo.add %3206, %3207 : tensor<384x512xf32>
    %3209 = stablehlo.reshape %3208 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3210 = stablehlo.maximum %3209, %9 : tensor<1x384x512xf32>
    %3211 = stablehlo.reshape %3210 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3212 = stablehlo.dot %3211, %386 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3213 = stablehlo.broadcast_in_dim %385, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3214 = stablehlo.add %3212, %3213 : tensor<384x128xf32>
    %3215 = stablehlo.reshape %3214 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3216 = stablehlo.add %3215, %3204 : tensor<1x384x128xf32>
    %3217 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3218 = stablehlo.multiply %3216, %3217 : tensor<1x384x128xf32>
    %3219 = stablehlo.broadcast_in_dim %379, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3220 = stablehlo.add %3218, %3219 : tensor<1x384x128xf32>
    %3221 = stablehlo.reshape %3220 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3222 = stablehlo.dot %3221, %384 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3223 = stablehlo.broadcast_in_dim %383, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3224 = stablehlo.add %3222, %3223 : tensor<384x512xf32>
    %3225 = stablehlo.reshape %3224 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3226 = stablehlo.add %3225, %3100 : tensor<1x384x512xf32>
    %3227 = stablehlo.broadcast_in_dim %382, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3228 = stablehlo.multiply %3226, %3227 : tensor<1x384x512xf32>
    %3229 = stablehlo.broadcast_in_dim %381, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3230 = stablehlo.add %3228, %3229 : tensor<1x384x512xf32>
    %3231 = stablehlo.reshape %3230 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3232 = stablehlo.dot %3231, %396 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3233 = stablehlo.broadcast_in_dim %395, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3234 = stablehlo.add %3232, %3233 : tensor<384x128xf32>
    %3235 = stablehlo.reshape %3234 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3236 = stablehlo.transpose %3235, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3237 = stablehlo.dot %3231, %400 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3238 = stablehlo.reshape %3237 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3239 = stablehlo.broadcast_in_dim %399, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3240 = stablehlo.add %3238, %3239 : tensor<1x384x128xf32>
    %3241 = stablehlo.broadcast_in_dim %398, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3242 = stablehlo.multiply %3240, %3241 : tensor<1x384x128xf32>
    %3243 = stablehlo.broadcast_in_dim %397, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3244 = stablehlo.add %3242, %3243 : tensor<1x384x128xf32>
    %3245 = stablehlo.reshape %3244 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3246 = stablehlo.dot %3245, %392 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3247 = stablehlo.broadcast_in_dim %391, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3248 = stablehlo.add %3246, %3247 : tensor<384x128xf32>
    %3249 = stablehlo.reshape %3248 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3250 = stablehlo.transpose %3249, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3251 = stablehlo.dot %3245, %394 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3252 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3253 = stablehlo.add %3251, %3252 : tensor<384x128xf32>
    %3254 = stablehlo.reshape %3253 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3255 = stablehlo.transpose %3254, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3256 = stablehlo.dot_general %3255, %3250, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3257 = stablehlo.multiply %3256, %4 : tensor<1x4x384x384xf32>
    %3258 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3259 = stablehlo.add %3257, %3258 : tensor<1x4x384x384xf32>
    %3260 = stablehlo.reduce(%3259 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3261 = stablehlo.broadcast_in_dim %3260, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3262 = stablehlo.subtract %3259, %3261 : tensor<1x4x384x384xf32>
    %3263 = stablehlo.exponential %3262 : tensor<1x4x384x384xf32>
    %3264 = stablehlo.reduce(%3263 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3265 = stablehlo.broadcast_in_dim %3264, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3266 = stablehlo.divide %3263, %3265 : tensor<1x4x384x384xf32>
    %3267 = stablehlo.dot_general %3266, %3236, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3268 = stablehlo.transpose %3267, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3269 = stablehlo.reshape %3268 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3270 = stablehlo.dot %3269, %390 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3271 = stablehlo.broadcast_in_dim %389, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3272 = stablehlo.add %3270, %3271 : tensor<384x128xf32>
    %3273 = stablehlo.reshape %3272 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3274 = stablehlo.dot %3231, %404 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3275 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3276 = stablehlo.add %3274, %3275 : tensor<384x128xf32>
    %3277 = stablehlo.reshape %3276 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3278 = stablehlo.broadcast_in_dim %402, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3279 = stablehlo.multiply %3277, %3278 : tensor<1x384x128xf32>
    %3280 = stablehlo.broadcast_in_dim %401, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3281 = stablehlo.add %3279, %3280 : tensor<1x384x128xf32>
    %3282 = stablehlo.add %3273, %3281 : tensor<1x384x128xf32>
    %3283 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3284 = stablehlo.multiply %3282, %3283 : tensor<1x384x128xf32>
    %3285 = stablehlo.broadcast_in_dim %387, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3286 = stablehlo.add %3284, %3285 : tensor<1x384x128xf32>
    %3287 = stablehlo.reshape %3286 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3288 = stablehlo.dot %3287, %406 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3289 = stablehlo.broadcast_in_dim %405, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3290 = stablehlo.add %3288, %3289 : tensor<384x512xf32>
    %3291 = stablehlo.reshape %3290 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3292 = stablehlo.maximum %3291, %9 : tensor<1x384x512xf32>
    %3293 = stablehlo.reshape %3292 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3294 = stablehlo.dot %3293, %410 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3295 = stablehlo.broadcast_in_dim %409, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3296 = stablehlo.add %3294, %3295 : tensor<384x128xf32>
    %3297 = stablehlo.reshape %3296 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3298 = stablehlo.add %3297, %3286 : tensor<1x384x128xf32>
    %3299 = stablehlo.broadcast_in_dim %408, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3300 = stablehlo.multiply %3298, %3299 : tensor<1x384x128xf32>
    %3301 = stablehlo.broadcast_in_dim %407, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3302 = stablehlo.add %3300, %3301 : tensor<1x384x128xf32>
    %3303 = stablehlo.reshape %3302 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3304 = stablehlo.dot %3303, %412 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3305 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3306 = stablehlo.add %3304, %3305 : tensor<384x512xf32>
    %3307 = stablehlo.reshape %3306 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3308 = stablehlo.maximum %3307, %9 : tensor<1x384x512xf32>
    %3309 = stablehlo.reshape %3308 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3310 = stablehlo.dot %3309, %416 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3311 = stablehlo.broadcast_in_dim %415, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3312 = stablehlo.add %3310, %3311 : tensor<384x128xf32>
    %3313 = stablehlo.reshape %3312 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3314 = stablehlo.add %3313, %3302 : tensor<1x384x128xf32>
    %3315 = stablehlo.broadcast_in_dim %414, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3316 = stablehlo.multiply %3314, %3315 : tensor<1x384x128xf32>
    %3317 = stablehlo.broadcast_in_dim %413, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3318 = stablehlo.add %3316, %3317 : tensor<1x384x128xf32>
    %3319 = stablehlo.reshape %3318 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3320 = stablehlo.dot %3319, %418 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3321 = stablehlo.broadcast_in_dim %417, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3322 = stablehlo.add %3320, %3321 : tensor<384x512xf32>
    %3323 = stablehlo.reshape %3322 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3324 = stablehlo.maximum %3323, %9 : tensor<1x384x512xf32>
    %3325 = stablehlo.reshape %3324 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3326 = stablehlo.dot %3325, %422 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3327 = stablehlo.broadcast_in_dim %421, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3328 = stablehlo.add %3326, %3327 : tensor<384x128xf32>
    %3329 = stablehlo.reshape %3328 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3330 = stablehlo.add %3329, %3318 : tensor<1x384x128xf32>
    %3331 = stablehlo.broadcast_in_dim %420, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3332 = stablehlo.multiply %3330, %3331 : tensor<1x384x128xf32>
    %3333 = stablehlo.broadcast_in_dim %419, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3334 = stablehlo.add %3332, %3333 : tensor<1x384x128xf32>
    %3335 = stablehlo.reshape %3334 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3336 = stablehlo.dot %3335, %424 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3337 = stablehlo.broadcast_in_dim %423, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3338 = stablehlo.add %3336, %3337 : tensor<384x512xf32>
    %3339 = stablehlo.reshape %3338 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3340 = stablehlo.maximum %3339, %9 : tensor<1x384x512xf32>
    %3341 = stablehlo.reshape %3340 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3342 = stablehlo.dot %3341, %432 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3343 = stablehlo.broadcast_in_dim %431, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3344 = stablehlo.add %3342, %3343 : tensor<384x128xf32>
    %3345 = stablehlo.reshape %3344 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3346 = stablehlo.add %3345, %3334 : tensor<1x384x128xf32>
    %3347 = stablehlo.broadcast_in_dim %426, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3348 = stablehlo.multiply %3346, %3347 : tensor<1x384x128xf32>
    %3349 = stablehlo.broadcast_in_dim %425, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3350 = stablehlo.add %3348, %3349 : tensor<1x384x128xf32>
    %3351 = stablehlo.reshape %3350 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3352 = stablehlo.dot %3351, %430 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3353 = stablehlo.broadcast_in_dim %429, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3354 = stablehlo.add %3352, %3353 : tensor<384x512xf32>
    %3355 = stablehlo.reshape %3354 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3356 = stablehlo.add %3355, %3230 : tensor<1x384x512xf32>
    %3357 = stablehlo.broadcast_in_dim %428, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3358 = stablehlo.multiply %3356, %3357 : tensor<1x384x512xf32>
    %3359 = stablehlo.broadcast_in_dim %427, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3360 = stablehlo.add %3358, %3359 : tensor<1x384x512xf32>
    %3361 = stablehlo.reshape %3360 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3362 = stablehlo.dot %3361, %442 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3363 = stablehlo.broadcast_in_dim %441, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3364 = stablehlo.add %3362, %3363 : tensor<384x128xf32>
    %3365 = stablehlo.reshape %3364 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3366 = stablehlo.transpose %3365, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3367 = stablehlo.dot %3361, %446 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3368 = stablehlo.reshape %3367 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3369 = stablehlo.broadcast_in_dim %445, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3370 = stablehlo.add %3368, %3369 : tensor<1x384x128xf32>
    %3371 = stablehlo.broadcast_in_dim %444, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3372 = stablehlo.multiply %3370, %3371 : tensor<1x384x128xf32>
    %3373 = stablehlo.broadcast_in_dim %443, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3374 = stablehlo.add %3372, %3373 : tensor<1x384x128xf32>
    %3375 = stablehlo.reshape %3374 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3376 = stablehlo.dot %3375, %438 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3377 = stablehlo.broadcast_in_dim %437, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3378 = stablehlo.add %3376, %3377 : tensor<384x128xf32>
    %3379 = stablehlo.reshape %3378 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3380 = stablehlo.transpose %3379, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3381 = stablehlo.dot %3375, %440 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3382 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3383 = stablehlo.add %3381, %3382 : tensor<384x128xf32>
    %3384 = stablehlo.reshape %3383 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3385 = stablehlo.transpose %3384, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3386 = stablehlo.dot_general %3385, %3380, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3387 = stablehlo.multiply %3386, %4 : tensor<1x4x384x384xf32>
    %3388 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3389 = stablehlo.add %3387, %3388 : tensor<1x4x384x384xf32>
    %3390 = stablehlo.reduce(%3389 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3391 = stablehlo.broadcast_in_dim %3390, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3392 = stablehlo.subtract %3389, %3391 : tensor<1x4x384x384xf32>
    %3393 = stablehlo.exponential %3392 : tensor<1x4x384x384xf32>
    %3394 = stablehlo.reduce(%3393 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3395 = stablehlo.broadcast_in_dim %3394, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3396 = stablehlo.divide %3393, %3395 : tensor<1x4x384x384xf32>
    %3397 = stablehlo.dot_general %3396, %3366, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3398 = stablehlo.transpose %3397, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3399 = stablehlo.reshape %3398 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3400 = stablehlo.dot %3399, %436 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3401 = stablehlo.broadcast_in_dim %435, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3402 = stablehlo.add %3400, %3401 : tensor<384x128xf32>
    %3403 = stablehlo.reshape %3402 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3404 = stablehlo.dot %3361, %450 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3405 = stablehlo.broadcast_in_dim %449, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3406 = stablehlo.add %3404, %3405 : tensor<384x128xf32>
    %3407 = stablehlo.reshape %3406 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3408 = stablehlo.broadcast_in_dim %448, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3409 = stablehlo.multiply %3407, %3408 : tensor<1x384x128xf32>
    %3410 = stablehlo.broadcast_in_dim %447, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3411 = stablehlo.add %3409, %3410 : tensor<1x384x128xf32>
    %3412 = stablehlo.add %3403, %3411 : tensor<1x384x128xf32>
    %3413 = stablehlo.broadcast_in_dim %434, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3414 = stablehlo.multiply %3412, %3413 : tensor<1x384x128xf32>
    %3415 = stablehlo.broadcast_in_dim %433, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3416 = stablehlo.add %3414, %3415 : tensor<1x384x128xf32>
    %3417 = stablehlo.reshape %3416 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3418 = stablehlo.dot %3417, %452 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3419 = stablehlo.broadcast_in_dim %451, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3420 = stablehlo.add %3418, %3419 : tensor<384x512xf32>
    %3421 = stablehlo.reshape %3420 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3422 = stablehlo.maximum %3421, %9 : tensor<1x384x512xf32>
    %3423 = stablehlo.reshape %3422 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3424 = stablehlo.dot %3423, %456 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3425 = stablehlo.broadcast_in_dim %455, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3426 = stablehlo.add %3424, %3425 : tensor<384x128xf32>
    %3427 = stablehlo.reshape %3426 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3428 = stablehlo.add %3427, %3416 : tensor<1x384x128xf32>
    %3429 = stablehlo.broadcast_in_dim %454, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3430 = stablehlo.multiply %3428, %3429 : tensor<1x384x128xf32>
    %3431 = stablehlo.broadcast_in_dim %453, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3432 = stablehlo.add %3430, %3431 : tensor<1x384x128xf32>
    %3433 = stablehlo.reshape %3432 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3434 = stablehlo.dot %3433, %458 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3435 = stablehlo.broadcast_in_dim %457, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3436 = stablehlo.add %3434, %3435 : tensor<384x512xf32>
    %3437 = stablehlo.reshape %3436 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3438 = stablehlo.maximum %3437, %9 : tensor<1x384x512xf32>
    %3439 = stablehlo.reshape %3438 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3440 = stablehlo.dot %3439, %462 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3441 = stablehlo.broadcast_in_dim %461, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3442 = stablehlo.add %3440, %3441 : tensor<384x128xf32>
    %3443 = stablehlo.reshape %3442 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3444 = stablehlo.add %3443, %3432 : tensor<1x384x128xf32>
    %3445 = stablehlo.broadcast_in_dim %460, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3446 = stablehlo.multiply %3444, %3445 : tensor<1x384x128xf32>
    %3447 = stablehlo.broadcast_in_dim %459, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3448 = stablehlo.add %3446, %3447 : tensor<1x384x128xf32>
    %3449 = stablehlo.reshape %3448 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3450 = stablehlo.dot %3449, %464 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3451 = stablehlo.broadcast_in_dim %463, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3452 = stablehlo.add %3450, %3451 : tensor<384x512xf32>
    %3453 = stablehlo.reshape %3452 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3454 = stablehlo.maximum %3453, %9 : tensor<1x384x512xf32>
    %3455 = stablehlo.reshape %3454 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3456 = stablehlo.dot %3455, %468 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3457 = stablehlo.broadcast_in_dim %467, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3458 = stablehlo.add %3456, %3457 : tensor<384x128xf32>
    %3459 = stablehlo.reshape %3458 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3460 = stablehlo.add %3459, %3448 : tensor<1x384x128xf32>
    %3461 = stablehlo.broadcast_in_dim %466, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3462 = stablehlo.multiply %3460, %3461 : tensor<1x384x128xf32>
    %3463 = stablehlo.broadcast_in_dim %465, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3464 = stablehlo.add %3462, %3463 : tensor<1x384x128xf32>
    %3465 = stablehlo.reshape %3464 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3466 = stablehlo.dot %3465, %470 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3467 = stablehlo.broadcast_in_dim %469, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3468 = stablehlo.add %3466, %3467 : tensor<384x512xf32>
    %3469 = stablehlo.reshape %3468 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3470 = stablehlo.maximum %3469, %9 : tensor<1x384x512xf32>
    %3471 = stablehlo.reshape %3470 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3472 = stablehlo.dot %3471, %478 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3473 = stablehlo.broadcast_in_dim %477, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3474 = stablehlo.add %3472, %3473 : tensor<384x128xf32>
    %3475 = stablehlo.reshape %3474 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3476 = stablehlo.add %3475, %3464 : tensor<1x384x128xf32>
    %3477 = stablehlo.broadcast_in_dim %472, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3478 = stablehlo.multiply %3476, %3477 : tensor<1x384x128xf32>
    %3479 = stablehlo.broadcast_in_dim %471, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3480 = stablehlo.add %3478, %3479 : tensor<1x384x128xf32>
    %3481 = stablehlo.reshape %3480 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3482 = stablehlo.dot %3481, %476 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3483 = stablehlo.broadcast_in_dim %475, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3484 = stablehlo.add %3482, %3483 : tensor<384x512xf32>
    %3485 = stablehlo.reshape %3484 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3486 = stablehlo.add %3485, %3360 : tensor<1x384x512xf32>
    %3487 = stablehlo.broadcast_in_dim %474, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3488 = stablehlo.multiply %3486, %3487 : tensor<1x384x512xf32>
    %3489 = stablehlo.broadcast_in_dim %473, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3490 = stablehlo.add %3488, %3489 : tensor<1x384x512xf32>
    %3491 = stablehlo.reshape %3490 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3492 = stablehlo.dot %3491, %488 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3493 = stablehlo.broadcast_in_dim %487, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3494 = stablehlo.add %3492, %3493 : tensor<384x128xf32>
    %3495 = stablehlo.reshape %3494 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3496 = stablehlo.transpose %3495, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3497 = stablehlo.dot %3491, %492 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3498 = stablehlo.reshape %3497 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3499 = stablehlo.broadcast_in_dim %491, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3500 = stablehlo.add %3498, %3499 : tensor<1x384x128xf32>
    %3501 = stablehlo.broadcast_in_dim %490, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3502 = stablehlo.multiply %3500, %3501 : tensor<1x384x128xf32>
    %3503 = stablehlo.broadcast_in_dim %489, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3504 = stablehlo.add %3502, %3503 : tensor<1x384x128xf32>
    %3505 = stablehlo.reshape %3504 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3506 = stablehlo.dot %3505, %484 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3507 = stablehlo.broadcast_in_dim %483, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3508 = stablehlo.add %3506, %3507 : tensor<384x128xf32>
    %3509 = stablehlo.reshape %3508 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3510 = stablehlo.transpose %3509, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3511 = stablehlo.dot %3505, %486 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3512 = stablehlo.broadcast_in_dim %485, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3513 = stablehlo.add %3511, %3512 : tensor<384x128xf32>
    %3514 = stablehlo.reshape %3513 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3515 = stablehlo.transpose %3514, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3516 = stablehlo.dot_general %3515, %3510, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3517 = stablehlo.multiply %3516, %4 : tensor<1x4x384x384xf32>
    %3518 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3519 = stablehlo.add %3517, %3518 : tensor<1x4x384x384xf32>
    %3520 = stablehlo.reduce(%3519 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3521 = stablehlo.broadcast_in_dim %3520, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3522 = stablehlo.subtract %3519, %3521 : tensor<1x4x384x384xf32>
    %3523 = stablehlo.exponential %3522 : tensor<1x4x384x384xf32>
    %3524 = stablehlo.reduce(%3523 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3525 = stablehlo.broadcast_in_dim %3524, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3526 = stablehlo.divide %3523, %3525 : tensor<1x4x384x384xf32>
    %3527 = stablehlo.dot_general %3526, %3496, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3528 = stablehlo.transpose %3527, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3529 = stablehlo.reshape %3528 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3530 = stablehlo.dot %3529, %482 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3531 = stablehlo.broadcast_in_dim %481, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3532 = stablehlo.add %3530, %3531 : tensor<384x128xf32>
    %3533 = stablehlo.reshape %3532 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3534 = stablehlo.dot %3491, %496 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3535 = stablehlo.broadcast_in_dim %495, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3536 = stablehlo.add %3534, %3535 : tensor<384x128xf32>
    %3537 = stablehlo.reshape %3536 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3538 = stablehlo.broadcast_in_dim %494, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3539 = stablehlo.multiply %3537, %3538 : tensor<1x384x128xf32>
    %3540 = stablehlo.broadcast_in_dim %493, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3541 = stablehlo.add %3539, %3540 : tensor<1x384x128xf32>
    %3542 = stablehlo.add %3533, %3541 : tensor<1x384x128xf32>
    %3543 = stablehlo.broadcast_in_dim %480, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3544 = stablehlo.multiply %3542, %3543 : tensor<1x384x128xf32>
    %3545 = stablehlo.broadcast_in_dim %479, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3546 = stablehlo.add %3544, %3545 : tensor<1x384x128xf32>
    %3547 = stablehlo.reshape %3546 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3548 = stablehlo.dot %3547, %498 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3549 = stablehlo.broadcast_in_dim %497, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3550 = stablehlo.add %3548, %3549 : tensor<384x512xf32>
    %3551 = stablehlo.reshape %3550 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3552 = stablehlo.maximum %3551, %9 : tensor<1x384x512xf32>
    %3553 = stablehlo.reshape %3552 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3554 = stablehlo.dot %3553, %502 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3555 = stablehlo.broadcast_in_dim %501, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3556 = stablehlo.add %3554, %3555 : tensor<384x128xf32>
    %3557 = stablehlo.reshape %3556 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3558 = stablehlo.add %3557, %3546 : tensor<1x384x128xf32>
    %3559 = stablehlo.broadcast_in_dim %500, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3560 = stablehlo.multiply %3558, %3559 : tensor<1x384x128xf32>
    %3561 = stablehlo.broadcast_in_dim %499, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3562 = stablehlo.add %3560, %3561 : tensor<1x384x128xf32>
    %3563 = stablehlo.reshape %3562 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3564 = stablehlo.dot %3563, %504 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3565 = stablehlo.broadcast_in_dim %503, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3566 = stablehlo.add %3564, %3565 : tensor<384x512xf32>
    %3567 = stablehlo.reshape %3566 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3568 = stablehlo.maximum %3567, %9 : tensor<1x384x512xf32>
    %3569 = stablehlo.reshape %3568 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3570 = stablehlo.dot %3569, %508 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3571 = stablehlo.broadcast_in_dim %507, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3572 = stablehlo.add %3570, %3571 : tensor<384x128xf32>
    %3573 = stablehlo.reshape %3572 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3574 = stablehlo.add %3573, %3562 : tensor<1x384x128xf32>
    %3575 = stablehlo.broadcast_in_dim %506, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3576 = stablehlo.multiply %3574, %3575 : tensor<1x384x128xf32>
    %3577 = stablehlo.broadcast_in_dim %505, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3578 = stablehlo.add %3576, %3577 : tensor<1x384x128xf32>
    %3579 = stablehlo.reshape %3578 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3580 = stablehlo.dot %3579, %510 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3581 = stablehlo.broadcast_in_dim %509, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3582 = stablehlo.add %3580, %3581 : tensor<384x512xf32>
    %3583 = stablehlo.reshape %3582 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3584 = stablehlo.maximum %3583, %9 : tensor<1x384x512xf32>
    %3585 = stablehlo.reshape %3584 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3586 = stablehlo.dot %3585, %514 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3587 = stablehlo.broadcast_in_dim %513, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3588 = stablehlo.add %3586, %3587 : tensor<384x128xf32>
    %3589 = stablehlo.reshape %3588 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3590 = stablehlo.add %3589, %3578 : tensor<1x384x128xf32>
    %3591 = stablehlo.broadcast_in_dim %512, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3592 = stablehlo.multiply %3590, %3591 : tensor<1x384x128xf32>
    %3593 = stablehlo.broadcast_in_dim %511, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3594 = stablehlo.add %3592, %3593 : tensor<1x384x128xf32>
    %3595 = stablehlo.reshape %3594 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3596 = stablehlo.dot %3595, %516 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3597 = stablehlo.broadcast_in_dim %515, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3598 = stablehlo.add %3596, %3597 : tensor<384x512xf32>
    %3599 = stablehlo.reshape %3598 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3600 = stablehlo.maximum %3599, %9 : tensor<1x384x512xf32>
    %3601 = stablehlo.reshape %3600 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3602 = stablehlo.dot %3601, %524 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3603 = stablehlo.broadcast_in_dim %523, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3604 = stablehlo.add %3602, %3603 : tensor<384x128xf32>
    %3605 = stablehlo.reshape %3604 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3606 = stablehlo.add %3605, %3594 : tensor<1x384x128xf32>
    %3607 = stablehlo.broadcast_in_dim %518, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3608 = stablehlo.multiply %3606, %3607 : tensor<1x384x128xf32>
    %3609 = stablehlo.broadcast_in_dim %517, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3610 = stablehlo.add %3608, %3609 : tensor<1x384x128xf32>
    %3611 = stablehlo.reshape %3610 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3612 = stablehlo.dot %3611, %522 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3613 = stablehlo.broadcast_in_dim %521, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3614 = stablehlo.add %3612, %3613 : tensor<384x512xf32>
    %3615 = stablehlo.reshape %3614 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3616 = stablehlo.add %3615, %3490 : tensor<1x384x512xf32>
    %3617 = stablehlo.broadcast_in_dim %520, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3618 = stablehlo.multiply %3616, %3617 : tensor<1x384x512xf32>
    %3619 = stablehlo.broadcast_in_dim %519, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3620 = stablehlo.add %3618, %3619 : tensor<1x384x512xf32>
    %3621 = stablehlo.reshape %3620 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3622 = stablehlo.dot %3621, %534 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3623 = stablehlo.broadcast_in_dim %533, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3624 = stablehlo.add %3622, %3623 : tensor<384x128xf32>
    %3625 = stablehlo.reshape %3624 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3626 = stablehlo.transpose %3625, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3627 = stablehlo.dot %3621, %538 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3628 = stablehlo.reshape %3627 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3629 = stablehlo.broadcast_in_dim %537, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3630 = stablehlo.add %3628, %3629 : tensor<1x384x128xf32>
    %3631 = stablehlo.broadcast_in_dim %536, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3632 = stablehlo.multiply %3630, %3631 : tensor<1x384x128xf32>
    %3633 = stablehlo.broadcast_in_dim %535, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3634 = stablehlo.add %3632, %3633 : tensor<1x384x128xf32>
    %3635 = stablehlo.reshape %3634 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3636 = stablehlo.dot %3635, %530 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3637 = stablehlo.broadcast_in_dim %529, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3638 = stablehlo.add %3636, %3637 : tensor<384x128xf32>
    %3639 = stablehlo.reshape %3638 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3640 = stablehlo.transpose %3639, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3641 = stablehlo.dot %3635, %532 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3642 = stablehlo.broadcast_in_dim %531, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3643 = stablehlo.add %3641, %3642 : tensor<384x128xf32>
    %3644 = stablehlo.reshape %3643 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3645 = stablehlo.transpose %3644, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3646 = stablehlo.dot_general %3645, %3640, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3647 = stablehlo.multiply %3646, %4 : tensor<1x4x384x384xf32>
    %3648 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3649 = stablehlo.add %3647, %3648 : tensor<1x4x384x384xf32>
    %3650 = stablehlo.reduce(%3649 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3651 = stablehlo.broadcast_in_dim %3650, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3652 = stablehlo.subtract %3649, %3651 : tensor<1x4x384x384xf32>
    %3653 = stablehlo.exponential %3652 : tensor<1x4x384x384xf32>
    %3654 = stablehlo.reduce(%3653 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3655 = stablehlo.broadcast_in_dim %3654, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3656 = stablehlo.divide %3653, %3655 : tensor<1x4x384x384xf32>
    %3657 = stablehlo.dot_general %3656, %3626, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3658 = stablehlo.transpose %3657, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3659 = stablehlo.reshape %3658 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3660 = stablehlo.dot %3659, %528 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3661 = stablehlo.broadcast_in_dim %527, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3662 = stablehlo.add %3660, %3661 : tensor<384x128xf32>
    %3663 = stablehlo.reshape %3662 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3664 = stablehlo.dot %3621, %542 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3665 = stablehlo.broadcast_in_dim %541, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3666 = stablehlo.add %3664, %3665 : tensor<384x128xf32>
    %3667 = stablehlo.reshape %3666 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3668 = stablehlo.broadcast_in_dim %540, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3669 = stablehlo.multiply %3667, %3668 : tensor<1x384x128xf32>
    %3670 = stablehlo.broadcast_in_dim %539, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3671 = stablehlo.add %3669, %3670 : tensor<1x384x128xf32>
    %3672 = stablehlo.add %3663, %3671 : tensor<1x384x128xf32>
    %3673 = stablehlo.broadcast_in_dim %526, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3674 = stablehlo.multiply %3672, %3673 : tensor<1x384x128xf32>
    %3675 = stablehlo.broadcast_in_dim %525, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3676 = stablehlo.add %3674, %3675 : tensor<1x384x128xf32>
    %3677 = stablehlo.reshape %3676 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3678 = stablehlo.dot %3677, %544 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3679 = stablehlo.broadcast_in_dim %543, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3680 = stablehlo.add %3678, %3679 : tensor<384x512xf32>
    %3681 = stablehlo.reshape %3680 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3682 = stablehlo.maximum %3681, %9 : tensor<1x384x512xf32>
    %3683 = stablehlo.reshape %3682 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3684 = stablehlo.dot %3683, %548 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3685 = stablehlo.broadcast_in_dim %547, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3686 = stablehlo.add %3684, %3685 : tensor<384x128xf32>
    %3687 = stablehlo.reshape %3686 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3688 = stablehlo.add %3687, %3676 : tensor<1x384x128xf32>
    %3689 = stablehlo.broadcast_in_dim %546, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3690 = stablehlo.multiply %3688, %3689 : tensor<1x384x128xf32>
    %3691 = stablehlo.broadcast_in_dim %545, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3692 = stablehlo.add %3690, %3691 : tensor<1x384x128xf32>
    %3693 = stablehlo.reshape %3692 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3694 = stablehlo.dot %3693, %550 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3695 = stablehlo.broadcast_in_dim %549, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3696 = stablehlo.add %3694, %3695 : tensor<384x512xf32>
    %3697 = stablehlo.reshape %3696 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3698 = stablehlo.maximum %3697, %9 : tensor<1x384x512xf32>
    %3699 = stablehlo.reshape %3698 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3700 = stablehlo.dot %3699, %554 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3701 = stablehlo.broadcast_in_dim %553, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3702 = stablehlo.add %3700, %3701 : tensor<384x128xf32>
    %3703 = stablehlo.reshape %3702 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3704 = stablehlo.add %3703, %3692 : tensor<1x384x128xf32>
    %3705 = stablehlo.broadcast_in_dim %552, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3706 = stablehlo.multiply %3704, %3705 : tensor<1x384x128xf32>
    %3707 = stablehlo.broadcast_in_dim %551, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3708 = stablehlo.add %3706, %3707 : tensor<1x384x128xf32>
    %3709 = stablehlo.reshape %3708 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3710 = stablehlo.dot %3709, %556 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3711 = stablehlo.broadcast_in_dim %555, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3712 = stablehlo.add %3710, %3711 : tensor<384x512xf32>
    %3713 = stablehlo.reshape %3712 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3714 = stablehlo.maximum %3713, %9 : tensor<1x384x512xf32>
    %3715 = stablehlo.reshape %3714 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3716 = stablehlo.dot %3715, %560 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3717 = stablehlo.broadcast_in_dim %559, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3718 = stablehlo.add %3716, %3717 : tensor<384x128xf32>
    %3719 = stablehlo.reshape %3718 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3720 = stablehlo.add %3719, %3708 : tensor<1x384x128xf32>
    %3721 = stablehlo.broadcast_in_dim %558, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3722 = stablehlo.multiply %3720, %3721 : tensor<1x384x128xf32>
    %3723 = stablehlo.broadcast_in_dim %557, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3724 = stablehlo.add %3722, %3723 : tensor<1x384x128xf32>
    %3725 = stablehlo.reshape %3724 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3726 = stablehlo.dot %3725, %562 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3727 = stablehlo.broadcast_in_dim %561, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3728 = stablehlo.add %3726, %3727 : tensor<384x512xf32>
    %3729 = stablehlo.reshape %3728 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3730 = stablehlo.maximum %3729, %9 : tensor<1x384x512xf32>
    %3731 = stablehlo.reshape %3730 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3732 = stablehlo.dot %3731, %570 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3733 = stablehlo.broadcast_in_dim %569, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3734 = stablehlo.add %3732, %3733 : tensor<384x128xf32>
    %3735 = stablehlo.reshape %3734 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3736 = stablehlo.add %3735, %3724 : tensor<1x384x128xf32>
    %3737 = stablehlo.broadcast_in_dim %564, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3738 = stablehlo.multiply %3736, %3737 : tensor<1x384x128xf32>
    %3739 = stablehlo.broadcast_in_dim %563, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3740 = stablehlo.add %3738, %3739 : tensor<1x384x128xf32>
    %3741 = stablehlo.reshape %3740 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3742 = stablehlo.dot %3741, %568 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3743 = stablehlo.broadcast_in_dim %567, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3744 = stablehlo.add %3742, %3743 : tensor<384x512xf32>
    %3745 = stablehlo.reshape %3744 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3746 = stablehlo.add %3745, %3620 : tensor<1x384x512xf32>
    %3747 = stablehlo.broadcast_in_dim %566, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3748 = stablehlo.multiply %3746, %3747 : tensor<1x384x512xf32>
    %3749 = stablehlo.broadcast_in_dim %565, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3750 = stablehlo.add %3748, %3749 : tensor<1x384x512xf32>
    %3751 = stablehlo.reshape %3750 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3752 = stablehlo.dot %3751, %626 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3753 = stablehlo.broadcast_in_dim %625, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3754 = stablehlo.add %3752, %3753 : tensor<384x128xf32>
    %3755 = stablehlo.reshape %3754 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3756 = stablehlo.transpose %3755, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3757 = stablehlo.dot %3751, %630 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3758 = stablehlo.reshape %3757 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3759 = stablehlo.broadcast_in_dim %629, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3760 = stablehlo.add %3758, %3759 : tensor<1x384x128xf32>
    %3761 = stablehlo.broadcast_in_dim %628, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3762 = stablehlo.multiply %3760, %3761 : tensor<1x384x128xf32>
    %3763 = stablehlo.broadcast_in_dim %627, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3764 = stablehlo.add %3762, %3763 : tensor<1x384x128xf32>
    %3765 = stablehlo.reshape %3764 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3766 = stablehlo.dot %3765, %622 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3767 = stablehlo.broadcast_in_dim %621, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3768 = stablehlo.add %3766, %3767 : tensor<384x128xf32>
    %3769 = stablehlo.reshape %3768 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3770 = stablehlo.transpose %3769, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3771 = stablehlo.dot %3765, %624 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3772 = stablehlo.broadcast_in_dim %623, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3773 = stablehlo.add %3771, %3772 : tensor<384x128xf32>
    %3774 = stablehlo.reshape %3773 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3775 = stablehlo.transpose %3774, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3776 = stablehlo.dot_general %3775, %3770, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3777 = stablehlo.multiply %3776, %4 : tensor<1x4x384x384xf32>
    %3778 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3779 = stablehlo.add %3777, %3778 : tensor<1x4x384x384xf32>
    %3780 = stablehlo.reduce(%3779 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3781 = stablehlo.broadcast_in_dim %3780, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3782 = stablehlo.subtract %3779, %3781 : tensor<1x4x384x384xf32>
    %3783 = stablehlo.exponential %3782 : tensor<1x4x384x384xf32>
    %3784 = stablehlo.reduce(%3783 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3785 = stablehlo.broadcast_in_dim %3784, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3786 = stablehlo.divide %3783, %3785 : tensor<1x4x384x384xf32>
    %3787 = stablehlo.dot_general %3786, %3756, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3788 = stablehlo.transpose %3787, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3789 = stablehlo.reshape %3788 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3790 = stablehlo.dot %3789, %620 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3791 = stablehlo.broadcast_in_dim %619, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3792 = stablehlo.add %3790, %3791 : tensor<384x128xf32>
    %3793 = stablehlo.reshape %3792 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3794 = stablehlo.dot %3751, %634 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3795 = stablehlo.broadcast_in_dim %633, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3796 = stablehlo.add %3794, %3795 : tensor<384x128xf32>
    %3797 = stablehlo.reshape %3796 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3798 = stablehlo.broadcast_in_dim %632, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3799 = stablehlo.multiply %3797, %3798 : tensor<1x384x128xf32>
    %3800 = stablehlo.broadcast_in_dim %631, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3801 = stablehlo.add %3799, %3800 : tensor<1x384x128xf32>
    %3802 = stablehlo.add %3793, %3801 : tensor<1x384x128xf32>
    %3803 = stablehlo.broadcast_in_dim %618, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3804 = stablehlo.multiply %3802, %3803 : tensor<1x384x128xf32>
    %3805 = stablehlo.broadcast_in_dim %617, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3806 = stablehlo.add %3804, %3805 : tensor<1x384x128xf32>
    %3807 = stablehlo.reshape %3806 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3808 = stablehlo.dot %3807, %636 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3809 = stablehlo.broadcast_in_dim %635, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3810 = stablehlo.add %3808, %3809 : tensor<384x512xf32>
    %3811 = stablehlo.reshape %3810 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3812 = stablehlo.maximum %3811, %9 : tensor<1x384x512xf32>
    %3813 = stablehlo.reshape %3812 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3814 = stablehlo.dot %3813, %640 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3815 = stablehlo.broadcast_in_dim %639, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3816 = stablehlo.add %3814, %3815 : tensor<384x128xf32>
    %3817 = stablehlo.reshape %3816 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3818 = stablehlo.add %3817, %3806 : tensor<1x384x128xf32>
    %3819 = stablehlo.broadcast_in_dim %638, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3820 = stablehlo.multiply %3818, %3819 : tensor<1x384x128xf32>
    %3821 = stablehlo.broadcast_in_dim %637, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3822 = stablehlo.add %3820, %3821 : tensor<1x384x128xf32>
    %3823 = stablehlo.reshape %3822 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3824 = stablehlo.dot %3823, %642 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3825 = stablehlo.broadcast_in_dim %641, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3826 = stablehlo.add %3824, %3825 : tensor<384x512xf32>
    %3827 = stablehlo.reshape %3826 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3828 = stablehlo.maximum %3827, %9 : tensor<1x384x512xf32>
    %3829 = stablehlo.reshape %3828 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3830 = stablehlo.dot %3829, %646 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3831 = stablehlo.broadcast_in_dim %645, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3832 = stablehlo.add %3830, %3831 : tensor<384x128xf32>
    %3833 = stablehlo.reshape %3832 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3834 = stablehlo.add %3833, %3822 : tensor<1x384x128xf32>
    %3835 = stablehlo.broadcast_in_dim %644, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3836 = stablehlo.multiply %3834, %3835 : tensor<1x384x128xf32>
    %3837 = stablehlo.broadcast_in_dim %643, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3838 = stablehlo.add %3836, %3837 : tensor<1x384x128xf32>
    %3839 = stablehlo.reshape %3838 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3840 = stablehlo.dot %3839, %648 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3841 = stablehlo.broadcast_in_dim %647, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3842 = stablehlo.add %3840, %3841 : tensor<384x512xf32>
    %3843 = stablehlo.reshape %3842 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3844 = stablehlo.maximum %3843, %9 : tensor<1x384x512xf32>
    %3845 = stablehlo.reshape %3844 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3846 = stablehlo.dot %3845, %652 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3847 = stablehlo.broadcast_in_dim %651, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3848 = stablehlo.add %3846, %3847 : tensor<384x128xf32>
    %3849 = stablehlo.reshape %3848 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3850 = stablehlo.add %3849, %3838 : tensor<1x384x128xf32>
    %3851 = stablehlo.broadcast_in_dim %650, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3852 = stablehlo.multiply %3850, %3851 : tensor<1x384x128xf32>
    %3853 = stablehlo.broadcast_in_dim %649, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3854 = stablehlo.add %3852, %3853 : tensor<1x384x128xf32>
    %3855 = stablehlo.reshape %3854 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3856 = stablehlo.dot %3855, %654 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3857 = stablehlo.broadcast_in_dim %653, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3858 = stablehlo.add %3856, %3857 : tensor<384x512xf32>
    %3859 = stablehlo.reshape %3858 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3860 = stablehlo.maximum %3859, %9 : tensor<1x384x512xf32>
    %3861 = stablehlo.reshape %3860 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3862 = stablehlo.dot %3861, %662 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3863 = stablehlo.broadcast_in_dim %661, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3864 = stablehlo.add %3862, %3863 : tensor<384x128xf32>
    %3865 = stablehlo.reshape %3864 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3866 = stablehlo.add %3865, %3854 : tensor<1x384x128xf32>
    %3867 = stablehlo.broadcast_in_dim %656, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3868 = stablehlo.multiply %3866, %3867 : tensor<1x384x128xf32>
    %3869 = stablehlo.broadcast_in_dim %655, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3870 = stablehlo.add %3868, %3869 : tensor<1x384x128xf32>
    %3871 = stablehlo.reshape %3870 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3872 = stablehlo.dot %3871, %660 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3873 = stablehlo.broadcast_in_dim %659, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3874 = stablehlo.add %3872, %3873 : tensor<384x512xf32>
    %3875 = stablehlo.reshape %3874 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3876 = stablehlo.add %3875, %3750 : tensor<1x384x512xf32>
    %3877 = stablehlo.broadcast_in_dim %658, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3878 = stablehlo.multiply %3876, %3877 : tensor<1x384x512xf32>
    %3879 = stablehlo.broadcast_in_dim %657, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3880 = stablehlo.add %3878, %3879 : tensor<1x384x512xf32>
    %3881 = stablehlo.reshape %3880 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3882 = stablehlo.dot %3881, %672 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3883 = stablehlo.broadcast_in_dim %671, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3884 = stablehlo.add %3882, %3883 : tensor<384x128xf32>
    %3885 = stablehlo.reshape %3884 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3886 = stablehlo.transpose %3885, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3887 = stablehlo.dot %3881, %676 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3888 = stablehlo.reshape %3887 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3889 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3890 = stablehlo.add %3888, %3889 : tensor<1x384x128xf32>
    %3891 = stablehlo.broadcast_in_dim %674, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3892 = stablehlo.multiply %3890, %3891 : tensor<1x384x128xf32>
    %3893 = stablehlo.broadcast_in_dim %673, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3894 = stablehlo.add %3892, %3893 : tensor<1x384x128xf32>
    %3895 = stablehlo.reshape %3894 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3896 = stablehlo.dot %3895, %668 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3897 = stablehlo.broadcast_in_dim %667, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3898 = stablehlo.add %3896, %3897 : tensor<384x128xf32>
    %3899 = stablehlo.reshape %3898 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3900 = stablehlo.transpose %3899, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3901 = stablehlo.dot %3895, %670 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3902 = stablehlo.broadcast_in_dim %669, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3903 = stablehlo.add %3901, %3902 : tensor<384x128xf32>
    %3904 = stablehlo.reshape %3903 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3905 = stablehlo.transpose %3904, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3906 = stablehlo.dot_general %3905, %3900, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3907 = stablehlo.multiply %3906, %4 : tensor<1x4x384x384xf32>
    %3908 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3909 = stablehlo.add %3907, %3908 : tensor<1x4x384x384xf32>
    %3910 = stablehlo.reduce(%3909 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3911 = stablehlo.broadcast_in_dim %3910, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3912 = stablehlo.subtract %3909, %3911 : tensor<1x4x384x384xf32>
    %3913 = stablehlo.exponential %3912 : tensor<1x4x384x384xf32>
    %3914 = stablehlo.reduce(%3913 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %3915 = stablehlo.broadcast_in_dim %3914, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3916 = stablehlo.divide %3913, %3915 : tensor<1x4x384x384xf32>
    %3917 = stablehlo.dot_general %3916, %3886, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3918 = stablehlo.transpose %3917, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3919 = stablehlo.reshape %3918 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3920 = stablehlo.dot %3919, %666 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3921 = stablehlo.broadcast_in_dim %665, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3922 = stablehlo.add %3920, %3921 : tensor<384x128xf32>
    %3923 = stablehlo.reshape %3922 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3924 = stablehlo.dot %3881, %680 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3925 = stablehlo.broadcast_in_dim %679, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3926 = stablehlo.add %3924, %3925 : tensor<384x128xf32>
    %3927 = stablehlo.reshape %3926 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3928 = stablehlo.broadcast_in_dim %678, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3929 = stablehlo.multiply %3927, %3928 : tensor<1x384x128xf32>
    %3930 = stablehlo.broadcast_in_dim %677, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3931 = stablehlo.add %3929, %3930 : tensor<1x384x128xf32>
    %3932 = stablehlo.add %3923, %3931 : tensor<1x384x128xf32>
    %3933 = stablehlo.broadcast_in_dim %664, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3934 = stablehlo.multiply %3932, %3933 : tensor<1x384x128xf32>
    %3935 = stablehlo.broadcast_in_dim %663, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3936 = stablehlo.add %3934, %3935 : tensor<1x384x128xf32>
    %3937 = stablehlo.reshape %3936 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3938 = stablehlo.dot %3937, %682 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3939 = stablehlo.broadcast_in_dim %681, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3940 = stablehlo.add %3938, %3939 : tensor<384x512xf32>
    %3941 = stablehlo.reshape %3940 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3942 = stablehlo.maximum %3941, %9 : tensor<1x384x512xf32>
    %3943 = stablehlo.reshape %3942 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3944 = stablehlo.dot %3943, %686 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3945 = stablehlo.broadcast_in_dim %685, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3946 = stablehlo.add %3944, %3945 : tensor<384x128xf32>
    %3947 = stablehlo.reshape %3946 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3948 = stablehlo.add %3947, %3936 : tensor<1x384x128xf32>
    %3949 = stablehlo.broadcast_in_dim %684, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3950 = stablehlo.multiply %3948, %3949 : tensor<1x384x128xf32>
    %3951 = stablehlo.broadcast_in_dim %683, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3952 = stablehlo.add %3950, %3951 : tensor<1x384x128xf32>
    %3953 = stablehlo.reshape %3952 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3954 = stablehlo.dot %3953, %688 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3955 = stablehlo.broadcast_in_dim %687, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3956 = stablehlo.add %3954, %3955 : tensor<384x512xf32>
    %3957 = stablehlo.reshape %3956 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3958 = stablehlo.maximum %3957, %9 : tensor<1x384x512xf32>
    %3959 = stablehlo.reshape %3958 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3960 = stablehlo.dot %3959, %692 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3961 = stablehlo.broadcast_in_dim %691, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3962 = stablehlo.add %3960, %3961 : tensor<384x128xf32>
    %3963 = stablehlo.reshape %3962 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3964 = stablehlo.add %3963, %3952 : tensor<1x384x128xf32>
    %3965 = stablehlo.broadcast_in_dim %690, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3966 = stablehlo.multiply %3964, %3965 : tensor<1x384x128xf32>
    %3967 = stablehlo.broadcast_in_dim %689, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3968 = stablehlo.add %3966, %3967 : tensor<1x384x128xf32>
    %3969 = stablehlo.reshape %3968 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3970 = stablehlo.dot %3969, %694 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3971 = stablehlo.broadcast_in_dim %693, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3972 = stablehlo.add %3970, %3971 : tensor<384x512xf32>
    %3973 = stablehlo.reshape %3972 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3974 = stablehlo.maximum %3973, %9 : tensor<1x384x512xf32>
    %3975 = stablehlo.reshape %3974 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3976 = stablehlo.dot %3975, %698 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3977 = stablehlo.broadcast_in_dim %697, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3978 = stablehlo.add %3976, %3977 : tensor<384x128xf32>
    %3979 = stablehlo.reshape %3978 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3980 = stablehlo.add %3979, %3968 : tensor<1x384x128xf32>
    %3981 = stablehlo.broadcast_in_dim %696, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3982 = stablehlo.multiply %3980, %3981 : tensor<1x384x128xf32>
    %3983 = stablehlo.broadcast_in_dim %695, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3984 = stablehlo.add %3982, %3983 : tensor<1x384x128xf32>
    %3985 = stablehlo.reshape %3984 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3986 = stablehlo.dot %3985, %700 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3987 = stablehlo.broadcast_in_dim %699, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %3988 = stablehlo.add %3986, %3987 : tensor<384x512xf32>
    %3989 = stablehlo.reshape %3988 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3990 = stablehlo.maximum %3989, %9 : tensor<1x384x512xf32>
    %3991 = stablehlo.reshape %3990 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3992 = stablehlo.dot %3991, %708 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3993 = stablehlo.broadcast_in_dim %707, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %3994 = stablehlo.add %3992, %3993 : tensor<384x128xf32>
    %3995 = stablehlo.reshape %3994 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3996 = stablehlo.add %3995, %3984 : tensor<1x384x128xf32>
    %3997 = stablehlo.broadcast_in_dim %702, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3998 = stablehlo.multiply %3996, %3997 : tensor<1x384x128xf32>
    %3999 = stablehlo.broadcast_in_dim %701, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4000 = stablehlo.add %3998, %3999 : tensor<1x384x128xf32>
    %4001 = stablehlo.reshape %4000 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4002 = stablehlo.dot %4001, %706 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4003 = stablehlo.broadcast_in_dim %705, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4004 = stablehlo.add %4002, %4003 : tensor<384x512xf32>
    %4005 = stablehlo.reshape %4004 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4006 = stablehlo.add %4005, %3880 : tensor<1x384x512xf32>
    %4007 = stablehlo.broadcast_in_dim %704, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4008 = stablehlo.multiply %4006, %4007 : tensor<1x384x512xf32>
    %4009 = stablehlo.broadcast_in_dim %703, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4010 = stablehlo.add %4008, %4009 : tensor<1x384x512xf32>
    %4011 = stablehlo.reshape %4010 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4012 = stablehlo.dot %4011, %718 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4013 = stablehlo.broadcast_in_dim %717, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4014 = stablehlo.add %4012, %4013 : tensor<384x128xf32>
    %4015 = stablehlo.reshape %4014 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4016 = stablehlo.transpose %4015, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4017 = stablehlo.dot %4011, %722 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4018 = stablehlo.reshape %4017 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4019 = stablehlo.broadcast_in_dim %721, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4020 = stablehlo.add %4018, %4019 : tensor<1x384x128xf32>
    %4021 = stablehlo.broadcast_in_dim %720, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4022 = stablehlo.multiply %4020, %4021 : tensor<1x384x128xf32>
    %4023 = stablehlo.broadcast_in_dim %719, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4024 = stablehlo.add %4022, %4023 : tensor<1x384x128xf32>
    %4025 = stablehlo.reshape %4024 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4026 = stablehlo.dot %4025, %714 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4027 = stablehlo.broadcast_in_dim %713, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4028 = stablehlo.add %4026, %4027 : tensor<384x128xf32>
    %4029 = stablehlo.reshape %4028 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4030 = stablehlo.transpose %4029, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4031 = stablehlo.dot %4025, %716 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4032 = stablehlo.broadcast_in_dim %715, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4033 = stablehlo.add %4031, %4032 : tensor<384x128xf32>
    %4034 = stablehlo.reshape %4033 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4035 = stablehlo.transpose %4034, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4036 = stablehlo.dot_general %4035, %4030, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4037 = stablehlo.multiply %4036, %4 : tensor<1x4x384x384xf32>
    %4038 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4039 = stablehlo.add %4037, %4038 : tensor<1x4x384x384xf32>
    %4040 = stablehlo.reduce(%4039 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %4041 = stablehlo.broadcast_in_dim %4040, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4042 = stablehlo.subtract %4039, %4041 : tensor<1x4x384x384xf32>
    %4043 = stablehlo.exponential %4042 : tensor<1x4x384x384xf32>
    %4044 = stablehlo.reduce(%4043 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %4045 = stablehlo.broadcast_in_dim %4044, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4046 = stablehlo.divide %4043, %4045 : tensor<1x4x384x384xf32>
    %4047 = stablehlo.dot_general %4046, %4016, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4048 = stablehlo.transpose %4047, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4049 = stablehlo.reshape %4048 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4050 = stablehlo.dot %4049, %712 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4051 = stablehlo.broadcast_in_dim %711, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4052 = stablehlo.add %4050, %4051 : tensor<384x128xf32>
    %4053 = stablehlo.reshape %4052 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4054 = stablehlo.dot %4011, %726 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4055 = stablehlo.broadcast_in_dim %725, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4056 = stablehlo.add %4054, %4055 : tensor<384x128xf32>
    %4057 = stablehlo.reshape %4056 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4058 = stablehlo.broadcast_in_dim %724, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4059 = stablehlo.multiply %4057, %4058 : tensor<1x384x128xf32>
    %4060 = stablehlo.broadcast_in_dim %723, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4061 = stablehlo.add %4059, %4060 : tensor<1x384x128xf32>
    %4062 = stablehlo.add %4053, %4061 : tensor<1x384x128xf32>
    %4063 = stablehlo.broadcast_in_dim %710, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4064 = stablehlo.multiply %4062, %4063 : tensor<1x384x128xf32>
    %4065 = stablehlo.broadcast_in_dim %709, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4066 = stablehlo.add %4064, %4065 : tensor<1x384x128xf32>
    %4067 = stablehlo.reshape %4066 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4068 = stablehlo.dot %4067, %728 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4069 = stablehlo.broadcast_in_dim %727, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4070 = stablehlo.add %4068, %4069 : tensor<384x512xf32>
    %4071 = stablehlo.reshape %4070 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4072 = stablehlo.maximum %4071, %9 : tensor<1x384x512xf32>
    %4073 = stablehlo.reshape %4072 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4074 = stablehlo.dot %4073, %732 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4075 = stablehlo.broadcast_in_dim %731, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4076 = stablehlo.add %4074, %4075 : tensor<384x128xf32>
    %4077 = stablehlo.reshape %4076 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4078 = stablehlo.add %4077, %4066 : tensor<1x384x128xf32>
    %4079 = stablehlo.broadcast_in_dim %730, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4080 = stablehlo.multiply %4078, %4079 : tensor<1x384x128xf32>
    %4081 = stablehlo.broadcast_in_dim %729, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4082 = stablehlo.add %4080, %4081 : tensor<1x384x128xf32>
    %4083 = stablehlo.reshape %4082 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4084 = stablehlo.dot %4083, %734 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4085 = stablehlo.broadcast_in_dim %733, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4086 = stablehlo.add %4084, %4085 : tensor<384x512xf32>
    %4087 = stablehlo.reshape %4086 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4088 = stablehlo.maximum %4087, %9 : tensor<1x384x512xf32>
    %4089 = stablehlo.reshape %4088 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4090 = stablehlo.dot %4089, %738 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4091 = stablehlo.broadcast_in_dim %737, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4092 = stablehlo.add %4090, %4091 : tensor<384x128xf32>
    %4093 = stablehlo.reshape %4092 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4094 = stablehlo.add %4093, %4082 : tensor<1x384x128xf32>
    %4095 = stablehlo.broadcast_in_dim %736, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4096 = stablehlo.multiply %4094, %4095 : tensor<1x384x128xf32>
    %4097 = stablehlo.broadcast_in_dim %735, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4098 = stablehlo.add %4096, %4097 : tensor<1x384x128xf32>
    %4099 = stablehlo.reshape %4098 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4100 = stablehlo.dot %4099, %740 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4101 = stablehlo.broadcast_in_dim %739, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4102 = stablehlo.add %4100, %4101 : tensor<384x512xf32>
    %4103 = stablehlo.reshape %4102 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4104 = stablehlo.maximum %4103, %9 : tensor<1x384x512xf32>
    %4105 = stablehlo.reshape %4104 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4106 = stablehlo.dot %4105, %744 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4107 = stablehlo.broadcast_in_dim %743, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4108 = stablehlo.add %4106, %4107 : tensor<384x128xf32>
    %4109 = stablehlo.reshape %4108 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4110 = stablehlo.add %4109, %4098 : tensor<1x384x128xf32>
    %4111 = stablehlo.broadcast_in_dim %742, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4112 = stablehlo.multiply %4110, %4111 : tensor<1x384x128xf32>
    %4113 = stablehlo.broadcast_in_dim %741, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4114 = stablehlo.add %4112, %4113 : tensor<1x384x128xf32>
    %4115 = stablehlo.reshape %4114 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4116 = stablehlo.dot %4115, %746 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4117 = stablehlo.broadcast_in_dim %745, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4118 = stablehlo.add %4116, %4117 : tensor<384x512xf32>
    %4119 = stablehlo.reshape %4118 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4120 = stablehlo.maximum %4119, %9 : tensor<1x384x512xf32>
    %4121 = stablehlo.reshape %4120 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4122 = stablehlo.dot %4121, %754 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4123 = stablehlo.broadcast_in_dim %753, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4124 = stablehlo.add %4122, %4123 : tensor<384x128xf32>
    %4125 = stablehlo.reshape %4124 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4126 = stablehlo.add %4125, %4114 : tensor<1x384x128xf32>
    %4127 = stablehlo.broadcast_in_dim %748, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4128 = stablehlo.multiply %4126, %4127 : tensor<1x384x128xf32>
    %4129 = stablehlo.broadcast_in_dim %747, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4130 = stablehlo.add %4128, %4129 : tensor<1x384x128xf32>
    %4131 = stablehlo.reshape %4130 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4132 = stablehlo.dot %4131, %752 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4133 = stablehlo.broadcast_in_dim %751, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4134 = stablehlo.add %4132, %4133 : tensor<384x512xf32>
    %4135 = stablehlo.reshape %4134 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4136 = stablehlo.add %4135, %4010 : tensor<1x384x512xf32>
    %4137 = stablehlo.broadcast_in_dim %750, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4138 = stablehlo.multiply %4136, %4137 : tensor<1x384x512xf32>
    %4139 = stablehlo.broadcast_in_dim %749, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4140 = stablehlo.add %4138, %4139 : tensor<1x384x512xf32>
    %4141 = stablehlo.reshape %4140 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4142 = stablehlo.dot %4141, %764 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4143 = stablehlo.broadcast_in_dim %763, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4144 = stablehlo.add %4142, %4143 : tensor<384x128xf32>
    %4145 = stablehlo.reshape %4144 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4146 = stablehlo.transpose %4145, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4147 = stablehlo.dot %4141, %768 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4148 = stablehlo.reshape %4147 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4149 = stablehlo.broadcast_in_dim %767, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4150 = stablehlo.add %4148, %4149 : tensor<1x384x128xf32>
    %4151 = stablehlo.broadcast_in_dim %766, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4152 = stablehlo.multiply %4150, %4151 : tensor<1x384x128xf32>
    %4153 = stablehlo.broadcast_in_dim %765, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4154 = stablehlo.add %4152, %4153 : tensor<1x384x128xf32>
    %4155 = stablehlo.reshape %4154 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4156 = stablehlo.dot %4155, %760 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4157 = stablehlo.broadcast_in_dim %759, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4158 = stablehlo.add %4156, %4157 : tensor<384x128xf32>
    %4159 = stablehlo.reshape %4158 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4160 = stablehlo.transpose %4159, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4161 = stablehlo.dot %4155, %762 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4162 = stablehlo.broadcast_in_dim %761, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4163 = stablehlo.add %4161, %4162 : tensor<384x128xf32>
    %4164 = stablehlo.reshape %4163 : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4165 = stablehlo.transpose %4164, dims = [0, 2, 1, 3] : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4166 = stablehlo.dot_general %4165, %4160, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4167 = stablehlo.multiply %4166, %4 : tensor<1x4x384x384xf32>
    %4168 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4169 = stablehlo.add %4167, %4168 : tensor<1x4x384x384xf32>
    %4170 = stablehlo.reduce(%4169 init: %7) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %4171 = stablehlo.broadcast_in_dim %4170, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4172 = stablehlo.subtract %4169, %4171 : tensor<1x4x384x384xf32>
    %4173 = stablehlo.exponential %4172 : tensor<1x4x384x384xf32>
    %4174 = stablehlo.reduce(%4173 init: %8) across dimensions = [3] : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %4282 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %4282 : tensor<f32>
    }
    %4175 = stablehlo.broadcast_in_dim %4174, dims = [0, 1, 2] : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4176 = stablehlo.divide %4173, %4175 : tensor<1x4x384x384xf32>
    %4177 = stablehlo.dot_general %4176, %4146, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4178 = stablehlo.transpose %4177, dims = [0, 2, 1, 3] : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4179 = stablehlo.reshape %4178 : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4180 = stablehlo.dot %4179, %758 : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4181 = stablehlo.broadcast_in_dim %757, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4182 = stablehlo.add %4180, %4181 : tensor<384x128xf32>
    %4183 = stablehlo.reshape %4182 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4184 = stablehlo.dot %4141, %772 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4185 = stablehlo.broadcast_in_dim %771, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4186 = stablehlo.add %4184, %4185 : tensor<384x128xf32>
    %4187 = stablehlo.reshape %4186 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4188 = stablehlo.broadcast_in_dim %770, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4189 = stablehlo.multiply %4187, %4188 : tensor<1x384x128xf32>
    %4190 = stablehlo.broadcast_in_dim %769, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4191 = stablehlo.add %4189, %4190 : tensor<1x384x128xf32>
    %4192 = stablehlo.add %4183, %4191 : tensor<1x384x128xf32>
    %4193 = stablehlo.broadcast_in_dim %756, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4194 = stablehlo.multiply %4192, %4193 : tensor<1x384x128xf32>
    %4195 = stablehlo.broadcast_in_dim %755, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4196 = stablehlo.add %4194, %4195 : tensor<1x384x128xf32>
    %4197 = stablehlo.reshape %4196 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4198 = stablehlo.dot %4197, %774 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4199 = stablehlo.broadcast_in_dim %773, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4200 = stablehlo.add %4198, %4199 : tensor<384x512xf32>
    %4201 = stablehlo.reshape %4200 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4202 = stablehlo.maximum %4201, %9 : tensor<1x384x512xf32>
    %4203 = stablehlo.reshape %4202 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4204 = stablehlo.dot %4203, %778 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4205 = stablehlo.broadcast_in_dim %777, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4206 = stablehlo.add %4204, %4205 : tensor<384x128xf32>
    %4207 = stablehlo.reshape %4206 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4208 = stablehlo.add %4207, %4196 : tensor<1x384x128xf32>
    %4209 = stablehlo.broadcast_in_dim %776, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4210 = stablehlo.multiply %4208, %4209 : tensor<1x384x128xf32>
    %4211 = stablehlo.broadcast_in_dim %775, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4212 = stablehlo.add %4210, %4211 : tensor<1x384x128xf32>
    %4213 = stablehlo.reshape %4212 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4214 = stablehlo.dot %4213, %780 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4215 = stablehlo.broadcast_in_dim %779, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4216 = stablehlo.add %4214, %4215 : tensor<384x512xf32>
    %4217 = stablehlo.reshape %4216 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4218 = stablehlo.maximum %4217, %9 : tensor<1x384x512xf32>
    %4219 = stablehlo.reshape %4218 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4220 = stablehlo.dot %4219, %784 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4221 = stablehlo.broadcast_in_dim %783, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4222 = stablehlo.add %4220, %4221 : tensor<384x128xf32>
    %4223 = stablehlo.reshape %4222 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4224 = stablehlo.add %4223, %4212 : tensor<1x384x128xf32>
    %4225 = stablehlo.broadcast_in_dim %782, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4226 = stablehlo.multiply %4224, %4225 : tensor<1x384x128xf32>
    %4227 = stablehlo.broadcast_in_dim %781, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4228 = stablehlo.add %4226, %4227 : tensor<1x384x128xf32>
    %4229 = stablehlo.reshape %4228 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4230 = stablehlo.dot %4229, %786 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4231 = stablehlo.broadcast_in_dim %785, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4232 = stablehlo.add %4230, %4231 : tensor<384x512xf32>
    %4233 = stablehlo.reshape %4232 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4234 = stablehlo.maximum %4233, %9 : tensor<1x384x512xf32>
    %4235 = stablehlo.reshape %4234 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4236 = stablehlo.dot %4235, %790 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4237 = stablehlo.broadcast_in_dim %789, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4238 = stablehlo.add %4236, %4237 : tensor<384x128xf32>
    %4239 = stablehlo.reshape %4238 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4240 = stablehlo.add %4239, %4228 : tensor<1x384x128xf32>
    %4241 = stablehlo.broadcast_in_dim %788, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4242 = stablehlo.multiply %4240, %4241 : tensor<1x384x128xf32>
    %4243 = stablehlo.broadcast_in_dim %787, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4244 = stablehlo.add %4242, %4243 : tensor<1x384x128xf32>
    %4245 = stablehlo.reshape %4244 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4246 = stablehlo.dot %4245, %792 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4247 = stablehlo.broadcast_in_dim %791, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4248 = stablehlo.add %4246, %4247 : tensor<384x512xf32>
    %4249 = stablehlo.reshape %4248 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4250 = stablehlo.maximum %4249, %9 : tensor<1x384x512xf32>
    %4251 = stablehlo.reshape %4250 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4252 = stablehlo.dot %4251, %800 : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4253 = stablehlo.broadcast_in_dim %799, dims = [1] : (tensor<128xf32>) -> tensor<384x128xf32>
    %4254 = stablehlo.add %4252, %4253 : tensor<384x128xf32>
    %4255 = stablehlo.reshape %4254 : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4256 = stablehlo.add %4255, %4244 : tensor<1x384x128xf32>
    %4257 = stablehlo.broadcast_in_dim %794, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4258 = stablehlo.multiply %4256, %4257 : tensor<1x384x128xf32>
    %4259 = stablehlo.broadcast_in_dim %793, dims = [2] : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4260 = stablehlo.add %4258, %4259 : tensor<1x384x128xf32>
    %4261 = stablehlo.reshape %4260 : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4262 = stablehlo.dot %4261, %798 : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4263 = stablehlo.broadcast_in_dim %797, dims = [1] : (tensor<512xf32>) -> tensor<384x512xf32>
    %4264 = stablehlo.add %4262, %4263 : tensor<384x512xf32>
    %4265 = stablehlo.reshape %4264 : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4266 = stablehlo.add %4265, %4140 : tensor<1x384x512xf32>
    %4267 = stablehlo.broadcast_in_dim %796, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4268 = stablehlo.multiply %4266, %4267 : tensor<1x384x512xf32>
    %4269 = stablehlo.broadcast_in_dim %795, dims = [2] : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4270 = stablehlo.add %4268, %4269 : tensor<1x384x512xf32>
    %4271 = stablehlo.reshape %4270 : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4272 = stablehlo.transpose %1124, dims = [1, 0] : (tensor<2x512xf32>) -> tensor<512x2xf32>
    %4273 = stablehlo.dot %4271, %4272 : (tensor<384x512xf32>, tensor<512x2xf32>) -> tensor<384x2xf32>
    %4274 = stablehlo.broadcast_in_dim %1123, dims = [1] : (tensor<2xf32>) -> tensor<384x2xf32>
    %4275 = stablehlo.add %4273, %4274 : tensor<384x2xf32>
    %4276 = stablehlo.reshape %4275 : (tensor<384x2xf32>) -> tensor<1x384x2xf32>
    %4277 = stablehlo.transpose %4276, dims = [2, 0, 1] : (tensor<1x384x2xf32>) -> tensor<2x1x384xf32>
    %4278 = "stablehlo.slice"(%4277) {limit_indices = dense<[1, 1, 384]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x1x384xf32>) -> tensor<1x1x384xf32>
    %4279 = stablehlo.reshape %4278 : (tensor<1x1x384xf32>) -> tensor<1x384xf32>
    %4280 = "stablehlo.slice"(%4277) {limit_indices = dense<[2, 1, 384]> : tensor<3xi64>, start_indices = dense<[1, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x1x384xf32>) -> tensor<1x1x384xf32>
    %4281 = stablehlo.reshape %4280 : (tensor<1x1x384xf32>) -> tensor<1x384xf32>
    check.expect_almost_eq_const(%4279, dense<895.130676> : tensor<1x384xf32>) : tensor<1x384xf32>
    check.expect_almost_eq_const(%4281, dense<895.130676> : tensor<1x384xf32>) : tensor<1x384xf32>
    return
  }
}
