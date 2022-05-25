// MobileBert encoder model with placeholder weights, for testing.

module {
  util.global private @"__iree_flow_bert/embeddings/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/embeddings/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/embeddings/embedding_transformation/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/embeddings/embedding_transformation/kernel" {noinline} = dense<0.0001> : tensor<384x512xf32>
  util.global private @"__iree_flow_bert/embeddings/position_embeddings" {noinline} = dense<0.0> : tensor<512x512xf32>
  util.global private @"__iree_flow_bert/embeddings/token_type_embeddings" {noinline} = dense<0.0> : tensor<2x512xf32>
  util.global private @"__iree_flow_bert/embeddings/word_embeddings" {noinline} = dense<0.0> : tensor<30522x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_10/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_11/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_12/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_13/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_14/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_15/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_16/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_17/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_18/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_19/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_20/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_21/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_22/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_23/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_3/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_4/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_5/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_6/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_7/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_8/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/output/dense/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/key/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/key/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/query/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/query/kernel" {noinline} = dense<0.0001> : tensor<128x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/value/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/attention/self/value/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/input/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/input/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/input/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/bottleneck/input/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/intermediate/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/intermediate/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/FakeLayerNorm/beta" = dense<1.0> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/FakeLayerNorm/gamma" = dense<0.4> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/bottleneck/FakeLayerNorm/beta" = dense<1.0> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/bottleneck/FakeLayerNorm/gamma" = dense<0.4> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/bottleneck/dense/bias" = dense<0.1> : tensor<512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/bottleneck/dense/kernel" {noinline} = dense<0.0001> : tensor<128x512xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/dense/bias" = dense<0.1> : tensor<128xf32>
  util.global private @"__iree_flow_bert/encoder/layer_9/output/dense/kernel" {noinline} = dense<0.0001> : tensor<512x128xf32>
  util.global private @"__iree_flow_cls/squad/output_bias" = dense<0.1> : tensor<2xf32>
  util.global private @"__iree_flow_cls/squad/output_weights" = dense<1.0> : tensor<2x512xf32>
  func.func @serving_default() attributes { iree.module.export} {
    %arg0 = util.unfoldable_constant dense<0> : tensor<1x384xi32>
    %arg1 = util.unfoldable_constant dense<0> : tensor<1x384xi32>
    %arg2 = util.unfoldable_constant dense<0> : tensor<1x384xi32>
    %0 = util.global.address @"__iree_flow_bert/embeddings/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %1 = util.global.address @"__iree_flow_bert/embeddings/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %2 = util.global.address @"__iree_flow_bert/embeddings/embedding_transformation/bias" : !util.ptr<tensor<512xf32>>
    %3 = util.global.address @"__iree_flow_bert/embeddings/embedding_transformation/kernel" : !util.ptr<tensor<384x512xf32>>
    %4 = util.global.address @"__iree_flow_bert/embeddings/position_embeddings" : !util.ptr<tensor<512x512xf32>>
    %5 = util.global.address @"__iree_flow_bert/embeddings/token_type_embeddings" : !util.ptr<tensor<2x512xf32>>
    %6 = util.global.address @"__iree_flow_bert/embeddings/word_embeddings" : !util.ptr<tensor<30522x128xf32>>
    %7 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %8 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %9 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %10 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %11 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %12 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %13 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %14 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %15 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %16 = util.global.address @"__iree_flow_bert/encoder/layer_0/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %17 = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %18 = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %19 = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %20 = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %21 = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %22 = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %23 = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %24 = util.global.address @"__iree_flow_bert/encoder/layer_0/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %25 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %26 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %27 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %28 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %29 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %30 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %31 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %32 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %33 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %34 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %35 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %36 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %37 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %38 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %39 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %40 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %41 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %42 = util.global.address @"__iree_flow_bert/encoder/layer_0/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %43 = util.global.address @"__iree_flow_bert/encoder/layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %44 = util.global.address @"__iree_flow_bert/encoder/layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %45 = util.global.address @"__iree_flow_bert/encoder/layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %46 = util.global.address @"__iree_flow_bert/encoder/layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %47 = util.global.address @"__iree_flow_bert/encoder/layer_0/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %48 = util.global.address @"__iree_flow_bert/encoder/layer_0/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %49 = util.global.address @"__iree_flow_bert/encoder/layer_0/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %50 = util.global.address @"__iree_flow_bert/encoder/layer_0/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %51 = util.global.address @"__iree_flow_bert/encoder/layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %52 = util.global.address @"__iree_flow_bert/encoder/layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %53 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %54 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %55 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %56 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %57 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %58 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %59 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %60 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %61 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %62 = util.global.address @"__iree_flow_bert/encoder/layer_1/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %63 = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %64 = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %65 = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %66 = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %67 = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %68 = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %69 = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %70 = util.global.address @"__iree_flow_bert/encoder/layer_1/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %71 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %72 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %73 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %74 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %75 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %76 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %77 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %78 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %79 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %80 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %81 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %82 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %83 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %84 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %85 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %86 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %87 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %88 = util.global.address @"__iree_flow_bert/encoder/layer_1/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %89 = util.global.address @"__iree_flow_bert/encoder/layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %90 = util.global.address @"__iree_flow_bert/encoder/layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %91 = util.global.address @"__iree_flow_bert/encoder/layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %92 = util.global.address @"__iree_flow_bert/encoder/layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %93 = util.global.address @"__iree_flow_bert/encoder/layer_1/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %94 = util.global.address @"__iree_flow_bert/encoder/layer_1/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %95 = util.global.address @"__iree_flow_bert/encoder/layer_1/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %96 = util.global.address @"__iree_flow_bert/encoder/layer_1/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %97 = util.global.address @"__iree_flow_bert/encoder/layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %98 = util.global.address @"__iree_flow_bert/encoder/layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %99 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %100 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %101 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %102 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %103 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %104 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %105 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %106 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %107 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %108 = util.global.address @"__iree_flow_bert/encoder/layer_10/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %109 = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %110 = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %111 = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %112 = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %113 = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %114 = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %115 = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %116 = util.global.address @"__iree_flow_bert/encoder/layer_10/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %117 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %118 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %119 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %120 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %121 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %122 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %123 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %124 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %125 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %126 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %127 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %128 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %129 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %130 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %131 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %132 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %133 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %134 = util.global.address @"__iree_flow_bert/encoder/layer_10/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %135 = util.global.address @"__iree_flow_bert/encoder/layer_10/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %136 = util.global.address @"__iree_flow_bert/encoder/layer_10/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %137 = util.global.address @"__iree_flow_bert/encoder/layer_10/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %138 = util.global.address @"__iree_flow_bert/encoder/layer_10/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %139 = util.global.address @"__iree_flow_bert/encoder/layer_10/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %140 = util.global.address @"__iree_flow_bert/encoder/layer_10/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %141 = util.global.address @"__iree_flow_bert/encoder/layer_10/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %142 = util.global.address @"__iree_flow_bert/encoder/layer_10/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %143 = util.global.address @"__iree_flow_bert/encoder/layer_10/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %144 = util.global.address @"__iree_flow_bert/encoder/layer_10/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %145 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %146 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %147 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %148 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %149 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %150 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %151 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %152 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %153 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %154 = util.global.address @"__iree_flow_bert/encoder/layer_11/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %155 = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %156 = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %157 = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %158 = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %159 = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %160 = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %161 = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %162 = util.global.address @"__iree_flow_bert/encoder/layer_11/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %163 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %164 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %165 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %166 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %167 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %168 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %169 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %170 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %171 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %172 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %173 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %174 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %175 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %176 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %177 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %178 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %179 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %180 = util.global.address @"__iree_flow_bert/encoder/layer_11/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %181 = util.global.address @"__iree_flow_bert/encoder/layer_11/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %182 = util.global.address @"__iree_flow_bert/encoder/layer_11/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %183 = util.global.address @"__iree_flow_bert/encoder/layer_11/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %184 = util.global.address @"__iree_flow_bert/encoder/layer_11/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %185 = util.global.address @"__iree_flow_bert/encoder/layer_11/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %186 = util.global.address @"__iree_flow_bert/encoder/layer_11/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %187 = util.global.address @"__iree_flow_bert/encoder/layer_11/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %188 = util.global.address @"__iree_flow_bert/encoder/layer_11/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %189 = util.global.address @"__iree_flow_bert/encoder/layer_11/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %190 = util.global.address @"__iree_flow_bert/encoder/layer_11/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %191 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %192 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %193 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %194 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %195 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %196 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %197 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %198 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %199 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %200 = util.global.address @"__iree_flow_bert/encoder/layer_12/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %201 = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %202 = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %203 = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %204 = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %205 = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %206 = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %207 = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %208 = util.global.address @"__iree_flow_bert/encoder/layer_12/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %209 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %210 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %211 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %212 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %213 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %214 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %215 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %216 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %217 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %218 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %219 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %220 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %221 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %222 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %223 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %224 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %225 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %226 = util.global.address @"__iree_flow_bert/encoder/layer_12/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %227 = util.global.address @"__iree_flow_bert/encoder/layer_12/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %228 = util.global.address @"__iree_flow_bert/encoder/layer_12/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %229 = util.global.address @"__iree_flow_bert/encoder/layer_12/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %230 = util.global.address @"__iree_flow_bert/encoder/layer_12/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %231 = util.global.address @"__iree_flow_bert/encoder/layer_12/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %232 = util.global.address @"__iree_flow_bert/encoder/layer_12/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %233 = util.global.address @"__iree_flow_bert/encoder/layer_12/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %234 = util.global.address @"__iree_flow_bert/encoder/layer_12/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %235 = util.global.address @"__iree_flow_bert/encoder/layer_12/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %236 = util.global.address @"__iree_flow_bert/encoder/layer_12/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %237 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %238 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %239 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %240 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %241 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %242 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %243 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %244 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %245 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %246 = util.global.address @"__iree_flow_bert/encoder/layer_13/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %247 = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %248 = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %249 = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %250 = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %251 = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %252 = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %253 = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %254 = util.global.address @"__iree_flow_bert/encoder/layer_13/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %255 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %256 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %257 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %258 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %259 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %260 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %261 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %262 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %263 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %264 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %265 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %266 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %267 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %268 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %269 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %270 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %271 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %272 = util.global.address @"__iree_flow_bert/encoder/layer_13/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %273 = util.global.address @"__iree_flow_bert/encoder/layer_13/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %274 = util.global.address @"__iree_flow_bert/encoder/layer_13/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %275 = util.global.address @"__iree_flow_bert/encoder/layer_13/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %276 = util.global.address @"__iree_flow_bert/encoder/layer_13/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %277 = util.global.address @"__iree_flow_bert/encoder/layer_13/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %278 = util.global.address @"__iree_flow_bert/encoder/layer_13/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %279 = util.global.address @"__iree_flow_bert/encoder/layer_13/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %280 = util.global.address @"__iree_flow_bert/encoder/layer_13/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %281 = util.global.address @"__iree_flow_bert/encoder/layer_13/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %282 = util.global.address @"__iree_flow_bert/encoder/layer_13/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %283 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %284 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %285 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %286 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %287 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %288 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %289 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %290 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %291 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %292 = util.global.address @"__iree_flow_bert/encoder/layer_14/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %293 = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %294 = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %295 = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %296 = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %297 = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %298 = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %299 = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %300 = util.global.address @"__iree_flow_bert/encoder/layer_14/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %301 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %302 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %303 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %304 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %305 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %306 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %307 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %308 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %309 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %310 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %311 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %312 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %313 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %314 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %315 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %316 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %317 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %318 = util.global.address @"__iree_flow_bert/encoder/layer_14/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %319 = util.global.address @"__iree_flow_bert/encoder/layer_14/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %320 = util.global.address @"__iree_flow_bert/encoder/layer_14/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %321 = util.global.address @"__iree_flow_bert/encoder/layer_14/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %322 = util.global.address @"__iree_flow_bert/encoder/layer_14/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %323 = util.global.address @"__iree_flow_bert/encoder/layer_14/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %324 = util.global.address @"__iree_flow_bert/encoder/layer_14/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %325 = util.global.address @"__iree_flow_bert/encoder/layer_14/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %326 = util.global.address @"__iree_flow_bert/encoder/layer_14/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %327 = util.global.address @"__iree_flow_bert/encoder/layer_14/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %328 = util.global.address @"__iree_flow_bert/encoder/layer_14/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %329 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %330 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %331 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %332 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %333 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %334 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %335 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %336 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %337 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %338 = util.global.address @"__iree_flow_bert/encoder/layer_15/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %339 = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %340 = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %341 = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %342 = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %343 = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %344 = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %345 = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %346 = util.global.address @"__iree_flow_bert/encoder/layer_15/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %347 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %348 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %349 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %350 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %351 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %352 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %353 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %354 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %355 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %356 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %357 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %358 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %359 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %360 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %361 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %362 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %363 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %364 = util.global.address @"__iree_flow_bert/encoder/layer_15/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %365 = util.global.address @"__iree_flow_bert/encoder/layer_15/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %366 = util.global.address @"__iree_flow_bert/encoder/layer_15/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %367 = util.global.address @"__iree_flow_bert/encoder/layer_15/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %368 = util.global.address @"__iree_flow_bert/encoder/layer_15/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %369 = util.global.address @"__iree_flow_bert/encoder/layer_15/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %370 = util.global.address @"__iree_flow_bert/encoder/layer_15/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %371 = util.global.address @"__iree_flow_bert/encoder/layer_15/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %372 = util.global.address @"__iree_flow_bert/encoder/layer_15/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %373 = util.global.address @"__iree_flow_bert/encoder/layer_15/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %374 = util.global.address @"__iree_flow_bert/encoder/layer_15/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %375 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %376 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %377 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %378 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %379 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %380 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %381 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %382 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %383 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %384 = util.global.address @"__iree_flow_bert/encoder/layer_16/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %385 = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %386 = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %387 = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %388 = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %389 = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %390 = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %391 = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %392 = util.global.address @"__iree_flow_bert/encoder/layer_16/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %393 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %394 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %395 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %396 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %397 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %398 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %399 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %400 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %401 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %402 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %403 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %404 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %405 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %406 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %407 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %408 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %409 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %410 = util.global.address @"__iree_flow_bert/encoder/layer_16/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %411 = util.global.address @"__iree_flow_bert/encoder/layer_16/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %412 = util.global.address @"__iree_flow_bert/encoder/layer_16/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %413 = util.global.address @"__iree_flow_bert/encoder/layer_16/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %414 = util.global.address @"__iree_flow_bert/encoder/layer_16/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %415 = util.global.address @"__iree_flow_bert/encoder/layer_16/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %416 = util.global.address @"__iree_flow_bert/encoder/layer_16/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %417 = util.global.address @"__iree_flow_bert/encoder/layer_16/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %418 = util.global.address @"__iree_flow_bert/encoder/layer_16/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %419 = util.global.address @"__iree_flow_bert/encoder/layer_16/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %420 = util.global.address @"__iree_flow_bert/encoder/layer_16/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %421 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %422 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %423 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %424 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %425 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %426 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %427 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %428 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %429 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %430 = util.global.address @"__iree_flow_bert/encoder/layer_17/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %431 = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %432 = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %433 = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %434 = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %435 = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %436 = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %437 = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %438 = util.global.address @"__iree_flow_bert/encoder/layer_17/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %439 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %440 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %441 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %442 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %443 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %444 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %445 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %446 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %447 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %448 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %449 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %450 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %451 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %452 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %453 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %454 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %455 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %456 = util.global.address @"__iree_flow_bert/encoder/layer_17/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %457 = util.global.address @"__iree_flow_bert/encoder/layer_17/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %458 = util.global.address @"__iree_flow_bert/encoder/layer_17/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %459 = util.global.address @"__iree_flow_bert/encoder/layer_17/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %460 = util.global.address @"__iree_flow_bert/encoder/layer_17/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %461 = util.global.address @"__iree_flow_bert/encoder/layer_17/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %462 = util.global.address @"__iree_flow_bert/encoder/layer_17/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %463 = util.global.address @"__iree_flow_bert/encoder/layer_17/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %464 = util.global.address @"__iree_flow_bert/encoder/layer_17/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %465 = util.global.address @"__iree_flow_bert/encoder/layer_17/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %466 = util.global.address @"__iree_flow_bert/encoder/layer_17/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %467 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %468 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %469 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %470 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %471 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %472 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %473 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %474 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %475 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %476 = util.global.address @"__iree_flow_bert/encoder/layer_18/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %477 = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %478 = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %479 = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %480 = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %481 = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %482 = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %483 = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %484 = util.global.address @"__iree_flow_bert/encoder/layer_18/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %485 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %486 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %487 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %488 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %489 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %490 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %491 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %492 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %493 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %494 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %495 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %496 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %497 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %498 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %499 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %500 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %501 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %502 = util.global.address @"__iree_flow_bert/encoder/layer_18/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %503 = util.global.address @"__iree_flow_bert/encoder/layer_18/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %504 = util.global.address @"__iree_flow_bert/encoder/layer_18/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %505 = util.global.address @"__iree_flow_bert/encoder/layer_18/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %506 = util.global.address @"__iree_flow_bert/encoder/layer_18/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %507 = util.global.address @"__iree_flow_bert/encoder/layer_18/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %508 = util.global.address @"__iree_flow_bert/encoder/layer_18/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %509 = util.global.address @"__iree_flow_bert/encoder/layer_18/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %510 = util.global.address @"__iree_flow_bert/encoder/layer_18/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %511 = util.global.address @"__iree_flow_bert/encoder/layer_18/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %512 = util.global.address @"__iree_flow_bert/encoder/layer_18/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %513 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %514 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %515 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %516 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %517 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %518 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %519 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %520 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %521 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %522 = util.global.address @"__iree_flow_bert/encoder/layer_19/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %523 = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %524 = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %525 = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %526 = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %527 = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %528 = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %529 = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %530 = util.global.address @"__iree_flow_bert/encoder/layer_19/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %531 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %532 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %533 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %534 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %535 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %536 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %537 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %538 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %539 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %540 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %541 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %542 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %543 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %544 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %545 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %546 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %547 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %548 = util.global.address @"__iree_flow_bert/encoder/layer_19/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %549 = util.global.address @"__iree_flow_bert/encoder/layer_19/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %550 = util.global.address @"__iree_flow_bert/encoder/layer_19/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %551 = util.global.address @"__iree_flow_bert/encoder/layer_19/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %552 = util.global.address @"__iree_flow_bert/encoder/layer_19/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %553 = util.global.address @"__iree_flow_bert/encoder/layer_19/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %554 = util.global.address @"__iree_flow_bert/encoder/layer_19/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %555 = util.global.address @"__iree_flow_bert/encoder/layer_19/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %556 = util.global.address @"__iree_flow_bert/encoder/layer_19/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %557 = util.global.address @"__iree_flow_bert/encoder/layer_19/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %558 = util.global.address @"__iree_flow_bert/encoder/layer_19/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %559 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %560 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %561 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %562 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %563 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %564 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %565 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %566 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %567 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %568 = util.global.address @"__iree_flow_bert/encoder/layer_2/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %569 = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %570 = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %571 = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %572 = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %573 = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %574 = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %575 = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %576 = util.global.address @"__iree_flow_bert/encoder/layer_2/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %577 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %578 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %579 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %580 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %581 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %582 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %583 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %584 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %585 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %586 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %587 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %588 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %589 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %590 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %591 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %592 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %593 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %594 = util.global.address @"__iree_flow_bert/encoder/layer_2/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %595 = util.global.address @"__iree_flow_bert/encoder/layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %596 = util.global.address @"__iree_flow_bert/encoder/layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %597 = util.global.address @"__iree_flow_bert/encoder/layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %598 = util.global.address @"__iree_flow_bert/encoder/layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %599 = util.global.address @"__iree_flow_bert/encoder/layer_2/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %600 = util.global.address @"__iree_flow_bert/encoder/layer_2/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %601 = util.global.address @"__iree_flow_bert/encoder/layer_2/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %602 = util.global.address @"__iree_flow_bert/encoder/layer_2/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %603 = util.global.address @"__iree_flow_bert/encoder/layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %604 = util.global.address @"__iree_flow_bert/encoder/layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %605 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %606 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %607 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %608 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %609 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %610 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %611 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %612 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %613 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %614 = util.global.address @"__iree_flow_bert/encoder/layer_20/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %615 = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %616 = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %617 = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %618 = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %619 = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %620 = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %621 = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %622 = util.global.address @"__iree_flow_bert/encoder/layer_20/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %623 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %624 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %625 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %626 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %627 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %628 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %629 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %630 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %631 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %632 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %633 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %634 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %635 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %636 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %637 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %638 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %639 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %640 = util.global.address @"__iree_flow_bert/encoder/layer_20/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %641 = util.global.address @"__iree_flow_bert/encoder/layer_20/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %642 = util.global.address @"__iree_flow_bert/encoder/layer_20/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %643 = util.global.address @"__iree_flow_bert/encoder/layer_20/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %644 = util.global.address @"__iree_flow_bert/encoder/layer_20/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %645 = util.global.address @"__iree_flow_bert/encoder/layer_20/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %646 = util.global.address @"__iree_flow_bert/encoder/layer_20/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %647 = util.global.address @"__iree_flow_bert/encoder/layer_20/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %648 = util.global.address @"__iree_flow_bert/encoder/layer_20/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %649 = util.global.address @"__iree_flow_bert/encoder/layer_20/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %650 = util.global.address @"__iree_flow_bert/encoder/layer_20/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %651 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %652 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %653 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %654 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %655 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %656 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %657 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %658 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %659 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %660 = util.global.address @"__iree_flow_bert/encoder/layer_21/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %661 = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %662 = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %663 = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %664 = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %665 = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %666 = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %667 = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %668 = util.global.address @"__iree_flow_bert/encoder/layer_21/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %669 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %670 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %671 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %672 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %673 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %674 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %675 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %676 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %677 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %678 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %679 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %680 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %681 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %682 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %683 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %684 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %685 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %686 = util.global.address @"__iree_flow_bert/encoder/layer_21/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %687 = util.global.address @"__iree_flow_bert/encoder/layer_21/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %688 = util.global.address @"__iree_flow_bert/encoder/layer_21/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %689 = util.global.address @"__iree_flow_bert/encoder/layer_21/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %690 = util.global.address @"__iree_flow_bert/encoder/layer_21/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %691 = util.global.address @"__iree_flow_bert/encoder/layer_21/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %692 = util.global.address @"__iree_flow_bert/encoder/layer_21/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %693 = util.global.address @"__iree_flow_bert/encoder/layer_21/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %694 = util.global.address @"__iree_flow_bert/encoder/layer_21/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %695 = util.global.address @"__iree_flow_bert/encoder/layer_21/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %696 = util.global.address @"__iree_flow_bert/encoder/layer_21/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %697 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %698 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %699 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %700 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %701 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %702 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %703 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %704 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %705 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %706 = util.global.address @"__iree_flow_bert/encoder/layer_22/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %707 = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %708 = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %709 = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %710 = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %711 = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %712 = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %713 = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %714 = util.global.address @"__iree_flow_bert/encoder/layer_22/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %715 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %716 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %717 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %718 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %719 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %720 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %721 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %722 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %723 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %724 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %725 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %726 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %727 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %728 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %729 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %730 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %731 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %732 = util.global.address @"__iree_flow_bert/encoder/layer_22/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %733 = util.global.address @"__iree_flow_bert/encoder/layer_22/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %734 = util.global.address @"__iree_flow_bert/encoder/layer_22/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %735 = util.global.address @"__iree_flow_bert/encoder/layer_22/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %736 = util.global.address @"__iree_flow_bert/encoder/layer_22/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %737 = util.global.address @"__iree_flow_bert/encoder/layer_22/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %738 = util.global.address @"__iree_flow_bert/encoder/layer_22/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %739 = util.global.address @"__iree_flow_bert/encoder/layer_22/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %740 = util.global.address @"__iree_flow_bert/encoder/layer_22/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %741 = util.global.address @"__iree_flow_bert/encoder/layer_22/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %742 = util.global.address @"__iree_flow_bert/encoder/layer_22/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %743 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %744 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %745 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %746 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %747 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %748 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %749 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %750 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %751 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %752 = util.global.address @"__iree_flow_bert/encoder/layer_23/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %753 = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %754 = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %755 = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %756 = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %757 = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %758 = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %759 = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %760 = util.global.address @"__iree_flow_bert/encoder/layer_23/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %761 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %762 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %763 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %764 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %765 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %766 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %767 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %768 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %769 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %770 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %771 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %772 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %773 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %774 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %775 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %776 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %777 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %778 = util.global.address @"__iree_flow_bert/encoder/layer_23/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %779 = util.global.address @"__iree_flow_bert/encoder/layer_23/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %780 = util.global.address @"__iree_flow_bert/encoder/layer_23/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %781 = util.global.address @"__iree_flow_bert/encoder/layer_23/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %782 = util.global.address @"__iree_flow_bert/encoder/layer_23/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %783 = util.global.address @"__iree_flow_bert/encoder/layer_23/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %784 = util.global.address @"__iree_flow_bert/encoder/layer_23/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %785 = util.global.address @"__iree_flow_bert/encoder/layer_23/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %786 = util.global.address @"__iree_flow_bert/encoder/layer_23/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %787 = util.global.address @"__iree_flow_bert/encoder/layer_23/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %788 = util.global.address @"__iree_flow_bert/encoder/layer_23/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %789 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %790 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %791 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %792 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %793 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %794 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %795 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %796 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %797 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %798 = util.global.address @"__iree_flow_bert/encoder/layer_3/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %799 = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %800 = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %801 = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %802 = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %803 = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %804 = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %805 = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %806 = util.global.address @"__iree_flow_bert/encoder/layer_3/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %807 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %808 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %809 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %810 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %811 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %812 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %813 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %814 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %815 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %816 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %817 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %818 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %819 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %820 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %821 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %822 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %823 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %824 = util.global.address @"__iree_flow_bert/encoder/layer_3/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %825 = util.global.address @"__iree_flow_bert/encoder/layer_3/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %826 = util.global.address @"__iree_flow_bert/encoder/layer_3/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %827 = util.global.address @"__iree_flow_bert/encoder/layer_3/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %828 = util.global.address @"__iree_flow_bert/encoder/layer_3/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %829 = util.global.address @"__iree_flow_bert/encoder/layer_3/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %830 = util.global.address @"__iree_flow_bert/encoder/layer_3/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %831 = util.global.address @"__iree_flow_bert/encoder/layer_3/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %832 = util.global.address @"__iree_flow_bert/encoder/layer_3/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %833 = util.global.address @"__iree_flow_bert/encoder/layer_3/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %834 = util.global.address @"__iree_flow_bert/encoder/layer_3/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %835 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %836 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %837 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %838 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %839 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %840 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %841 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %842 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %843 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %844 = util.global.address @"__iree_flow_bert/encoder/layer_4/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %845 = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %846 = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %847 = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %848 = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %849 = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %850 = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %851 = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %852 = util.global.address @"__iree_flow_bert/encoder/layer_4/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %853 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %854 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %855 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %856 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %857 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %858 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %859 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %860 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %861 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %862 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %863 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %864 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %865 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %866 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %867 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %868 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %869 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %870 = util.global.address @"__iree_flow_bert/encoder/layer_4/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %871 = util.global.address @"__iree_flow_bert/encoder/layer_4/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %872 = util.global.address @"__iree_flow_bert/encoder/layer_4/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %873 = util.global.address @"__iree_flow_bert/encoder/layer_4/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %874 = util.global.address @"__iree_flow_bert/encoder/layer_4/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %875 = util.global.address @"__iree_flow_bert/encoder/layer_4/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %876 = util.global.address @"__iree_flow_bert/encoder/layer_4/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %877 = util.global.address @"__iree_flow_bert/encoder/layer_4/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %878 = util.global.address @"__iree_flow_bert/encoder/layer_4/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %879 = util.global.address @"__iree_flow_bert/encoder/layer_4/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %880 = util.global.address @"__iree_flow_bert/encoder/layer_4/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %881 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %882 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %883 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %884 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %885 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %886 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %887 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %888 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %889 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %890 = util.global.address @"__iree_flow_bert/encoder/layer_5/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %891 = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %892 = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %893 = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %894 = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %895 = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %896 = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %897 = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %898 = util.global.address @"__iree_flow_bert/encoder/layer_5/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %899 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %900 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %901 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %902 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %903 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %904 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %905 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %906 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %907 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %908 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %909 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %910 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %911 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %912 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %913 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %914 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %915 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %916 = util.global.address @"__iree_flow_bert/encoder/layer_5/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %917 = util.global.address @"__iree_flow_bert/encoder/layer_5/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %918 = util.global.address @"__iree_flow_bert/encoder/layer_5/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %919 = util.global.address @"__iree_flow_bert/encoder/layer_5/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %920 = util.global.address @"__iree_flow_bert/encoder/layer_5/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %921 = util.global.address @"__iree_flow_bert/encoder/layer_5/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %922 = util.global.address @"__iree_flow_bert/encoder/layer_5/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %923 = util.global.address @"__iree_flow_bert/encoder/layer_5/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %924 = util.global.address @"__iree_flow_bert/encoder/layer_5/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %925 = util.global.address @"__iree_flow_bert/encoder/layer_5/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %926 = util.global.address @"__iree_flow_bert/encoder/layer_5/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %927 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %928 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %929 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %930 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %931 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %932 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %933 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %934 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %935 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %936 = util.global.address @"__iree_flow_bert/encoder/layer_6/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %937 = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %938 = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %939 = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %940 = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %941 = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %942 = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %943 = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %944 = util.global.address @"__iree_flow_bert/encoder/layer_6/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %945 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %946 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %947 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %948 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %949 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %950 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %951 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %952 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %953 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %954 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %955 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %956 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %957 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %958 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %959 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %960 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %961 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %962 = util.global.address @"__iree_flow_bert/encoder/layer_6/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %963 = util.global.address @"__iree_flow_bert/encoder/layer_6/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %964 = util.global.address @"__iree_flow_bert/encoder/layer_6/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %965 = util.global.address @"__iree_flow_bert/encoder/layer_6/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %966 = util.global.address @"__iree_flow_bert/encoder/layer_6/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %967 = util.global.address @"__iree_flow_bert/encoder/layer_6/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %968 = util.global.address @"__iree_flow_bert/encoder/layer_6/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %969 = util.global.address @"__iree_flow_bert/encoder/layer_6/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %970 = util.global.address @"__iree_flow_bert/encoder/layer_6/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %971 = util.global.address @"__iree_flow_bert/encoder/layer_6/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %972 = util.global.address @"__iree_flow_bert/encoder/layer_6/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %973 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %974 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %975 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %976 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %977 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %978 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %979 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %980 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %981 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %982 = util.global.address @"__iree_flow_bert/encoder/layer_7/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %983 = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %984 = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %985 = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %986 = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %987 = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %988 = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %989 = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %990 = util.global.address @"__iree_flow_bert/encoder/layer_7/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %991 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %992 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %993 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %994 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %995 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %996 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %997 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %998 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %999 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1000 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1001 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1002 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1003 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1004 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1005 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1006 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1007 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1008 = util.global.address @"__iree_flow_bert/encoder/layer_7/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1009 = util.global.address @"__iree_flow_bert/encoder/layer_7/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1010 = util.global.address @"__iree_flow_bert/encoder/layer_7/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1011 = util.global.address @"__iree_flow_bert/encoder/layer_7/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1012 = util.global.address @"__iree_flow_bert/encoder/layer_7/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1013 = util.global.address @"__iree_flow_bert/encoder/layer_7/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %1014 = util.global.address @"__iree_flow_bert/encoder/layer_7/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %1015 = util.global.address @"__iree_flow_bert/encoder/layer_7/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %1016 = util.global.address @"__iree_flow_bert/encoder/layer_7/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1017 = util.global.address @"__iree_flow_bert/encoder/layer_7/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1018 = util.global.address @"__iree_flow_bert/encoder/layer_7/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1019 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1020 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1021 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1022 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %1023 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %1024 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %1025 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %1026 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %1027 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %1028 = util.global.address @"__iree_flow_bert/encoder/layer_8/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %1029 = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1030 = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1031 = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %1032 = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1033 = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1034 = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1035 = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %1036 = util.global.address @"__iree_flow_bert/encoder/layer_8/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1037 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1038 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1039 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1040 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1041 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1042 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1043 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1044 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1045 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1046 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1047 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1048 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1049 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1050 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1051 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1052 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1053 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1054 = util.global.address @"__iree_flow_bert/encoder/layer_8/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1055 = util.global.address @"__iree_flow_bert/encoder/layer_8/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1056 = util.global.address @"__iree_flow_bert/encoder/layer_8/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1057 = util.global.address @"__iree_flow_bert/encoder/layer_8/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1058 = util.global.address @"__iree_flow_bert/encoder/layer_8/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1059 = util.global.address @"__iree_flow_bert/encoder/layer_8/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %1060 = util.global.address @"__iree_flow_bert/encoder/layer_8/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %1061 = util.global.address @"__iree_flow_bert/encoder/layer_8/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %1062 = util.global.address @"__iree_flow_bert/encoder/layer_8/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1063 = util.global.address @"__iree_flow_bert/encoder/layer_8/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1064 = util.global.address @"__iree_flow_bert/encoder/layer_8/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1065 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1066 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1067 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1068 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/output/dense/kernel" : !util.ptr<tensor<128x128xf32>>
    %1069 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/key/bias" : !util.ptr<tensor<128xf32>>
    %1070 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/key/kernel" : !util.ptr<tensor<128x128xf32>>
    %1071 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/query/bias" : !util.ptr<tensor<128xf32>>
    %1072 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/query/kernel" : !util.ptr<tensor<128x128xf32>>
    %1073 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/value/bias" : !util.ptr<tensor<128xf32>>
    %1074 = util.global.address @"__iree_flow_bert/encoder/layer_9/attention/self/value/kernel" : !util.ptr<tensor<512x128xf32>>
    %1075 = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1076 = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1077 = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/dense/bias" : !util.ptr<tensor<128xf32>>
    %1078 = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/attention/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1079 = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/input/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1080 = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/input/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1081 = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/input/dense/bias" : !util.ptr<tensor<128xf32>>
    %1082 = util.global.address @"__iree_flow_bert/encoder/layer_9/bottleneck/input/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1083 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1084 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1085 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1086 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1087 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1088 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_0/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1089 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1090 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1091 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1092 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1093 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1094 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_1/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1095 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1096 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1097 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1098 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1099 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1100 = util.global.address @"__iree_flow_bert/encoder/layer_9/ffn_layer_2/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1101 = util.global.address @"__iree_flow_bert/encoder/layer_9/intermediate/dense/bias" : !util.ptr<tensor<512xf32>>
    %1102 = util.global.address @"__iree_flow_bert/encoder/layer_9/intermediate/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1103 = util.global.address @"__iree_flow_bert/encoder/layer_9/output/FakeLayerNorm/beta" : !util.ptr<tensor<128xf32>>
    %1104 = util.global.address @"__iree_flow_bert/encoder/layer_9/output/FakeLayerNorm/gamma" : !util.ptr<tensor<128xf32>>
    %1105 = util.global.address @"__iree_flow_bert/encoder/layer_9/output/bottleneck/FakeLayerNorm/beta" : !util.ptr<tensor<512xf32>>
    %1106 = util.global.address @"__iree_flow_bert/encoder/layer_9/output/bottleneck/FakeLayerNorm/gamma" : !util.ptr<tensor<512xf32>>
    %1107 = util.global.address @"__iree_flow_bert/encoder/layer_9/output/bottleneck/dense/bias" : !util.ptr<tensor<512xf32>>
    %1108 = util.global.address @"__iree_flow_bert/encoder/layer_9/output/bottleneck/dense/kernel" : !util.ptr<tensor<128x512xf32>>
    %1109 = util.global.address @"__iree_flow_bert/encoder/layer_9/output/dense/bias" : !util.ptr<tensor<128xf32>>
    %1110 = util.global.address @"__iree_flow_bert/encoder/layer_9/output/dense/kernel" : !util.ptr<tensor<512x128xf32>>
    %1111 = util.global.address @"__iree_flow_cls/squad/output_bias" : !util.ptr<tensor<2xf32>>
    %1112 = util.global.address @"__iree_flow_cls/squad/output_weights" : !util.ptr<tensor<2x512xf32>>
    %1113 = mhlo.constant dense<-1.000000e+04> : tensor<1x1x384x384xf32>
    %1114 = mhlo.constant dense<0.176776692> : tensor<1x4x384x384xf32>
    %1115 = mhlo.constant dense<1.000000e+04> : tensor<1x1x384x384xf32>
    %1116 = mhlo.constant dense<1.000000e+00> : tensor<1x384x384xf32>
    %1117 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %1118 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1119 = mhlo.constant dense<0.000000e+00> : tensor<1x384x512xf32>
    %1120 = util.global.load.indirect %0 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1121 = util.global.load.indirect %1 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1122 = util.global.load.indirect %2 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1123 = util.global.load.indirect %3 : !util.ptr<tensor<384x512xf32>> -> tensor<384x512xf32>
    %1124 = util.global.load.indirect %4 : !util.ptr<tensor<512x512xf32>> -> tensor<512x512xf32>
    %1125 = "mhlo.slice"(%1124) {limit_indices = dense<[384, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x512xf32>) -> tensor<384x512xf32>
    %1126 = "mhlo.reshape"(%1125) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %1127 = util.global.load.indirect %5 : !util.ptr<tensor<2x512xf32>> -> tensor<2x512xf32>
    %1128 = util.global.load.indirect %6 : !util.ptr<tensor<30522x128xf32>> -> tensor<30522x128xf32>
    %1129 = util.global.load.indirect %7 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1130 = util.global.load.indirect %8 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1131 = util.global.load.indirect %9 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1132 = util.global.load.indirect %10 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1133 = util.global.load.indirect %11 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1134 = util.global.load.indirect %12 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1135 = util.global.load.indirect %13 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1136 = util.global.load.indirect %14 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1137 = util.global.load.indirect %15 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1138 = util.global.load.indirect %16 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1139 = util.global.load.indirect %17 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1140 = util.global.load.indirect %18 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1141 = util.global.load.indirect %19 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1142 = util.global.load.indirect %20 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1143 = util.global.load.indirect %21 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1144 = util.global.load.indirect %22 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1145 = util.global.load.indirect %23 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1146 = util.global.load.indirect %24 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1147 = util.global.load.indirect %25 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1148 = util.global.load.indirect %26 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1149 = util.global.load.indirect %27 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1150 = util.global.load.indirect %28 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1151 = util.global.load.indirect %29 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1152 = util.global.load.indirect %30 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1153 = util.global.load.indirect %31 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1154 = util.global.load.indirect %32 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1155 = util.global.load.indirect %33 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1156 = util.global.load.indirect %34 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1157 = util.global.load.indirect %35 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1158 = util.global.load.indirect %36 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1159 = util.global.load.indirect %37 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1160 = util.global.load.indirect %38 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1161 = util.global.load.indirect %39 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1162 = util.global.load.indirect %40 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1163 = util.global.load.indirect %41 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1164 = util.global.load.indirect %42 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1165 = util.global.load.indirect %43 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1166 = util.global.load.indirect %44 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1167 = util.global.load.indirect %45 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1168 = util.global.load.indirect %46 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1169 = util.global.load.indirect %47 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1170 = util.global.load.indirect %48 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1171 = util.global.load.indirect %49 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1172 = util.global.load.indirect %50 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1173 = util.global.load.indirect %51 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1174 = util.global.load.indirect %52 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1175 = util.global.load.indirect %53 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1176 = util.global.load.indirect %54 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1177 = util.global.load.indirect %55 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1178 = util.global.load.indirect %56 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1179 = util.global.load.indirect %57 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1180 = util.global.load.indirect %58 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1181 = util.global.load.indirect %59 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1182 = util.global.load.indirect %60 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1183 = util.global.load.indirect %61 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1184 = util.global.load.indirect %62 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1185 = util.global.load.indirect %63 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1186 = util.global.load.indirect %64 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1187 = util.global.load.indirect %65 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1188 = util.global.load.indirect %66 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1189 = util.global.load.indirect %67 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1190 = util.global.load.indirect %68 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1191 = util.global.load.indirect %69 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1192 = util.global.load.indirect %70 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1193 = util.global.load.indirect %71 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1194 = util.global.load.indirect %72 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1195 = util.global.load.indirect %73 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1196 = util.global.load.indirect %74 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1197 = util.global.load.indirect %75 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1198 = util.global.load.indirect %76 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1199 = util.global.load.indirect %77 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1200 = util.global.load.indirect %78 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1201 = util.global.load.indirect %79 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1202 = util.global.load.indirect %80 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1203 = util.global.load.indirect %81 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1204 = util.global.load.indirect %82 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1205 = util.global.load.indirect %83 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1206 = util.global.load.indirect %84 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1207 = util.global.load.indirect %85 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1208 = util.global.load.indirect %86 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1209 = util.global.load.indirect %87 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1210 = util.global.load.indirect %88 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1211 = util.global.load.indirect %89 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1212 = util.global.load.indirect %90 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1213 = util.global.load.indirect %91 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1214 = util.global.load.indirect %92 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1215 = util.global.load.indirect %93 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1216 = util.global.load.indirect %94 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1217 = util.global.load.indirect %95 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1218 = util.global.load.indirect %96 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1219 = util.global.load.indirect %97 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1220 = util.global.load.indirect %98 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1221 = util.global.load.indirect %99 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1222 = util.global.load.indirect %100 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1223 = util.global.load.indirect %101 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1224 = util.global.load.indirect %102 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1225 = util.global.load.indirect %103 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1226 = util.global.load.indirect %104 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1227 = util.global.load.indirect %105 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1228 = util.global.load.indirect %106 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1229 = util.global.load.indirect %107 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1230 = util.global.load.indirect %108 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1231 = util.global.load.indirect %109 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1232 = util.global.load.indirect %110 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1233 = util.global.load.indirect %111 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1234 = util.global.load.indirect %112 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1235 = util.global.load.indirect %113 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1236 = util.global.load.indirect %114 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1237 = util.global.load.indirect %115 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1238 = util.global.load.indirect %116 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1239 = util.global.load.indirect %117 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1240 = util.global.load.indirect %118 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1241 = util.global.load.indirect %119 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1242 = util.global.load.indirect %120 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1243 = util.global.load.indirect %121 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1244 = util.global.load.indirect %122 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1245 = util.global.load.indirect %123 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1246 = util.global.load.indirect %124 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1247 = util.global.load.indirect %125 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1248 = util.global.load.indirect %126 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1249 = util.global.load.indirect %127 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1250 = util.global.load.indirect %128 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1251 = util.global.load.indirect %129 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1252 = util.global.load.indirect %130 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1253 = util.global.load.indirect %131 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1254 = util.global.load.indirect %132 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1255 = util.global.load.indirect %133 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1256 = util.global.load.indirect %134 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1257 = util.global.load.indirect %135 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1258 = util.global.load.indirect %136 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1259 = util.global.load.indirect %137 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1260 = util.global.load.indirect %138 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1261 = util.global.load.indirect %139 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1262 = util.global.load.indirect %140 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1263 = util.global.load.indirect %141 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1264 = util.global.load.indirect %142 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1265 = util.global.load.indirect %143 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1266 = util.global.load.indirect %144 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1267 = util.global.load.indirect %145 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1268 = util.global.load.indirect %146 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1269 = util.global.load.indirect %147 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1270 = util.global.load.indirect %148 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1271 = util.global.load.indirect %149 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1272 = util.global.load.indirect %150 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1273 = util.global.load.indirect %151 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1274 = util.global.load.indirect %152 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1275 = util.global.load.indirect %153 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1276 = util.global.load.indirect %154 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1277 = util.global.load.indirect %155 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1278 = util.global.load.indirect %156 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1279 = util.global.load.indirect %157 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1280 = util.global.load.indirect %158 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1281 = util.global.load.indirect %159 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1282 = util.global.load.indirect %160 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1283 = util.global.load.indirect %161 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1284 = util.global.load.indirect %162 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1285 = util.global.load.indirect %163 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1286 = util.global.load.indirect %164 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1287 = util.global.load.indirect %165 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1288 = util.global.load.indirect %166 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1289 = util.global.load.indirect %167 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1290 = util.global.load.indirect %168 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1291 = util.global.load.indirect %169 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1292 = util.global.load.indirect %170 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1293 = util.global.load.indirect %171 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1294 = util.global.load.indirect %172 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1295 = util.global.load.indirect %173 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1296 = util.global.load.indirect %174 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1297 = util.global.load.indirect %175 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1298 = util.global.load.indirect %176 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1299 = util.global.load.indirect %177 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1300 = util.global.load.indirect %178 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1301 = util.global.load.indirect %179 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1302 = util.global.load.indirect %180 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1303 = util.global.load.indirect %181 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1304 = util.global.load.indirect %182 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1305 = util.global.load.indirect %183 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1306 = util.global.load.indirect %184 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1307 = util.global.load.indirect %185 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1308 = util.global.load.indirect %186 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1309 = util.global.load.indirect %187 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1310 = util.global.load.indirect %188 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1311 = util.global.load.indirect %189 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1312 = util.global.load.indirect %190 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1313 = util.global.load.indirect %191 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1314 = util.global.load.indirect %192 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1315 = util.global.load.indirect %193 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1316 = util.global.load.indirect %194 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1317 = util.global.load.indirect %195 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1318 = util.global.load.indirect %196 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1319 = util.global.load.indirect %197 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1320 = util.global.load.indirect %198 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1321 = util.global.load.indirect %199 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1322 = util.global.load.indirect %200 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1323 = util.global.load.indirect %201 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1324 = util.global.load.indirect %202 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1325 = util.global.load.indirect %203 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1326 = util.global.load.indirect %204 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1327 = util.global.load.indirect %205 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1328 = util.global.load.indirect %206 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1329 = util.global.load.indirect %207 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1330 = util.global.load.indirect %208 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1331 = util.global.load.indirect %209 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1332 = util.global.load.indirect %210 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1333 = util.global.load.indirect %211 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1334 = util.global.load.indirect %212 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1335 = util.global.load.indirect %213 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1336 = util.global.load.indirect %214 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1337 = util.global.load.indirect %215 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1338 = util.global.load.indirect %216 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1339 = util.global.load.indirect %217 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1340 = util.global.load.indirect %218 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1341 = util.global.load.indirect %219 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1342 = util.global.load.indirect %220 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1343 = util.global.load.indirect %221 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1344 = util.global.load.indirect %222 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1345 = util.global.load.indirect %223 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1346 = util.global.load.indirect %224 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1347 = util.global.load.indirect %225 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1348 = util.global.load.indirect %226 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1349 = util.global.load.indirect %227 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1350 = util.global.load.indirect %228 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1351 = util.global.load.indirect %229 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1352 = util.global.load.indirect %230 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1353 = util.global.load.indirect %231 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1354 = util.global.load.indirect %232 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1355 = util.global.load.indirect %233 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1356 = util.global.load.indirect %234 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1357 = util.global.load.indirect %235 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1358 = util.global.load.indirect %236 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1359 = util.global.load.indirect %237 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1360 = util.global.load.indirect %238 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1361 = util.global.load.indirect %239 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1362 = util.global.load.indirect %240 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1363 = util.global.load.indirect %241 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1364 = util.global.load.indirect %242 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1365 = util.global.load.indirect %243 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1366 = util.global.load.indirect %244 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1367 = util.global.load.indirect %245 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1368 = util.global.load.indirect %246 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1369 = util.global.load.indirect %247 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1370 = util.global.load.indirect %248 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1371 = util.global.load.indirect %249 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1372 = util.global.load.indirect %250 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1373 = util.global.load.indirect %251 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1374 = util.global.load.indirect %252 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1375 = util.global.load.indirect %253 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1376 = util.global.load.indirect %254 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1377 = util.global.load.indirect %255 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1378 = util.global.load.indirect %256 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1379 = util.global.load.indirect %257 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1380 = util.global.load.indirect %258 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1381 = util.global.load.indirect %259 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1382 = util.global.load.indirect %260 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1383 = util.global.load.indirect %261 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1384 = util.global.load.indirect %262 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1385 = util.global.load.indirect %263 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1386 = util.global.load.indirect %264 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1387 = util.global.load.indirect %265 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1388 = util.global.load.indirect %266 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1389 = util.global.load.indirect %267 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1390 = util.global.load.indirect %268 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1391 = util.global.load.indirect %269 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1392 = util.global.load.indirect %270 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1393 = util.global.load.indirect %271 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1394 = util.global.load.indirect %272 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1395 = util.global.load.indirect %273 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1396 = util.global.load.indirect %274 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1397 = util.global.load.indirect %275 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1398 = util.global.load.indirect %276 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1399 = util.global.load.indirect %277 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1400 = util.global.load.indirect %278 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1401 = util.global.load.indirect %279 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1402 = util.global.load.indirect %280 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1403 = util.global.load.indirect %281 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1404 = util.global.load.indirect %282 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1405 = util.global.load.indirect %283 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1406 = util.global.load.indirect %284 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1407 = util.global.load.indirect %285 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1408 = util.global.load.indirect %286 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1409 = util.global.load.indirect %287 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1410 = util.global.load.indirect %288 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1411 = util.global.load.indirect %289 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1412 = util.global.load.indirect %290 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1413 = util.global.load.indirect %291 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1414 = util.global.load.indirect %292 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1415 = util.global.load.indirect %293 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1416 = util.global.load.indirect %294 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1417 = util.global.load.indirect %295 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1418 = util.global.load.indirect %296 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1419 = util.global.load.indirect %297 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1420 = util.global.load.indirect %298 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1421 = util.global.load.indirect %299 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1422 = util.global.load.indirect %300 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1423 = util.global.load.indirect %301 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1424 = util.global.load.indirect %302 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1425 = util.global.load.indirect %303 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1426 = util.global.load.indirect %304 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1427 = util.global.load.indirect %305 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1428 = util.global.load.indirect %306 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1429 = util.global.load.indirect %307 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1430 = util.global.load.indirect %308 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1431 = util.global.load.indirect %309 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1432 = util.global.load.indirect %310 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1433 = util.global.load.indirect %311 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1434 = util.global.load.indirect %312 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1435 = util.global.load.indirect %313 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1436 = util.global.load.indirect %314 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1437 = util.global.load.indirect %315 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1438 = util.global.load.indirect %316 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1439 = util.global.load.indirect %317 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1440 = util.global.load.indirect %318 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1441 = util.global.load.indirect %319 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1442 = util.global.load.indirect %320 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1443 = util.global.load.indirect %321 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1444 = util.global.load.indirect %322 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1445 = util.global.load.indirect %323 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1446 = util.global.load.indirect %324 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1447 = util.global.load.indirect %325 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1448 = util.global.load.indirect %326 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1449 = util.global.load.indirect %327 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1450 = util.global.load.indirect %328 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1451 = util.global.load.indirect %329 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1452 = util.global.load.indirect %330 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1453 = util.global.load.indirect %331 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1454 = util.global.load.indirect %332 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1455 = util.global.load.indirect %333 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1456 = util.global.load.indirect %334 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1457 = util.global.load.indirect %335 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1458 = util.global.load.indirect %336 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1459 = util.global.load.indirect %337 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1460 = util.global.load.indirect %338 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1461 = util.global.load.indirect %339 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1462 = util.global.load.indirect %340 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1463 = util.global.load.indirect %341 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1464 = util.global.load.indirect %342 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1465 = util.global.load.indirect %343 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1466 = util.global.load.indirect %344 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1467 = util.global.load.indirect %345 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1468 = util.global.load.indirect %346 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1469 = util.global.load.indirect %347 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1470 = util.global.load.indirect %348 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1471 = util.global.load.indirect %349 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1472 = util.global.load.indirect %350 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1473 = util.global.load.indirect %351 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1474 = util.global.load.indirect %352 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1475 = util.global.load.indirect %353 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1476 = util.global.load.indirect %354 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1477 = util.global.load.indirect %355 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1478 = util.global.load.indirect %356 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1479 = util.global.load.indirect %357 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1480 = util.global.load.indirect %358 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1481 = util.global.load.indirect %359 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1482 = util.global.load.indirect %360 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1483 = util.global.load.indirect %361 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1484 = util.global.load.indirect %362 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1485 = util.global.load.indirect %363 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1486 = util.global.load.indirect %364 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1487 = util.global.load.indirect %365 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1488 = util.global.load.indirect %366 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1489 = util.global.load.indirect %367 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1490 = util.global.load.indirect %368 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1491 = util.global.load.indirect %369 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1492 = util.global.load.indirect %370 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1493 = util.global.load.indirect %371 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1494 = util.global.load.indirect %372 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1495 = util.global.load.indirect %373 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1496 = util.global.load.indirect %374 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1497 = util.global.load.indirect %375 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1498 = util.global.load.indirect %376 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1499 = util.global.load.indirect %377 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1500 = util.global.load.indirect %378 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1501 = util.global.load.indirect %379 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1502 = util.global.load.indirect %380 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1503 = util.global.load.indirect %381 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1504 = util.global.load.indirect %382 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1505 = util.global.load.indirect %383 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1506 = util.global.load.indirect %384 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1507 = util.global.load.indirect %385 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1508 = util.global.load.indirect %386 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1509 = util.global.load.indirect %387 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1510 = util.global.load.indirect %388 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1511 = util.global.load.indirect %389 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1512 = util.global.load.indirect %390 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1513 = util.global.load.indirect %391 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1514 = util.global.load.indirect %392 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1515 = util.global.load.indirect %393 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1516 = util.global.load.indirect %394 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1517 = util.global.load.indirect %395 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1518 = util.global.load.indirect %396 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1519 = util.global.load.indirect %397 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1520 = util.global.load.indirect %398 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1521 = util.global.load.indirect %399 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1522 = util.global.load.indirect %400 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1523 = util.global.load.indirect %401 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1524 = util.global.load.indirect %402 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1525 = util.global.load.indirect %403 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1526 = util.global.load.indirect %404 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1527 = util.global.load.indirect %405 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1528 = util.global.load.indirect %406 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1529 = util.global.load.indirect %407 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1530 = util.global.load.indirect %408 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1531 = util.global.load.indirect %409 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1532 = util.global.load.indirect %410 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1533 = util.global.load.indirect %411 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1534 = util.global.load.indirect %412 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1535 = util.global.load.indirect %413 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1536 = util.global.load.indirect %414 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1537 = util.global.load.indirect %415 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1538 = util.global.load.indirect %416 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1539 = util.global.load.indirect %417 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1540 = util.global.load.indirect %418 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1541 = util.global.load.indirect %419 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1542 = util.global.load.indirect %420 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1543 = util.global.load.indirect %421 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1544 = util.global.load.indirect %422 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1545 = util.global.load.indirect %423 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1546 = util.global.load.indirect %424 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1547 = util.global.load.indirect %425 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1548 = util.global.load.indirect %426 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1549 = util.global.load.indirect %427 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1550 = util.global.load.indirect %428 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1551 = util.global.load.indirect %429 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1552 = util.global.load.indirect %430 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1553 = util.global.load.indirect %431 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1554 = util.global.load.indirect %432 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1555 = util.global.load.indirect %433 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1556 = util.global.load.indirect %434 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1557 = util.global.load.indirect %435 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1558 = util.global.load.indirect %436 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1559 = util.global.load.indirect %437 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1560 = util.global.load.indirect %438 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1561 = util.global.load.indirect %439 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1562 = util.global.load.indirect %440 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1563 = util.global.load.indirect %441 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1564 = util.global.load.indirect %442 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1565 = util.global.load.indirect %443 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1566 = util.global.load.indirect %444 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1567 = util.global.load.indirect %445 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1568 = util.global.load.indirect %446 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1569 = util.global.load.indirect %447 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1570 = util.global.load.indirect %448 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1571 = util.global.load.indirect %449 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1572 = util.global.load.indirect %450 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1573 = util.global.load.indirect %451 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1574 = util.global.load.indirect %452 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1575 = util.global.load.indirect %453 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1576 = util.global.load.indirect %454 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1577 = util.global.load.indirect %455 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1578 = util.global.load.indirect %456 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1579 = util.global.load.indirect %457 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1580 = util.global.load.indirect %458 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1581 = util.global.load.indirect %459 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1582 = util.global.load.indirect %460 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1583 = util.global.load.indirect %461 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1584 = util.global.load.indirect %462 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1585 = util.global.load.indirect %463 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1586 = util.global.load.indirect %464 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1587 = util.global.load.indirect %465 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1588 = util.global.load.indirect %466 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1589 = util.global.load.indirect %467 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1590 = util.global.load.indirect %468 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1591 = util.global.load.indirect %469 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1592 = util.global.load.indirect %470 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1593 = util.global.load.indirect %471 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1594 = util.global.load.indirect %472 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1595 = util.global.load.indirect %473 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1596 = util.global.load.indirect %474 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1597 = util.global.load.indirect %475 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1598 = util.global.load.indirect %476 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1599 = util.global.load.indirect %477 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1600 = util.global.load.indirect %478 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1601 = util.global.load.indirect %479 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1602 = util.global.load.indirect %480 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1603 = util.global.load.indirect %481 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1604 = util.global.load.indirect %482 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1605 = util.global.load.indirect %483 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1606 = util.global.load.indirect %484 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1607 = util.global.load.indirect %485 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1608 = util.global.load.indirect %486 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1609 = util.global.load.indirect %487 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1610 = util.global.load.indirect %488 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1611 = util.global.load.indirect %489 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1612 = util.global.load.indirect %490 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1613 = util.global.load.indirect %491 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1614 = util.global.load.indirect %492 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1615 = util.global.load.indirect %493 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1616 = util.global.load.indirect %494 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1617 = util.global.load.indirect %495 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1618 = util.global.load.indirect %496 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1619 = util.global.load.indirect %497 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1620 = util.global.load.indirect %498 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1621 = util.global.load.indirect %499 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1622 = util.global.load.indirect %500 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1623 = util.global.load.indirect %501 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1624 = util.global.load.indirect %502 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1625 = util.global.load.indirect %503 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1626 = util.global.load.indirect %504 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1627 = util.global.load.indirect %505 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1628 = util.global.load.indirect %506 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1629 = util.global.load.indirect %507 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1630 = util.global.load.indirect %508 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1631 = util.global.load.indirect %509 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1632 = util.global.load.indirect %510 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1633 = util.global.load.indirect %511 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1634 = util.global.load.indirect %512 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1635 = util.global.load.indirect %513 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1636 = util.global.load.indirect %514 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1637 = util.global.load.indirect %515 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1638 = util.global.load.indirect %516 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1639 = util.global.load.indirect %517 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1640 = util.global.load.indirect %518 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1641 = util.global.load.indirect %519 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1642 = util.global.load.indirect %520 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1643 = util.global.load.indirect %521 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1644 = util.global.load.indirect %522 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1645 = util.global.load.indirect %523 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1646 = util.global.load.indirect %524 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1647 = util.global.load.indirect %525 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1648 = util.global.load.indirect %526 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1649 = util.global.load.indirect %527 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1650 = util.global.load.indirect %528 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1651 = util.global.load.indirect %529 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1652 = util.global.load.indirect %530 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1653 = util.global.load.indirect %531 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1654 = util.global.load.indirect %532 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1655 = util.global.load.indirect %533 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1656 = util.global.load.indirect %534 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1657 = util.global.load.indirect %535 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1658 = util.global.load.indirect %536 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1659 = util.global.load.indirect %537 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1660 = util.global.load.indirect %538 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1661 = util.global.load.indirect %539 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1662 = util.global.load.indirect %540 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1663 = util.global.load.indirect %541 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1664 = util.global.load.indirect %542 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1665 = util.global.load.indirect %543 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1666 = util.global.load.indirect %544 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1667 = util.global.load.indirect %545 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1668 = util.global.load.indirect %546 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1669 = util.global.load.indirect %547 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1670 = util.global.load.indirect %548 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1671 = util.global.load.indirect %549 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1672 = util.global.load.indirect %550 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1673 = util.global.load.indirect %551 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1674 = util.global.load.indirect %552 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1675 = util.global.load.indirect %553 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1676 = util.global.load.indirect %554 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1677 = util.global.load.indirect %555 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1678 = util.global.load.indirect %556 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1679 = util.global.load.indirect %557 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1680 = util.global.load.indirect %558 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1681 = util.global.load.indirect %559 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1682 = util.global.load.indirect %560 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1683 = util.global.load.indirect %561 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1684 = util.global.load.indirect %562 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1685 = util.global.load.indirect %563 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1686 = util.global.load.indirect %564 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1687 = util.global.load.indirect %565 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1688 = util.global.load.indirect %566 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1689 = util.global.load.indirect %567 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1690 = util.global.load.indirect %568 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1691 = util.global.load.indirect %569 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1692 = util.global.load.indirect %570 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1693 = util.global.load.indirect %571 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1694 = util.global.load.indirect %572 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1695 = util.global.load.indirect %573 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1696 = util.global.load.indirect %574 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1697 = util.global.load.indirect %575 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1698 = util.global.load.indirect %576 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1699 = util.global.load.indirect %577 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1700 = util.global.load.indirect %578 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1701 = util.global.load.indirect %579 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1702 = util.global.load.indirect %580 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1703 = util.global.load.indirect %581 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1704 = util.global.load.indirect %582 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1705 = util.global.load.indirect %583 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1706 = util.global.load.indirect %584 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1707 = util.global.load.indirect %585 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1708 = util.global.load.indirect %586 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1709 = util.global.load.indirect %587 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1710 = util.global.load.indirect %588 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1711 = util.global.load.indirect %589 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1712 = util.global.load.indirect %590 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1713 = util.global.load.indirect %591 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1714 = util.global.load.indirect %592 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1715 = util.global.load.indirect %593 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1716 = util.global.load.indirect %594 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1717 = util.global.load.indirect %595 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1718 = util.global.load.indirect %596 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1719 = util.global.load.indirect %597 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1720 = util.global.load.indirect %598 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1721 = util.global.load.indirect %599 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1722 = util.global.load.indirect %600 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1723 = util.global.load.indirect %601 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1724 = util.global.load.indirect %602 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1725 = util.global.load.indirect %603 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1726 = util.global.load.indirect %604 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1727 = util.global.load.indirect %605 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1728 = util.global.load.indirect %606 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1729 = util.global.load.indirect %607 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1730 = util.global.load.indirect %608 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1731 = util.global.load.indirect %609 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1732 = util.global.load.indirect %610 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1733 = util.global.load.indirect %611 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1734 = util.global.load.indirect %612 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1735 = util.global.load.indirect %613 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1736 = util.global.load.indirect %614 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1737 = util.global.load.indirect %615 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1738 = util.global.load.indirect %616 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1739 = util.global.load.indirect %617 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1740 = util.global.load.indirect %618 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1741 = util.global.load.indirect %619 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1742 = util.global.load.indirect %620 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1743 = util.global.load.indirect %621 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1744 = util.global.load.indirect %622 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1745 = util.global.load.indirect %623 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1746 = util.global.load.indirect %624 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1747 = util.global.load.indirect %625 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1748 = util.global.load.indirect %626 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1749 = util.global.load.indirect %627 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1750 = util.global.load.indirect %628 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1751 = util.global.load.indirect %629 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1752 = util.global.load.indirect %630 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1753 = util.global.load.indirect %631 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1754 = util.global.load.indirect %632 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1755 = util.global.load.indirect %633 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1756 = util.global.load.indirect %634 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1757 = util.global.load.indirect %635 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1758 = util.global.load.indirect %636 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1759 = util.global.load.indirect %637 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1760 = util.global.load.indirect %638 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1761 = util.global.load.indirect %639 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1762 = util.global.load.indirect %640 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1763 = util.global.load.indirect %641 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1764 = util.global.load.indirect %642 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1765 = util.global.load.indirect %643 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1766 = util.global.load.indirect %644 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1767 = util.global.load.indirect %645 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1768 = util.global.load.indirect %646 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1769 = util.global.load.indirect %647 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1770 = util.global.load.indirect %648 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1771 = util.global.load.indirect %649 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1772 = util.global.load.indirect %650 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1773 = util.global.load.indirect %651 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1774 = util.global.load.indirect %652 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1775 = util.global.load.indirect %653 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1776 = util.global.load.indirect %654 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1777 = util.global.load.indirect %655 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1778 = util.global.load.indirect %656 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1779 = util.global.load.indirect %657 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1780 = util.global.load.indirect %658 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1781 = util.global.load.indirect %659 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1782 = util.global.load.indirect %660 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1783 = util.global.load.indirect %661 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1784 = util.global.load.indirect %662 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1785 = util.global.load.indirect %663 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1786 = util.global.load.indirect %664 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1787 = util.global.load.indirect %665 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1788 = util.global.load.indirect %666 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1789 = util.global.load.indirect %667 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1790 = util.global.load.indirect %668 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1791 = util.global.load.indirect %669 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1792 = util.global.load.indirect %670 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1793 = util.global.load.indirect %671 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1794 = util.global.load.indirect %672 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1795 = util.global.load.indirect %673 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1796 = util.global.load.indirect %674 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1797 = util.global.load.indirect %675 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1798 = util.global.load.indirect %676 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1799 = util.global.load.indirect %677 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1800 = util.global.load.indirect %678 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1801 = util.global.load.indirect %679 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1802 = util.global.load.indirect %680 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1803 = util.global.load.indirect %681 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1804 = util.global.load.indirect %682 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1805 = util.global.load.indirect %683 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1806 = util.global.load.indirect %684 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1807 = util.global.load.indirect %685 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1808 = util.global.load.indirect %686 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1809 = util.global.load.indirect %687 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1810 = util.global.load.indirect %688 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1811 = util.global.load.indirect %689 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1812 = util.global.load.indirect %690 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1813 = util.global.load.indirect %691 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1814 = util.global.load.indirect %692 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1815 = util.global.load.indirect %693 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1816 = util.global.load.indirect %694 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1817 = util.global.load.indirect %695 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1818 = util.global.load.indirect %696 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1819 = util.global.load.indirect %697 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1820 = util.global.load.indirect %698 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1821 = util.global.load.indirect %699 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1822 = util.global.load.indirect %700 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1823 = util.global.load.indirect %701 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1824 = util.global.load.indirect %702 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1825 = util.global.load.indirect %703 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1826 = util.global.load.indirect %704 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1827 = util.global.load.indirect %705 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1828 = util.global.load.indirect %706 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1829 = util.global.load.indirect %707 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1830 = util.global.load.indirect %708 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1831 = util.global.load.indirect %709 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1832 = util.global.load.indirect %710 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1833 = util.global.load.indirect %711 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1834 = util.global.load.indirect %712 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1835 = util.global.load.indirect %713 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1836 = util.global.load.indirect %714 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1837 = util.global.load.indirect %715 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1838 = util.global.load.indirect %716 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1839 = util.global.load.indirect %717 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1840 = util.global.load.indirect %718 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1841 = util.global.load.indirect %719 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1842 = util.global.load.indirect %720 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1843 = util.global.load.indirect %721 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1844 = util.global.load.indirect %722 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1845 = util.global.load.indirect %723 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1846 = util.global.load.indirect %724 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1847 = util.global.load.indirect %725 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1848 = util.global.load.indirect %726 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1849 = util.global.load.indirect %727 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1850 = util.global.load.indirect %728 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1851 = util.global.load.indirect %729 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1852 = util.global.load.indirect %730 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1853 = util.global.load.indirect %731 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1854 = util.global.load.indirect %732 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1855 = util.global.load.indirect %733 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1856 = util.global.load.indirect %734 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1857 = util.global.load.indirect %735 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1858 = util.global.load.indirect %736 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1859 = util.global.load.indirect %737 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1860 = util.global.load.indirect %738 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1861 = util.global.load.indirect %739 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1862 = util.global.load.indirect %740 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1863 = util.global.load.indirect %741 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1864 = util.global.load.indirect %742 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1865 = util.global.load.indirect %743 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1866 = util.global.load.indirect %744 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1867 = util.global.load.indirect %745 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1868 = util.global.load.indirect %746 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1869 = util.global.load.indirect %747 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1870 = util.global.load.indirect %748 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1871 = util.global.load.indirect %749 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1872 = util.global.load.indirect %750 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1873 = util.global.load.indirect %751 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1874 = util.global.load.indirect %752 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1875 = util.global.load.indirect %753 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1876 = util.global.load.indirect %754 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1877 = util.global.load.indirect %755 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1878 = util.global.load.indirect %756 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1879 = util.global.load.indirect %757 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1880 = util.global.load.indirect %758 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1881 = util.global.load.indirect %759 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1882 = util.global.load.indirect %760 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1883 = util.global.load.indirect %761 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1884 = util.global.load.indirect %762 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1885 = util.global.load.indirect %763 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1886 = util.global.load.indirect %764 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1887 = util.global.load.indirect %765 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1888 = util.global.load.indirect %766 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1889 = util.global.load.indirect %767 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1890 = util.global.load.indirect %768 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1891 = util.global.load.indirect %769 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1892 = util.global.load.indirect %770 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1893 = util.global.load.indirect %771 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1894 = util.global.load.indirect %772 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1895 = util.global.load.indirect %773 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1896 = util.global.load.indirect %774 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1897 = util.global.load.indirect %775 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1898 = util.global.load.indirect %776 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1899 = util.global.load.indirect %777 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1900 = util.global.load.indirect %778 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1901 = util.global.load.indirect %779 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1902 = util.global.load.indirect %780 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1903 = util.global.load.indirect %781 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1904 = util.global.load.indirect %782 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1905 = util.global.load.indirect %783 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1906 = util.global.load.indirect %784 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1907 = util.global.load.indirect %785 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1908 = util.global.load.indirect %786 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1909 = util.global.load.indirect %787 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1910 = util.global.load.indirect %788 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1911 = util.global.load.indirect %789 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1912 = util.global.load.indirect %790 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1913 = util.global.load.indirect %791 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1914 = util.global.load.indirect %792 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1915 = util.global.load.indirect %793 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1916 = util.global.load.indirect %794 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1917 = util.global.load.indirect %795 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1918 = util.global.load.indirect %796 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1919 = util.global.load.indirect %797 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1920 = util.global.load.indirect %798 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1921 = util.global.load.indirect %799 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1922 = util.global.load.indirect %800 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1923 = util.global.load.indirect %801 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1924 = util.global.load.indirect %802 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1925 = util.global.load.indirect %803 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1926 = util.global.load.indirect %804 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1927 = util.global.load.indirect %805 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1928 = util.global.load.indirect %806 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1929 = util.global.load.indirect %807 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1930 = util.global.load.indirect %808 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1931 = util.global.load.indirect %809 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1932 = util.global.load.indirect %810 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1933 = util.global.load.indirect %811 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1934 = util.global.load.indirect %812 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1935 = util.global.load.indirect %813 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1936 = util.global.load.indirect %814 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1937 = util.global.load.indirect %815 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1938 = util.global.load.indirect %816 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1939 = util.global.load.indirect %817 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1940 = util.global.load.indirect %818 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1941 = util.global.load.indirect %819 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1942 = util.global.load.indirect %820 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1943 = util.global.load.indirect %821 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1944 = util.global.load.indirect %822 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1945 = util.global.load.indirect %823 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1946 = util.global.load.indirect %824 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1947 = util.global.load.indirect %825 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1948 = util.global.load.indirect %826 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1949 = util.global.load.indirect %827 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1950 = util.global.load.indirect %828 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1951 = util.global.load.indirect %829 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1952 = util.global.load.indirect %830 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1953 = util.global.load.indirect %831 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1954 = util.global.load.indirect %832 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1955 = util.global.load.indirect %833 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1956 = util.global.load.indirect %834 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1957 = util.global.load.indirect %835 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1958 = util.global.load.indirect %836 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1959 = util.global.load.indirect %837 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1960 = util.global.load.indirect %838 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1961 = util.global.load.indirect %839 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1962 = util.global.load.indirect %840 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1963 = util.global.load.indirect %841 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1964 = util.global.load.indirect %842 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %1965 = util.global.load.indirect %843 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1966 = util.global.load.indirect %844 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1967 = util.global.load.indirect %845 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1968 = util.global.load.indirect %846 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1969 = util.global.load.indirect %847 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1970 = util.global.load.indirect %848 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1971 = util.global.load.indirect %849 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1972 = util.global.load.indirect %850 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1973 = util.global.load.indirect %851 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1974 = util.global.load.indirect %852 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1975 = util.global.load.indirect %853 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1976 = util.global.load.indirect %854 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1977 = util.global.load.indirect %855 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1978 = util.global.load.indirect %856 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1979 = util.global.load.indirect %857 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1980 = util.global.load.indirect %858 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1981 = util.global.load.indirect %859 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1982 = util.global.load.indirect %860 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1983 = util.global.load.indirect %861 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1984 = util.global.load.indirect %862 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1985 = util.global.load.indirect %863 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1986 = util.global.load.indirect %864 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1987 = util.global.load.indirect %865 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1988 = util.global.load.indirect %866 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1989 = util.global.load.indirect %867 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1990 = util.global.load.indirect %868 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1991 = util.global.load.indirect %869 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1992 = util.global.load.indirect %870 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %1993 = util.global.load.indirect %871 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1994 = util.global.load.indirect %872 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %1995 = util.global.load.indirect %873 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1996 = util.global.load.indirect %874 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %1997 = util.global.load.indirect %875 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1998 = util.global.load.indirect %876 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %1999 = util.global.load.indirect %877 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2000 = util.global.load.indirect %878 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2001 = util.global.load.indirect %879 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2002 = util.global.load.indirect %880 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2003 = util.global.load.indirect %881 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2004 = util.global.load.indirect %882 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2005 = util.global.load.indirect %883 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2006 = util.global.load.indirect %884 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2007 = util.global.load.indirect %885 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2008 = util.global.load.indirect %886 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2009 = util.global.load.indirect %887 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2010 = util.global.load.indirect %888 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2011 = util.global.load.indirect %889 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2012 = util.global.load.indirect %890 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2013 = util.global.load.indirect %891 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2014 = util.global.load.indirect %892 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2015 = util.global.load.indirect %893 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2016 = util.global.load.indirect %894 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2017 = util.global.load.indirect %895 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2018 = util.global.load.indirect %896 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2019 = util.global.load.indirect %897 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2020 = util.global.load.indirect %898 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2021 = util.global.load.indirect %899 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2022 = util.global.load.indirect %900 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2023 = util.global.load.indirect %901 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2024 = util.global.load.indirect %902 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2025 = util.global.load.indirect %903 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2026 = util.global.load.indirect %904 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2027 = util.global.load.indirect %905 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2028 = util.global.load.indirect %906 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2029 = util.global.load.indirect %907 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2030 = util.global.load.indirect %908 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2031 = util.global.load.indirect %909 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2032 = util.global.load.indirect %910 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2033 = util.global.load.indirect %911 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2034 = util.global.load.indirect %912 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2035 = util.global.load.indirect %913 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2036 = util.global.load.indirect %914 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2037 = util.global.load.indirect %915 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2038 = util.global.load.indirect %916 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2039 = util.global.load.indirect %917 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2040 = util.global.load.indirect %918 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2041 = util.global.load.indirect %919 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2042 = util.global.load.indirect %920 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2043 = util.global.load.indirect %921 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2044 = util.global.load.indirect %922 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2045 = util.global.load.indirect %923 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2046 = util.global.load.indirect %924 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2047 = util.global.load.indirect %925 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2048 = util.global.load.indirect %926 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2049 = util.global.load.indirect %927 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2050 = util.global.load.indirect %928 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2051 = util.global.load.indirect %929 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2052 = util.global.load.indirect %930 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2053 = util.global.load.indirect %931 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2054 = util.global.load.indirect %932 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2055 = util.global.load.indirect %933 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2056 = util.global.load.indirect %934 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2057 = util.global.load.indirect %935 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2058 = util.global.load.indirect %936 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2059 = util.global.load.indirect %937 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2060 = util.global.load.indirect %938 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2061 = util.global.load.indirect %939 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2062 = util.global.load.indirect %940 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2063 = util.global.load.indirect %941 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2064 = util.global.load.indirect %942 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2065 = util.global.load.indirect %943 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2066 = util.global.load.indirect %944 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2067 = util.global.load.indirect %945 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2068 = util.global.load.indirect %946 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2069 = util.global.load.indirect %947 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2070 = util.global.load.indirect %948 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2071 = util.global.load.indirect %949 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2072 = util.global.load.indirect %950 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2073 = util.global.load.indirect %951 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2074 = util.global.load.indirect %952 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2075 = util.global.load.indirect %953 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2076 = util.global.load.indirect %954 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2077 = util.global.load.indirect %955 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2078 = util.global.load.indirect %956 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2079 = util.global.load.indirect %957 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2080 = util.global.load.indirect %958 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2081 = util.global.load.indirect %959 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2082 = util.global.load.indirect %960 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2083 = util.global.load.indirect %961 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2084 = util.global.load.indirect %962 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2085 = util.global.load.indirect %963 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2086 = util.global.load.indirect %964 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2087 = util.global.load.indirect %965 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2088 = util.global.load.indirect %966 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2089 = util.global.load.indirect %967 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2090 = util.global.load.indirect %968 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2091 = util.global.load.indirect %969 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2092 = util.global.load.indirect %970 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2093 = util.global.load.indirect %971 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2094 = util.global.load.indirect %972 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2095 = util.global.load.indirect %973 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2096 = util.global.load.indirect %974 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2097 = util.global.load.indirect %975 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2098 = util.global.load.indirect %976 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2099 = util.global.load.indirect %977 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2100 = util.global.load.indirect %978 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2101 = util.global.load.indirect %979 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2102 = util.global.load.indirect %980 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2103 = util.global.load.indirect %981 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2104 = util.global.load.indirect %982 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2105 = util.global.load.indirect %983 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2106 = util.global.load.indirect %984 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2107 = util.global.load.indirect %985 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2108 = util.global.load.indirect %986 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2109 = util.global.load.indirect %987 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2110 = util.global.load.indirect %988 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2111 = util.global.load.indirect %989 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2112 = util.global.load.indirect %990 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2113 = util.global.load.indirect %991 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2114 = util.global.load.indirect %992 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2115 = util.global.load.indirect %993 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2116 = util.global.load.indirect %994 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2117 = util.global.load.indirect %995 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2118 = util.global.load.indirect %996 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2119 = util.global.load.indirect %997 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2120 = util.global.load.indirect %998 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2121 = util.global.load.indirect %999 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2122 = util.global.load.indirect %1000 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2123 = util.global.load.indirect %1001 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2124 = util.global.load.indirect %1002 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2125 = util.global.load.indirect %1003 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2126 = util.global.load.indirect %1004 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2127 = util.global.load.indirect %1005 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2128 = util.global.load.indirect %1006 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2129 = util.global.load.indirect %1007 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2130 = util.global.load.indirect %1008 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2131 = util.global.load.indirect %1009 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2132 = util.global.load.indirect %1010 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2133 = util.global.load.indirect %1011 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2134 = util.global.load.indirect %1012 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2135 = util.global.load.indirect %1013 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2136 = util.global.load.indirect %1014 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2137 = util.global.load.indirect %1015 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2138 = util.global.load.indirect %1016 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2139 = util.global.load.indirect %1017 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2140 = util.global.load.indirect %1018 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2141 = util.global.load.indirect %1019 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2142 = util.global.load.indirect %1020 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2143 = util.global.load.indirect %1021 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2144 = util.global.load.indirect %1022 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2145 = util.global.load.indirect %1023 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2146 = util.global.load.indirect %1024 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2147 = util.global.load.indirect %1025 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2148 = util.global.load.indirect %1026 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2149 = util.global.load.indirect %1027 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2150 = util.global.load.indirect %1028 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2151 = util.global.load.indirect %1029 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2152 = util.global.load.indirect %1030 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2153 = util.global.load.indirect %1031 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2154 = util.global.load.indirect %1032 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2155 = util.global.load.indirect %1033 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2156 = util.global.load.indirect %1034 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2157 = util.global.load.indirect %1035 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2158 = util.global.load.indirect %1036 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2159 = util.global.load.indirect %1037 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2160 = util.global.load.indirect %1038 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2161 = util.global.load.indirect %1039 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2162 = util.global.load.indirect %1040 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2163 = util.global.load.indirect %1041 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2164 = util.global.load.indirect %1042 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2165 = util.global.load.indirect %1043 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2166 = util.global.load.indirect %1044 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2167 = util.global.load.indirect %1045 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2168 = util.global.load.indirect %1046 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2169 = util.global.load.indirect %1047 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2170 = util.global.load.indirect %1048 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2171 = util.global.load.indirect %1049 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2172 = util.global.load.indirect %1050 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2173 = util.global.load.indirect %1051 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2174 = util.global.load.indirect %1052 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2175 = util.global.load.indirect %1053 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2176 = util.global.load.indirect %1054 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2177 = util.global.load.indirect %1055 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2178 = util.global.load.indirect %1056 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2179 = util.global.load.indirect %1057 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2180 = util.global.load.indirect %1058 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2181 = util.global.load.indirect %1059 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2182 = util.global.load.indirect %1060 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2183 = util.global.load.indirect %1061 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2184 = util.global.load.indirect %1062 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2185 = util.global.load.indirect %1063 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2186 = util.global.load.indirect %1064 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2187 = util.global.load.indirect %1065 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2188 = util.global.load.indirect %1066 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2189 = util.global.load.indirect %1067 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2190 = util.global.load.indirect %1068 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2191 = util.global.load.indirect %1069 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2192 = util.global.load.indirect %1070 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2193 = util.global.load.indirect %1071 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2194 = util.global.load.indirect %1072 : !util.ptr<tensor<128x128xf32>> -> tensor<128x128xf32>
    %2195 = util.global.load.indirect %1073 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2196 = util.global.load.indirect %1074 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2197 = util.global.load.indirect %1075 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2198 = util.global.load.indirect %1076 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2199 = util.global.load.indirect %1077 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2200 = util.global.load.indirect %1078 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2201 = util.global.load.indirect %1079 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2202 = util.global.load.indirect %1080 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2203 = util.global.load.indirect %1081 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2204 = util.global.load.indirect %1082 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2205 = util.global.load.indirect %1083 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2206 = util.global.load.indirect %1084 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2207 = util.global.load.indirect %1085 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2208 = util.global.load.indirect %1086 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2209 = util.global.load.indirect %1087 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2210 = util.global.load.indirect %1088 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2211 = util.global.load.indirect %1089 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2212 = util.global.load.indirect %1090 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2213 = util.global.load.indirect %1091 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2214 = util.global.load.indirect %1092 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2215 = util.global.load.indirect %1093 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2216 = util.global.load.indirect %1094 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2217 = util.global.load.indirect %1095 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2218 = util.global.load.indirect %1096 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2219 = util.global.load.indirect %1097 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2220 = util.global.load.indirect %1098 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2221 = util.global.load.indirect %1099 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2222 = util.global.load.indirect %1100 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2223 = util.global.load.indirect %1101 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2224 = util.global.load.indirect %1102 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2225 = util.global.load.indirect %1103 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2226 = util.global.load.indirect %1104 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2227 = util.global.load.indirect %1105 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2228 = util.global.load.indirect %1106 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2229 = util.global.load.indirect %1107 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
    %2230 = util.global.load.indirect %1108 : !util.ptr<tensor<128x512xf32>> -> tensor<128x512xf32>
    %2231 = util.global.load.indirect %1109 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %2232 = util.global.load.indirect %1110 : !util.ptr<tensor<512x128xf32>> -> tensor<512x128xf32>
    %2233 = util.global.load.indirect %1111 : !util.ptr<tensor<2xf32>> -> tensor<2xf32>
    %2234 = util.global.load.indirect %1112 : !util.ptr<tensor<2x512xf32>> -> tensor<2x512xf32>
    %2235 = "mhlo.reshape"(%arg1) : (tensor<1x384xi32>) -> tensor<1x384x1xi32>
    %2236 = "mhlo.torch_index_select"(%1128, %2235) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<30522x128xf32>, tensor<1x384x1xi32>) -> tensor<1x384x1x128xf32>
    %2237 = "mhlo.reshape"(%2236) : (tensor<1x384x1x128xf32>) -> tensor<1x384x128xf32>
    %2238 = "mhlo.slice"(%2237) {limit_indices = dense<[1, 384, 128]> : tensor<3xi64>, start_indices = dense<[0, 1, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x384x128xf32>) -> tensor<1x383x128xf32>
    %2239 = "mhlo.pad"(%2238, %1118) {edge_padding_high = dense<[0, 1, 0]> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>} : (tensor<1x383x128xf32>, tensor<f32>) -> tensor<1x384x128xf32>
    %2240 = "mhlo.slice"(%2237) {limit_indices = dense<[1, 383, 128]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x384x128xf32>) -> tensor<1x383x128xf32>
    %2241 = "mhlo.pad"(%2240, %1118) {edge_padding_high = dense<0> : tensor<3xi64>, edge_padding_low = dense<[0, 1, 0]> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>} : (tensor<1x383x128xf32>, tensor<f32>) -> tensor<1x384x128xf32>
    %2242 = "mhlo.concatenate"(%2239, %2237, %2241) {dimension = 2 : i64} : (tensor<1x384x128xf32>, tensor<1x384x128xf32>, tensor<1x384x128xf32>) -> tensor<1x384x384xf32>
    %2243 = "mhlo.reshape"(%2242) : (tensor<1x384x384xf32>) -> tensor<384x384xf32>
    %2244 = "mhlo.dot"(%2243, %1123) : (tensor<384x384xf32>, tensor<384x512xf32>) -> tensor<384x512xf32>
    %2245 = "mhlo.broadcast_in_dim"(%1122) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2246 = mhlo.add %2244, %2245 : tensor<384x512xf32>
    %2247 = "mhlo.reshape"(%2246) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2248 = "mhlo.convert"(%arg0) : (tensor<1x384xi32>) -> tensor<1x384xf32>
    %2249 = "mhlo.reshape"(%2248) : (tensor<1x384xf32>) -> tensor<1x1x384xf32>
    %2250 = "mhlo.broadcast_in_dim"(%2249) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
    %2251 = mhlo.multiply %2250, %1116 : tensor<1x384x384xf32>
    %2252 = "mhlo.reshape"(%2251) : (tensor<1x384x384xf32>) -> tensor<1x1x384x384xf32>
    %2253 = mhlo.multiply %2252, %1115 : tensor<1x1x384x384xf32>
    %2254 = mhlo.add %2253, %1113 : tensor<1x1x384x384xf32>
    %2255 = "mhlo.torch_index_select"(%1127, %arg2) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<2x512xf32>, tensor<1x384xi32>) -> tensor<1x384x512xf32>
    %2256 = mhlo.add %2247, %2255 : tensor<1x384x512xf32>
    %2257 = mhlo.add %2256, %1126 : tensor<1x384x512xf32>
    %2258 = "mhlo.broadcast_in_dim"(%1121) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2259 = mhlo.multiply %2257, %2258 : tensor<1x384x512xf32>
    %2260 = "mhlo.broadcast_in_dim"(%1120) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2261 = mhlo.add %2259, %2260 : tensor<1x384x512xf32>
    %2262 = "mhlo.reshape"(%2261) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2263 = "mhlo.dot"(%2262, %1138) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2264 = "mhlo.broadcast_in_dim"(%1137) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2265 = mhlo.add %2263, %2264 : tensor<384x128xf32>
    %2266 = "mhlo.reshape"(%2265) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2267 = "mhlo.transpose"(%2266) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2268 = "mhlo.dot"(%2262, %1142) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2269 = "mhlo.reshape"(%2268) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2270 = "mhlo.broadcast_in_dim"(%1141) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2271 = mhlo.add %2269, %2270 : tensor<1x384x128xf32>
    %2272 = "mhlo.broadcast_in_dim"(%1140) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2273 = mhlo.multiply %2271, %2272 : tensor<1x384x128xf32>
    %2274 = "mhlo.broadcast_in_dim"(%1139) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2275 = mhlo.add %2273, %2274 : tensor<1x384x128xf32>
    %2276 = "mhlo.reshape"(%2275) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2277 = "mhlo.dot"(%2276, %1134) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2278 = "mhlo.broadcast_in_dim"(%1133) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2279 = mhlo.add %2277, %2278 : tensor<384x128xf32>
    %2280 = "mhlo.reshape"(%2279) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2281 = "mhlo.transpose"(%2280) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2282 = "mhlo.dot"(%2276, %1136) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2283 = "mhlo.broadcast_in_dim"(%1135) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2284 = mhlo.add %2282, %2283 : tensor<384x128xf32>
    %2285 = "mhlo.reshape"(%2284) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2286 = "mhlo.transpose"(%2285) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2287 = "mhlo.dot_general"(%2286, %2281) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2288 = mhlo.multiply %2287, %1114 : tensor<1x4x384x384xf32>
    %2289 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2290 = mhlo.add %2288, %2289 : tensor<1x4x384x384xf32>
    %2291 = "mhlo.reduce"(%2290, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2292 = "mhlo.broadcast_in_dim"(%2291) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2293 = mhlo.subtract %2290, %2292 : tensor<1x4x384x384xf32>
    %2294 = "mhlo.exponential"(%2293) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2295 = "mhlo.reduce"(%2294, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2296 = "mhlo.broadcast_in_dim"(%2295) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2297 = mhlo.divide %2294, %2296 : tensor<1x4x384x384xf32>
    %2298 = "mhlo.dot_general"(%2297, %2267) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2299 = "mhlo.transpose"(%2298) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2300 = "mhlo.reshape"(%2299) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2301 = "mhlo.dot"(%2300, %1132) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2302 = "mhlo.broadcast_in_dim"(%1131) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2303 = mhlo.add %2301, %2302 : tensor<384x128xf32>
    %2304 = "mhlo.reshape"(%2303) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2305 = "mhlo.dot"(%2262, %1146) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2306 = "mhlo.broadcast_in_dim"(%1145) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2307 = mhlo.add %2305, %2306 : tensor<384x128xf32>
    %2308 = "mhlo.reshape"(%2307) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2309 = "mhlo.broadcast_in_dim"(%1144) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2310 = mhlo.multiply %2308, %2309 : tensor<1x384x128xf32>
    %2311 = "mhlo.broadcast_in_dim"(%1143) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2312 = mhlo.add %2310, %2311 : tensor<1x384x128xf32>
    %2313 = mhlo.add %2304, %2312 : tensor<1x384x128xf32>
    %2314 = "mhlo.broadcast_in_dim"(%1130) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2315 = mhlo.multiply %2313, %2314 : tensor<1x384x128xf32>
    %2316 = "mhlo.broadcast_in_dim"(%1129) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2317 = mhlo.add %2315, %2316 : tensor<1x384x128xf32>
    %2318 = "mhlo.reshape"(%2317) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2319 = "mhlo.dot"(%2318, %1148) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2320 = "mhlo.broadcast_in_dim"(%1147) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2321 = mhlo.add %2319, %2320 : tensor<384x512xf32>
    %2322 = "mhlo.reshape"(%2321) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2323 = mhlo.maximum %2322, %1119 : tensor<1x384x512xf32>
    %2324 = "mhlo.reshape"(%2323) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2325 = "mhlo.dot"(%2324, %1152) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2326 = "mhlo.broadcast_in_dim"(%1151) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2327 = mhlo.add %2325, %2326 : tensor<384x128xf32>
    %2328 = "mhlo.reshape"(%2327) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2329 = mhlo.add %2328, %2317 : tensor<1x384x128xf32>
    %2330 = "mhlo.broadcast_in_dim"(%1150) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2331 = mhlo.multiply %2329, %2330 : tensor<1x384x128xf32>
    %2332 = "mhlo.broadcast_in_dim"(%1149) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2333 = mhlo.add %2331, %2332 : tensor<1x384x128xf32>
    %2334 = "mhlo.reshape"(%2333) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2335 = "mhlo.dot"(%2334, %1154) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2336 = "mhlo.broadcast_in_dim"(%1153) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2337 = mhlo.add %2335, %2336 : tensor<384x512xf32>
    %2338 = "mhlo.reshape"(%2337) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2339 = mhlo.maximum %2338, %1119 : tensor<1x384x512xf32>
    %2340 = "mhlo.reshape"(%2339) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2341 = "mhlo.dot"(%2340, %1158) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2342 = "mhlo.broadcast_in_dim"(%1157) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2343 = mhlo.add %2341, %2342 : tensor<384x128xf32>
    %2344 = "mhlo.reshape"(%2343) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2345 = mhlo.add %2344, %2333 : tensor<1x384x128xf32>
    %2346 = "mhlo.broadcast_in_dim"(%1156) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2347 = mhlo.multiply %2345, %2346 : tensor<1x384x128xf32>
    %2348 = "mhlo.broadcast_in_dim"(%1155) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2349 = mhlo.add %2347, %2348 : tensor<1x384x128xf32>
    %2350 = "mhlo.reshape"(%2349) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2351 = "mhlo.dot"(%2350, %1160) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2352 = "mhlo.broadcast_in_dim"(%1159) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2353 = mhlo.add %2351, %2352 : tensor<384x512xf32>
    %2354 = "mhlo.reshape"(%2353) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2355 = mhlo.maximum %2354, %1119 : tensor<1x384x512xf32>
    %2356 = "mhlo.reshape"(%2355) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2357 = "mhlo.dot"(%2356, %1164) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2358 = "mhlo.broadcast_in_dim"(%1163) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2359 = mhlo.add %2357, %2358 : tensor<384x128xf32>
    %2360 = "mhlo.reshape"(%2359) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2361 = mhlo.add %2360, %2349 : tensor<1x384x128xf32>
    %2362 = "mhlo.broadcast_in_dim"(%1162) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2363 = mhlo.multiply %2361, %2362 : tensor<1x384x128xf32>
    %2364 = "mhlo.broadcast_in_dim"(%1161) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2365 = mhlo.add %2363, %2364 : tensor<1x384x128xf32>
    %2366 = "mhlo.reshape"(%2365) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2367 = "mhlo.dot"(%2366, %1166) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2368 = "mhlo.broadcast_in_dim"(%1165) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2369 = mhlo.add %2367, %2368 : tensor<384x512xf32>
    %2370 = "mhlo.reshape"(%2369) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2371 = mhlo.maximum %2370, %1119 : tensor<1x384x512xf32>
    %2372 = "mhlo.reshape"(%2371) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2373 = "mhlo.dot"(%2372, %1174) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2374 = "mhlo.broadcast_in_dim"(%1173) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2375 = mhlo.add %2373, %2374 : tensor<384x128xf32>
    %2376 = "mhlo.reshape"(%2375) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2377 = mhlo.add %2376, %2365 : tensor<1x384x128xf32>
    %2378 = "mhlo.broadcast_in_dim"(%1168) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2379 = mhlo.multiply %2377, %2378 : tensor<1x384x128xf32>
    %2380 = "mhlo.broadcast_in_dim"(%1167) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2381 = mhlo.add %2379, %2380 : tensor<1x384x128xf32>
    %2382 = "mhlo.reshape"(%2381) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2383 = "mhlo.dot"(%2382, %1172) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2384 = "mhlo.broadcast_in_dim"(%1171) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2385 = mhlo.add %2383, %2384 : tensor<384x512xf32>
    %2386 = "mhlo.reshape"(%2385) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2387 = mhlo.add %2386, %2261 : tensor<1x384x512xf32>
    %2388 = "mhlo.broadcast_in_dim"(%1170) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2389 = mhlo.multiply %2387, %2388 : tensor<1x384x512xf32>
    %2390 = "mhlo.broadcast_in_dim"(%1169) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2391 = mhlo.add %2389, %2390 : tensor<1x384x512xf32>
    %2392 = "mhlo.reshape"(%2391) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2393 = "mhlo.dot"(%2392, %1184) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2394 = "mhlo.broadcast_in_dim"(%1183) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2395 = mhlo.add %2393, %2394 : tensor<384x128xf32>
    %2396 = "mhlo.reshape"(%2395) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2397 = "mhlo.transpose"(%2396) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2398 = "mhlo.dot"(%2392, %1188) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2399 = "mhlo.reshape"(%2398) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2400 = "mhlo.broadcast_in_dim"(%1187) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2401 = mhlo.add %2399, %2400 : tensor<1x384x128xf32>
    %2402 = "mhlo.broadcast_in_dim"(%1186) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2403 = mhlo.multiply %2401, %2402 : tensor<1x384x128xf32>
    %2404 = "mhlo.broadcast_in_dim"(%1185) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2405 = mhlo.add %2403, %2404 : tensor<1x384x128xf32>
    %2406 = "mhlo.reshape"(%2405) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2407 = "mhlo.dot"(%2406, %1180) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2408 = "mhlo.broadcast_in_dim"(%1179) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2409 = mhlo.add %2407, %2408 : tensor<384x128xf32>
    %2410 = "mhlo.reshape"(%2409) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2411 = "mhlo.transpose"(%2410) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2412 = "mhlo.dot"(%2406, %1182) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2413 = "mhlo.broadcast_in_dim"(%1181) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2414 = mhlo.add %2412, %2413 : tensor<384x128xf32>
    %2415 = "mhlo.reshape"(%2414) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2416 = "mhlo.transpose"(%2415) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2417 = "mhlo.dot_general"(%2416, %2411) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2418 = mhlo.multiply %2417, %1114 : tensor<1x4x384x384xf32>
    %2419 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2420 = mhlo.add %2418, %2419 : tensor<1x4x384x384xf32>
    %2421 = "mhlo.reduce"(%2420, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2422 = "mhlo.broadcast_in_dim"(%2421) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2423 = mhlo.subtract %2420, %2422 : tensor<1x4x384x384xf32>
    %2424 = "mhlo.exponential"(%2423) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2425 = "mhlo.reduce"(%2424, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2426 = "mhlo.broadcast_in_dim"(%2425) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2427 = mhlo.divide %2424, %2426 : tensor<1x4x384x384xf32>
    %2428 = "mhlo.dot_general"(%2427, %2397) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2429 = "mhlo.transpose"(%2428) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2430 = "mhlo.reshape"(%2429) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2431 = "mhlo.dot"(%2430, %1178) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2432 = "mhlo.broadcast_in_dim"(%1177) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2433 = mhlo.add %2431, %2432 : tensor<384x128xf32>
    %2434 = "mhlo.reshape"(%2433) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2435 = "mhlo.dot"(%2392, %1192) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2436 = "mhlo.broadcast_in_dim"(%1191) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2437 = mhlo.add %2435, %2436 : tensor<384x128xf32>
    %2438 = "mhlo.reshape"(%2437) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2439 = "mhlo.broadcast_in_dim"(%1190) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2440 = mhlo.multiply %2438, %2439 : tensor<1x384x128xf32>
    %2441 = "mhlo.broadcast_in_dim"(%1189) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2442 = mhlo.add %2440, %2441 : tensor<1x384x128xf32>
    %2443 = mhlo.add %2434, %2442 : tensor<1x384x128xf32>
    %2444 = "mhlo.broadcast_in_dim"(%1176) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2445 = mhlo.multiply %2443, %2444 : tensor<1x384x128xf32>
    %2446 = "mhlo.broadcast_in_dim"(%1175) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2447 = mhlo.add %2445, %2446 : tensor<1x384x128xf32>
    %2448 = "mhlo.reshape"(%2447) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2449 = "mhlo.dot"(%2448, %1194) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2450 = "mhlo.broadcast_in_dim"(%1193) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2451 = mhlo.add %2449, %2450 : tensor<384x512xf32>
    %2452 = "mhlo.reshape"(%2451) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2453 = mhlo.maximum %2452, %1119 : tensor<1x384x512xf32>
    %2454 = "mhlo.reshape"(%2453) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2455 = "mhlo.dot"(%2454, %1198) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2456 = "mhlo.broadcast_in_dim"(%1197) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2457 = mhlo.add %2455, %2456 : tensor<384x128xf32>
    %2458 = "mhlo.reshape"(%2457) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2459 = mhlo.add %2458, %2447 : tensor<1x384x128xf32>
    %2460 = "mhlo.broadcast_in_dim"(%1196) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2461 = mhlo.multiply %2459, %2460 : tensor<1x384x128xf32>
    %2462 = "mhlo.broadcast_in_dim"(%1195) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2463 = mhlo.add %2461, %2462 : tensor<1x384x128xf32>
    %2464 = "mhlo.reshape"(%2463) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2465 = "mhlo.dot"(%2464, %1200) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2466 = "mhlo.broadcast_in_dim"(%1199) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2467 = mhlo.add %2465, %2466 : tensor<384x512xf32>
    %2468 = "mhlo.reshape"(%2467) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2469 = mhlo.maximum %2468, %1119 : tensor<1x384x512xf32>
    %2470 = "mhlo.reshape"(%2469) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2471 = "mhlo.dot"(%2470, %1204) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2472 = "mhlo.broadcast_in_dim"(%1203) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2473 = mhlo.add %2471, %2472 : tensor<384x128xf32>
    %2474 = "mhlo.reshape"(%2473) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2475 = mhlo.add %2474, %2463 : tensor<1x384x128xf32>
    %2476 = "mhlo.broadcast_in_dim"(%1202) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2477 = mhlo.multiply %2475, %2476 : tensor<1x384x128xf32>
    %2478 = "mhlo.broadcast_in_dim"(%1201) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2479 = mhlo.add %2477, %2478 : tensor<1x384x128xf32>
    %2480 = "mhlo.reshape"(%2479) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2481 = "mhlo.dot"(%2480, %1206) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2482 = "mhlo.broadcast_in_dim"(%1205) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2483 = mhlo.add %2481, %2482 : tensor<384x512xf32>
    %2484 = "mhlo.reshape"(%2483) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2485 = mhlo.maximum %2484, %1119 : tensor<1x384x512xf32>
    %2486 = "mhlo.reshape"(%2485) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2487 = "mhlo.dot"(%2486, %1210) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2488 = "mhlo.broadcast_in_dim"(%1209) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2489 = mhlo.add %2487, %2488 : tensor<384x128xf32>
    %2490 = "mhlo.reshape"(%2489) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2491 = mhlo.add %2490, %2479 : tensor<1x384x128xf32>
    %2492 = "mhlo.broadcast_in_dim"(%1208) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2493 = mhlo.multiply %2491, %2492 : tensor<1x384x128xf32>
    %2494 = "mhlo.broadcast_in_dim"(%1207) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2495 = mhlo.add %2493, %2494 : tensor<1x384x128xf32>
    %2496 = "mhlo.reshape"(%2495) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2497 = "mhlo.dot"(%2496, %1212) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2498 = "mhlo.broadcast_in_dim"(%1211) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2499 = mhlo.add %2497, %2498 : tensor<384x512xf32>
    %2500 = "mhlo.reshape"(%2499) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2501 = mhlo.maximum %2500, %1119 : tensor<1x384x512xf32>
    %2502 = "mhlo.reshape"(%2501) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2503 = "mhlo.dot"(%2502, %1220) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2504 = "mhlo.broadcast_in_dim"(%1219) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2505 = mhlo.add %2503, %2504 : tensor<384x128xf32>
    %2506 = "mhlo.reshape"(%2505) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2507 = mhlo.add %2506, %2495 : tensor<1x384x128xf32>
    %2508 = "mhlo.broadcast_in_dim"(%1214) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2509 = mhlo.multiply %2507, %2508 : tensor<1x384x128xf32>
    %2510 = "mhlo.broadcast_in_dim"(%1213) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2511 = mhlo.add %2509, %2510 : tensor<1x384x128xf32>
    %2512 = "mhlo.reshape"(%2511) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2513 = "mhlo.dot"(%2512, %1218) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2514 = "mhlo.broadcast_in_dim"(%1217) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2515 = mhlo.add %2513, %2514 : tensor<384x512xf32>
    %2516 = "mhlo.reshape"(%2515) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2517 = mhlo.add %2516, %2391 : tensor<1x384x512xf32>
    %2518 = "mhlo.broadcast_in_dim"(%1216) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2519 = mhlo.multiply %2517, %2518 : tensor<1x384x512xf32>
    %2520 = "mhlo.broadcast_in_dim"(%1215) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2521 = mhlo.add %2519, %2520 : tensor<1x384x512xf32>
    %2522 = "mhlo.reshape"(%2521) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2523 = "mhlo.dot"(%2522, %1690) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2524 = "mhlo.broadcast_in_dim"(%1689) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2525 = mhlo.add %2523, %2524 : tensor<384x128xf32>
    %2526 = "mhlo.reshape"(%2525) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2527 = "mhlo.transpose"(%2526) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2528 = "mhlo.dot"(%2522, %1694) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2529 = "mhlo.reshape"(%2528) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2530 = "mhlo.broadcast_in_dim"(%1693) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2531 = mhlo.add %2529, %2530 : tensor<1x384x128xf32>
    %2532 = "mhlo.broadcast_in_dim"(%1692) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2533 = mhlo.multiply %2531, %2532 : tensor<1x384x128xf32>
    %2534 = "mhlo.broadcast_in_dim"(%1691) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2535 = mhlo.add %2533, %2534 : tensor<1x384x128xf32>
    %2536 = "mhlo.reshape"(%2535) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2537 = "mhlo.dot"(%2536, %1686) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2538 = "mhlo.broadcast_in_dim"(%1685) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2539 = mhlo.add %2537, %2538 : tensor<384x128xf32>
    %2540 = "mhlo.reshape"(%2539) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2541 = "mhlo.transpose"(%2540) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2542 = "mhlo.dot"(%2536, %1688) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2543 = "mhlo.broadcast_in_dim"(%1687) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2544 = mhlo.add %2542, %2543 : tensor<384x128xf32>
    %2545 = "mhlo.reshape"(%2544) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2546 = "mhlo.transpose"(%2545) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2547 = "mhlo.dot_general"(%2546, %2541) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2548 = mhlo.multiply %2547, %1114 : tensor<1x4x384x384xf32>
    %2549 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2550 = mhlo.add %2548, %2549 : tensor<1x4x384x384xf32>
    %2551 = "mhlo.reduce"(%2550, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2552 = "mhlo.broadcast_in_dim"(%2551) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2553 = mhlo.subtract %2550, %2552 : tensor<1x4x384x384xf32>
    %2554 = "mhlo.exponential"(%2553) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2555 = "mhlo.reduce"(%2554, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2556 = "mhlo.broadcast_in_dim"(%2555) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2557 = mhlo.divide %2554, %2556 : tensor<1x4x384x384xf32>
    %2558 = "mhlo.dot_general"(%2557, %2527) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2559 = "mhlo.transpose"(%2558) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2560 = "mhlo.reshape"(%2559) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2561 = "mhlo.dot"(%2560, %1684) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2562 = "mhlo.broadcast_in_dim"(%1683) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2563 = mhlo.add %2561, %2562 : tensor<384x128xf32>
    %2564 = "mhlo.reshape"(%2563) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2565 = "mhlo.dot"(%2522, %1698) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2566 = "mhlo.broadcast_in_dim"(%1697) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2567 = mhlo.add %2565, %2566 : tensor<384x128xf32>
    %2568 = "mhlo.reshape"(%2567) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2569 = "mhlo.broadcast_in_dim"(%1696) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2570 = mhlo.multiply %2568, %2569 : tensor<1x384x128xf32>
    %2571 = "mhlo.broadcast_in_dim"(%1695) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2572 = mhlo.add %2570, %2571 : tensor<1x384x128xf32>
    %2573 = mhlo.add %2564, %2572 : tensor<1x384x128xf32>
    %2574 = "mhlo.broadcast_in_dim"(%1682) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2575 = mhlo.multiply %2573, %2574 : tensor<1x384x128xf32>
    %2576 = "mhlo.broadcast_in_dim"(%1681) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2577 = mhlo.add %2575, %2576 : tensor<1x384x128xf32>
    %2578 = "mhlo.reshape"(%2577) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2579 = "mhlo.dot"(%2578, %1700) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2580 = "mhlo.broadcast_in_dim"(%1699) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2581 = mhlo.add %2579, %2580 : tensor<384x512xf32>
    %2582 = "mhlo.reshape"(%2581) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2583 = mhlo.maximum %2582, %1119 : tensor<1x384x512xf32>
    %2584 = "mhlo.reshape"(%2583) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2585 = "mhlo.dot"(%2584, %1704) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2586 = "mhlo.broadcast_in_dim"(%1703) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2587 = mhlo.add %2585, %2586 : tensor<384x128xf32>
    %2588 = "mhlo.reshape"(%2587) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2589 = mhlo.add %2588, %2577 : tensor<1x384x128xf32>
    %2590 = "mhlo.broadcast_in_dim"(%1702) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2591 = mhlo.multiply %2589, %2590 : tensor<1x384x128xf32>
    %2592 = "mhlo.broadcast_in_dim"(%1701) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2593 = mhlo.add %2591, %2592 : tensor<1x384x128xf32>
    %2594 = "mhlo.reshape"(%2593) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2595 = "mhlo.dot"(%2594, %1706) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2596 = "mhlo.broadcast_in_dim"(%1705) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2597 = mhlo.add %2595, %2596 : tensor<384x512xf32>
    %2598 = "mhlo.reshape"(%2597) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2599 = mhlo.maximum %2598, %1119 : tensor<1x384x512xf32>
    %2600 = "mhlo.reshape"(%2599) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2601 = "mhlo.dot"(%2600, %1710) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2602 = "mhlo.broadcast_in_dim"(%1709) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2603 = mhlo.add %2601, %2602 : tensor<384x128xf32>
    %2604 = "mhlo.reshape"(%2603) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2605 = mhlo.add %2604, %2593 : tensor<1x384x128xf32>
    %2606 = "mhlo.broadcast_in_dim"(%1708) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2607 = mhlo.multiply %2605, %2606 : tensor<1x384x128xf32>
    %2608 = "mhlo.broadcast_in_dim"(%1707) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2609 = mhlo.add %2607, %2608 : tensor<1x384x128xf32>
    %2610 = "mhlo.reshape"(%2609) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2611 = "mhlo.dot"(%2610, %1712) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2612 = "mhlo.broadcast_in_dim"(%1711) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2613 = mhlo.add %2611, %2612 : tensor<384x512xf32>
    %2614 = "mhlo.reshape"(%2613) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2615 = mhlo.maximum %2614, %1119 : tensor<1x384x512xf32>
    %2616 = "mhlo.reshape"(%2615) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2617 = "mhlo.dot"(%2616, %1716) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2618 = "mhlo.broadcast_in_dim"(%1715) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2619 = mhlo.add %2617, %2618 : tensor<384x128xf32>
    %2620 = "mhlo.reshape"(%2619) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2621 = mhlo.add %2620, %2609 : tensor<1x384x128xf32>
    %2622 = "mhlo.broadcast_in_dim"(%1714) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2623 = mhlo.multiply %2621, %2622 : tensor<1x384x128xf32>
    %2624 = "mhlo.broadcast_in_dim"(%1713) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2625 = mhlo.add %2623, %2624 : tensor<1x384x128xf32>
    %2626 = "mhlo.reshape"(%2625) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2627 = "mhlo.dot"(%2626, %1718) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2628 = "mhlo.broadcast_in_dim"(%1717) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2629 = mhlo.add %2627, %2628 : tensor<384x512xf32>
    %2630 = "mhlo.reshape"(%2629) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2631 = mhlo.maximum %2630, %1119 : tensor<1x384x512xf32>
    %2632 = "mhlo.reshape"(%2631) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2633 = "mhlo.dot"(%2632, %1726) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2634 = "mhlo.broadcast_in_dim"(%1725) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2635 = mhlo.add %2633, %2634 : tensor<384x128xf32>
    %2636 = "mhlo.reshape"(%2635) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2637 = mhlo.add %2636, %2625 : tensor<1x384x128xf32>
    %2638 = "mhlo.broadcast_in_dim"(%1720) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2639 = mhlo.multiply %2637, %2638 : tensor<1x384x128xf32>
    %2640 = "mhlo.broadcast_in_dim"(%1719) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2641 = mhlo.add %2639, %2640 : tensor<1x384x128xf32>
    %2642 = "mhlo.reshape"(%2641) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2643 = "mhlo.dot"(%2642, %1724) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2644 = "mhlo.broadcast_in_dim"(%1723) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2645 = mhlo.add %2643, %2644 : tensor<384x512xf32>
    %2646 = "mhlo.reshape"(%2645) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2647 = mhlo.add %2646, %2521 : tensor<1x384x512xf32>
    %2648 = "mhlo.broadcast_in_dim"(%1722) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2649 = mhlo.multiply %2647, %2648 : tensor<1x384x512xf32>
    %2650 = "mhlo.broadcast_in_dim"(%1721) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2651 = mhlo.add %2649, %2650 : tensor<1x384x512xf32>
    %2652 = "mhlo.reshape"(%2651) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2653 = "mhlo.dot"(%2652, %1920) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2654 = "mhlo.broadcast_in_dim"(%1919) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2655 = mhlo.add %2653, %2654 : tensor<384x128xf32>
    %2656 = "mhlo.reshape"(%2655) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2657 = "mhlo.transpose"(%2656) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2658 = "mhlo.dot"(%2652, %1924) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2659 = "mhlo.reshape"(%2658) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2660 = "mhlo.broadcast_in_dim"(%1923) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2661 = mhlo.add %2659, %2660 : tensor<1x384x128xf32>
    %2662 = "mhlo.broadcast_in_dim"(%1922) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2663 = mhlo.multiply %2661, %2662 : tensor<1x384x128xf32>
    %2664 = "mhlo.broadcast_in_dim"(%1921) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2665 = mhlo.add %2663, %2664 : tensor<1x384x128xf32>
    %2666 = "mhlo.reshape"(%2665) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2667 = "mhlo.dot"(%2666, %1916) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2668 = "mhlo.broadcast_in_dim"(%1915) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2669 = mhlo.add %2667, %2668 : tensor<384x128xf32>
    %2670 = "mhlo.reshape"(%2669) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2671 = "mhlo.transpose"(%2670) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2672 = "mhlo.dot"(%2666, %1918) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2673 = "mhlo.broadcast_in_dim"(%1917) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2674 = mhlo.add %2672, %2673 : tensor<384x128xf32>
    %2675 = "mhlo.reshape"(%2674) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2676 = "mhlo.transpose"(%2675) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2677 = "mhlo.dot_general"(%2676, %2671) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2678 = mhlo.multiply %2677, %1114 : tensor<1x4x384x384xf32>
    %2679 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2680 = mhlo.add %2678, %2679 : tensor<1x4x384x384xf32>
    %2681 = "mhlo.reduce"(%2680, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2682 = "mhlo.broadcast_in_dim"(%2681) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2683 = mhlo.subtract %2680, %2682 : tensor<1x4x384x384xf32>
    %2684 = "mhlo.exponential"(%2683) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2685 = "mhlo.reduce"(%2684, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2686 = "mhlo.broadcast_in_dim"(%2685) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2687 = mhlo.divide %2684, %2686 : tensor<1x4x384x384xf32>
    %2688 = "mhlo.dot_general"(%2687, %2657) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2689 = "mhlo.transpose"(%2688) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2690 = "mhlo.reshape"(%2689) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2691 = "mhlo.dot"(%2690, %1914) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2692 = "mhlo.broadcast_in_dim"(%1913) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2693 = mhlo.add %2691, %2692 : tensor<384x128xf32>
    %2694 = "mhlo.reshape"(%2693) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2695 = "mhlo.dot"(%2652, %1928) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2696 = "mhlo.broadcast_in_dim"(%1927) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2697 = mhlo.add %2695, %2696 : tensor<384x128xf32>
    %2698 = "mhlo.reshape"(%2697) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2699 = "mhlo.broadcast_in_dim"(%1926) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2700 = mhlo.multiply %2698, %2699 : tensor<1x384x128xf32>
    %2701 = "mhlo.broadcast_in_dim"(%1925) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2702 = mhlo.add %2700, %2701 : tensor<1x384x128xf32>
    %2703 = mhlo.add %2694, %2702 : tensor<1x384x128xf32>
    %2704 = "mhlo.broadcast_in_dim"(%1912) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2705 = mhlo.multiply %2703, %2704 : tensor<1x384x128xf32>
    %2706 = "mhlo.broadcast_in_dim"(%1911) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2707 = mhlo.add %2705, %2706 : tensor<1x384x128xf32>
    %2708 = "mhlo.reshape"(%2707) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2709 = "mhlo.dot"(%2708, %1930) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2710 = "mhlo.broadcast_in_dim"(%1929) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2711 = mhlo.add %2709, %2710 : tensor<384x512xf32>
    %2712 = "mhlo.reshape"(%2711) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2713 = mhlo.maximum %2712, %1119 : tensor<1x384x512xf32>
    %2714 = "mhlo.reshape"(%2713) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2715 = "mhlo.dot"(%2714, %1934) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2716 = "mhlo.broadcast_in_dim"(%1933) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2717 = mhlo.add %2715, %2716 : tensor<384x128xf32>
    %2718 = "mhlo.reshape"(%2717) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2719 = mhlo.add %2718, %2707 : tensor<1x384x128xf32>
    %2720 = "mhlo.broadcast_in_dim"(%1932) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2721 = mhlo.multiply %2719, %2720 : tensor<1x384x128xf32>
    %2722 = "mhlo.broadcast_in_dim"(%1931) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2723 = mhlo.add %2721, %2722 : tensor<1x384x128xf32>
    %2724 = "mhlo.reshape"(%2723) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2725 = "mhlo.dot"(%2724, %1936) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2726 = "mhlo.broadcast_in_dim"(%1935) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2727 = mhlo.add %2725, %2726 : tensor<384x512xf32>
    %2728 = "mhlo.reshape"(%2727) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2729 = mhlo.maximum %2728, %1119 : tensor<1x384x512xf32>
    %2730 = "mhlo.reshape"(%2729) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2731 = "mhlo.dot"(%2730, %1940) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2732 = "mhlo.broadcast_in_dim"(%1939) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2733 = mhlo.add %2731, %2732 : tensor<384x128xf32>
    %2734 = "mhlo.reshape"(%2733) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2735 = mhlo.add %2734, %2723 : tensor<1x384x128xf32>
    %2736 = "mhlo.broadcast_in_dim"(%1938) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2737 = mhlo.multiply %2735, %2736 : tensor<1x384x128xf32>
    %2738 = "mhlo.broadcast_in_dim"(%1937) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2739 = mhlo.add %2737, %2738 : tensor<1x384x128xf32>
    %2740 = "mhlo.reshape"(%2739) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2741 = "mhlo.dot"(%2740, %1942) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2742 = "mhlo.broadcast_in_dim"(%1941) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2743 = mhlo.add %2741, %2742 : tensor<384x512xf32>
    %2744 = "mhlo.reshape"(%2743) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2745 = mhlo.maximum %2744, %1119 : tensor<1x384x512xf32>
    %2746 = "mhlo.reshape"(%2745) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2747 = "mhlo.dot"(%2746, %1946) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2748 = "mhlo.broadcast_in_dim"(%1945) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2749 = mhlo.add %2747, %2748 : tensor<384x128xf32>
    %2750 = "mhlo.reshape"(%2749) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2751 = mhlo.add %2750, %2739 : tensor<1x384x128xf32>
    %2752 = "mhlo.broadcast_in_dim"(%1944) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2753 = mhlo.multiply %2751, %2752 : tensor<1x384x128xf32>
    %2754 = "mhlo.broadcast_in_dim"(%1943) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2755 = mhlo.add %2753, %2754 : tensor<1x384x128xf32>
    %2756 = "mhlo.reshape"(%2755) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2757 = "mhlo.dot"(%2756, %1948) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2758 = "mhlo.broadcast_in_dim"(%1947) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2759 = mhlo.add %2757, %2758 : tensor<384x512xf32>
    %2760 = "mhlo.reshape"(%2759) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2761 = mhlo.maximum %2760, %1119 : tensor<1x384x512xf32>
    %2762 = "mhlo.reshape"(%2761) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2763 = "mhlo.dot"(%2762, %1956) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2764 = "mhlo.broadcast_in_dim"(%1955) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2765 = mhlo.add %2763, %2764 : tensor<384x128xf32>
    %2766 = "mhlo.reshape"(%2765) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2767 = mhlo.add %2766, %2755 : tensor<1x384x128xf32>
    %2768 = "mhlo.broadcast_in_dim"(%1950) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2769 = mhlo.multiply %2767, %2768 : tensor<1x384x128xf32>
    %2770 = "mhlo.broadcast_in_dim"(%1949) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2771 = mhlo.add %2769, %2770 : tensor<1x384x128xf32>
    %2772 = "mhlo.reshape"(%2771) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2773 = "mhlo.dot"(%2772, %1954) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2774 = "mhlo.broadcast_in_dim"(%1953) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2775 = mhlo.add %2773, %2774 : tensor<384x512xf32>
    %2776 = "mhlo.reshape"(%2775) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2777 = mhlo.add %2776, %2651 : tensor<1x384x512xf32>
    %2778 = "mhlo.broadcast_in_dim"(%1952) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2779 = mhlo.multiply %2777, %2778 : tensor<1x384x512xf32>
    %2780 = "mhlo.broadcast_in_dim"(%1951) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2781 = mhlo.add %2779, %2780 : tensor<1x384x512xf32>
    %2782 = "mhlo.reshape"(%2781) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2783 = "mhlo.dot"(%2782, %1966) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2784 = "mhlo.broadcast_in_dim"(%1965) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2785 = mhlo.add %2783, %2784 : tensor<384x128xf32>
    %2786 = "mhlo.reshape"(%2785) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2787 = "mhlo.transpose"(%2786) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2788 = "mhlo.dot"(%2782, %1970) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2789 = "mhlo.reshape"(%2788) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2790 = "mhlo.broadcast_in_dim"(%1969) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2791 = mhlo.add %2789, %2790 : tensor<1x384x128xf32>
    %2792 = "mhlo.broadcast_in_dim"(%1968) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2793 = mhlo.multiply %2791, %2792 : tensor<1x384x128xf32>
    %2794 = "mhlo.broadcast_in_dim"(%1967) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2795 = mhlo.add %2793, %2794 : tensor<1x384x128xf32>
    %2796 = "mhlo.reshape"(%2795) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2797 = "mhlo.dot"(%2796, %1962) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2798 = "mhlo.broadcast_in_dim"(%1961) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2799 = mhlo.add %2797, %2798 : tensor<384x128xf32>
    %2800 = "mhlo.reshape"(%2799) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2801 = "mhlo.transpose"(%2800) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2802 = "mhlo.dot"(%2796, %1964) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2803 = "mhlo.broadcast_in_dim"(%1963) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2804 = mhlo.add %2802, %2803 : tensor<384x128xf32>
    %2805 = "mhlo.reshape"(%2804) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2806 = "mhlo.transpose"(%2805) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2807 = "mhlo.dot_general"(%2806, %2801) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2808 = mhlo.multiply %2807, %1114 : tensor<1x4x384x384xf32>
    %2809 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2810 = mhlo.add %2808, %2809 : tensor<1x4x384x384xf32>
    %2811 = "mhlo.reduce"(%2810, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2812 = "mhlo.broadcast_in_dim"(%2811) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2813 = mhlo.subtract %2810, %2812 : tensor<1x4x384x384xf32>
    %2814 = "mhlo.exponential"(%2813) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2815 = "mhlo.reduce"(%2814, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2816 = "mhlo.broadcast_in_dim"(%2815) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2817 = mhlo.divide %2814, %2816 : tensor<1x4x384x384xf32>
    %2818 = "mhlo.dot_general"(%2817, %2787) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2819 = "mhlo.transpose"(%2818) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2820 = "mhlo.reshape"(%2819) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2821 = "mhlo.dot"(%2820, %1960) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2822 = "mhlo.broadcast_in_dim"(%1959) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2823 = mhlo.add %2821, %2822 : tensor<384x128xf32>
    %2824 = "mhlo.reshape"(%2823) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2825 = "mhlo.dot"(%2782, %1974) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2826 = "mhlo.broadcast_in_dim"(%1973) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2827 = mhlo.add %2825, %2826 : tensor<384x128xf32>
    %2828 = "mhlo.reshape"(%2827) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2829 = "mhlo.broadcast_in_dim"(%1972) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2830 = mhlo.multiply %2828, %2829 : tensor<1x384x128xf32>
    %2831 = "mhlo.broadcast_in_dim"(%1971) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2832 = mhlo.add %2830, %2831 : tensor<1x384x128xf32>
    %2833 = mhlo.add %2824, %2832 : tensor<1x384x128xf32>
    %2834 = "mhlo.broadcast_in_dim"(%1958) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2835 = mhlo.multiply %2833, %2834 : tensor<1x384x128xf32>
    %2836 = "mhlo.broadcast_in_dim"(%1957) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2837 = mhlo.add %2835, %2836 : tensor<1x384x128xf32>
    %2838 = "mhlo.reshape"(%2837) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2839 = "mhlo.dot"(%2838, %1976) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2840 = "mhlo.broadcast_in_dim"(%1975) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2841 = mhlo.add %2839, %2840 : tensor<384x512xf32>
    %2842 = "mhlo.reshape"(%2841) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2843 = mhlo.maximum %2842, %1119 : tensor<1x384x512xf32>
    %2844 = "mhlo.reshape"(%2843) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2845 = "mhlo.dot"(%2844, %1980) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2846 = "mhlo.broadcast_in_dim"(%1979) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2847 = mhlo.add %2845, %2846 : tensor<384x128xf32>
    %2848 = "mhlo.reshape"(%2847) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2849 = mhlo.add %2848, %2837 : tensor<1x384x128xf32>
    %2850 = "mhlo.broadcast_in_dim"(%1978) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2851 = mhlo.multiply %2849, %2850 : tensor<1x384x128xf32>
    %2852 = "mhlo.broadcast_in_dim"(%1977) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2853 = mhlo.add %2851, %2852 : tensor<1x384x128xf32>
    %2854 = "mhlo.reshape"(%2853) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2855 = "mhlo.dot"(%2854, %1982) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2856 = "mhlo.broadcast_in_dim"(%1981) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2857 = mhlo.add %2855, %2856 : tensor<384x512xf32>
    %2858 = "mhlo.reshape"(%2857) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2859 = mhlo.maximum %2858, %1119 : tensor<1x384x512xf32>
    %2860 = "mhlo.reshape"(%2859) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2861 = "mhlo.dot"(%2860, %1986) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2862 = "mhlo.broadcast_in_dim"(%1985) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2863 = mhlo.add %2861, %2862 : tensor<384x128xf32>
    %2864 = "mhlo.reshape"(%2863) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2865 = mhlo.add %2864, %2853 : tensor<1x384x128xf32>
    %2866 = "mhlo.broadcast_in_dim"(%1984) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2867 = mhlo.multiply %2865, %2866 : tensor<1x384x128xf32>
    %2868 = "mhlo.broadcast_in_dim"(%1983) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2869 = mhlo.add %2867, %2868 : tensor<1x384x128xf32>
    %2870 = "mhlo.reshape"(%2869) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2871 = "mhlo.dot"(%2870, %1988) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2872 = "mhlo.broadcast_in_dim"(%1987) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2873 = mhlo.add %2871, %2872 : tensor<384x512xf32>
    %2874 = "mhlo.reshape"(%2873) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2875 = mhlo.maximum %2874, %1119 : tensor<1x384x512xf32>
    %2876 = "mhlo.reshape"(%2875) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2877 = "mhlo.dot"(%2876, %1992) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2878 = "mhlo.broadcast_in_dim"(%1991) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2879 = mhlo.add %2877, %2878 : tensor<384x128xf32>
    %2880 = "mhlo.reshape"(%2879) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2881 = mhlo.add %2880, %2869 : tensor<1x384x128xf32>
    %2882 = "mhlo.broadcast_in_dim"(%1990) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2883 = mhlo.multiply %2881, %2882 : tensor<1x384x128xf32>
    %2884 = "mhlo.broadcast_in_dim"(%1989) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2885 = mhlo.add %2883, %2884 : tensor<1x384x128xf32>
    %2886 = "mhlo.reshape"(%2885) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2887 = "mhlo.dot"(%2886, %1994) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2888 = "mhlo.broadcast_in_dim"(%1993) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2889 = mhlo.add %2887, %2888 : tensor<384x512xf32>
    %2890 = "mhlo.reshape"(%2889) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2891 = mhlo.maximum %2890, %1119 : tensor<1x384x512xf32>
    %2892 = "mhlo.reshape"(%2891) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2893 = "mhlo.dot"(%2892, %2002) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2894 = "mhlo.broadcast_in_dim"(%2001) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2895 = mhlo.add %2893, %2894 : tensor<384x128xf32>
    %2896 = "mhlo.reshape"(%2895) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2897 = mhlo.add %2896, %2885 : tensor<1x384x128xf32>
    %2898 = "mhlo.broadcast_in_dim"(%1996) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2899 = mhlo.multiply %2897, %2898 : tensor<1x384x128xf32>
    %2900 = "mhlo.broadcast_in_dim"(%1995) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2901 = mhlo.add %2899, %2900 : tensor<1x384x128xf32>
    %2902 = "mhlo.reshape"(%2901) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2903 = "mhlo.dot"(%2902, %2000) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2904 = "mhlo.broadcast_in_dim"(%1999) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2905 = mhlo.add %2903, %2904 : tensor<384x512xf32>
    %2906 = "mhlo.reshape"(%2905) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2907 = mhlo.add %2906, %2781 : tensor<1x384x512xf32>
    %2908 = "mhlo.broadcast_in_dim"(%1998) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2909 = mhlo.multiply %2907, %2908 : tensor<1x384x512xf32>
    %2910 = "mhlo.broadcast_in_dim"(%1997) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %2911 = mhlo.add %2909, %2910 : tensor<1x384x512xf32>
    %2912 = "mhlo.reshape"(%2911) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2913 = "mhlo.dot"(%2912, %2012) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2914 = "mhlo.broadcast_in_dim"(%2011) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2915 = mhlo.add %2913, %2914 : tensor<384x128xf32>
    %2916 = "mhlo.reshape"(%2915) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2917 = "mhlo.transpose"(%2916) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2918 = "mhlo.dot"(%2912, %2016) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2919 = "mhlo.reshape"(%2918) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2920 = "mhlo.broadcast_in_dim"(%2015) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2921 = mhlo.add %2919, %2920 : tensor<1x384x128xf32>
    %2922 = "mhlo.broadcast_in_dim"(%2014) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2923 = mhlo.multiply %2921, %2922 : tensor<1x384x128xf32>
    %2924 = "mhlo.broadcast_in_dim"(%2013) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2925 = mhlo.add %2923, %2924 : tensor<1x384x128xf32>
    %2926 = "mhlo.reshape"(%2925) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2927 = "mhlo.dot"(%2926, %2008) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2928 = "mhlo.broadcast_in_dim"(%2007) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2929 = mhlo.add %2927, %2928 : tensor<384x128xf32>
    %2930 = "mhlo.reshape"(%2929) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2931 = "mhlo.transpose"(%2930) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2932 = "mhlo.dot"(%2926, %2010) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2933 = "mhlo.broadcast_in_dim"(%2009) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2934 = mhlo.add %2932, %2933 : tensor<384x128xf32>
    %2935 = "mhlo.reshape"(%2934) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %2936 = "mhlo.transpose"(%2935) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %2937 = "mhlo.dot_general"(%2936, %2931) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %2938 = mhlo.multiply %2937, %1114 : tensor<1x4x384x384xf32>
    %2939 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2940 = mhlo.add %2938, %2939 : tensor<1x4x384x384xf32>
    %2941 = "mhlo.reduce"(%2940, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2942 = "mhlo.broadcast_in_dim"(%2941) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2943 = mhlo.subtract %2940, %2942 : tensor<1x4x384x384xf32>
    %2944 = "mhlo.exponential"(%2943) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %2945 = "mhlo.reduce"(%2944, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %2946 = "mhlo.broadcast_in_dim"(%2945) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %2947 = mhlo.divide %2944, %2946 : tensor<1x4x384x384xf32>
    %2948 = "mhlo.dot_general"(%2947, %2917) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %2949 = "mhlo.transpose"(%2948) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %2950 = "mhlo.reshape"(%2949) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %2951 = "mhlo.dot"(%2950, %2006) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %2952 = "mhlo.broadcast_in_dim"(%2005) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2953 = mhlo.add %2951, %2952 : tensor<384x128xf32>
    %2954 = "mhlo.reshape"(%2953) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2955 = "mhlo.dot"(%2912, %2020) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2956 = "mhlo.broadcast_in_dim"(%2019) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2957 = mhlo.add %2955, %2956 : tensor<384x128xf32>
    %2958 = "mhlo.reshape"(%2957) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2959 = "mhlo.broadcast_in_dim"(%2018) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2960 = mhlo.multiply %2958, %2959 : tensor<1x384x128xf32>
    %2961 = "mhlo.broadcast_in_dim"(%2017) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2962 = mhlo.add %2960, %2961 : tensor<1x384x128xf32>
    %2963 = mhlo.add %2954, %2962 : tensor<1x384x128xf32>
    %2964 = "mhlo.broadcast_in_dim"(%2004) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2965 = mhlo.multiply %2963, %2964 : tensor<1x384x128xf32>
    %2966 = "mhlo.broadcast_in_dim"(%2003) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2967 = mhlo.add %2965, %2966 : tensor<1x384x128xf32>
    %2968 = "mhlo.reshape"(%2967) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2969 = "mhlo.dot"(%2968, %2022) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2970 = "mhlo.broadcast_in_dim"(%2021) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2971 = mhlo.add %2969, %2970 : tensor<384x512xf32>
    %2972 = "mhlo.reshape"(%2971) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2973 = mhlo.maximum %2972, %1119 : tensor<1x384x512xf32>
    %2974 = "mhlo.reshape"(%2973) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2975 = "mhlo.dot"(%2974, %2026) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2976 = "mhlo.broadcast_in_dim"(%2025) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2977 = mhlo.add %2975, %2976 : tensor<384x128xf32>
    %2978 = "mhlo.reshape"(%2977) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2979 = mhlo.add %2978, %2967 : tensor<1x384x128xf32>
    %2980 = "mhlo.broadcast_in_dim"(%2024) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2981 = mhlo.multiply %2979, %2980 : tensor<1x384x128xf32>
    %2982 = "mhlo.broadcast_in_dim"(%2023) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2983 = mhlo.add %2981, %2982 : tensor<1x384x128xf32>
    %2984 = "mhlo.reshape"(%2983) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %2985 = "mhlo.dot"(%2984, %2028) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %2986 = "mhlo.broadcast_in_dim"(%2027) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %2987 = mhlo.add %2985, %2986 : tensor<384x512xf32>
    %2988 = "mhlo.reshape"(%2987) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %2989 = mhlo.maximum %2988, %1119 : tensor<1x384x512xf32>
    %2990 = "mhlo.reshape"(%2989) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %2991 = "mhlo.dot"(%2990, %2032) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %2992 = "mhlo.broadcast_in_dim"(%2031) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %2993 = mhlo.add %2991, %2992 : tensor<384x128xf32>
    %2994 = "mhlo.reshape"(%2993) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %2995 = mhlo.add %2994, %2983 : tensor<1x384x128xf32>
    %2996 = "mhlo.broadcast_in_dim"(%2030) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2997 = mhlo.multiply %2995, %2996 : tensor<1x384x128xf32>
    %2998 = "mhlo.broadcast_in_dim"(%2029) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %2999 = mhlo.add %2997, %2998 : tensor<1x384x128xf32>
    %3000 = "mhlo.reshape"(%2999) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3001 = "mhlo.dot"(%3000, %2034) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3002 = "mhlo.broadcast_in_dim"(%2033) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3003 = mhlo.add %3001, %3002 : tensor<384x512xf32>
    %3004 = "mhlo.reshape"(%3003) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3005 = mhlo.maximum %3004, %1119 : tensor<1x384x512xf32>
    %3006 = "mhlo.reshape"(%3005) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3007 = "mhlo.dot"(%3006, %2038) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3008 = "mhlo.broadcast_in_dim"(%2037) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3009 = mhlo.add %3007, %3008 : tensor<384x128xf32>
    %3010 = "mhlo.reshape"(%3009) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3011 = mhlo.add %3010, %2999 : tensor<1x384x128xf32>
    %3012 = "mhlo.broadcast_in_dim"(%2036) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3013 = mhlo.multiply %3011, %3012 : tensor<1x384x128xf32>
    %3014 = "mhlo.broadcast_in_dim"(%2035) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3015 = mhlo.add %3013, %3014 : tensor<1x384x128xf32>
    %3016 = "mhlo.reshape"(%3015) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3017 = "mhlo.dot"(%3016, %2040) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3018 = "mhlo.broadcast_in_dim"(%2039) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3019 = mhlo.add %3017, %3018 : tensor<384x512xf32>
    %3020 = "mhlo.reshape"(%3019) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3021 = mhlo.maximum %3020, %1119 : tensor<1x384x512xf32>
    %3022 = "mhlo.reshape"(%3021) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3023 = "mhlo.dot"(%3022, %2048) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3024 = "mhlo.broadcast_in_dim"(%2047) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3025 = mhlo.add %3023, %3024 : tensor<384x128xf32>
    %3026 = "mhlo.reshape"(%3025) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3027 = mhlo.add %3026, %3015 : tensor<1x384x128xf32>
    %3028 = "mhlo.broadcast_in_dim"(%2042) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3029 = mhlo.multiply %3027, %3028 : tensor<1x384x128xf32>
    %3030 = "mhlo.broadcast_in_dim"(%2041) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3031 = mhlo.add %3029, %3030 : tensor<1x384x128xf32>
    %3032 = "mhlo.reshape"(%3031) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3033 = "mhlo.dot"(%3032, %2046) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3034 = "mhlo.broadcast_in_dim"(%2045) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3035 = mhlo.add %3033, %3034 : tensor<384x512xf32>
    %3036 = "mhlo.reshape"(%3035) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3037 = mhlo.add %3036, %2911 : tensor<1x384x512xf32>
    %3038 = "mhlo.broadcast_in_dim"(%2044) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3039 = mhlo.multiply %3037, %3038 : tensor<1x384x512xf32>
    %3040 = "mhlo.broadcast_in_dim"(%2043) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3041 = mhlo.add %3039, %3040 : tensor<1x384x512xf32>
    %3042 = "mhlo.reshape"(%3041) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3043 = "mhlo.dot"(%3042, %2058) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3044 = "mhlo.broadcast_in_dim"(%2057) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3045 = mhlo.add %3043, %3044 : tensor<384x128xf32>
    %3046 = "mhlo.reshape"(%3045) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3047 = "mhlo.transpose"(%3046) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3048 = "mhlo.dot"(%3042, %2062) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3049 = "mhlo.reshape"(%3048) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3050 = "mhlo.broadcast_in_dim"(%2061) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3051 = mhlo.add %3049, %3050 : tensor<1x384x128xf32>
    %3052 = "mhlo.broadcast_in_dim"(%2060) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3053 = mhlo.multiply %3051, %3052 : tensor<1x384x128xf32>
    %3054 = "mhlo.broadcast_in_dim"(%2059) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3055 = mhlo.add %3053, %3054 : tensor<1x384x128xf32>
    %3056 = "mhlo.reshape"(%3055) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3057 = "mhlo.dot"(%3056, %2054) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3058 = "mhlo.broadcast_in_dim"(%2053) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3059 = mhlo.add %3057, %3058 : tensor<384x128xf32>
    %3060 = "mhlo.reshape"(%3059) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3061 = "mhlo.transpose"(%3060) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3062 = "mhlo.dot"(%3056, %2056) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3063 = "mhlo.broadcast_in_dim"(%2055) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3064 = mhlo.add %3062, %3063 : tensor<384x128xf32>
    %3065 = "mhlo.reshape"(%3064) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3066 = "mhlo.transpose"(%3065) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3067 = "mhlo.dot_general"(%3066, %3061) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3068 = mhlo.multiply %3067, %1114 : tensor<1x4x384x384xf32>
    %3069 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3070 = mhlo.add %3068, %3069 : tensor<1x4x384x384xf32>
    %3071 = "mhlo.reduce"(%3070, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3072 = "mhlo.broadcast_in_dim"(%3071) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3073 = mhlo.subtract %3070, %3072 : tensor<1x4x384x384xf32>
    %3074 = "mhlo.exponential"(%3073) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3075 = "mhlo.reduce"(%3074, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3076 = "mhlo.broadcast_in_dim"(%3075) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3077 = mhlo.divide %3074, %3076 : tensor<1x4x384x384xf32>
    %3078 = "mhlo.dot_general"(%3077, %3047) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3079 = "mhlo.transpose"(%3078) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3080 = "mhlo.reshape"(%3079) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3081 = "mhlo.dot"(%3080, %2052) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3082 = "mhlo.broadcast_in_dim"(%2051) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3083 = mhlo.add %3081, %3082 : tensor<384x128xf32>
    %3084 = "mhlo.reshape"(%3083) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3085 = "mhlo.dot"(%3042, %2066) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3086 = "mhlo.broadcast_in_dim"(%2065) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3087 = mhlo.add %3085, %3086 : tensor<384x128xf32>
    %3088 = "mhlo.reshape"(%3087) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3089 = "mhlo.broadcast_in_dim"(%2064) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3090 = mhlo.multiply %3088, %3089 : tensor<1x384x128xf32>
    %3091 = "mhlo.broadcast_in_dim"(%2063) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3092 = mhlo.add %3090, %3091 : tensor<1x384x128xf32>
    %3093 = mhlo.add %3084, %3092 : tensor<1x384x128xf32>
    %3094 = "mhlo.broadcast_in_dim"(%2050) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3095 = mhlo.multiply %3093, %3094 : tensor<1x384x128xf32>
    %3096 = "mhlo.broadcast_in_dim"(%2049) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3097 = mhlo.add %3095, %3096 : tensor<1x384x128xf32>
    %3098 = "mhlo.reshape"(%3097) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3099 = "mhlo.dot"(%3098, %2068) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3100 = "mhlo.broadcast_in_dim"(%2067) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3101 = mhlo.add %3099, %3100 : tensor<384x512xf32>
    %3102 = "mhlo.reshape"(%3101) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3103 = mhlo.maximum %3102, %1119 : tensor<1x384x512xf32>
    %3104 = "mhlo.reshape"(%3103) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3105 = "mhlo.dot"(%3104, %2072) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3106 = "mhlo.broadcast_in_dim"(%2071) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3107 = mhlo.add %3105, %3106 : tensor<384x128xf32>
    %3108 = "mhlo.reshape"(%3107) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3109 = mhlo.add %3108, %3097 : tensor<1x384x128xf32>
    %3110 = "mhlo.broadcast_in_dim"(%2070) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3111 = mhlo.multiply %3109, %3110 : tensor<1x384x128xf32>
    %3112 = "mhlo.broadcast_in_dim"(%2069) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3113 = mhlo.add %3111, %3112 : tensor<1x384x128xf32>
    %3114 = "mhlo.reshape"(%3113) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3115 = "mhlo.dot"(%3114, %2074) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3116 = "mhlo.broadcast_in_dim"(%2073) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3117 = mhlo.add %3115, %3116 : tensor<384x512xf32>
    %3118 = "mhlo.reshape"(%3117) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3119 = mhlo.maximum %3118, %1119 : tensor<1x384x512xf32>
    %3120 = "mhlo.reshape"(%3119) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3121 = "mhlo.dot"(%3120, %2078) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3122 = "mhlo.broadcast_in_dim"(%2077) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3123 = mhlo.add %3121, %3122 : tensor<384x128xf32>
    %3124 = "mhlo.reshape"(%3123) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3125 = mhlo.add %3124, %3113 : tensor<1x384x128xf32>
    %3126 = "mhlo.broadcast_in_dim"(%2076) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3127 = mhlo.multiply %3125, %3126 : tensor<1x384x128xf32>
    %3128 = "mhlo.broadcast_in_dim"(%2075) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3129 = mhlo.add %3127, %3128 : tensor<1x384x128xf32>
    %3130 = "mhlo.reshape"(%3129) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3131 = "mhlo.dot"(%3130, %2080) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3132 = "mhlo.broadcast_in_dim"(%2079) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3133 = mhlo.add %3131, %3132 : tensor<384x512xf32>
    %3134 = "mhlo.reshape"(%3133) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3135 = mhlo.maximum %3134, %1119 : tensor<1x384x512xf32>
    %3136 = "mhlo.reshape"(%3135) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3137 = "mhlo.dot"(%3136, %2084) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3138 = "mhlo.broadcast_in_dim"(%2083) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3139 = mhlo.add %3137, %3138 : tensor<384x128xf32>
    %3140 = "mhlo.reshape"(%3139) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3141 = mhlo.add %3140, %3129 : tensor<1x384x128xf32>
    %3142 = "mhlo.broadcast_in_dim"(%2082) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3143 = mhlo.multiply %3141, %3142 : tensor<1x384x128xf32>
    %3144 = "mhlo.broadcast_in_dim"(%2081) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3145 = mhlo.add %3143, %3144 : tensor<1x384x128xf32>
    %3146 = "mhlo.reshape"(%3145) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3147 = "mhlo.dot"(%3146, %2086) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3148 = "mhlo.broadcast_in_dim"(%2085) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3149 = mhlo.add %3147, %3148 : tensor<384x512xf32>
    %3150 = "mhlo.reshape"(%3149) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3151 = mhlo.maximum %3150, %1119 : tensor<1x384x512xf32>
    %3152 = "mhlo.reshape"(%3151) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3153 = "mhlo.dot"(%3152, %2094) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3154 = "mhlo.broadcast_in_dim"(%2093) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3155 = mhlo.add %3153, %3154 : tensor<384x128xf32>
    %3156 = "mhlo.reshape"(%3155) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3157 = mhlo.add %3156, %3145 : tensor<1x384x128xf32>
    %3158 = "mhlo.broadcast_in_dim"(%2088) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3159 = mhlo.multiply %3157, %3158 : tensor<1x384x128xf32>
    %3160 = "mhlo.broadcast_in_dim"(%2087) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3161 = mhlo.add %3159, %3160 : tensor<1x384x128xf32>
    %3162 = "mhlo.reshape"(%3161) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3163 = "mhlo.dot"(%3162, %2092) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3164 = "mhlo.broadcast_in_dim"(%2091) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3165 = mhlo.add %3163, %3164 : tensor<384x512xf32>
    %3166 = "mhlo.reshape"(%3165) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3167 = mhlo.add %3166, %3041 : tensor<1x384x512xf32>
    %3168 = "mhlo.broadcast_in_dim"(%2090) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3169 = mhlo.multiply %3167, %3168 : tensor<1x384x512xf32>
    %3170 = "mhlo.broadcast_in_dim"(%2089) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3171 = mhlo.add %3169, %3170 : tensor<1x384x512xf32>
    %3172 = "mhlo.reshape"(%3171) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3173 = "mhlo.dot"(%3172, %2104) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3174 = "mhlo.broadcast_in_dim"(%2103) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3175 = mhlo.add %3173, %3174 : tensor<384x128xf32>
    %3176 = "mhlo.reshape"(%3175) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3177 = "mhlo.transpose"(%3176) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3178 = "mhlo.dot"(%3172, %2108) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3179 = "mhlo.reshape"(%3178) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3180 = "mhlo.broadcast_in_dim"(%2107) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3181 = mhlo.add %3179, %3180 : tensor<1x384x128xf32>
    %3182 = "mhlo.broadcast_in_dim"(%2106) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3183 = mhlo.multiply %3181, %3182 : tensor<1x384x128xf32>
    %3184 = "mhlo.broadcast_in_dim"(%2105) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3185 = mhlo.add %3183, %3184 : tensor<1x384x128xf32>
    %3186 = "mhlo.reshape"(%3185) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3187 = "mhlo.dot"(%3186, %2100) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3188 = "mhlo.broadcast_in_dim"(%2099) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3189 = mhlo.add %3187, %3188 : tensor<384x128xf32>
    %3190 = "mhlo.reshape"(%3189) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3191 = "mhlo.transpose"(%3190) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3192 = "mhlo.dot"(%3186, %2102) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3193 = "mhlo.broadcast_in_dim"(%2101) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3194 = mhlo.add %3192, %3193 : tensor<384x128xf32>
    %3195 = "mhlo.reshape"(%3194) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3196 = "mhlo.transpose"(%3195) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3197 = "mhlo.dot_general"(%3196, %3191) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3198 = mhlo.multiply %3197, %1114 : tensor<1x4x384x384xf32>
    %3199 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3200 = mhlo.add %3198, %3199 : tensor<1x4x384x384xf32>
    %3201 = "mhlo.reduce"(%3200, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3202 = "mhlo.broadcast_in_dim"(%3201) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3203 = mhlo.subtract %3200, %3202 : tensor<1x4x384x384xf32>
    %3204 = "mhlo.exponential"(%3203) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3205 = "mhlo.reduce"(%3204, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3206 = "mhlo.broadcast_in_dim"(%3205) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3207 = mhlo.divide %3204, %3206 : tensor<1x4x384x384xf32>
    %3208 = "mhlo.dot_general"(%3207, %3177) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3209 = "mhlo.transpose"(%3208) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3210 = "mhlo.reshape"(%3209) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3211 = "mhlo.dot"(%3210, %2098) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3212 = "mhlo.broadcast_in_dim"(%2097) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3213 = mhlo.add %3211, %3212 : tensor<384x128xf32>
    %3214 = "mhlo.reshape"(%3213) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3215 = "mhlo.dot"(%3172, %2112) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3216 = "mhlo.broadcast_in_dim"(%2111) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3217 = mhlo.add %3215, %3216 : tensor<384x128xf32>
    %3218 = "mhlo.reshape"(%3217) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3219 = "mhlo.broadcast_in_dim"(%2110) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3220 = mhlo.multiply %3218, %3219 : tensor<1x384x128xf32>
    %3221 = "mhlo.broadcast_in_dim"(%2109) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3222 = mhlo.add %3220, %3221 : tensor<1x384x128xf32>
    %3223 = mhlo.add %3214, %3222 : tensor<1x384x128xf32>
    %3224 = "mhlo.broadcast_in_dim"(%2096) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3225 = mhlo.multiply %3223, %3224 : tensor<1x384x128xf32>
    %3226 = "mhlo.broadcast_in_dim"(%2095) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3227 = mhlo.add %3225, %3226 : tensor<1x384x128xf32>
    %3228 = "mhlo.reshape"(%3227) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3229 = "mhlo.dot"(%3228, %2114) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3230 = "mhlo.broadcast_in_dim"(%2113) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3231 = mhlo.add %3229, %3230 : tensor<384x512xf32>
    %3232 = "mhlo.reshape"(%3231) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3233 = mhlo.maximum %3232, %1119 : tensor<1x384x512xf32>
    %3234 = "mhlo.reshape"(%3233) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3235 = "mhlo.dot"(%3234, %2118) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3236 = "mhlo.broadcast_in_dim"(%2117) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3237 = mhlo.add %3235, %3236 : tensor<384x128xf32>
    %3238 = "mhlo.reshape"(%3237) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3239 = mhlo.add %3238, %3227 : tensor<1x384x128xf32>
    %3240 = "mhlo.broadcast_in_dim"(%2116) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3241 = mhlo.multiply %3239, %3240 : tensor<1x384x128xf32>
    %3242 = "mhlo.broadcast_in_dim"(%2115) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3243 = mhlo.add %3241, %3242 : tensor<1x384x128xf32>
    %3244 = "mhlo.reshape"(%3243) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3245 = "mhlo.dot"(%3244, %2120) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3246 = "mhlo.broadcast_in_dim"(%2119) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3247 = mhlo.add %3245, %3246 : tensor<384x512xf32>
    %3248 = "mhlo.reshape"(%3247) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3249 = mhlo.maximum %3248, %1119 : tensor<1x384x512xf32>
    %3250 = "mhlo.reshape"(%3249) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3251 = "mhlo.dot"(%3250, %2124) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3252 = "mhlo.broadcast_in_dim"(%2123) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3253 = mhlo.add %3251, %3252 : tensor<384x128xf32>
    %3254 = "mhlo.reshape"(%3253) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3255 = mhlo.add %3254, %3243 : tensor<1x384x128xf32>
    %3256 = "mhlo.broadcast_in_dim"(%2122) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3257 = mhlo.multiply %3255, %3256 : tensor<1x384x128xf32>
    %3258 = "mhlo.broadcast_in_dim"(%2121) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3259 = mhlo.add %3257, %3258 : tensor<1x384x128xf32>
    %3260 = "mhlo.reshape"(%3259) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3261 = "mhlo.dot"(%3260, %2126) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3262 = "mhlo.broadcast_in_dim"(%2125) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3263 = mhlo.add %3261, %3262 : tensor<384x512xf32>
    %3264 = "mhlo.reshape"(%3263) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3265 = mhlo.maximum %3264, %1119 : tensor<1x384x512xf32>
    %3266 = "mhlo.reshape"(%3265) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3267 = "mhlo.dot"(%3266, %2130) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3268 = "mhlo.broadcast_in_dim"(%2129) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3269 = mhlo.add %3267, %3268 : tensor<384x128xf32>
    %3270 = "mhlo.reshape"(%3269) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3271 = mhlo.add %3270, %3259 : tensor<1x384x128xf32>
    %3272 = "mhlo.broadcast_in_dim"(%2128) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3273 = mhlo.multiply %3271, %3272 : tensor<1x384x128xf32>
    %3274 = "mhlo.broadcast_in_dim"(%2127) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3275 = mhlo.add %3273, %3274 : tensor<1x384x128xf32>
    %3276 = "mhlo.reshape"(%3275) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3277 = "mhlo.dot"(%3276, %2132) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3278 = "mhlo.broadcast_in_dim"(%2131) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3279 = mhlo.add %3277, %3278 : tensor<384x512xf32>
    %3280 = "mhlo.reshape"(%3279) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3281 = mhlo.maximum %3280, %1119 : tensor<1x384x512xf32>
    %3282 = "mhlo.reshape"(%3281) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3283 = "mhlo.dot"(%3282, %2140) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3284 = "mhlo.broadcast_in_dim"(%2139) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3285 = mhlo.add %3283, %3284 : tensor<384x128xf32>
    %3286 = "mhlo.reshape"(%3285) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3287 = mhlo.add %3286, %3275 : tensor<1x384x128xf32>
    %3288 = "mhlo.broadcast_in_dim"(%2134) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3289 = mhlo.multiply %3287, %3288 : tensor<1x384x128xf32>
    %3290 = "mhlo.broadcast_in_dim"(%2133) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3291 = mhlo.add %3289, %3290 : tensor<1x384x128xf32>
    %3292 = "mhlo.reshape"(%3291) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3293 = "mhlo.dot"(%3292, %2138) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3294 = "mhlo.broadcast_in_dim"(%2137) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3295 = mhlo.add %3293, %3294 : tensor<384x512xf32>
    %3296 = "mhlo.reshape"(%3295) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3297 = mhlo.add %3296, %3171 : tensor<1x384x512xf32>
    %3298 = "mhlo.broadcast_in_dim"(%2136) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3299 = mhlo.multiply %3297, %3298 : tensor<1x384x512xf32>
    %3300 = "mhlo.broadcast_in_dim"(%2135) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3301 = mhlo.add %3299, %3300 : tensor<1x384x512xf32>
    %3302 = "mhlo.reshape"(%3301) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3303 = "mhlo.dot"(%3302, %2150) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3304 = "mhlo.broadcast_in_dim"(%2149) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3305 = mhlo.add %3303, %3304 : tensor<384x128xf32>
    %3306 = "mhlo.reshape"(%3305) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3307 = "mhlo.transpose"(%3306) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3308 = "mhlo.dot"(%3302, %2154) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3309 = "mhlo.reshape"(%3308) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3310 = "mhlo.broadcast_in_dim"(%2153) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3311 = mhlo.add %3309, %3310 : tensor<1x384x128xf32>
    %3312 = "mhlo.broadcast_in_dim"(%2152) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3313 = mhlo.multiply %3311, %3312 : tensor<1x384x128xf32>
    %3314 = "mhlo.broadcast_in_dim"(%2151) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3315 = mhlo.add %3313, %3314 : tensor<1x384x128xf32>
    %3316 = "mhlo.reshape"(%3315) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3317 = "mhlo.dot"(%3316, %2146) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3318 = "mhlo.broadcast_in_dim"(%2145) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3319 = mhlo.add %3317, %3318 : tensor<384x128xf32>
    %3320 = "mhlo.reshape"(%3319) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3321 = "mhlo.transpose"(%3320) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3322 = "mhlo.dot"(%3316, %2148) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3323 = "mhlo.broadcast_in_dim"(%2147) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3324 = mhlo.add %3322, %3323 : tensor<384x128xf32>
    %3325 = "mhlo.reshape"(%3324) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3326 = "mhlo.transpose"(%3325) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3327 = "mhlo.dot_general"(%3326, %3321) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3328 = mhlo.multiply %3327, %1114 : tensor<1x4x384x384xf32>
    %3329 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3330 = mhlo.add %3328, %3329 : tensor<1x4x384x384xf32>
    %3331 = "mhlo.reduce"(%3330, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3332 = "mhlo.broadcast_in_dim"(%3331) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3333 = mhlo.subtract %3330, %3332 : tensor<1x4x384x384xf32>
    %3334 = "mhlo.exponential"(%3333) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3335 = "mhlo.reduce"(%3334, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3336 = "mhlo.broadcast_in_dim"(%3335) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3337 = mhlo.divide %3334, %3336 : tensor<1x4x384x384xf32>
    %3338 = "mhlo.dot_general"(%3337, %3307) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3339 = "mhlo.transpose"(%3338) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3340 = "mhlo.reshape"(%3339) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3341 = "mhlo.dot"(%3340, %2144) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3342 = "mhlo.broadcast_in_dim"(%2143) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3343 = mhlo.add %3341, %3342 : tensor<384x128xf32>
    %3344 = "mhlo.reshape"(%3343) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3345 = "mhlo.dot"(%3302, %2158) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3346 = "mhlo.broadcast_in_dim"(%2157) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3347 = mhlo.add %3345, %3346 : tensor<384x128xf32>
    %3348 = "mhlo.reshape"(%3347) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3349 = "mhlo.broadcast_in_dim"(%2156) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3350 = mhlo.multiply %3348, %3349 : tensor<1x384x128xf32>
    %3351 = "mhlo.broadcast_in_dim"(%2155) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3352 = mhlo.add %3350, %3351 : tensor<1x384x128xf32>
    %3353 = mhlo.add %3344, %3352 : tensor<1x384x128xf32>
    %3354 = "mhlo.broadcast_in_dim"(%2142) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3355 = mhlo.multiply %3353, %3354 : tensor<1x384x128xf32>
    %3356 = "mhlo.broadcast_in_dim"(%2141) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3357 = mhlo.add %3355, %3356 : tensor<1x384x128xf32>
    %3358 = "mhlo.reshape"(%3357) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3359 = "mhlo.dot"(%3358, %2160) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3360 = "mhlo.broadcast_in_dim"(%2159) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3361 = mhlo.add %3359, %3360 : tensor<384x512xf32>
    %3362 = "mhlo.reshape"(%3361) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3363 = mhlo.maximum %3362, %1119 : tensor<1x384x512xf32>
    %3364 = "mhlo.reshape"(%3363) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3365 = "mhlo.dot"(%3364, %2164) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3366 = "mhlo.broadcast_in_dim"(%2163) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3367 = mhlo.add %3365, %3366 : tensor<384x128xf32>
    %3368 = "mhlo.reshape"(%3367) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3369 = mhlo.add %3368, %3357 : tensor<1x384x128xf32>
    %3370 = "mhlo.broadcast_in_dim"(%2162) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3371 = mhlo.multiply %3369, %3370 : tensor<1x384x128xf32>
    %3372 = "mhlo.broadcast_in_dim"(%2161) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3373 = mhlo.add %3371, %3372 : tensor<1x384x128xf32>
    %3374 = "mhlo.reshape"(%3373) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3375 = "mhlo.dot"(%3374, %2166) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3376 = "mhlo.broadcast_in_dim"(%2165) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3377 = mhlo.add %3375, %3376 : tensor<384x512xf32>
    %3378 = "mhlo.reshape"(%3377) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3379 = mhlo.maximum %3378, %1119 : tensor<1x384x512xf32>
    %3380 = "mhlo.reshape"(%3379) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3381 = "mhlo.dot"(%3380, %2170) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3382 = "mhlo.broadcast_in_dim"(%2169) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3383 = mhlo.add %3381, %3382 : tensor<384x128xf32>
    %3384 = "mhlo.reshape"(%3383) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3385 = mhlo.add %3384, %3373 : tensor<1x384x128xf32>
    %3386 = "mhlo.broadcast_in_dim"(%2168) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3387 = mhlo.multiply %3385, %3386 : tensor<1x384x128xf32>
    %3388 = "mhlo.broadcast_in_dim"(%2167) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3389 = mhlo.add %3387, %3388 : tensor<1x384x128xf32>
    %3390 = "mhlo.reshape"(%3389) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3391 = "mhlo.dot"(%3390, %2172) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3392 = "mhlo.broadcast_in_dim"(%2171) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3393 = mhlo.add %3391, %3392 : tensor<384x512xf32>
    %3394 = "mhlo.reshape"(%3393) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3395 = mhlo.maximum %3394, %1119 : tensor<1x384x512xf32>
    %3396 = "mhlo.reshape"(%3395) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3397 = "mhlo.dot"(%3396, %2176) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3398 = "mhlo.broadcast_in_dim"(%2175) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3399 = mhlo.add %3397, %3398 : tensor<384x128xf32>
    %3400 = "mhlo.reshape"(%3399) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3401 = mhlo.add %3400, %3389 : tensor<1x384x128xf32>
    %3402 = "mhlo.broadcast_in_dim"(%2174) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3403 = mhlo.multiply %3401, %3402 : tensor<1x384x128xf32>
    %3404 = "mhlo.broadcast_in_dim"(%2173) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3405 = mhlo.add %3403, %3404 : tensor<1x384x128xf32>
    %3406 = "mhlo.reshape"(%3405) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3407 = "mhlo.dot"(%3406, %2178) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3408 = "mhlo.broadcast_in_dim"(%2177) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3409 = mhlo.add %3407, %3408 : tensor<384x512xf32>
    %3410 = "mhlo.reshape"(%3409) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3411 = mhlo.maximum %3410, %1119 : tensor<1x384x512xf32>
    %3412 = "mhlo.reshape"(%3411) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3413 = "mhlo.dot"(%3412, %2186) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3414 = "mhlo.broadcast_in_dim"(%2185) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3415 = mhlo.add %3413, %3414 : tensor<384x128xf32>
    %3416 = "mhlo.reshape"(%3415) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3417 = mhlo.add %3416, %3405 : tensor<1x384x128xf32>
    %3418 = "mhlo.broadcast_in_dim"(%2180) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3419 = mhlo.multiply %3417, %3418 : tensor<1x384x128xf32>
    %3420 = "mhlo.broadcast_in_dim"(%2179) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3421 = mhlo.add %3419, %3420 : tensor<1x384x128xf32>
    %3422 = "mhlo.reshape"(%3421) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3423 = "mhlo.dot"(%3422, %2184) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3424 = "mhlo.broadcast_in_dim"(%2183) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3425 = mhlo.add %3423, %3424 : tensor<384x512xf32>
    %3426 = "mhlo.reshape"(%3425) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3427 = mhlo.add %3426, %3301 : tensor<1x384x512xf32>
    %3428 = "mhlo.broadcast_in_dim"(%2182) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3429 = mhlo.multiply %3427, %3428 : tensor<1x384x512xf32>
    %3430 = "mhlo.broadcast_in_dim"(%2181) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3431 = mhlo.add %3429, %3430 : tensor<1x384x512xf32>
    %3432 = "mhlo.reshape"(%3431) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3433 = "mhlo.dot"(%3432, %2196) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3434 = "mhlo.broadcast_in_dim"(%2195) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3435 = mhlo.add %3433, %3434 : tensor<384x128xf32>
    %3436 = "mhlo.reshape"(%3435) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3437 = "mhlo.transpose"(%3436) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3438 = "mhlo.dot"(%3432, %2200) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3439 = "mhlo.reshape"(%3438) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3440 = "mhlo.broadcast_in_dim"(%2199) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3441 = mhlo.add %3439, %3440 : tensor<1x384x128xf32>
    %3442 = "mhlo.broadcast_in_dim"(%2198) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3443 = mhlo.multiply %3441, %3442 : tensor<1x384x128xf32>
    %3444 = "mhlo.broadcast_in_dim"(%2197) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3445 = mhlo.add %3443, %3444 : tensor<1x384x128xf32>
    %3446 = "mhlo.reshape"(%3445) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3447 = "mhlo.dot"(%3446, %2192) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3448 = "mhlo.broadcast_in_dim"(%2191) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3449 = mhlo.add %3447, %3448 : tensor<384x128xf32>
    %3450 = "mhlo.reshape"(%3449) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3451 = "mhlo.transpose"(%3450) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3452 = "mhlo.dot"(%3446, %2194) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3453 = "mhlo.broadcast_in_dim"(%2193) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3454 = mhlo.add %3452, %3453 : tensor<384x128xf32>
    %3455 = "mhlo.reshape"(%3454) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3456 = "mhlo.transpose"(%3455) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3457 = "mhlo.dot_general"(%3456, %3451) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3458 = mhlo.multiply %3457, %1114 : tensor<1x4x384x384xf32>
    %3459 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3460 = mhlo.add %3458, %3459 : tensor<1x4x384x384xf32>
    %3461 = "mhlo.reduce"(%3460, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3462 = "mhlo.broadcast_in_dim"(%3461) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3463 = mhlo.subtract %3460, %3462 : tensor<1x4x384x384xf32>
    %3464 = "mhlo.exponential"(%3463) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3465 = "mhlo.reduce"(%3464, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3466 = "mhlo.broadcast_in_dim"(%3465) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3467 = mhlo.divide %3464, %3466 : tensor<1x4x384x384xf32>
    %3468 = "mhlo.dot_general"(%3467, %3437) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3469 = "mhlo.transpose"(%3468) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3470 = "mhlo.reshape"(%3469) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3471 = "mhlo.dot"(%3470, %2190) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3472 = "mhlo.broadcast_in_dim"(%2189) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3473 = mhlo.add %3471, %3472 : tensor<384x128xf32>
    %3474 = "mhlo.reshape"(%3473) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3475 = "mhlo.dot"(%3432, %2204) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3476 = "mhlo.broadcast_in_dim"(%2203) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3477 = mhlo.add %3475, %3476 : tensor<384x128xf32>
    %3478 = "mhlo.reshape"(%3477) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3479 = "mhlo.broadcast_in_dim"(%2202) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3480 = mhlo.multiply %3478, %3479 : tensor<1x384x128xf32>
    %3481 = "mhlo.broadcast_in_dim"(%2201) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3482 = mhlo.add %3480, %3481 : tensor<1x384x128xf32>
    %3483 = mhlo.add %3474, %3482 : tensor<1x384x128xf32>
    %3484 = "mhlo.broadcast_in_dim"(%2188) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3485 = mhlo.multiply %3483, %3484 : tensor<1x384x128xf32>
    %3486 = "mhlo.broadcast_in_dim"(%2187) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3487 = mhlo.add %3485, %3486 : tensor<1x384x128xf32>
    %3488 = "mhlo.reshape"(%3487) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3489 = "mhlo.dot"(%3488, %2206) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3490 = "mhlo.broadcast_in_dim"(%2205) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3491 = mhlo.add %3489, %3490 : tensor<384x512xf32>
    %3492 = "mhlo.reshape"(%3491) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3493 = mhlo.maximum %3492, %1119 : tensor<1x384x512xf32>
    %3494 = "mhlo.reshape"(%3493) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3495 = "mhlo.dot"(%3494, %2210) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3496 = "mhlo.broadcast_in_dim"(%2209) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3497 = mhlo.add %3495, %3496 : tensor<384x128xf32>
    %3498 = "mhlo.reshape"(%3497) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3499 = mhlo.add %3498, %3487 : tensor<1x384x128xf32>
    %3500 = "mhlo.broadcast_in_dim"(%2208) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3501 = mhlo.multiply %3499, %3500 : tensor<1x384x128xf32>
    %3502 = "mhlo.broadcast_in_dim"(%2207) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3503 = mhlo.add %3501, %3502 : tensor<1x384x128xf32>
    %3504 = "mhlo.reshape"(%3503) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3505 = "mhlo.dot"(%3504, %2212) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3506 = "mhlo.broadcast_in_dim"(%2211) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3507 = mhlo.add %3505, %3506 : tensor<384x512xf32>
    %3508 = "mhlo.reshape"(%3507) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3509 = mhlo.maximum %3508, %1119 : tensor<1x384x512xf32>
    %3510 = "mhlo.reshape"(%3509) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3511 = "mhlo.dot"(%3510, %2216) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3512 = "mhlo.broadcast_in_dim"(%2215) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3513 = mhlo.add %3511, %3512 : tensor<384x128xf32>
    %3514 = "mhlo.reshape"(%3513) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3515 = mhlo.add %3514, %3503 : tensor<1x384x128xf32>
    %3516 = "mhlo.broadcast_in_dim"(%2214) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3517 = mhlo.multiply %3515, %3516 : tensor<1x384x128xf32>
    %3518 = "mhlo.broadcast_in_dim"(%2213) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3519 = mhlo.add %3517, %3518 : tensor<1x384x128xf32>
    %3520 = "mhlo.reshape"(%3519) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3521 = "mhlo.dot"(%3520, %2218) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3522 = "mhlo.broadcast_in_dim"(%2217) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3523 = mhlo.add %3521, %3522 : tensor<384x512xf32>
    %3524 = "mhlo.reshape"(%3523) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3525 = mhlo.maximum %3524, %1119 : tensor<1x384x512xf32>
    %3526 = "mhlo.reshape"(%3525) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3527 = "mhlo.dot"(%3526, %2222) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3528 = "mhlo.broadcast_in_dim"(%2221) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3529 = mhlo.add %3527, %3528 : tensor<384x128xf32>
    %3530 = "mhlo.reshape"(%3529) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3531 = mhlo.add %3530, %3519 : tensor<1x384x128xf32>
    %3532 = "mhlo.broadcast_in_dim"(%2220) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3533 = mhlo.multiply %3531, %3532 : tensor<1x384x128xf32>
    %3534 = "mhlo.broadcast_in_dim"(%2219) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3535 = mhlo.add %3533, %3534 : tensor<1x384x128xf32>
    %3536 = "mhlo.reshape"(%3535) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3537 = "mhlo.dot"(%3536, %2224) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3538 = "mhlo.broadcast_in_dim"(%2223) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3539 = mhlo.add %3537, %3538 : tensor<384x512xf32>
    %3540 = "mhlo.reshape"(%3539) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3541 = mhlo.maximum %3540, %1119 : tensor<1x384x512xf32>
    %3542 = "mhlo.reshape"(%3541) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3543 = "mhlo.dot"(%3542, %2232) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3544 = "mhlo.broadcast_in_dim"(%2231) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3545 = mhlo.add %3543, %3544 : tensor<384x128xf32>
    %3546 = "mhlo.reshape"(%3545) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3547 = mhlo.add %3546, %3535 : tensor<1x384x128xf32>
    %3548 = "mhlo.broadcast_in_dim"(%2226) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3549 = mhlo.multiply %3547, %3548 : tensor<1x384x128xf32>
    %3550 = "mhlo.broadcast_in_dim"(%2225) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3551 = mhlo.add %3549, %3550 : tensor<1x384x128xf32>
    %3552 = "mhlo.reshape"(%3551) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3553 = "mhlo.dot"(%3552, %2230) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3554 = "mhlo.broadcast_in_dim"(%2229) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3555 = mhlo.add %3553, %3554 : tensor<384x512xf32>
    %3556 = "mhlo.reshape"(%3555) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3557 = mhlo.add %3556, %3431 : tensor<1x384x512xf32>
    %3558 = "mhlo.broadcast_in_dim"(%2228) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3559 = mhlo.multiply %3557, %3558 : tensor<1x384x512xf32>
    %3560 = "mhlo.broadcast_in_dim"(%2227) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3561 = mhlo.add %3559, %3560 : tensor<1x384x512xf32>
    %3562 = "mhlo.reshape"(%3561) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3563 = "mhlo.dot"(%3562, %1230) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3564 = "mhlo.broadcast_in_dim"(%1229) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3565 = mhlo.add %3563, %3564 : tensor<384x128xf32>
    %3566 = "mhlo.reshape"(%3565) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3567 = "mhlo.transpose"(%3566) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3568 = "mhlo.dot"(%3562, %1234) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3569 = "mhlo.reshape"(%3568) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3570 = "mhlo.broadcast_in_dim"(%1233) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3571 = mhlo.add %3569, %3570 : tensor<1x384x128xf32>
    %3572 = "mhlo.broadcast_in_dim"(%1232) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3573 = mhlo.multiply %3571, %3572 : tensor<1x384x128xf32>
    %3574 = "mhlo.broadcast_in_dim"(%1231) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3575 = mhlo.add %3573, %3574 : tensor<1x384x128xf32>
    %3576 = "mhlo.reshape"(%3575) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3577 = "mhlo.dot"(%3576, %1226) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3578 = "mhlo.broadcast_in_dim"(%1225) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3579 = mhlo.add %3577, %3578 : tensor<384x128xf32>
    %3580 = "mhlo.reshape"(%3579) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3581 = "mhlo.transpose"(%3580) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3582 = "mhlo.dot"(%3576, %1228) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3583 = "mhlo.broadcast_in_dim"(%1227) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3584 = mhlo.add %3582, %3583 : tensor<384x128xf32>
    %3585 = "mhlo.reshape"(%3584) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3586 = "mhlo.transpose"(%3585) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3587 = "mhlo.dot_general"(%3586, %3581) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3588 = mhlo.multiply %3587, %1114 : tensor<1x4x384x384xf32>
    %3589 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3590 = mhlo.add %3588, %3589 : tensor<1x4x384x384xf32>
    %3591 = "mhlo.reduce"(%3590, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3592 = "mhlo.broadcast_in_dim"(%3591) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3593 = mhlo.subtract %3590, %3592 : tensor<1x4x384x384xf32>
    %3594 = "mhlo.exponential"(%3593) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3595 = "mhlo.reduce"(%3594, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3596 = "mhlo.broadcast_in_dim"(%3595) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3597 = mhlo.divide %3594, %3596 : tensor<1x4x384x384xf32>
    %3598 = "mhlo.dot_general"(%3597, %3567) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3599 = "mhlo.transpose"(%3598) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3600 = "mhlo.reshape"(%3599) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3601 = "mhlo.dot"(%3600, %1224) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3602 = "mhlo.broadcast_in_dim"(%1223) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3603 = mhlo.add %3601, %3602 : tensor<384x128xf32>
    %3604 = "mhlo.reshape"(%3603) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3605 = "mhlo.dot"(%3562, %1238) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3606 = "mhlo.broadcast_in_dim"(%1237) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3607 = mhlo.add %3605, %3606 : tensor<384x128xf32>
    %3608 = "mhlo.reshape"(%3607) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3609 = "mhlo.broadcast_in_dim"(%1236) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3610 = mhlo.multiply %3608, %3609 : tensor<1x384x128xf32>
    %3611 = "mhlo.broadcast_in_dim"(%1235) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3612 = mhlo.add %3610, %3611 : tensor<1x384x128xf32>
    %3613 = mhlo.add %3604, %3612 : tensor<1x384x128xf32>
    %3614 = "mhlo.broadcast_in_dim"(%1222) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3615 = mhlo.multiply %3613, %3614 : tensor<1x384x128xf32>
    %3616 = "mhlo.broadcast_in_dim"(%1221) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3617 = mhlo.add %3615, %3616 : tensor<1x384x128xf32>
    %3618 = "mhlo.reshape"(%3617) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3619 = "mhlo.dot"(%3618, %1240) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3620 = "mhlo.broadcast_in_dim"(%1239) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3621 = mhlo.add %3619, %3620 : tensor<384x512xf32>
    %3622 = "mhlo.reshape"(%3621) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3623 = mhlo.maximum %3622, %1119 : tensor<1x384x512xf32>
    %3624 = "mhlo.reshape"(%3623) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3625 = "mhlo.dot"(%3624, %1244) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3626 = "mhlo.broadcast_in_dim"(%1243) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3627 = mhlo.add %3625, %3626 : tensor<384x128xf32>
    %3628 = "mhlo.reshape"(%3627) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3629 = mhlo.add %3628, %3617 : tensor<1x384x128xf32>
    %3630 = "mhlo.broadcast_in_dim"(%1242) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3631 = mhlo.multiply %3629, %3630 : tensor<1x384x128xf32>
    %3632 = "mhlo.broadcast_in_dim"(%1241) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3633 = mhlo.add %3631, %3632 : tensor<1x384x128xf32>
    %3634 = "mhlo.reshape"(%3633) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3635 = "mhlo.dot"(%3634, %1246) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3636 = "mhlo.broadcast_in_dim"(%1245) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3637 = mhlo.add %3635, %3636 : tensor<384x512xf32>
    %3638 = "mhlo.reshape"(%3637) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3639 = mhlo.maximum %3638, %1119 : tensor<1x384x512xf32>
    %3640 = "mhlo.reshape"(%3639) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3641 = "mhlo.dot"(%3640, %1250) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3642 = "mhlo.broadcast_in_dim"(%1249) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3643 = mhlo.add %3641, %3642 : tensor<384x128xf32>
    %3644 = "mhlo.reshape"(%3643) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3645 = mhlo.add %3644, %3633 : tensor<1x384x128xf32>
    %3646 = "mhlo.broadcast_in_dim"(%1248) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3647 = mhlo.multiply %3645, %3646 : tensor<1x384x128xf32>
    %3648 = "mhlo.broadcast_in_dim"(%1247) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3649 = mhlo.add %3647, %3648 : tensor<1x384x128xf32>
    %3650 = "mhlo.reshape"(%3649) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3651 = "mhlo.dot"(%3650, %1252) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3652 = "mhlo.broadcast_in_dim"(%1251) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3653 = mhlo.add %3651, %3652 : tensor<384x512xf32>
    %3654 = "mhlo.reshape"(%3653) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3655 = mhlo.maximum %3654, %1119 : tensor<1x384x512xf32>
    %3656 = "mhlo.reshape"(%3655) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3657 = "mhlo.dot"(%3656, %1256) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3658 = "mhlo.broadcast_in_dim"(%1255) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3659 = mhlo.add %3657, %3658 : tensor<384x128xf32>
    %3660 = "mhlo.reshape"(%3659) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3661 = mhlo.add %3660, %3649 : tensor<1x384x128xf32>
    %3662 = "mhlo.broadcast_in_dim"(%1254) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3663 = mhlo.multiply %3661, %3662 : tensor<1x384x128xf32>
    %3664 = "mhlo.broadcast_in_dim"(%1253) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3665 = mhlo.add %3663, %3664 : tensor<1x384x128xf32>
    %3666 = "mhlo.reshape"(%3665) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3667 = "mhlo.dot"(%3666, %1258) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3668 = "mhlo.broadcast_in_dim"(%1257) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3669 = mhlo.add %3667, %3668 : tensor<384x512xf32>
    %3670 = "mhlo.reshape"(%3669) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3671 = mhlo.maximum %3670, %1119 : tensor<1x384x512xf32>
    %3672 = "mhlo.reshape"(%3671) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3673 = "mhlo.dot"(%3672, %1266) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3674 = "mhlo.broadcast_in_dim"(%1265) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3675 = mhlo.add %3673, %3674 : tensor<384x128xf32>
    %3676 = "mhlo.reshape"(%3675) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3677 = mhlo.add %3676, %3665 : tensor<1x384x128xf32>
    %3678 = "mhlo.broadcast_in_dim"(%1260) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3679 = mhlo.multiply %3677, %3678 : tensor<1x384x128xf32>
    %3680 = "mhlo.broadcast_in_dim"(%1259) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3681 = mhlo.add %3679, %3680 : tensor<1x384x128xf32>
    %3682 = "mhlo.reshape"(%3681) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3683 = "mhlo.dot"(%3682, %1264) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3684 = "mhlo.broadcast_in_dim"(%1263) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3685 = mhlo.add %3683, %3684 : tensor<384x512xf32>
    %3686 = "mhlo.reshape"(%3685) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3687 = mhlo.add %3686, %3561 : tensor<1x384x512xf32>
    %3688 = "mhlo.broadcast_in_dim"(%1262) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3689 = mhlo.multiply %3687, %3688 : tensor<1x384x512xf32>
    %3690 = "mhlo.broadcast_in_dim"(%1261) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3691 = mhlo.add %3689, %3690 : tensor<1x384x512xf32>
    %3692 = "mhlo.reshape"(%3691) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3693 = "mhlo.dot"(%3692, %1276) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3694 = "mhlo.broadcast_in_dim"(%1275) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3695 = mhlo.add %3693, %3694 : tensor<384x128xf32>
    %3696 = "mhlo.reshape"(%3695) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3697 = "mhlo.transpose"(%3696) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3698 = "mhlo.dot"(%3692, %1280) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3699 = "mhlo.reshape"(%3698) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3700 = "mhlo.broadcast_in_dim"(%1279) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3701 = mhlo.add %3699, %3700 : tensor<1x384x128xf32>
    %3702 = "mhlo.broadcast_in_dim"(%1278) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3703 = mhlo.multiply %3701, %3702 : tensor<1x384x128xf32>
    %3704 = "mhlo.broadcast_in_dim"(%1277) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3705 = mhlo.add %3703, %3704 : tensor<1x384x128xf32>
    %3706 = "mhlo.reshape"(%3705) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3707 = "mhlo.dot"(%3706, %1272) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3708 = "mhlo.broadcast_in_dim"(%1271) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3709 = mhlo.add %3707, %3708 : tensor<384x128xf32>
    %3710 = "mhlo.reshape"(%3709) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3711 = "mhlo.transpose"(%3710) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3712 = "mhlo.dot"(%3706, %1274) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3713 = "mhlo.broadcast_in_dim"(%1273) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3714 = mhlo.add %3712, %3713 : tensor<384x128xf32>
    %3715 = "mhlo.reshape"(%3714) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3716 = "mhlo.transpose"(%3715) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3717 = "mhlo.dot_general"(%3716, %3711) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3718 = mhlo.multiply %3717, %1114 : tensor<1x4x384x384xf32>
    %3719 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3720 = mhlo.add %3718, %3719 : tensor<1x4x384x384xf32>
    %3721 = "mhlo.reduce"(%3720, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3722 = "mhlo.broadcast_in_dim"(%3721) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3723 = mhlo.subtract %3720, %3722 : tensor<1x4x384x384xf32>
    %3724 = "mhlo.exponential"(%3723) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3725 = "mhlo.reduce"(%3724, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3726 = "mhlo.broadcast_in_dim"(%3725) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3727 = mhlo.divide %3724, %3726 : tensor<1x4x384x384xf32>
    %3728 = "mhlo.dot_general"(%3727, %3697) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3729 = "mhlo.transpose"(%3728) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3730 = "mhlo.reshape"(%3729) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3731 = "mhlo.dot"(%3730, %1270) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3732 = "mhlo.broadcast_in_dim"(%1269) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3733 = mhlo.add %3731, %3732 : tensor<384x128xf32>
    %3734 = "mhlo.reshape"(%3733) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3735 = "mhlo.dot"(%3692, %1284) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3736 = "mhlo.broadcast_in_dim"(%1283) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3737 = mhlo.add %3735, %3736 : tensor<384x128xf32>
    %3738 = "mhlo.reshape"(%3737) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3739 = "mhlo.broadcast_in_dim"(%1282) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3740 = mhlo.multiply %3738, %3739 : tensor<1x384x128xf32>
    %3741 = "mhlo.broadcast_in_dim"(%1281) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3742 = mhlo.add %3740, %3741 : tensor<1x384x128xf32>
    %3743 = mhlo.add %3734, %3742 : tensor<1x384x128xf32>
    %3744 = "mhlo.broadcast_in_dim"(%1268) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3745 = mhlo.multiply %3743, %3744 : tensor<1x384x128xf32>
    %3746 = "mhlo.broadcast_in_dim"(%1267) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3747 = mhlo.add %3745, %3746 : tensor<1x384x128xf32>
    %3748 = "mhlo.reshape"(%3747) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3749 = "mhlo.dot"(%3748, %1286) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3750 = "mhlo.broadcast_in_dim"(%1285) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3751 = mhlo.add %3749, %3750 : tensor<384x512xf32>
    %3752 = "mhlo.reshape"(%3751) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3753 = mhlo.maximum %3752, %1119 : tensor<1x384x512xf32>
    %3754 = "mhlo.reshape"(%3753) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3755 = "mhlo.dot"(%3754, %1290) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3756 = "mhlo.broadcast_in_dim"(%1289) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3757 = mhlo.add %3755, %3756 : tensor<384x128xf32>
    %3758 = "mhlo.reshape"(%3757) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3759 = mhlo.add %3758, %3747 : tensor<1x384x128xf32>
    %3760 = "mhlo.broadcast_in_dim"(%1288) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3761 = mhlo.multiply %3759, %3760 : tensor<1x384x128xf32>
    %3762 = "mhlo.broadcast_in_dim"(%1287) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3763 = mhlo.add %3761, %3762 : tensor<1x384x128xf32>
    %3764 = "mhlo.reshape"(%3763) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3765 = "mhlo.dot"(%3764, %1292) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3766 = "mhlo.broadcast_in_dim"(%1291) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3767 = mhlo.add %3765, %3766 : tensor<384x512xf32>
    %3768 = "mhlo.reshape"(%3767) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3769 = mhlo.maximum %3768, %1119 : tensor<1x384x512xf32>
    %3770 = "mhlo.reshape"(%3769) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3771 = "mhlo.dot"(%3770, %1296) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3772 = "mhlo.broadcast_in_dim"(%1295) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3773 = mhlo.add %3771, %3772 : tensor<384x128xf32>
    %3774 = "mhlo.reshape"(%3773) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3775 = mhlo.add %3774, %3763 : tensor<1x384x128xf32>
    %3776 = "mhlo.broadcast_in_dim"(%1294) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3777 = mhlo.multiply %3775, %3776 : tensor<1x384x128xf32>
    %3778 = "mhlo.broadcast_in_dim"(%1293) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3779 = mhlo.add %3777, %3778 : tensor<1x384x128xf32>
    %3780 = "mhlo.reshape"(%3779) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3781 = "mhlo.dot"(%3780, %1298) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3782 = "mhlo.broadcast_in_dim"(%1297) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3783 = mhlo.add %3781, %3782 : tensor<384x512xf32>
    %3784 = "mhlo.reshape"(%3783) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3785 = mhlo.maximum %3784, %1119 : tensor<1x384x512xf32>
    %3786 = "mhlo.reshape"(%3785) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3787 = "mhlo.dot"(%3786, %1302) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3788 = "mhlo.broadcast_in_dim"(%1301) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3789 = mhlo.add %3787, %3788 : tensor<384x128xf32>
    %3790 = "mhlo.reshape"(%3789) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3791 = mhlo.add %3790, %3779 : tensor<1x384x128xf32>
    %3792 = "mhlo.broadcast_in_dim"(%1300) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3793 = mhlo.multiply %3791, %3792 : tensor<1x384x128xf32>
    %3794 = "mhlo.broadcast_in_dim"(%1299) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3795 = mhlo.add %3793, %3794 : tensor<1x384x128xf32>
    %3796 = "mhlo.reshape"(%3795) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3797 = "mhlo.dot"(%3796, %1304) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3798 = "mhlo.broadcast_in_dim"(%1303) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3799 = mhlo.add %3797, %3798 : tensor<384x512xf32>
    %3800 = "mhlo.reshape"(%3799) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3801 = mhlo.maximum %3800, %1119 : tensor<1x384x512xf32>
    %3802 = "mhlo.reshape"(%3801) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3803 = "mhlo.dot"(%3802, %1312) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3804 = "mhlo.broadcast_in_dim"(%1311) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3805 = mhlo.add %3803, %3804 : tensor<384x128xf32>
    %3806 = "mhlo.reshape"(%3805) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3807 = mhlo.add %3806, %3795 : tensor<1x384x128xf32>
    %3808 = "mhlo.broadcast_in_dim"(%1306) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3809 = mhlo.multiply %3807, %3808 : tensor<1x384x128xf32>
    %3810 = "mhlo.broadcast_in_dim"(%1305) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3811 = mhlo.add %3809, %3810 : tensor<1x384x128xf32>
    %3812 = "mhlo.reshape"(%3811) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3813 = "mhlo.dot"(%3812, %1310) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3814 = "mhlo.broadcast_in_dim"(%1309) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3815 = mhlo.add %3813, %3814 : tensor<384x512xf32>
    %3816 = "mhlo.reshape"(%3815) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3817 = mhlo.add %3816, %3691 : tensor<1x384x512xf32>
    %3818 = "mhlo.broadcast_in_dim"(%1308) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3819 = mhlo.multiply %3817, %3818 : tensor<1x384x512xf32>
    %3820 = "mhlo.broadcast_in_dim"(%1307) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3821 = mhlo.add %3819, %3820 : tensor<1x384x512xf32>
    %3822 = "mhlo.reshape"(%3821) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3823 = "mhlo.dot"(%3822, %1322) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3824 = "mhlo.broadcast_in_dim"(%1321) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3825 = mhlo.add %3823, %3824 : tensor<384x128xf32>
    %3826 = "mhlo.reshape"(%3825) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3827 = "mhlo.transpose"(%3826) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3828 = "mhlo.dot"(%3822, %1326) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3829 = "mhlo.reshape"(%3828) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3830 = "mhlo.broadcast_in_dim"(%1325) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3831 = mhlo.add %3829, %3830 : tensor<1x384x128xf32>
    %3832 = "mhlo.broadcast_in_dim"(%1324) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3833 = mhlo.multiply %3831, %3832 : tensor<1x384x128xf32>
    %3834 = "mhlo.broadcast_in_dim"(%1323) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3835 = mhlo.add %3833, %3834 : tensor<1x384x128xf32>
    %3836 = "mhlo.reshape"(%3835) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3837 = "mhlo.dot"(%3836, %1318) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3838 = "mhlo.broadcast_in_dim"(%1317) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3839 = mhlo.add %3837, %3838 : tensor<384x128xf32>
    %3840 = "mhlo.reshape"(%3839) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3841 = "mhlo.transpose"(%3840) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3842 = "mhlo.dot"(%3836, %1320) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3843 = "mhlo.broadcast_in_dim"(%1319) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3844 = mhlo.add %3842, %3843 : tensor<384x128xf32>
    %3845 = "mhlo.reshape"(%3844) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3846 = "mhlo.transpose"(%3845) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3847 = "mhlo.dot_general"(%3846, %3841) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3848 = mhlo.multiply %3847, %1114 : tensor<1x4x384x384xf32>
    %3849 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3850 = mhlo.add %3848, %3849 : tensor<1x4x384x384xf32>
    %3851 = "mhlo.reduce"(%3850, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3852 = "mhlo.broadcast_in_dim"(%3851) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3853 = mhlo.subtract %3850, %3852 : tensor<1x4x384x384xf32>
    %3854 = "mhlo.exponential"(%3853) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3855 = "mhlo.reduce"(%3854, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3856 = "mhlo.broadcast_in_dim"(%3855) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3857 = mhlo.divide %3854, %3856 : tensor<1x4x384x384xf32>
    %3858 = "mhlo.dot_general"(%3857, %3827) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3859 = "mhlo.transpose"(%3858) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3860 = "mhlo.reshape"(%3859) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3861 = "mhlo.dot"(%3860, %1316) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3862 = "mhlo.broadcast_in_dim"(%1315) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3863 = mhlo.add %3861, %3862 : tensor<384x128xf32>
    %3864 = "mhlo.reshape"(%3863) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3865 = "mhlo.dot"(%3822, %1330) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3866 = "mhlo.broadcast_in_dim"(%1329) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3867 = mhlo.add %3865, %3866 : tensor<384x128xf32>
    %3868 = "mhlo.reshape"(%3867) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3869 = "mhlo.broadcast_in_dim"(%1328) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3870 = mhlo.multiply %3868, %3869 : tensor<1x384x128xf32>
    %3871 = "mhlo.broadcast_in_dim"(%1327) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3872 = mhlo.add %3870, %3871 : tensor<1x384x128xf32>
    %3873 = mhlo.add %3864, %3872 : tensor<1x384x128xf32>
    %3874 = "mhlo.broadcast_in_dim"(%1314) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3875 = mhlo.multiply %3873, %3874 : tensor<1x384x128xf32>
    %3876 = "mhlo.broadcast_in_dim"(%1313) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3877 = mhlo.add %3875, %3876 : tensor<1x384x128xf32>
    %3878 = "mhlo.reshape"(%3877) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3879 = "mhlo.dot"(%3878, %1332) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3880 = "mhlo.broadcast_in_dim"(%1331) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3881 = mhlo.add %3879, %3880 : tensor<384x512xf32>
    %3882 = "mhlo.reshape"(%3881) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3883 = mhlo.maximum %3882, %1119 : tensor<1x384x512xf32>
    %3884 = "mhlo.reshape"(%3883) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3885 = "mhlo.dot"(%3884, %1336) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3886 = "mhlo.broadcast_in_dim"(%1335) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3887 = mhlo.add %3885, %3886 : tensor<384x128xf32>
    %3888 = "mhlo.reshape"(%3887) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3889 = mhlo.add %3888, %3877 : tensor<1x384x128xf32>
    %3890 = "mhlo.broadcast_in_dim"(%1334) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3891 = mhlo.multiply %3889, %3890 : tensor<1x384x128xf32>
    %3892 = "mhlo.broadcast_in_dim"(%1333) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3893 = mhlo.add %3891, %3892 : tensor<1x384x128xf32>
    %3894 = "mhlo.reshape"(%3893) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3895 = "mhlo.dot"(%3894, %1338) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3896 = "mhlo.broadcast_in_dim"(%1337) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3897 = mhlo.add %3895, %3896 : tensor<384x512xf32>
    %3898 = "mhlo.reshape"(%3897) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3899 = mhlo.maximum %3898, %1119 : tensor<1x384x512xf32>
    %3900 = "mhlo.reshape"(%3899) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3901 = "mhlo.dot"(%3900, %1342) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3902 = "mhlo.broadcast_in_dim"(%1341) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3903 = mhlo.add %3901, %3902 : tensor<384x128xf32>
    %3904 = "mhlo.reshape"(%3903) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3905 = mhlo.add %3904, %3893 : tensor<1x384x128xf32>
    %3906 = "mhlo.broadcast_in_dim"(%1340) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3907 = mhlo.multiply %3905, %3906 : tensor<1x384x128xf32>
    %3908 = "mhlo.broadcast_in_dim"(%1339) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3909 = mhlo.add %3907, %3908 : tensor<1x384x128xf32>
    %3910 = "mhlo.reshape"(%3909) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3911 = "mhlo.dot"(%3910, %1344) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3912 = "mhlo.broadcast_in_dim"(%1343) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3913 = mhlo.add %3911, %3912 : tensor<384x512xf32>
    %3914 = "mhlo.reshape"(%3913) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3915 = mhlo.maximum %3914, %1119 : tensor<1x384x512xf32>
    %3916 = "mhlo.reshape"(%3915) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3917 = "mhlo.dot"(%3916, %1348) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3918 = "mhlo.broadcast_in_dim"(%1347) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3919 = mhlo.add %3917, %3918 : tensor<384x128xf32>
    %3920 = "mhlo.reshape"(%3919) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3921 = mhlo.add %3920, %3909 : tensor<1x384x128xf32>
    %3922 = "mhlo.broadcast_in_dim"(%1346) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3923 = mhlo.multiply %3921, %3922 : tensor<1x384x128xf32>
    %3924 = "mhlo.broadcast_in_dim"(%1345) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3925 = mhlo.add %3923, %3924 : tensor<1x384x128xf32>
    %3926 = "mhlo.reshape"(%3925) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3927 = "mhlo.dot"(%3926, %1350) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3928 = "mhlo.broadcast_in_dim"(%1349) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3929 = mhlo.add %3927, %3928 : tensor<384x512xf32>
    %3930 = "mhlo.reshape"(%3929) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3931 = mhlo.maximum %3930, %1119 : tensor<1x384x512xf32>
    %3932 = "mhlo.reshape"(%3931) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3933 = "mhlo.dot"(%3932, %1358) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3934 = "mhlo.broadcast_in_dim"(%1357) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3935 = mhlo.add %3933, %3934 : tensor<384x128xf32>
    %3936 = "mhlo.reshape"(%3935) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3937 = mhlo.add %3936, %3925 : tensor<1x384x128xf32>
    %3938 = "mhlo.broadcast_in_dim"(%1352) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3939 = mhlo.multiply %3937, %3938 : tensor<1x384x128xf32>
    %3940 = "mhlo.broadcast_in_dim"(%1351) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3941 = mhlo.add %3939, %3940 : tensor<1x384x128xf32>
    %3942 = "mhlo.reshape"(%3941) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3943 = "mhlo.dot"(%3942, %1356) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %3944 = "mhlo.broadcast_in_dim"(%1355) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %3945 = mhlo.add %3943, %3944 : tensor<384x512xf32>
    %3946 = "mhlo.reshape"(%3945) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %3947 = mhlo.add %3946, %3821 : tensor<1x384x512xf32>
    %3948 = "mhlo.broadcast_in_dim"(%1354) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3949 = mhlo.multiply %3947, %3948 : tensor<1x384x512xf32>
    %3950 = "mhlo.broadcast_in_dim"(%1353) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %3951 = mhlo.add %3949, %3950 : tensor<1x384x512xf32>
    %3952 = "mhlo.reshape"(%3951) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %3953 = "mhlo.dot"(%3952, %1368) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3954 = "mhlo.broadcast_in_dim"(%1367) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3955 = mhlo.add %3953, %3954 : tensor<384x128xf32>
    %3956 = "mhlo.reshape"(%3955) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3957 = "mhlo.transpose"(%3956) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3958 = "mhlo.dot"(%3952, %1372) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3959 = "mhlo.reshape"(%3958) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3960 = "mhlo.broadcast_in_dim"(%1371) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3961 = mhlo.add %3959, %3960 : tensor<1x384x128xf32>
    %3962 = "mhlo.broadcast_in_dim"(%1370) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3963 = mhlo.multiply %3961, %3962 : tensor<1x384x128xf32>
    %3964 = "mhlo.broadcast_in_dim"(%1369) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %3965 = mhlo.add %3963, %3964 : tensor<1x384x128xf32>
    %3966 = "mhlo.reshape"(%3965) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %3967 = "mhlo.dot"(%3966, %1364) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3968 = "mhlo.broadcast_in_dim"(%1363) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3969 = mhlo.add %3967, %3968 : tensor<384x128xf32>
    %3970 = "mhlo.reshape"(%3969) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3971 = "mhlo.transpose"(%3970) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3972 = "mhlo.dot"(%3966, %1366) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3973 = "mhlo.broadcast_in_dim"(%1365) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3974 = mhlo.add %3972, %3973 : tensor<384x128xf32>
    %3975 = "mhlo.reshape"(%3974) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %3976 = "mhlo.transpose"(%3975) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %3977 = "mhlo.dot_general"(%3976, %3971) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %3978 = mhlo.multiply %3977, %1114 : tensor<1x4x384x384xf32>
    %3979 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3980 = mhlo.add %3978, %3979 : tensor<1x4x384x384xf32>
    %3981 = "mhlo.reduce"(%3980, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3982 = "mhlo.broadcast_in_dim"(%3981) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3983 = mhlo.subtract %3980, %3982 : tensor<1x4x384x384xf32>
    %3984 = "mhlo.exponential"(%3983) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %3985 = "mhlo.reduce"(%3984, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %3986 = "mhlo.broadcast_in_dim"(%3985) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %3987 = mhlo.divide %3984, %3986 : tensor<1x4x384x384xf32>
    %3988 = "mhlo.dot_general"(%3987, %3957) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %3989 = "mhlo.transpose"(%3988) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %3990 = "mhlo.reshape"(%3989) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %3991 = "mhlo.dot"(%3990, %1362) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %3992 = "mhlo.broadcast_in_dim"(%1361) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3993 = mhlo.add %3991, %3992 : tensor<384x128xf32>
    %3994 = "mhlo.reshape"(%3993) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3995 = "mhlo.dot"(%3952, %1376) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %3996 = "mhlo.broadcast_in_dim"(%1375) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %3997 = mhlo.add %3995, %3996 : tensor<384x128xf32>
    %3998 = "mhlo.reshape"(%3997) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %3999 = "mhlo.broadcast_in_dim"(%1374) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4000 = mhlo.multiply %3998, %3999 : tensor<1x384x128xf32>
    %4001 = "mhlo.broadcast_in_dim"(%1373) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4002 = mhlo.add %4000, %4001 : tensor<1x384x128xf32>
    %4003 = mhlo.add %3994, %4002 : tensor<1x384x128xf32>
    %4004 = "mhlo.broadcast_in_dim"(%1360) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4005 = mhlo.multiply %4003, %4004 : tensor<1x384x128xf32>
    %4006 = "mhlo.broadcast_in_dim"(%1359) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4007 = mhlo.add %4005, %4006 : tensor<1x384x128xf32>
    %4008 = "mhlo.reshape"(%4007) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4009 = "mhlo.dot"(%4008, %1378) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4010 = "mhlo.broadcast_in_dim"(%1377) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4011 = mhlo.add %4009, %4010 : tensor<384x512xf32>
    %4012 = "mhlo.reshape"(%4011) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4013 = mhlo.maximum %4012, %1119 : tensor<1x384x512xf32>
    %4014 = "mhlo.reshape"(%4013) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4015 = "mhlo.dot"(%4014, %1382) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4016 = "mhlo.broadcast_in_dim"(%1381) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4017 = mhlo.add %4015, %4016 : tensor<384x128xf32>
    %4018 = "mhlo.reshape"(%4017) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4019 = mhlo.add %4018, %4007 : tensor<1x384x128xf32>
    %4020 = "mhlo.broadcast_in_dim"(%1380) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4021 = mhlo.multiply %4019, %4020 : tensor<1x384x128xf32>
    %4022 = "mhlo.broadcast_in_dim"(%1379) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4023 = mhlo.add %4021, %4022 : tensor<1x384x128xf32>
    %4024 = "mhlo.reshape"(%4023) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4025 = "mhlo.dot"(%4024, %1384) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4026 = "mhlo.broadcast_in_dim"(%1383) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4027 = mhlo.add %4025, %4026 : tensor<384x512xf32>
    %4028 = "mhlo.reshape"(%4027) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4029 = mhlo.maximum %4028, %1119 : tensor<1x384x512xf32>
    %4030 = "mhlo.reshape"(%4029) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4031 = "mhlo.dot"(%4030, %1388) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4032 = "mhlo.broadcast_in_dim"(%1387) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4033 = mhlo.add %4031, %4032 : tensor<384x128xf32>
    %4034 = "mhlo.reshape"(%4033) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4035 = mhlo.add %4034, %4023 : tensor<1x384x128xf32>
    %4036 = "mhlo.broadcast_in_dim"(%1386) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4037 = mhlo.multiply %4035, %4036 : tensor<1x384x128xf32>
    %4038 = "mhlo.broadcast_in_dim"(%1385) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4039 = mhlo.add %4037, %4038 : tensor<1x384x128xf32>
    %4040 = "mhlo.reshape"(%4039) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4041 = "mhlo.dot"(%4040, %1390) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4042 = "mhlo.broadcast_in_dim"(%1389) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4043 = mhlo.add %4041, %4042 : tensor<384x512xf32>
    %4044 = "mhlo.reshape"(%4043) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4045 = mhlo.maximum %4044, %1119 : tensor<1x384x512xf32>
    %4046 = "mhlo.reshape"(%4045) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4047 = "mhlo.dot"(%4046, %1394) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4048 = "mhlo.broadcast_in_dim"(%1393) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4049 = mhlo.add %4047, %4048 : tensor<384x128xf32>
    %4050 = "mhlo.reshape"(%4049) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4051 = mhlo.add %4050, %4039 : tensor<1x384x128xf32>
    %4052 = "mhlo.broadcast_in_dim"(%1392) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4053 = mhlo.multiply %4051, %4052 : tensor<1x384x128xf32>
    %4054 = "mhlo.broadcast_in_dim"(%1391) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4055 = mhlo.add %4053, %4054 : tensor<1x384x128xf32>
    %4056 = "mhlo.reshape"(%4055) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4057 = "mhlo.dot"(%4056, %1396) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4058 = "mhlo.broadcast_in_dim"(%1395) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4059 = mhlo.add %4057, %4058 : tensor<384x512xf32>
    %4060 = "mhlo.reshape"(%4059) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4061 = mhlo.maximum %4060, %1119 : tensor<1x384x512xf32>
    %4062 = "mhlo.reshape"(%4061) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4063 = "mhlo.dot"(%4062, %1404) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4064 = "mhlo.broadcast_in_dim"(%1403) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4065 = mhlo.add %4063, %4064 : tensor<384x128xf32>
    %4066 = "mhlo.reshape"(%4065) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4067 = mhlo.add %4066, %4055 : tensor<1x384x128xf32>
    %4068 = "mhlo.broadcast_in_dim"(%1398) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4069 = mhlo.multiply %4067, %4068 : tensor<1x384x128xf32>
    %4070 = "mhlo.broadcast_in_dim"(%1397) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4071 = mhlo.add %4069, %4070 : tensor<1x384x128xf32>
    %4072 = "mhlo.reshape"(%4071) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4073 = "mhlo.dot"(%4072, %1402) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4074 = "mhlo.broadcast_in_dim"(%1401) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4075 = mhlo.add %4073, %4074 : tensor<384x512xf32>
    %4076 = "mhlo.reshape"(%4075) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4077 = mhlo.add %4076, %3951 : tensor<1x384x512xf32>
    %4078 = "mhlo.broadcast_in_dim"(%1400) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4079 = mhlo.multiply %4077, %4078 : tensor<1x384x512xf32>
    %4080 = "mhlo.broadcast_in_dim"(%1399) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4081 = mhlo.add %4079, %4080 : tensor<1x384x512xf32>
    %4082 = "mhlo.reshape"(%4081) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4083 = "mhlo.dot"(%4082, %1414) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4084 = "mhlo.broadcast_in_dim"(%1413) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4085 = mhlo.add %4083, %4084 : tensor<384x128xf32>
    %4086 = "mhlo.reshape"(%4085) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4087 = "mhlo.transpose"(%4086) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4088 = "mhlo.dot"(%4082, %1418) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4089 = "mhlo.reshape"(%4088) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4090 = "mhlo.broadcast_in_dim"(%1417) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4091 = mhlo.add %4089, %4090 : tensor<1x384x128xf32>
    %4092 = "mhlo.broadcast_in_dim"(%1416) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4093 = mhlo.multiply %4091, %4092 : tensor<1x384x128xf32>
    %4094 = "mhlo.broadcast_in_dim"(%1415) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4095 = mhlo.add %4093, %4094 : tensor<1x384x128xf32>
    %4096 = "mhlo.reshape"(%4095) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4097 = "mhlo.dot"(%4096, %1410) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4098 = "mhlo.broadcast_in_dim"(%1409) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4099 = mhlo.add %4097, %4098 : tensor<384x128xf32>
    %4100 = "mhlo.reshape"(%4099) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4101 = "mhlo.transpose"(%4100) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4102 = "mhlo.dot"(%4096, %1412) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4103 = "mhlo.broadcast_in_dim"(%1411) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4104 = mhlo.add %4102, %4103 : tensor<384x128xf32>
    %4105 = "mhlo.reshape"(%4104) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4106 = "mhlo.transpose"(%4105) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4107 = "mhlo.dot_general"(%4106, %4101) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4108 = mhlo.multiply %4107, %1114 : tensor<1x4x384x384xf32>
    %4109 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4110 = mhlo.add %4108, %4109 : tensor<1x4x384x384xf32>
    %4111 = "mhlo.reduce"(%4110, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4112 = "mhlo.broadcast_in_dim"(%4111) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4113 = mhlo.subtract %4110, %4112 : tensor<1x4x384x384xf32>
    %4114 = "mhlo.exponential"(%4113) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4115 = "mhlo.reduce"(%4114, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4116 = "mhlo.broadcast_in_dim"(%4115) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4117 = mhlo.divide %4114, %4116 : tensor<1x4x384x384xf32>
    %4118 = "mhlo.dot_general"(%4117, %4087) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4119 = "mhlo.transpose"(%4118) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4120 = "mhlo.reshape"(%4119) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4121 = "mhlo.dot"(%4120, %1408) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4122 = "mhlo.broadcast_in_dim"(%1407) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4123 = mhlo.add %4121, %4122 : tensor<384x128xf32>
    %4124 = "mhlo.reshape"(%4123) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4125 = "mhlo.dot"(%4082, %1422) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4126 = "mhlo.broadcast_in_dim"(%1421) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4127 = mhlo.add %4125, %4126 : tensor<384x128xf32>
    %4128 = "mhlo.reshape"(%4127) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4129 = "mhlo.broadcast_in_dim"(%1420) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4130 = mhlo.multiply %4128, %4129 : tensor<1x384x128xf32>
    %4131 = "mhlo.broadcast_in_dim"(%1419) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4132 = mhlo.add %4130, %4131 : tensor<1x384x128xf32>
    %4133 = mhlo.add %4124, %4132 : tensor<1x384x128xf32>
    %4134 = "mhlo.broadcast_in_dim"(%1406) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4135 = mhlo.multiply %4133, %4134 : tensor<1x384x128xf32>
    %4136 = "mhlo.broadcast_in_dim"(%1405) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4137 = mhlo.add %4135, %4136 : tensor<1x384x128xf32>
    %4138 = "mhlo.reshape"(%4137) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4139 = "mhlo.dot"(%4138, %1424) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4140 = "mhlo.broadcast_in_dim"(%1423) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4141 = mhlo.add %4139, %4140 : tensor<384x512xf32>
    %4142 = "mhlo.reshape"(%4141) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4143 = mhlo.maximum %4142, %1119 : tensor<1x384x512xf32>
    %4144 = "mhlo.reshape"(%4143) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4145 = "mhlo.dot"(%4144, %1428) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4146 = "mhlo.broadcast_in_dim"(%1427) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4147 = mhlo.add %4145, %4146 : tensor<384x128xf32>
    %4148 = "mhlo.reshape"(%4147) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4149 = mhlo.add %4148, %4137 : tensor<1x384x128xf32>
    %4150 = "mhlo.broadcast_in_dim"(%1426) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4151 = mhlo.multiply %4149, %4150 : tensor<1x384x128xf32>
    %4152 = "mhlo.broadcast_in_dim"(%1425) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4153 = mhlo.add %4151, %4152 : tensor<1x384x128xf32>
    %4154 = "mhlo.reshape"(%4153) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4155 = "mhlo.dot"(%4154, %1430) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4156 = "mhlo.broadcast_in_dim"(%1429) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4157 = mhlo.add %4155, %4156 : tensor<384x512xf32>
    %4158 = "mhlo.reshape"(%4157) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4159 = mhlo.maximum %4158, %1119 : tensor<1x384x512xf32>
    %4160 = "mhlo.reshape"(%4159) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4161 = "mhlo.dot"(%4160, %1434) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4162 = "mhlo.broadcast_in_dim"(%1433) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4163 = mhlo.add %4161, %4162 : tensor<384x128xf32>
    %4164 = "mhlo.reshape"(%4163) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4165 = mhlo.add %4164, %4153 : tensor<1x384x128xf32>
    %4166 = "mhlo.broadcast_in_dim"(%1432) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4167 = mhlo.multiply %4165, %4166 : tensor<1x384x128xf32>
    %4168 = "mhlo.broadcast_in_dim"(%1431) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4169 = mhlo.add %4167, %4168 : tensor<1x384x128xf32>
    %4170 = "mhlo.reshape"(%4169) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4171 = "mhlo.dot"(%4170, %1436) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4172 = "mhlo.broadcast_in_dim"(%1435) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4173 = mhlo.add %4171, %4172 : tensor<384x512xf32>
    %4174 = "mhlo.reshape"(%4173) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4175 = mhlo.maximum %4174, %1119 : tensor<1x384x512xf32>
    %4176 = "mhlo.reshape"(%4175) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4177 = "mhlo.dot"(%4176, %1440) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4178 = "mhlo.broadcast_in_dim"(%1439) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4179 = mhlo.add %4177, %4178 : tensor<384x128xf32>
    %4180 = "mhlo.reshape"(%4179) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4181 = mhlo.add %4180, %4169 : tensor<1x384x128xf32>
    %4182 = "mhlo.broadcast_in_dim"(%1438) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4183 = mhlo.multiply %4181, %4182 : tensor<1x384x128xf32>
    %4184 = "mhlo.broadcast_in_dim"(%1437) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4185 = mhlo.add %4183, %4184 : tensor<1x384x128xf32>
    %4186 = "mhlo.reshape"(%4185) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4187 = "mhlo.dot"(%4186, %1442) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4188 = "mhlo.broadcast_in_dim"(%1441) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4189 = mhlo.add %4187, %4188 : tensor<384x512xf32>
    %4190 = "mhlo.reshape"(%4189) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4191 = mhlo.maximum %4190, %1119 : tensor<1x384x512xf32>
    %4192 = "mhlo.reshape"(%4191) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4193 = "mhlo.dot"(%4192, %1450) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4194 = "mhlo.broadcast_in_dim"(%1449) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4195 = mhlo.add %4193, %4194 : tensor<384x128xf32>
    %4196 = "mhlo.reshape"(%4195) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4197 = mhlo.add %4196, %4185 : tensor<1x384x128xf32>
    %4198 = "mhlo.broadcast_in_dim"(%1444) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4199 = mhlo.multiply %4197, %4198 : tensor<1x384x128xf32>
    %4200 = "mhlo.broadcast_in_dim"(%1443) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4201 = mhlo.add %4199, %4200 : tensor<1x384x128xf32>
    %4202 = "mhlo.reshape"(%4201) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4203 = "mhlo.dot"(%4202, %1448) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4204 = "mhlo.broadcast_in_dim"(%1447) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4205 = mhlo.add %4203, %4204 : tensor<384x512xf32>
    %4206 = "mhlo.reshape"(%4205) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4207 = mhlo.add %4206, %4081 : tensor<1x384x512xf32>
    %4208 = "mhlo.broadcast_in_dim"(%1446) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4209 = mhlo.multiply %4207, %4208 : tensor<1x384x512xf32>
    %4210 = "mhlo.broadcast_in_dim"(%1445) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4211 = mhlo.add %4209, %4210 : tensor<1x384x512xf32>
    %4212 = "mhlo.reshape"(%4211) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4213 = "mhlo.dot"(%4212, %1460) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4214 = "mhlo.broadcast_in_dim"(%1459) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4215 = mhlo.add %4213, %4214 : tensor<384x128xf32>
    %4216 = "mhlo.reshape"(%4215) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4217 = "mhlo.transpose"(%4216) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4218 = "mhlo.dot"(%4212, %1464) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4219 = "mhlo.reshape"(%4218) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4220 = "mhlo.broadcast_in_dim"(%1463) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4221 = mhlo.add %4219, %4220 : tensor<1x384x128xf32>
    %4222 = "mhlo.broadcast_in_dim"(%1462) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4223 = mhlo.multiply %4221, %4222 : tensor<1x384x128xf32>
    %4224 = "mhlo.broadcast_in_dim"(%1461) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4225 = mhlo.add %4223, %4224 : tensor<1x384x128xf32>
    %4226 = "mhlo.reshape"(%4225) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4227 = "mhlo.dot"(%4226, %1456) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4228 = "mhlo.broadcast_in_dim"(%1455) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4229 = mhlo.add %4227, %4228 : tensor<384x128xf32>
    %4230 = "mhlo.reshape"(%4229) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4231 = "mhlo.transpose"(%4230) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4232 = "mhlo.dot"(%4226, %1458) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4233 = "mhlo.broadcast_in_dim"(%1457) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4234 = mhlo.add %4232, %4233 : tensor<384x128xf32>
    %4235 = "mhlo.reshape"(%4234) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4236 = "mhlo.transpose"(%4235) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4237 = "mhlo.dot_general"(%4236, %4231) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4238 = mhlo.multiply %4237, %1114 : tensor<1x4x384x384xf32>
    %4239 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4240 = mhlo.add %4238, %4239 : tensor<1x4x384x384xf32>
    %4241 = "mhlo.reduce"(%4240, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4242 = "mhlo.broadcast_in_dim"(%4241) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4243 = mhlo.subtract %4240, %4242 : tensor<1x4x384x384xf32>
    %4244 = "mhlo.exponential"(%4243) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4245 = "mhlo.reduce"(%4244, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4246 = "mhlo.broadcast_in_dim"(%4245) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4247 = mhlo.divide %4244, %4246 : tensor<1x4x384x384xf32>
    %4248 = "mhlo.dot_general"(%4247, %4217) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4249 = "mhlo.transpose"(%4248) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4250 = "mhlo.reshape"(%4249) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4251 = "mhlo.dot"(%4250, %1454) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4252 = "mhlo.broadcast_in_dim"(%1453) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4253 = mhlo.add %4251, %4252 : tensor<384x128xf32>
    %4254 = "mhlo.reshape"(%4253) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4255 = "mhlo.dot"(%4212, %1468) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4256 = "mhlo.broadcast_in_dim"(%1467) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4257 = mhlo.add %4255, %4256 : tensor<384x128xf32>
    %4258 = "mhlo.reshape"(%4257) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4259 = "mhlo.broadcast_in_dim"(%1466) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4260 = mhlo.multiply %4258, %4259 : tensor<1x384x128xf32>
    %4261 = "mhlo.broadcast_in_dim"(%1465) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4262 = mhlo.add %4260, %4261 : tensor<1x384x128xf32>
    %4263 = mhlo.add %4254, %4262 : tensor<1x384x128xf32>
    %4264 = "mhlo.broadcast_in_dim"(%1452) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4265 = mhlo.multiply %4263, %4264 : tensor<1x384x128xf32>
    %4266 = "mhlo.broadcast_in_dim"(%1451) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4267 = mhlo.add %4265, %4266 : tensor<1x384x128xf32>
    %4268 = "mhlo.reshape"(%4267) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4269 = "mhlo.dot"(%4268, %1470) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4270 = "mhlo.broadcast_in_dim"(%1469) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4271 = mhlo.add %4269, %4270 : tensor<384x512xf32>
    %4272 = "mhlo.reshape"(%4271) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4273 = mhlo.maximum %4272, %1119 : tensor<1x384x512xf32>
    %4274 = "mhlo.reshape"(%4273) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4275 = "mhlo.dot"(%4274, %1474) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4276 = "mhlo.broadcast_in_dim"(%1473) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4277 = mhlo.add %4275, %4276 : tensor<384x128xf32>
    %4278 = "mhlo.reshape"(%4277) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4279 = mhlo.add %4278, %4267 : tensor<1x384x128xf32>
    %4280 = "mhlo.broadcast_in_dim"(%1472) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4281 = mhlo.multiply %4279, %4280 : tensor<1x384x128xf32>
    %4282 = "mhlo.broadcast_in_dim"(%1471) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4283 = mhlo.add %4281, %4282 : tensor<1x384x128xf32>
    %4284 = "mhlo.reshape"(%4283) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4285 = "mhlo.dot"(%4284, %1476) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4286 = "mhlo.broadcast_in_dim"(%1475) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4287 = mhlo.add %4285, %4286 : tensor<384x512xf32>
    %4288 = "mhlo.reshape"(%4287) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4289 = mhlo.maximum %4288, %1119 : tensor<1x384x512xf32>
    %4290 = "mhlo.reshape"(%4289) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4291 = "mhlo.dot"(%4290, %1480) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4292 = "mhlo.broadcast_in_dim"(%1479) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4293 = mhlo.add %4291, %4292 : tensor<384x128xf32>
    %4294 = "mhlo.reshape"(%4293) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4295 = mhlo.add %4294, %4283 : tensor<1x384x128xf32>
    %4296 = "mhlo.broadcast_in_dim"(%1478) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4297 = mhlo.multiply %4295, %4296 : tensor<1x384x128xf32>
    %4298 = "mhlo.broadcast_in_dim"(%1477) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4299 = mhlo.add %4297, %4298 : tensor<1x384x128xf32>
    %4300 = "mhlo.reshape"(%4299) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4301 = "mhlo.dot"(%4300, %1482) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4302 = "mhlo.broadcast_in_dim"(%1481) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4303 = mhlo.add %4301, %4302 : tensor<384x512xf32>
    %4304 = "mhlo.reshape"(%4303) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4305 = mhlo.maximum %4304, %1119 : tensor<1x384x512xf32>
    %4306 = "mhlo.reshape"(%4305) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4307 = "mhlo.dot"(%4306, %1486) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4308 = "mhlo.broadcast_in_dim"(%1485) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4309 = mhlo.add %4307, %4308 : tensor<384x128xf32>
    %4310 = "mhlo.reshape"(%4309) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4311 = mhlo.add %4310, %4299 : tensor<1x384x128xf32>
    %4312 = "mhlo.broadcast_in_dim"(%1484) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4313 = mhlo.multiply %4311, %4312 : tensor<1x384x128xf32>
    %4314 = "mhlo.broadcast_in_dim"(%1483) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4315 = mhlo.add %4313, %4314 : tensor<1x384x128xf32>
    %4316 = "mhlo.reshape"(%4315) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4317 = "mhlo.dot"(%4316, %1488) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4318 = "mhlo.broadcast_in_dim"(%1487) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4319 = mhlo.add %4317, %4318 : tensor<384x512xf32>
    %4320 = "mhlo.reshape"(%4319) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4321 = mhlo.maximum %4320, %1119 : tensor<1x384x512xf32>
    %4322 = "mhlo.reshape"(%4321) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4323 = "mhlo.dot"(%4322, %1496) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4324 = "mhlo.broadcast_in_dim"(%1495) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4325 = mhlo.add %4323, %4324 : tensor<384x128xf32>
    %4326 = "mhlo.reshape"(%4325) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4327 = mhlo.add %4326, %4315 : tensor<1x384x128xf32>
    %4328 = "mhlo.broadcast_in_dim"(%1490) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4329 = mhlo.multiply %4327, %4328 : tensor<1x384x128xf32>
    %4330 = "mhlo.broadcast_in_dim"(%1489) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4331 = mhlo.add %4329, %4330 : tensor<1x384x128xf32>
    %4332 = "mhlo.reshape"(%4331) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4333 = "mhlo.dot"(%4332, %1494) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4334 = "mhlo.broadcast_in_dim"(%1493) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4335 = mhlo.add %4333, %4334 : tensor<384x512xf32>
    %4336 = "mhlo.reshape"(%4335) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4337 = mhlo.add %4336, %4211 : tensor<1x384x512xf32>
    %4338 = "mhlo.broadcast_in_dim"(%1492) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4339 = mhlo.multiply %4337, %4338 : tensor<1x384x512xf32>
    %4340 = "mhlo.broadcast_in_dim"(%1491) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4341 = mhlo.add %4339, %4340 : tensor<1x384x512xf32>
    %4342 = "mhlo.reshape"(%4341) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4343 = "mhlo.dot"(%4342, %1506) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4344 = "mhlo.broadcast_in_dim"(%1505) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4345 = mhlo.add %4343, %4344 : tensor<384x128xf32>
    %4346 = "mhlo.reshape"(%4345) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4347 = "mhlo.transpose"(%4346) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4348 = "mhlo.dot"(%4342, %1510) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4349 = "mhlo.reshape"(%4348) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4350 = "mhlo.broadcast_in_dim"(%1509) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4351 = mhlo.add %4349, %4350 : tensor<1x384x128xf32>
    %4352 = "mhlo.broadcast_in_dim"(%1508) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4353 = mhlo.multiply %4351, %4352 : tensor<1x384x128xf32>
    %4354 = "mhlo.broadcast_in_dim"(%1507) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4355 = mhlo.add %4353, %4354 : tensor<1x384x128xf32>
    %4356 = "mhlo.reshape"(%4355) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4357 = "mhlo.dot"(%4356, %1502) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4358 = "mhlo.broadcast_in_dim"(%1501) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4359 = mhlo.add %4357, %4358 : tensor<384x128xf32>
    %4360 = "mhlo.reshape"(%4359) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4361 = "mhlo.transpose"(%4360) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4362 = "mhlo.dot"(%4356, %1504) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4363 = "mhlo.broadcast_in_dim"(%1503) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4364 = mhlo.add %4362, %4363 : tensor<384x128xf32>
    %4365 = "mhlo.reshape"(%4364) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4366 = "mhlo.transpose"(%4365) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4367 = "mhlo.dot_general"(%4366, %4361) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4368 = mhlo.multiply %4367, %1114 : tensor<1x4x384x384xf32>
    %4369 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4370 = mhlo.add %4368, %4369 : tensor<1x4x384x384xf32>
    %4371 = "mhlo.reduce"(%4370, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4372 = "mhlo.broadcast_in_dim"(%4371) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4373 = mhlo.subtract %4370, %4372 : tensor<1x4x384x384xf32>
    %4374 = "mhlo.exponential"(%4373) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4375 = "mhlo.reduce"(%4374, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4376 = "mhlo.broadcast_in_dim"(%4375) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4377 = mhlo.divide %4374, %4376 : tensor<1x4x384x384xf32>
    %4378 = "mhlo.dot_general"(%4377, %4347) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4379 = "mhlo.transpose"(%4378) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4380 = "mhlo.reshape"(%4379) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4381 = "mhlo.dot"(%4380, %1500) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4382 = "mhlo.broadcast_in_dim"(%1499) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4383 = mhlo.add %4381, %4382 : tensor<384x128xf32>
    %4384 = "mhlo.reshape"(%4383) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4385 = "mhlo.dot"(%4342, %1514) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4386 = "mhlo.broadcast_in_dim"(%1513) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4387 = mhlo.add %4385, %4386 : tensor<384x128xf32>
    %4388 = "mhlo.reshape"(%4387) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4389 = "mhlo.broadcast_in_dim"(%1512) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4390 = mhlo.multiply %4388, %4389 : tensor<1x384x128xf32>
    %4391 = "mhlo.broadcast_in_dim"(%1511) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4392 = mhlo.add %4390, %4391 : tensor<1x384x128xf32>
    %4393 = mhlo.add %4384, %4392 : tensor<1x384x128xf32>
    %4394 = "mhlo.broadcast_in_dim"(%1498) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4395 = mhlo.multiply %4393, %4394 : tensor<1x384x128xf32>
    %4396 = "mhlo.broadcast_in_dim"(%1497) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4397 = mhlo.add %4395, %4396 : tensor<1x384x128xf32>
    %4398 = "mhlo.reshape"(%4397) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4399 = "mhlo.dot"(%4398, %1516) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4400 = "mhlo.broadcast_in_dim"(%1515) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4401 = mhlo.add %4399, %4400 : tensor<384x512xf32>
    %4402 = "mhlo.reshape"(%4401) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4403 = mhlo.maximum %4402, %1119 : tensor<1x384x512xf32>
    %4404 = "mhlo.reshape"(%4403) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4405 = "mhlo.dot"(%4404, %1520) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4406 = "mhlo.broadcast_in_dim"(%1519) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4407 = mhlo.add %4405, %4406 : tensor<384x128xf32>
    %4408 = "mhlo.reshape"(%4407) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4409 = mhlo.add %4408, %4397 : tensor<1x384x128xf32>
    %4410 = "mhlo.broadcast_in_dim"(%1518) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4411 = mhlo.multiply %4409, %4410 : tensor<1x384x128xf32>
    %4412 = "mhlo.broadcast_in_dim"(%1517) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4413 = mhlo.add %4411, %4412 : tensor<1x384x128xf32>
    %4414 = "mhlo.reshape"(%4413) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4415 = "mhlo.dot"(%4414, %1522) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4416 = "mhlo.broadcast_in_dim"(%1521) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4417 = mhlo.add %4415, %4416 : tensor<384x512xf32>
    %4418 = "mhlo.reshape"(%4417) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4419 = mhlo.maximum %4418, %1119 : tensor<1x384x512xf32>
    %4420 = "mhlo.reshape"(%4419) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4421 = "mhlo.dot"(%4420, %1526) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4422 = "mhlo.broadcast_in_dim"(%1525) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4423 = mhlo.add %4421, %4422 : tensor<384x128xf32>
    %4424 = "mhlo.reshape"(%4423) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4425 = mhlo.add %4424, %4413 : tensor<1x384x128xf32>
    %4426 = "mhlo.broadcast_in_dim"(%1524) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4427 = mhlo.multiply %4425, %4426 : tensor<1x384x128xf32>
    %4428 = "mhlo.broadcast_in_dim"(%1523) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4429 = mhlo.add %4427, %4428 : tensor<1x384x128xf32>
    %4430 = "mhlo.reshape"(%4429) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4431 = "mhlo.dot"(%4430, %1528) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4432 = "mhlo.broadcast_in_dim"(%1527) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4433 = mhlo.add %4431, %4432 : tensor<384x512xf32>
    %4434 = "mhlo.reshape"(%4433) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4435 = mhlo.maximum %4434, %1119 : tensor<1x384x512xf32>
    %4436 = "mhlo.reshape"(%4435) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4437 = "mhlo.dot"(%4436, %1532) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4438 = "mhlo.broadcast_in_dim"(%1531) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4439 = mhlo.add %4437, %4438 : tensor<384x128xf32>
    %4440 = "mhlo.reshape"(%4439) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4441 = mhlo.add %4440, %4429 : tensor<1x384x128xf32>
    %4442 = "mhlo.broadcast_in_dim"(%1530) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4443 = mhlo.multiply %4441, %4442 : tensor<1x384x128xf32>
    %4444 = "mhlo.broadcast_in_dim"(%1529) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4445 = mhlo.add %4443, %4444 : tensor<1x384x128xf32>
    %4446 = "mhlo.reshape"(%4445) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4447 = "mhlo.dot"(%4446, %1534) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4448 = "mhlo.broadcast_in_dim"(%1533) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4449 = mhlo.add %4447, %4448 : tensor<384x512xf32>
    %4450 = "mhlo.reshape"(%4449) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4451 = mhlo.maximum %4450, %1119 : tensor<1x384x512xf32>
    %4452 = "mhlo.reshape"(%4451) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4453 = "mhlo.dot"(%4452, %1542) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4454 = "mhlo.broadcast_in_dim"(%1541) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4455 = mhlo.add %4453, %4454 : tensor<384x128xf32>
    %4456 = "mhlo.reshape"(%4455) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4457 = mhlo.add %4456, %4445 : tensor<1x384x128xf32>
    %4458 = "mhlo.broadcast_in_dim"(%1536) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4459 = mhlo.multiply %4457, %4458 : tensor<1x384x128xf32>
    %4460 = "mhlo.broadcast_in_dim"(%1535) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4461 = mhlo.add %4459, %4460 : tensor<1x384x128xf32>
    %4462 = "mhlo.reshape"(%4461) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4463 = "mhlo.dot"(%4462, %1540) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4464 = "mhlo.broadcast_in_dim"(%1539) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4465 = mhlo.add %4463, %4464 : tensor<384x512xf32>
    %4466 = "mhlo.reshape"(%4465) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4467 = mhlo.add %4466, %4341 : tensor<1x384x512xf32>
    %4468 = "mhlo.broadcast_in_dim"(%1538) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4469 = mhlo.multiply %4467, %4468 : tensor<1x384x512xf32>
    %4470 = "mhlo.broadcast_in_dim"(%1537) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4471 = mhlo.add %4469, %4470 : tensor<1x384x512xf32>
    %4472 = "mhlo.reshape"(%4471) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4473 = "mhlo.dot"(%4472, %1552) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4474 = "mhlo.broadcast_in_dim"(%1551) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4475 = mhlo.add %4473, %4474 : tensor<384x128xf32>
    %4476 = "mhlo.reshape"(%4475) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4477 = "mhlo.transpose"(%4476) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4478 = "mhlo.dot"(%4472, %1556) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4479 = "mhlo.reshape"(%4478) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4480 = "mhlo.broadcast_in_dim"(%1555) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4481 = mhlo.add %4479, %4480 : tensor<1x384x128xf32>
    %4482 = "mhlo.broadcast_in_dim"(%1554) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4483 = mhlo.multiply %4481, %4482 : tensor<1x384x128xf32>
    %4484 = "mhlo.broadcast_in_dim"(%1553) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4485 = mhlo.add %4483, %4484 : tensor<1x384x128xf32>
    %4486 = "mhlo.reshape"(%4485) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4487 = "mhlo.dot"(%4486, %1548) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4488 = "mhlo.broadcast_in_dim"(%1547) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4489 = mhlo.add %4487, %4488 : tensor<384x128xf32>
    %4490 = "mhlo.reshape"(%4489) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4491 = "mhlo.transpose"(%4490) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4492 = "mhlo.dot"(%4486, %1550) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4493 = "mhlo.broadcast_in_dim"(%1549) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4494 = mhlo.add %4492, %4493 : tensor<384x128xf32>
    %4495 = "mhlo.reshape"(%4494) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4496 = "mhlo.transpose"(%4495) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4497 = "mhlo.dot_general"(%4496, %4491) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4498 = mhlo.multiply %4497, %1114 : tensor<1x4x384x384xf32>
    %4499 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4500 = mhlo.add %4498, %4499 : tensor<1x4x384x384xf32>
    %4501 = "mhlo.reduce"(%4500, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4502 = "mhlo.broadcast_in_dim"(%4501) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4503 = mhlo.subtract %4500, %4502 : tensor<1x4x384x384xf32>
    %4504 = "mhlo.exponential"(%4503) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4505 = "mhlo.reduce"(%4504, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4506 = "mhlo.broadcast_in_dim"(%4505) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4507 = mhlo.divide %4504, %4506 : tensor<1x4x384x384xf32>
    %4508 = "mhlo.dot_general"(%4507, %4477) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4509 = "mhlo.transpose"(%4508) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4510 = "mhlo.reshape"(%4509) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4511 = "mhlo.dot"(%4510, %1546) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4512 = "mhlo.broadcast_in_dim"(%1545) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4513 = mhlo.add %4511, %4512 : tensor<384x128xf32>
    %4514 = "mhlo.reshape"(%4513) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4515 = "mhlo.dot"(%4472, %1560) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4516 = "mhlo.broadcast_in_dim"(%1559) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4517 = mhlo.add %4515, %4516 : tensor<384x128xf32>
    %4518 = "mhlo.reshape"(%4517) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4519 = "mhlo.broadcast_in_dim"(%1558) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4520 = mhlo.multiply %4518, %4519 : tensor<1x384x128xf32>
    %4521 = "mhlo.broadcast_in_dim"(%1557) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4522 = mhlo.add %4520, %4521 : tensor<1x384x128xf32>
    %4523 = mhlo.add %4514, %4522 : tensor<1x384x128xf32>
    %4524 = "mhlo.broadcast_in_dim"(%1544) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4525 = mhlo.multiply %4523, %4524 : tensor<1x384x128xf32>
    %4526 = "mhlo.broadcast_in_dim"(%1543) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4527 = mhlo.add %4525, %4526 : tensor<1x384x128xf32>
    %4528 = "mhlo.reshape"(%4527) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4529 = "mhlo.dot"(%4528, %1562) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4530 = "mhlo.broadcast_in_dim"(%1561) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4531 = mhlo.add %4529, %4530 : tensor<384x512xf32>
    %4532 = "mhlo.reshape"(%4531) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4533 = mhlo.maximum %4532, %1119 : tensor<1x384x512xf32>
    %4534 = "mhlo.reshape"(%4533) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4535 = "mhlo.dot"(%4534, %1566) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4536 = "mhlo.broadcast_in_dim"(%1565) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4537 = mhlo.add %4535, %4536 : tensor<384x128xf32>
    %4538 = "mhlo.reshape"(%4537) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4539 = mhlo.add %4538, %4527 : tensor<1x384x128xf32>
    %4540 = "mhlo.broadcast_in_dim"(%1564) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4541 = mhlo.multiply %4539, %4540 : tensor<1x384x128xf32>
    %4542 = "mhlo.broadcast_in_dim"(%1563) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4543 = mhlo.add %4541, %4542 : tensor<1x384x128xf32>
    %4544 = "mhlo.reshape"(%4543) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4545 = "mhlo.dot"(%4544, %1568) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4546 = "mhlo.broadcast_in_dim"(%1567) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4547 = mhlo.add %4545, %4546 : tensor<384x512xf32>
    %4548 = "mhlo.reshape"(%4547) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4549 = mhlo.maximum %4548, %1119 : tensor<1x384x512xf32>
    %4550 = "mhlo.reshape"(%4549) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4551 = "mhlo.dot"(%4550, %1572) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4552 = "mhlo.broadcast_in_dim"(%1571) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4553 = mhlo.add %4551, %4552 : tensor<384x128xf32>
    %4554 = "mhlo.reshape"(%4553) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4555 = mhlo.add %4554, %4543 : tensor<1x384x128xf32>
    %4556 = "mhlo.broadcast_in_dim"(%1570) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4557 = mhlo.multiply %4555, %4556 : tensor<1x384x128xf32>
    %4558 = "mhlo.broadcast_in_dim"(%1569) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4559 = mhlo.add %4557, %4558 : tensor<1x384x128xf32>
    %4560 = "mhlo.reshape"(%4559) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4561 = "mhlo.dot"(%4560, %1574) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4562 = "mhlo.broadcast_in_dim"(%1573) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4563 = mhlo.add %4561, %4562 : tensor<384x512xf32>
    %4564 = "mhlo.reshape"(%4563) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4565 = mhlo.maximum %4564, %1119 : tensor<1x384x512xf32>
    %4566 = "mhlo.reshape"(%4565) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4567 = "mhlo.dot"(%4566, %1578) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4568 = "mhlo.broadcast_in_dim"(%1577) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4569 = mhlo.add %4567, %4568 : tensor<384x128xf32>
    %4570 = "mhlo.reshape"(%4569) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4571 = mhlo.add %4570, %4559 : tensor<1x384x128xf32>
    %4572 = "mhlo.broadcast_in_dim"(%1576) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4573 = mhlo.multiply %4571, %4572 : tensor<1x384x128xf32>
    %4574 = "mhlo.broadcast_in_dim"(%1575) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4575 = mhlo.add %4573, %4574 : tensor<1x384x128xf32>
    %4576 = "mhlo.reshape"(%4575) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4577 = "mhlo.dot"(%4576, %1580) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4578 = "mhlo.broadcast_in_dim"(%1579) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4579 = mhlo.add %4577, %4578 : tensor<384x512xf32>
    %4580 = "mhlo.reshape"(%4579) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4581 = mhlo.maximum %4580, %1119 : tensor<1x384x512xf32>
    %4582 = "mhlo.reshape"(%4581) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4583 = "mhlo.dot"(%4582, %1588) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4584 = "mhlo.broadcast_in_dim"(%1587) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4585 = mhlo.add %4583, %4584 : tensor<384x128xf32>
    %4586 = "mhlo.reshape"(%4585) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4587 = mhlo.add %4586, %4575 : tensor<1x384x128xf32>
    %4588 = "mhlo.broadcast_in_dim"(%1582) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4589 = mhlo.multiply %4587, %4588 : tensor<1x384x128xf32>
    %4590 = "mhlo.broadcast_in_dim"(%1581) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4591 = mhlo.add %4589, %4590 : tensor<1x384x128xf32>
    %4592 = "mhlo.reshape"(%4591) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4593 = "mhlo.dot"(%4592, %1586) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4594 = "mhlo.broadcast_in_dim"(%1585) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4595 = mhlo.add %4593, %4594 : tensor<384x512xf32>
    %4596 = "mhlo.reshape"(%4595) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4597 = mhlo.add %4596, %4471 : tensor<1x384x512xf32>
    %4598 = "mhlo.broadcast_in_dim"(%1584) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4599 = mhlo.multiply %4597, %4598 : tensor<1x384x512xf32>
    %4600 = "mhlo.broadcast_in_dim"(%1583) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4601 = mhlo.add %4599, %4600 : tensor<1x384x512xf32>
    %4602 = "mhlo.reshape"(%4601) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4603 = "mhlo.dot"(%4602, %1598) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4604 = "mhlo.broadcast_in_dim"(%1597) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4605 = mhlo.add %4603, %4604 : tensor<384x128xf32>
    %4606 = "mhlo.reshape"(%4605) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4607 = "mhlo.transpose"(%4606) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4608 = "mhlo.dot"(%4602, %1602) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4609 = "mhlo.reshape"(%4608) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4610 = "mhlo.broadcast_in_dim"(%1601) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4611 = mhlo.add %4609, %4610 : tensor<1x384x128xf32>
    %4612 = "mhlo.broadcast_in_dim"(%1600) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4613 = mhlo.multiply %4611, %4612 : tensor<1x384x128xf32>
    %4614 = "mhlo.broadcast_in_dim"(%1599) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4615 = mhlo.add %4613, %4614 : tensor<1x384x128xf32>
    %4616 = "mhlo.reshape"(%4615) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4617 = "mhlo.dot"(%4616, %1594) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4618 = "mhlo.broadcast_in_dim"(%1593) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4619 = mhlo.add %4617, %4618 : tensor<384x128xf32>
    %4620 = "mhlo.reshape"(%4619) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4621 = "mhlo.transpose"(%4620) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4622 = "mhlo.dot"(%4616, %1596) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4623 = "mhlo.broadcast_in_dim"(%1595) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4624 = mhlo.add %4622, %4623 : tensor<384x128xf32>
    %4625 = "mhlo.reshape"(%4624) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4626 = "mhlo.transpose"(%4625) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4627 = "mhlo.dot_general"(%4626, %4621) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4628 = mhlo.multiply %4627, %1114 : tensor<1x4x384x384xf32>
    %4629 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4630 = mhlo.add %4628, %4629 : tensor<1x4x384x384xf32>
    %4631 = "mhlo.reduce"(%4630, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4632 = "mhlo.broadcast_in_dim"(%4631) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4633 = mhlo.subtract %4630, %4632 : tensor<1x4x384x384xf32>
    %4634 = "mhlo.exponential"(%4633) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4635 = "mhlo.reduce"(%4634, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4636 = "mhlo.broadcast_in_dim"(%4635) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4637 = mhlo.divide %4634, %4636 : tensor<1x4x384x384xf32>
    %4638 = "mhlo.dot_general"(%4637, %4607) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4639 = "mhlo.transpose"(%4638) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4640 = "mhlo.reshape"(%4639) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4641 = "mhlo.dot"(%4640, %1592) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4642 = "mhlo.broadcast_in_dim"(%1591) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4643 = mhlo.add %4641, %4642 : tensor<384x128xf32>
    %4644 = "mhlo.reshape"(%4643) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4645 = "mhlo.dot"(%4602, %1606) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4646 = "mhlo.broadcast_in_dim"(%1605) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4647 = mhlo.add %4645, %4646 : tensor<384x128xf32>
    %4648 = "mhlo.reshape"(%4647) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4649 = "mhlo.broadcast_in_dim"(%1604) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4650 = mhlo.multiply %4648, %4649 : tensor<1x384x128xf32>
    %4651 = "mhlo.broadcast_in_dim"(%1603) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4652 = mhlo.add %4650, %4651 : tensor<1x384x128xf32>
    %4653 = mhlo.add %4644, %4652 : tensor<1x384x128xf32>
    %4654 = "mhlo.broadcast_in_dim"(%1590) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4655 = mhlo.multiply %4653, %4654 : tensor<1x384x128xf32>
    %4656 = "mhlo.broadcast_in_dim"(%1589) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4657 = mhlo.add %4655, %4656 : tensor<1x384x128xf32>
    %4658 = "mhlo.reshape"(%4657) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4659 = "mhlo.dot"(%4658, %1608) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4660 = "mhlo.broadcast_in_dim"(%1607) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4661 = mhlo.add %4659, %4660 : tensor<384x512xf32>
    %4662 = "mhlo.reshape"(%4661) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4663 = mhlo.maximum %4662, %1119 : tensor<1x384x512xf32>
    %4664 = "mhlo.reshape"(%4663) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4665 = "mhlo.dot"(%4664, %1612) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4666 = "mhlo.broadcast_in_dim"(%1611) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4667 = mhlo.add %4665, %4666 : tensor<384x128xf32>
    %4668 = "mhlo.reshape"(%4667) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4669 = mhlo.add %4668, %4657 : tensor<1x384x128xf32>
    %4670 = "mhlo.broadcast_in_dim"(%1610) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4671 = mhlo.multiply %4669, %4670 : tensor<1x384x128xf32>
    %4672 = "mhlo.broadcast_in_dim"(%1609) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4673 = mhlo.add %4671, %4672 : tensor<1x384x128xf32>
    %4674 = "mhlo.reshape"(%4673) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4675 = "mhlo.dot"(%4674, %1614) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4676 = "mhlo.broadcast_in_dim"(%1613) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4677 = mhlo.add %4675, %4676 : tensor<384x512xf32>
    %4678 = "mhlo.reshape"(%4677) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4679 = mhlo.maximum %4678, %1119 : tensor<1x384x512xf32>
    %4680 = "mhlo.reshape"(%4679) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4681 = "mhlo.dot"(%4680, %1618) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4682 = "mhlo.broadcast_in_dim"(%1617) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4683 = mhlo.add %4681, %4682 : tensor<384x128xf32>
    %4684 = "mhlo.reshape"(%4683) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4685 = mhlo.add %4684, %4673 : tensor<1x384x128xf32>
    %4686 = "mhlo.broadcast_in_dim"(%1616) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4687 = mhlo.multiply %4685, %4686 : tensor<1x384x128xf32>
    %4688 = "mhlo.broadcast_in_dim"(%1615) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4689 = mhlo.add %4687, %4688 : tensor<1x384x128xf32>
    %4690 = "mhlo.reshape"(%4689) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4691 = "mhlo.dot"(%4690, %1620) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4692 = "mhlo.broadcast_in_dim"(%1619) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4693 = mhlo.add %4691, %4692 : tensor<384x512xf32>
    %4694 = "mhlo.reshape"(%4693) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4695 = mhlo.maximum %4694, %1119 : tensor<1x384x512xf32>
    %4696 = "mhlo.reshape"(%4695) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4697 = "mhlo.dot"(%4696, %1624) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4698 = "mhlo.broadcast_in_dim"(%1623) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4699 = mhlo.add %4697, %4698 : tensor<384x128xf32>
    %4700 = "mhlo.reshape"(%4699) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4701 = mhlo.add %4700, %4689 : tensor<1x384x128xf32>
    %4702 = "mhlo.broadcast_in_dim"(%1622) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4703 = mhlo.multiply %4701, %4702 : tensor<1x384x128xf32>
    %4704 = "mhlo.broadcast_in_dim"(%1621) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4705 = mhlo.add %4703, %4704 : tensor<1x384x128xf32>
    %4706 = "mhlo.reshape"(%4705) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4707 = "mhlo.dot"(%4706, %1626) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4708 = "mhlo.broadcast_in_dim"(%1625) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4709 = mhlo.add %4707, %4708 : tensor<384x512xf32>
    %4710 = "mhlo.reshape"(%4709) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4711 = mhlo.maximum %4710, %1119 : tensor<1x384x512xf32>
    %4712 = "mhlo.reshape"(%4711) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4713 = "mhlo.dot"(%4712, %1634) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4714 = "mhlo.broadcast_in_dim"(%1633) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4715 = mhlo.add %4713, %4714 : tensor<384x128xf32>
    %4716 = "mhlo.reshape"(%4715) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4717 = mhlo.add %4716, %4705 : tensor<1x384x128xf32>
    %4718 = "mhlo.broadcast_in_dim"(%1628) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4719 = mhlo.multiply %4717, %4718 : tensor<1x384x128xf32>
    %4720 = "mhlo.broadcast_in_dim"(%1627) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4721 = mhlo.add %4719, %4720 : tensor<1x384x128xf32>
    %4722 = "mhlo.reshape"(%4721) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4723 = "mhlo.dot"(%4722, %1632) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4724 = "mhlo.broadcast_in_dim"(%1631) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4725 = mhlo.add %4723, %4724 : tensor<384x512xf32>
    %4726 = "mhlo.reshape"(%4725) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4727 = mhlo.add %4726, %4601 : tensor<1x384x512xf32>
    %4728 = "mhlo.broadcast_in_dim"(%1630) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4729 = mhlo.multiply %4727, %4728 : tensor<1x384x512xf32>
    %4730 = "mhlo.broadcast_in_dim"(%1629) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4731 = mhlo.add %4729, %4730 : tensor<1x384x512xf32>
    %4732 = "mhlo.reshape"(%4731) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4733 = "mhlo.dot"(%4732, %1644) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4734 = "mhlo.broadcast_in_dim"(%1643) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4735 = mhlo.add %4733, %4734 : tensor<384x128xf32>
    %4736 = "mhlo.reshape"(%4735) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4737 = "mhlo.transpose"(%4736) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4738 = "mhlo.dot"(%4732, %1648) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4739 = "mhlo.reshape"(%4738) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4740 = "mhlo.broadcast_in_dim"(%1647) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4741 = mhlo.add %4739, %4740 : tensor<1x384x128xf32>
    %4742 = "mhlo.broadcast_in_dim"(%1646) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4743 = mhlo.multiply %4741, %4742 : tensor<1x384x128xf32>
    %4744 = "mhlo.broadcast_in_dim"(%1645) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4745 = mhlo.add %4743, %4744 : tensor<1x384x128xf32>
    %4746 = "mhlo.reshape"(%4745) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4747 = "mhlo.dot"(%4746, %1640) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4748 = "mhlo.broadcast_in_dim"(%1639) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4749 = mhlo.add %4747, %4748 : tensor<384x128xf32>
    %4750 = "mhlo.reshape"(%4749) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4751 = "mhlo.transpose"(%4750) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4752 = "mhlo.dot"(%4746, %1642) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4753 = "mhlo.broadcast_in_dim"(%1641) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4754 = mhlo.add %4752, %4753 : tensor<384x128xf32>
    %4755 = "mhlo.reshape"(%4754) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4756 = "mhlo.transpose"(%4755) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4757 = "mhlo.dot_general"(%4756, %4751) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4758 = mhlo.multiply %4757, %1114 : tensor<1x4x384x384xf32>
    %4759 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4760 = mhlo.add %4758, %4759 : tensor<1x4x384x384xf32>
    %4761 = "mhlo.reduce"(%4760, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4762 = "mhlo.broadcast_in_dim"(%4761) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4763 = mhlo.subtract %4760, %4762 : tensor<1x4x384x384xf32>
    %4764 = "mhlo.exponential"(%4763) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4765 = "mhlo.reduce"(%4764, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4766 = "mhlo.broadcast_in_dim"(%4765) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4767 = mhlo.divide %4764, %4766 : tensor<1x4x384x384xf32>
    %4768 = "mhlo.dot_general"(%4767, %4737) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4769 = "mhlo.transpose"(%4768) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4770 = "mhlo.reshape"(%4769) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4771 = "mhlo.dot"(%4770, %1638) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4772 = "mhlo.broadcast_in_dim"(%1637) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4773 = mhlo.add %4771, %4772 : tensor<384x128xf32>
    %4774 = "mhlo.reshape"(%4773) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4775 = "mhlo.dot"(%4732, %1652) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4776 = "mhlo.broadcast_in_dim"(%1651) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4777 = mhlo.add %4775, %4776 : tensor<384x128xf32>
    %4778 = "mhlo.reshape"(%4777) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4779 = "mhlo.broadcast_in_dim"(%1650) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4780 = mhlo.multiply %4778, %4779 : tensor<1x384x128xf32>
    %4781 = "mhlo.broadcast_in_dim"(%1649) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4782 = mhlo.add %4780, %4781 : tensor<1x384x128xf32>
    %4783 = mhlo.add %4774, %4782 : tensor<1x384x128xf32>
    %4784 = "mhlo.broadcast_in_dim"(%1636) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4785 = mhlo.multiply %4783, %4784 : tensor<1x384x128xf32>
    %4786 = "mhlo.broadcast_in_dim"(%1635) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4787 = mhlo.add %4785, %4786 : tensor<1x384x128xf32>
    %4788 = "mhlo.reshape"(%4787) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4789 = "mhlo.dot"(%4788, %1654) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4790 = "mhlo.broadcast_in_dim"(%1653) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4791 = mhlo.add %4789, %4790 : tensor<384x512xf32>
    %4792 = "mhlo.reshape"(%4791) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4793 = mhlo.maximum %4792, %1119 : tensor<1x384x512xf32>
    %4794 = "mhlo.reshape"(%4793) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4795 = "mhlo.dot"(%4794, %1658) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4796 = "mhlo.broadcast_in_dim"(%1657) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4797 = mhlo.add %4795, %4796 : tensor<384x128xf32>
    %4798 = "mhlo.reshape"(%4797) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4799 = mhlo.add %4798, %4787 : tensor<1x384x128xf32>
    %4800 = "mhlo.broadcast_in_dim"(%1656) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4801 = mhlo.multiply %4799, %4800 : tensor<1x384x128xf32>
    %4802 = "mhlo.broadcast_in_dim"(%1655) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4803 = mhlo.add %4801, %4802 : tensor<1x384x128xf32>
    %4804 = "mhlo.reshape"(%4803) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4805 = "mhlo.dot"(%4804, %1660) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4806 = "mhlo.broadcast_in_dim"(%1659) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4807 = mhlo.add %4805, %4806 : tensor<384x512xf32>
    %4808 = "mhlo.reshape"(%4807) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4809 = mhlo.maximum %4808, %1119 : tensor<1x384x512xf32>
    %4810 = "mhlo.reshape"(%4809) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4811 = "mhlo.dot"(%4810, %1664) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4812 = "mhlo.broadcast_in_dim"(%1663) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4813 = mhlo.add %4811, %4812 : tensor<384x128xf32>
    %4814 = "mhlo.reshape"(%4813) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4815 = mhlo.add %4814, %4803 : tensor<1x384x128xf32>
    %4816 = "mhlo.broadcast_in_dim"(%1662) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4817 = mhlo.multiply %4815, %4816 : tensor<1x384x128xf32>
    %4818 = "mhlo.broadcast_in_dim"(%1661) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4819 = mhlo.add %4817, %4818 : tensor<1x384x128xf32>
    %4820 = "mhlo.reshape"(%4819) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4821 = "mhlo.dot"(%4820, %1666) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4822 = "mhlo.broadcast_in_dim"(%1665) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4823 = mhlo.add %4821, %4822 : tensor<384x512xf32>
    %4824 = "mhlo.reshape"(%4823) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4825 = mhlo.maximum %4824, %1119 : tensor<1x384x512xf32>
    %4826 = "mhlo.reshape"(%4825) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4827 = "mhlo.dot"(%4826, %1670) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4828 = "mhlo.broadcast_in_dim"(%1669) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4829 = mhlo.add %4827, %4828 : tensor<384x128xf32>
    %4830 = "mhlo.reshape"(%4829) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4831 = mhlo.add %4830, %4819 : tensor<1x384x128xf32>
    %4832 = "mhlo.broadcast_in_dim"(%1668) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4833 = mhlo.multiply %4831, %4832 : tensor<1x384x128xf32>
    %4834 = "mhlo.broadcast_in_dim"(%1667) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4835 = mhlo.add %4833, %4834 : tensor<1x384x128xf32>
    %4836 = "mhlo.reshape"(%4835) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4837 = "mhlo.dot"(%4836, %1672) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4838 = "mhlo.broadcast_in_dim"(%1671) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4839 = mhlo.add %4837, %4838 : tensor<384x512xf32>
    %4840 = "mhlo.reshape"(%4839) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4841 = mhlo.maximum %4840, %1119 : tensor<1x384x512xf32>
    %4842 = "mhlo.reshape"(%4841) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4843 = "mhlo.dot"(%4842, %1680) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4844 = "mhlo.broadcast_in_dim"(%1679) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4845 = mhlo.add %4843, %4844 : tensor<384x128xf32>
    %4846 = "mhlo.reshape"(%4845) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4847 = mhlo.add %4846, %4835 : tensor<1x384x128xf32>
    %4848 = "mhlo.broadcast_in_dim"(%1674) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4849 = mhlo.multiply %4847, %4848 : tensor<1x384x128xf32>
    %4850 = "mhlo.broadcast_in_dim"(%1673) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4851 = mhlo.add %4849, %4850 : tensor<1x384x128xf32>
    %4852 = "mhlo.reshape"(%4851) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4853 = "mhlo.dot"(%4852, %1678) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4854 = "mhlo.broadcast_in_dim"(%1677) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4855 = mhlo.add %4853, %4854 : tensor<384x512xf32>
    %4856 = "mhlo.reshape"(%4855) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4857 = mhlo.add %4856, %4731 : tensor<1x384x512xf32>
    %4858 = "mhlo.broadcast_in_dim"(%1676) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4859 = mhlo.multiply %4857, %4858 : tensor<1x384x512xf32>
    %4860 = "mhlo.broadcast_in_dim"(%1675) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4861 = mhlo.add %4859, %4860 : tensor<1x384x512xf32>
    %4862 = "mhlo.reshape"(%4861) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4863 = "mhlo.dot"(%4862, %1736) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4864 = "mhlo.broadcast_in_dim"(%1735) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4865 = mhlo.add %4863, %4864 : tensor<384x128xf32>
    %4866 = "mhlo.reshape"(%4865) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4867 = "mhlo.transpose"(%4866) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4868 = "mhlo.dot"(%4862, %1740) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4869 = "mhlo.reshape"(%4868) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4870 = "mhlo.broadcast_in_dim"(%1739) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4871 = mhlo.add %4869, %4870 : tensor<1x384x128xf32>
    %4872 = "mhlo.broadcast_in_dim"(%1738) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4873 = mhlo.multiply %4871, %4872 : tensor<1x384x128xf32>
    %4874 = "mhlo.broadcast_in_dim"(%1737) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4875 = mhlo.add %4873, %4874 : tensor<1x384x128xf32>
    %4876 = "mhlo.reshape"(%4875) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4877 = "mhlo.dot"(%4876, %1732) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4878 = "mhlo.broadcast_in_dim"(%1731) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4879 = mhlo.add %4877, %4878 : tensor<384x128xf32>
    %4880 = "mhlo.reshape"(%4879) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4881 = "mhlo.transpose"(%4880) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4882 = "mhlo.dot"(%4876, %1734) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4883 = "mhlo.broadcast_in_dim"(%1733) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4884 = mhlo.add %4882, %4883 : tensor<384x128xf32>
    %4885 = "mhlo.reshape"(%4884) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4886 = "mhlo.transpose"(%4885) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4887 = "mhlo.dot_general"(%4886, %4881) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %4888 = mhlo.multiply %4887, %1114 : tensor<1x4x384x384xf32>
    %4889 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4890 = mhlo.add %4888, %4889 : tensor<1x4x384x384xf32>
    %4891 = "mhlo.reduce"(%4890, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4892 = "mhlo.broadcast_in_dim"(%4891) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4893 = mhlo.subtract %4890, %4892 : tensor<1x4x384x384xf32>
    %4894 = "mhlo.exponential"(%4893) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %4895 = "mhlo.reduce"(%4894, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %4896 = "mhlo.broadcast_in_dim"(%4895) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %4897 = mhlo.divide %4894, %4896 : tensor<1x4x384x384xf32>
    %4898 = "mhlo.dot_general"(%4897, %4867) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %4899 = "mhlo.transpose"(%4898) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %4900 = "mhlo.reshape"(%4899) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %4901 = "mhlo.dot"(%4900, %1730) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %4902 = "mhlo.broadcast_in_dim"(%1729) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4903 = mhlo.add %4901, %4902 : tensor<384x128xf32>
    %4904 = "mhlo.reshape"(%4903) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4905 = "mhlo.dot"(%4862, %1744) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4906 = "mhlo.broadcast_in_dim"(%1743) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4907 = mhlo.add %4905, %4906 : tensor<384x128xf32>
    %4908 = "mhlo.reshape"(%4907) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4909 = "mhlo.broadcast_in_dim"(%1742) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4910 = mhlo.multiply %4908, %4909 : tensor<1x384x128xf32>
    %4911 = "mhlo.broadcast_in_dim"(%1741) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4912 = mhlo.add %4910, %4911 : tensor<1x384x128xf32>
    %4913 = mhlo.add %4904, %4912 : tensor<1x384x128xf32>
    %4914 = "mhlo.broadcast_in_dim"(%1728) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4915 = mhlo.multiply %4913, %4914 : tensor<1x384x128xf32>
    %4916 = "mhlo.broadcast_in_dim"(%1727) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4917 = mhlo.add %4915, %4916 : tensor<1x384x128xf32>
    %4918 = "mhlo.reshape"(%4917) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4919 = "mhlo.dot"(%4918, %1746) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4920 = "mhlo.broadcast_in_dim"(%1745) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4921 = mhlo.add %4919, %4920 : tensor<384x512xf32>
    %4922 = "mhlo.reshape"(%4921) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4923 = mhlo.maximum %4922, %1119 : tensor<1x384x512xf32>
    %4924 = "mhlo.reshape"(%4923) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4925 = "mhlo.dot"(%4924, %1750) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4926 = "mhlo.broadcast_in_dim"(%1749) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4927 = mhlo.add %4925, %4926 : tensor<384x128xf32>
    %4928 = "mhlo.reshape"(%4927) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4929 = mhlo.add %4928, %4917 : tensor<1x384x128xf32>
    %4930 = "mhlo.broadcast_in_dim"(%1748) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4931 = mhlo.multiply %4929, %4930 : tensor<1x384x128xf32>
    %4932 = "mhlo.broadcast_in_dim"(%1747) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4933 = mhlo.add %4931, %4932 : tensor<1x384x128xf32>
    %4934 = "mhlo.reshape"(%4933) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4935 = "mhlo.dot"(%4934, %1752) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4936 = "mhlo.broadcast_in_dim"(%1751) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4937 = mhlo.add %4935, %4936 : tensor<384x512xf32>
    %4938 = "mhlo.reshape"(%4937) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4939 = mhlo.maximum %4938, %1119 : tensor<1x384x512xf32>
    %4940 = "mhlo.reshape"(%4939) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4941 = "mhlo.dot"(%4940, %1756) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4942 = "mhlo.broadcast_in_dim"(%1755) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4943 = mhlo.add %4941, %4942 : tensor<384x128xf32>
    %4944 = "mhlo.reshape"(%4943) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4945 = mhlo.add %4944, %4933 : tensor<1x384x128xf32>
    %4946 = "mhlo.broadcast_in_dim"(%1754) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4947 = mhlo.multiply %4945, %4946 : tensor<1x384x128xf32>
    %4948 = "mhlo.broadcast_in_dim"(%1753) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4949 = mhlo.add %4947, %4948 : tensor<1x384x128xf32>
    %4950 = "mhlo.reshape"(%4949) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4951 = "mhlo.dot"(%4950, %1758) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4952 = "mhlo.broadcast_in_dim"(%1757) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4953 = mhlo.add %4951, %4952 : tensor<384x512xf32>
    %4954 = "mhlo.reshape"(%4953) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4955 = mhlo.maximum %4954, %1119 : tensor<1x384x512xf32>
    %4956 = "mhlo.reshape"(%4955) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4957 = "mhlo.dot"(%4956, %1762) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4958 = "mhlo.broadcast_in_dim"(%1761) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4959 = mhlo.add %4957, %4958 : tensor<384x128xf32>
    %4960 = "mhlo.reshape"(%4959) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4961 = mhlo.add %4960, %4949 : tensor<1x384x128xf32>
    %4962 = "mhlo.broadcast_in_dim"(%1760) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4963 = mhlo.multiply %4961, %4962 : tensor<1x384x128xf32>
    %4964 = "mhlo.broadcast_in_dim"(%1759) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4965 = mhlo.add %4963, %4964 : tensor<1x384x128xf32>
    %4966 = "mhlo.reshape"(%4965) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4967 = "mhlo.dot"(%4966, %1764) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4968 = "mhlo.broadcast_in_dim"(%1763) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4969 = mhlo.add %4967, %4968 : tensor<384x512xf32>
    %4970 = "mhlo.reshape"(%4969) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4971 = mhlo.maximum %4970, %1119 : tensor<1x384x512xf32>
    %4972 = "mhlo.reshape"(%4971) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4973 = "mhlo.dot"(%4972, %1772) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4974 = "mhlo.broadcast_in_dim"(%1771) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4975 = mhlo.add %4973, %4974 : tensor<384x128xf32>
    %4976 = "mhlo.reshape"(%4975) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %4977 = mhlo.add %4976, %4965 : tensor<1x384x128xf32>
    %4978 = "mhlo.broadcast_in_dim"(%1766) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4979 = mhlo.multiply %4977, %4978 : tensor<1x384x128xf32>
    %4980 = "mhlo.broadcast_in_dim"(%1765) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %4981 = mhlo.add %4979, %4980 : tensor<1x384x128xf32>
    %4982 = "mhlo.reshape"(%4981) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %4983 = "mhlo.dot"(%4982, %1770) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %4984 = "mhlo.broadcast_in_dim"(%1769) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %4985 = mhlo.add %4983, %4984 : tensor<384x512xf32>
    %4986 = "mhlo.reshape"(%4985) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %4987 = mhlo.add %4986, %4861 : tensor<1x384x512xf32>
    %4988 = "mhlo.broadcast_in_dim"(%1768) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4989 = mhlo.multiply %4987, %4988 : tensor<1x384x512xf32>
    %4990 = "mhlo.broadcast_in_dim"(%1767) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %4991 = mhlo.add %4989, %4990 : tensor<1x384x512xf32>
    %4992 = "mhlo.reshape"(%4991) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %4993 = "mhlo.dot"(%4992, %1782) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4994 = "mhlo.broadcast_in_dim"(%1781) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %4995 = mhlo.add %4993, %4994 : tensor<384x128xf32>
    %4996 = "mhlo.reshape"(%4995) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %4997 = "mhlo.transpose"(%4996) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %4998 = "mhlo.dot"(%4992, %1786) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %4999 = "mhlo.reshape"(%4998) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5000 = "mhlo.broadcast_in_dim"(%1785) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5001 = mhlo.add %4999, %5000 : tensor<1x384x128xf32>
    %5002 = "mhlo.broadcast_in_dim"(%1784) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5003 = mhlo.multiply %5001, %5002 : tensor<1x384x128xf32>
    %5004 = "mhlo.broadcast_in_dim"(%1783) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5005 = mhlo.add %5003, %5004 : tensor<1x384x128xf32>
    %5006 = "mhlo.reshape"(%5005) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5007 = "mhlo.dot"(%5006, %1778) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5008 = "mhlo.broadcast_in_dim"(%1777) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5009 = mhlo.add %5007, %5008 : tensor<384x128xf32>
    %5010 = "mhlo.reshape"(%5009) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %5011 = "mhlo.transpose"(%5010) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %5012 = "mhlo.dot"(%5006, %1780) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5013 = "mhlo.broadcast_in_dim"(%1779) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5014 = mhlo.add %5012, %5013 : tensor<384x128xf32>
    %5015 = "mhlo.reshape"(%5014) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %5016 = "mhlo.transpose"(%5015) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %5017 = "mhlo.dot_general"(%5016, %5011) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %5018 = mhlo.multiply %5017, %1114 : tensor<1x4x384x384xf32>
    %5019 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %5020 = mhlo.add %5018, %5019 : tensor<1x4x384x384xf32>
    %5021 = "mhlo.reduce"(%5020, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %5022 = "mhlo.broadcast_in_dim"(%5021) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %5023 = mhlo.subtract %5020, %5022 : tensor<1x4x384x384xf32>
    %5024 = "mhlo.exponential"(%5023) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %5025 = "mhlo.reduce"(%5024, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %5026 = "mhlo.broadcast_in_dim"(%5025) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %5027 = mhlo.divide %5024, %5026 : tensor<1x4x384x384xf32>
    %5028 = "mhlo.dot_general"(%5027, %4997) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %5029 = "mhlo.transpose"(%5028) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %5030 = "mhlo.reshape"(%5029) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %5031 = "mhlo.dot"(%5030, %1776) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5032 = "mhlo.broadcast_in_dim"(%1775) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5033 = mhlo.add %5031, %5032 : tensor<384x128xf32>
    %5034 = "mhlo.reshape"(%5033) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5035 = "mhlo.dot"(%4992, %1790) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5036 = "mhlo.broadcast_in_dim"(%1789) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5037 = mhlo.add %5035, %5036 : tensor<384x128xf32>
    %5038 = "mhlo.reshape"(%5037) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5039 = "mhlo.broadcast_in_dim"(%1788) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5040 = mhlo.multiply %5038, %5039 : tensor<1x384x128xf32>
    %5041 = "mhlo.broadcast_in_dim"(%1787) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5042 = mhlo.add %5040, %5041 : tensor<1x384x128xf32>
    %5043 = mhlo.add %5034, %5042 : tensor<1x384x128xf32>
    %5044 = "mhlo.broadcast_in_dim"(%1774) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5045 = mhlo.multiply %5043, %5044 : tensor<1x384x128xf32>
    %5046 = "mhlo.broadcast_in_dim"(%1773) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5047 = mhlo.add %5045, %5046 : tensor<1x384x128xf32>
    %5048 = "mhlo.reshape"(%5047) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5049 = "mhlo.dot"(%5048, %1792) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5050 = "mhlo.broadcast_in_dim"(%1791) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5051 = mhlo.add %5049, %5050 : tensor<384x512xf32>
    %5052 = "mhlo.reshape"(%5051) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5053 = mhlo.maximum %5052, %1119 : tensor<1x384x512xf32>
    %5054 = "mhlo.reshape"(%5053) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5055 = "mhlo.dot"(%5054, %1796) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5056 = "mhlo.broadcast_in_dim"(%1795) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5057 = mhlo.add %5055, %5056 : tensor<384x128xf32>
    %5058 = "mhlo.reshape"(%5057) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5059 = mhlo.add %5058, %5047 : tensor<1x384x128xf32>
    %5060 = "mhlo.broadcast_in_dim"(%1794) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5061 = mhlo.multiply %5059, %5060 : tensor<1x384x128xf32>
    %5062 = "mhlo.broadcast_in_dim"(%1793) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5063 = mhlo.add %5061, %5062 : tensor<1x384x128xf32>
    %5064 = "mhlo.reshape"(%5063) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5065 = "mhlo.dot"(%5064, %1798) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5066 = "mhlo.broadcast_in_dim"(%1797) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5067 = mhlo.add %5065, %5066 : tensor<384x512xf32>
    %5068 = "mhlo.reshape"(%5067) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5069 = mhlo.maximum %5068, %1119 : tensor<1x384x512xf32>
    %5070 = "mhlo.reshape"(%5069) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5071 = "mhlo.dot"(%5070, %1802) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5072 = "mhlo.broadcast_in_dim"(%1801) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5073 = mhlo.add %5071, %5072 : tensor<384x128xf32>
    %5074 = "mhlo.reshape"(%5073) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5075 = mhlo.add %5074, %5063 : tensor<1x384x128xf32>
    %5076 = "mhlo.broadcast_in_dim"(%1800) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5077 = mhlo.multiply %5075, %5076 : tensor<1x384x128xf32>
    %5078 = "mhlo.broadcast_in_dim"(%1799) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5079 = mhlo.add %5077, %5078 : tensor<1x384x128xf32>
    %5080 = "mhlo.reshape"(%5079) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5081 = "mhlo.dot"(%5080, %1804) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5082 = "mhlo.broadcast_in_dim"(%1803) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5083 = mhlo.add %5081, %5082 : tensor<384x512xf32>
    %5084 = "mhlo.reshape"(%5083) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5085 = mhlo.maximum %5084, %1119 : tensor<1x384x512xf32>
    %5086 = "mhlo.reshape"(%5085) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5087 = "mhlo.dot"(%5086, %1808) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5088 = "mhlo.broadcast_in_dim"(%1807) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5089 = mhlo.add %5087, %5088 : tensor<384x128xf32>
    %5090 = "mhlo.reshape"(%5089) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5091 = mhlo.add %5090, %5079 : tensor<1x384x128xf32>
    %5092 = "mhlo.broadcast_in_dim"(%1806) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5093 = mhlo.multiply %5091, %5092 : tensor<1x384x128xf32>
    %5094 = "mhlo.broadcast_in_dim"(%1805) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5095 = mhlo.add %5093, %5094 : tensor<1x384x128xf32>
    %5096 = "mhlo.reshape"(%5095) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5097 = "mhlo.dot"(%5096, %1810) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5098 = "mhlo.broadcast_in_dim"(%1809) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5099 = mhlo.add %5097, %5098 : tensor<384x512xf32>
    %5100 = "mhlo.reshape"(%5099) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5101 = mhlo.maximum %5100, %1119 : tensor<1x384x512xf32>
    %5102 = "mhlo.reshape"(%5101) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5103 = "mhlo.dot"(%5102, %1818) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5104 = "mhlo.broadcast_in_dim"(%1817) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5105 = mhlo.add %5103, %5104 : tensor<384x128xf32>
    %5106 = "mhlo.reshape"(%5105) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5107 = mhlo.add %5106, %5095 : tensor<1x384x128xf32>
    %5108 = "mhlo.broadcast_in_dim"(%1812) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5109 = mhlo.multiply %5107, %5108 : tensor<1x384x128xf32>
    %5110 = "mhlo.broadcast_in_dim"(%1811) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5111 = mhlo.add %5109, %5110 : tensor<1x384x128xf32>
    %5112 = "mhlo.reshape"(%5111) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5113 = "mhlo.dot"(%5112, %1816) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5114 = "mhlo.broadcast_in_dim"(%1815) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5115 = mhlo.add %5113, %5114 : tensor<384x512xf32>
    %5116 = "mhlo.reshape"(%5115) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5117 = mhlo.add %5116, %4991 : tensor<1x384x512xf32>
    %5118 = "mhlo.broadcast_in_dim"(%1814) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %5119 = mhlo.multiply %5117, %5118 : tensor<1x384x512xf32>
    %5120 = "mhlo.broadcast_in_dim"(%1813) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %5121 = mhlo.add %5119, %5120 : tensor<1x384x512xf32>
    %5122 = "mhlo.reshape"(%5121) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5123 = "mhlo.dot"(%5122, %1828) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5124 = "mhlo.broadcast_in_dim"(%1827) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5125 = mhlo.add %5123, %5124 : tensor<384x128xf32>
    %5126 = "mhlo.reshape"(%5125) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %5127 = "mhlo.transpose"(%5126) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %5128 = "mhlo.dot"(%5122, %1832) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5129 = "mhlo.reshape"(%5128) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5130 = "mhlo.broadcast_in_dim"(%1831) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5131 = mhlo.add %5129, %5130 : tensor<1x384x128xf32>
    %5132 = "mhlo.broadcast_in_dim"(%1830) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5133 = mhlo.multiply %5131, %5132 : tensor<1x384x128xf32>
    %5134 = "mhlo.broadcast_in_dim"(%1829) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5135 = mhlo.add %5133, %5134 : tensor<1x384x128xf32>
    %5136 = "mhlo.reshape"(%5135) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5137 = "mhlo.dot"(%5136, %1824) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5138 = "mhlo.broadcast_in_dim"(%1823) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5139 = mhlo.add %5137, %5138 : tensor<384x128xf32>
    %5140 = "mhlo.reshape"(%5139) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %5141 = "mhlo.transpose"(%5140) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %5142 = "mhlo.dot"(%5136, %1826) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5143 = "mhlo.broadcast_in_dim"(%1825) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5144 = mhlo.add %5142, %5143 : tensor<384x128xf32>
    %5145 = "mhlo.reshape"(%5144) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %5146 = "mhlo.transpose"(%5145) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %5147 = "mhlo.dot_general"(%5146, %5141) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %5148 = mhlo.multiply %5147, %1114 : tensor<1x4x384x384xf32>
    %5149 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %5150 = mhlo.add %5148, %5149 : tensor<1x4x384x384xf32>
    %5151 = "mhlo.reduce"(%5150, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %5152 = "mhlo.broadcast_in_dim"(%5151) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %5153 = mhlo.subtract %5150, %5152 : tensor<1x4x384x384xf32>
    %5154 = "mhlo.exponential"(%5153) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %5155 = "mhlo.reduce"(%5154, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %5156 = "mhlo.broadcast_in_dim"(%5155) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %5157 = mhlo.divide %5154, %5156 : tensor<1x4x384x384xf32>
    %5158 = "mhlo.dot_general"(%5157, %5127) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %5159 = "mhlo.transpose"(%5158) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %5160 = "mhlo.reshape"(%5159) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %5161 = "mhlo.dot"(%5160, %1822) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5162 = "mhlo.broadcast_in_dim"(%1821) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5163 = mhlo.add %5161, %5162 : tensor<384x128xf32>
    %5164 = "mhlo.reshape"(%5163) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5165 = "mhlo.dot"(%5122, %1836) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5166 = "mhlo.broadcast_in_dim"(%1835) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5167 = mhlo.add %5165, %5166 : tensor<384x128xf32>
    %5168 = "mhlo.reshape"(%5167) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5169 = "mhlo.broadcast_in_dim"(%1834) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5170 = mhlo.multiply %5168, %5169 : tensor<1x384x128xf32>
    %5171 = "mhlo.broadcast_in_dim"(%1833) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5172 = mhlo.add %5170, %5171 : tensor<1x384x128xf32>
    %5173 = mhlo.add %5164, %5172 : tensor<1x384x128xf32>
    %5174 = "mhlo.broadcast_in_dim"(%1820) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5175 = mhlo.multiply %5173, %5174 : tensor<1x384x128xf32>
    %5176 = "mhlo.broadcast_in_dim"(%1819) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5177 = mhlo.add %5175, %5176 : tensor<1x384x128xf32>
    %5178 = "mhlo.reshape"(%5177) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5179 = "mhlo.dot"(%5178, %1838) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5180 = "mhlo.broadcast_in_dim"(%1837) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5181 = mhlo.add %5179, %5180 : tensor<384x512xf32>
    %5182 = "mhlo.reshape"(%5181) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5183 = mhlo.maximum %5182, %1119 : tensor<1x384x512xf32>
    %5184 = "mhlo.reshape"(%5183) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5185 = "mhlo.dot"(%5184, %1842) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5186 = "mhlo.broadcast_in_dim"(%1841) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5187 = mhlo.add %5185, %5186 : tensor<384x128xf32>
    %5188 = "mhlo.reshape"(%5187) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5189 = mhlo.add %5188, %5177 : tensor<1x384x128xf32>
    %5190 = "mhlo.broadcast_in_dim"(%1840) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5191 = mhlo.multiply %5189, %5190 : tensor<1x384x128xf32>
    %5192 = "mhlo.broadcast_in_dim"(%1839) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5193 = mhlo.add %5191, %5192 : tensor<1x384x128xf32>
    %5194 = "mhlo.reshape"(%5193) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5195 = "mhlo.dot"(%5194, %1844) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5196 = "mhlo.broadcast_in_dim"(%1843) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5197 = mhlo.add %5195, %5196 : tensor<384x512xf32>
    %5198 = "mhlo.reshape"(%5197) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5199 = mhlo.maximum %5198, %1119 : tensor<1x384x512xf32>
    %5200 = "mhlo.reshape"(%5199) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5201 = "mhlo.dot"(%5200, %1848) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5202 = "mhlo.broadcast_in_dim"(%1847) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5203 = mhlo.add %5201, %5202 : tensor<384x128xf32>
    %5204 = "mhlo.reshape"(%5203) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5205 = mhlo.add %5204, %5193 : tensor<1x384x128xf32>
    %5206 = "mhlo.broadcast_in_dim"(%1846) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5207 = mhlo.multiply %5205, %5206 : tensor<1x384x128xf32>
    %5208 = "mhlo.broadcast_in_dim"(%1845) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5209 = mhlo.add %5207, %5208 : tensor<1x384x128xf32>
    %5210 = "mhlo.reshape"(%5209) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5211 = "mhlo.dot"(%5210, %1850) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5212 = "mhlo.broadcast_in_dim"(%1849) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5213 = mhlo.add %5211, %5212 : tensor<384x512xf32>
    %5214 = "mhlo.reshape"(%5213) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5215 = mhlo.maximum %5214, %1119 : tensor<1x384x512xf32>
    %5216 = "mhlo.reshape"(%5215) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5217 = "mhlo.dot"(%5216, %1854) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5218 = "mhlo.broadcast_in_dim"(%1853) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5219 = mhlo.add %5217, %5218 : tensor<384x128xf32>
    %5220 = "mhlo.reshape"(%5219) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5221 = mhlo.add %5220, %5209 : tensor<1x384x128xf32>
    %5222 = "mhlo.broadcast_in_dim"(%1852) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5223 = mhlo.multiply %5221, %5222 : tensor<1x384x128xf32>
    %5224 = "mhlo.broadcast_in_dim"(%1851) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5225 = mhlo.add %5223, %5224 : tensor<1x384x128xf32>
    %5226 = "mhlo.reshape"(%5225) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5227 = "mhlo.dot"(%5226, %1856) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5228 = "mhlo.broadcast_in_dim"(%1855) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5229 = mhlo.add %5227, %5228 : tensor<384x512xf32>
    %5230 = "mhlo.reshape"(%5229) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5231 = mhlo.maximum %5230, %1119 : tensor<1x384x512xf32>
    %5232 = "mhlo.reshape"(%5231) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5233 = "mhlo.dot"(%5232, %1864) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5234 = "mhlo.broadcast_in_dim"(%1863) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5235 = mhlo.add %5233, %5234 : tensor<384x128xf32>
    %5236 = "mhlo.reshape"(%5235) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5237 = mhlo.add %5236, %5225 : tensor<1x384x128xf32>
    %5238 = "mhlo.broadcast_in_dim"(%1858) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5239 = mhlo.multiply %5237, %5238 : tensor<1x384x128xf32>
    %5240 = "mhlo.broadcast_in_dim"(%1857) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5241 = mhlo.add %5239, %5240 : tensor<1x384x128xf32>
    %5242 = "mhlo.reshape"(%5241) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5243 = "mhlo.dot"(%5242, %1862) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5244 = "mhlo.broadcast_in_dim"(%1861) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5245 = mhlo.add %5243, %5244 : tensor<384x512xf32>
    %5246 = "mhlo.reshape"(%5245) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5247 = mhlo.add %5246, %5121 : tensor<1x384x512xf32>
    %5248 = "mhlo.broadcast_in_dim"(%1860) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %5249 = mhlo.multiply %5247, %5248 : tensor<1x384x512xf32>
    %5250 = "mhlo.broadcast_in_dim"(%1859) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %5251 = mhlo.add %5249, %5250 : tensor<1x384x512xf32>
    %5252 = "mhlo.reshape"(%5251) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5253 = "mhlo.dot"(%5252, %1874) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5254 = "mhlo.broadcast_in_dim"(%1873) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5255 = mhlo.add %5253, %5254 : tensor<384x128xf32>
    %5256 = "mhlo.reshape"(%5255) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %5257 = "mhlo.transpose"(%5256) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %5258 = "mhlo.dot"(%5252, %1878) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5259 = "mhlo.reshape"(%5258) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5260 = "mhlo.broadcast_in_dim"(%1877) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5261 = mhlo.add %5259, %5260 : tensor<1x384x128xf32>
    %5262 = "mhlo.broadcast_in_dim"(%1876) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5263 = mhlo.multiply %5261, %5262 : tensor<1x384x128xf32>
    %5264 = "mhlo.broadcast_in_dim"(%1875) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5265 = mhlo.add %5263, %5264 : tensor<1x384x128xf32>
    %5266 = "mhlo.reshape"(%5265) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5267 = "mhlo.dot"(%5266, %1870) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5268 = "mhlo.broadcast_in_dim"(%1869) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5269 = mhlo.add %5267, %5268 : tensor<384x128xf32>
    %5270 = "mhlo.reshape"(%5269) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %5271 = "mhlo.transpose"(%5270) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %5272 = "mhlo.dot"(%5266, %1872) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5273 = "mhlo.broadcast_in_dim"(%1871) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5274 = mhlo.add %5272, %5273 : tensor<384x128xf32>
    %5275 = "mhlo.reshape"(%5274) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
    %5276 = "mhlo.transpose"(%5275) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
    %5277 = "mhlo.dot_general"(%5276, %5271) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [3]>} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x384xf32>
    %5278 = mhlo.multiply %5277, %1114 : tensor<1x4x384x384xf32>
    %5279 = "mhlo.broadcast_in_dim"(%2254) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x384x384xf32>) -> tensor<1x4x384x384xf32>
    %5280 = mhlo.add %5278, %5279 : tensor<1x4x384x384xf32>
    %5281 = "mhlo.reduce"(%5280, %1117) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %5282 = "mhlo.broadcast_in_dim"(%5281) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %5283 = mhlo.subtract %5280, %5282 : tensor<1x4x384x384xf32>
    %5284 = "mhlo.exponential"(%5283) : (tensor<1x4x384x384xf32>) -> tensor<1x4x384x384xf32>
    %5285 = "mhlo.reduce"(%5284, %1118) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<3> : tensor<1xi64>} : (tensor<1x4x384x384xf32>, tensor<f32>) -> tensor<1x4x384xf32>
    %5286 = "mhlo.broadcast_in_dim"(%5285) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x384xf32>) -> tensor<1x4x384x384xf32>
    %5287 = mhlo.divide %5284, %5286 : tensor<1x4x384x384xf32>
    %5288 = "mhlo.dot_general"(%5287, %5257) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_batching_dimensions = [0, 1], rhs_contracting_dimensions = [2]>} : (tensor<1x4x384x384xf32>, tensor<1x4x384x32xf32>) -> tensor<1x4x384x32xf32>
    %5289 = "mhlo.transpose"(%5288) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x4x384x32xf32>) -> tensor<1x384x4x32xf32>
    %5290 = "mhlo.reshape"(%5289) : (tensor<1x384x4x32xf32>) -> tensor<384x128xf32>
    %5291 = "mhlo.dot"(%5290, %1868) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    %5292 = "mhlo.broadcast_in_dim"(%1867) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5293 = mhlo.add %5291, %5292 : tensor<384x128xf32>
    %5294 = "mhlo.reshape"(%5293) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5295 = "mhlo.dot"(%5252, %1882) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5296 = "mhlo.broadcast_in_dim"(%1881) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5297 = mhlo.add %5295, %5296 : tensor<384x128xf32>
    %5298 = "mhlo.reshape"(%5297) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5299 = "mhlo.broadcast_in_dim"(%1880) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5300 = mhlo.multiply %5298, %5299 : tensor<1x384x128xf32>
    %5301 = "mhlo.broadcast_in_dim"(%1879) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5302 = mhlo.add %5300, %5301 : tensor<1x384x128xf32>
    %5303 = mhlo.add %5294, %5302 : tensor<1x384x128xf32>
    %5304 = "mhlo.broadcast_in_dim"(%1866) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5305 = mhlo.multiply %5303, %5304 : tensor<1x384x128xf32>
    %5306 = "mhlo.broadcast_in_dim"(%1865) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5307 = mhlo.add %5305, %5306 : tensor<1x384x128xf32>
    %5308 = "mhlo.reshape"(%5307) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5309 = "mhlo.dot"(%5308, %1884) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5310 = "mhlo.broadcast_in_dim"(%1883) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5311 = mhlo.add %5309, %5310 : tensor<384x512xf32>
    %5312 = "mhlo.reshape"(%5311) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5313 = mhlo.maximum %5312, %1119 : tensor<1x384x512xf32>
    %5314 = "mhlo.reshape"(%5313) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5315 = "mhlo.dot"(%5314, %1888) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5316 = "mhlo.broadcast_in_dim"(%1887) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5317 = mhlo.add %5315, %5316 : tensor<384x128xf32>
    %5318 = "mhlo.reshape"(%5317) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5319 = mhlo.add %5318, %5307 : tensor<1x384x128xf32>
    %5320 = "mhlo.broadcast_in_dim"(%1886) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5321 = mhlo.multiply %5319, %5320 : tensor<1x384x128xf32>
    %5322 = "mhlo.broadcast_in_dim"(%1885) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5323 = mhlo.add %5321, %5322 : tensor<1x384x128xf32>
    %5324 = "mhlo.reshape"(%5323) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5325 = "mhlo.dot"(%5324, %1890) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5326 = "mhlo.broadcast_in_dim"(%1889) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5327 = mhlo.add %5325, %5326 : tensor<384x512xf32>
    %5328 = "mhlo.reshape"(%5327) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5329 = mhlo.maximum %5328, %1119 : tensor<1x384x512xf32>
    %5330 = "mhlo.reshape"(%5329) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5331 = "mhlo.dot"(%5330, %1894) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5332 = "mhlo.broadcast_in_dim"(%1893) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5333 = mhlo.add %5331, %5332 : tensor<384x128xf32>
    %5334 = "mhlo.reshape"(%5333) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5335 = mhlo.add %5334, %5323 : tensor<1x384x128xf32>
    %5336 = "mhlo.broadcast_in_dim"(%1892) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5337 = mhlo.multiply %5335, %5336 : tensor<1x384x128xf32>
    %5338 = "mhlo.broadcast_in_dim"(%1891) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5339 = mhlo.add %5337, %5338 : tensor<1x384x128xf32>
    %5340 = "mhlo.reshape"(%5339) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5341 = "mhlo.dot"(%5340, %1896) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5342 = "mhlo.broadcast_in_dim"(%1895) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5343 = mhlo.add %5341, %5342 : tensor<384x512xf32>
    %5344 = "mhlo.reshape"(%5343) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5345 = mhlo.maximum %5344, %1119 : tensor<1x384x512xf32>
    %5346 = "mhlo.reshape"(%5345) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5347 = "mhlo.dot"(%5346, %1900) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5348 = "mhlo.broadcast_in_dim"(%1899) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5349 = mhlo.add %5347, %5348 : tensor<384x128xf32>
    %5350 = "mhlo.reshape"(%5349) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5351 = mhlo.add %5350, %5339 : tensor<1x384x128xf32>
    %5352 = "mhlo.broadcast_in_dim"(%1898) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5353 = mhlo.multiply %5351, %5352 : tensor<1x384x128xf32>
    %5354 = "mhlo.broadcast_in_dim"(%1897) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5355 = mhlo.add %5353, %5354 : tensor<1x384x128xf32>
    %5356 = "mhlo.reshape"(%5355) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5357 = "mhlo.dot"(%5356, %1902) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5358 = "mhlo.broadcast_in_dim"(%1901) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5359 = mhlo.add %5357, %5358 : tensor<384x512xf32>
    %5360 = "mhlo.reshape"(%5359) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5361 = mhlo.maximum %5360, %1119 : tensor<1x384x512xf32>
    %5362 = "mhlo.reshape"(%5361) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5363 = "mhlo.dot"(%5362, %1910) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    %5364 = "mhlo.broadcast_in_dim"(%1909) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<384x128xf32>
    %5365 = mhlo.add %5363, %5364 : tensor<384x128xf32>
    %5366 = "mhlo.reshape"(%5365) : (tensor<384x128xf32>) -> tensor<1x384x128xf32>
    %5367 = mhlo.add %5366, %5355 : tensor<1x384x128xf32>
    %5368 = "mhlo.broadcast_in_dim"(%1904) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5369 = mhlo.multiply %5367, %5368 : tensor<1x384x128xf32>
    %5370 = "mhlo.broadcast_in_dim"(%1903) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x384x128xf32>
    %5371 = mhlo.add %5369, %5370 : tensor<1x384x128xf32>
    %5372 = "mhlo.reshape"(%5371) : (tensor<1x384x128xf32>) -> tensor<384x128xf32>
    %5373 = "mhlo.dot"(%5372, %1908) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    %5374 = "mhlo.broadcast_in_dim"(%1907) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<384x512xf32>
    %5375 = mhlo.add %5373, %5374 : tensor<384x512xf32>
    %5376 = "mhlo.reshape"(%5375) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
    %5377 = mhlo.add %5376, %5251 : tensor<1x384x512xf32>
    %5378 = "mhlo.broadcast_in_dim"(%1906) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %5379 = mhlo.multiply %5377, %5378 : tensor<1x384x512xf32>
    %5380 = "mhlo.broadcast_in_dim"(%1905) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x384x512xf32>
    %5381 = mhlo.add %5379, %5380 : tensor<1x384x512xf32>
    %5382 = "mhlo.reshape"(%5381) : (tensor<1x384x512xf32>) -> tensor<384x512xf32>
    %5383 = "mhlo.transpose"(%2234) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x512xf32>) -> tensor<512x2xf32>
    %5384 = "mhlo.dot"(%5382, %5383) : (tensor<384x512xf32>, tensor<512x2xf32>) -> tensor<384x2xf32>
    %5385 = "mhlo.broadcast_in_dim"(%2233) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2xf32>) -> tensor<384x2xf32>
    %5386 = mhlo.add %5384, %5385 : tensor<384x2xf32>
    %5387 = "mhlo.reshape"(%5386) : (tensor<384x2xf32>) -> tensor<1x384x2xf32>
    %5388 = "mhlo.transpose"(%5387) {permutation = dense<[2, 0, 1]> : tensor<3xi64>} : (tensor<1x384x2xf32>) -> tensor<2x1x384xf32>
    %5389 = "mhlo.slice"(%5388) {limit_indices = dense<[1, 1, 384]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x1x384xf32>) -> tensor<1x1x384xf32>
    %5390 = "mhlo.reshape"(%5389) : (tensor<1x1x384xf32>) -> tensor<1x384xf32>
    %5391 = "mhlo.slice"(%5388) {limit_indices = dense<[2, 1, 384]> : tensor<3xi64>, start_indices = dense<[1, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x1x384xf32>) -> tensor<1x1x384xf32>
    %5392 = "mhlo.reshape"(%5391) : (tensor<1x1x384xf32>) -> tensor<1x384xf32>
    check.expect_almost_eq_const(%5390, dense<895.1307> : tensor<1x384xf32>) : tensor<1x384xf32>
    check.expect_almost_eq_const(%5392, dense<895.1307> : tensor<1x384xf32>) : tensor<1x384xf32>
    return
  }
}
