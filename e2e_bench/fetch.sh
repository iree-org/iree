#!/bin/bash

# wget -O EfficientNetV2SPT.mlirbc https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/EFFICIENTNET_V2_S/batch_1/linalg.mlir &
# cat<<EOF > EfficientNetV2SPT.mlirbc.run_flag
# --function=forward
# --input=1x3x384x384xf32=0
# EOF

wget -O BertLargeTF_Batch1.mlirbc https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/BertLargeTF_2023-05-07.timestamp_1683504734.mlirbc &
cat<<EOF > BertLargeTF_Batch1.mlirbc.run_flag
--function=serving_default
--input=1x384xi32=0
--input=1x384xi32=0
--input=1x384xi32=0
EOF

wget -O BertLargeTF_Batch32.mlirbc https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/BERT_LARGE_FP32_TF_384XI32_BATCH32/stablehlo.mlirbc &
cat<<EOF > BertLargeTF_Batch32.mlirbc.run_flag
--function=forward
--input=32x384xi32=0
--input=32x384xi32=0
--input=32x384xi32=0
EOF

wget -O T5LargeTF_Batch1.mlirbc https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/T5_LARGE_FP32_TF_512XI32_BATCH1/stablehlo.mlirbc &
cat<<EOF > T5LargeTF_Batch1.mlirbc.run_flag
--function=forward
--input=1x512xi32=0
--input=1x512xi32=0
EOF

wget -O T5LargeTF_Batch32.mlirbc https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/T5_LARGE_FP32_TF_512XI32_BATCH32/stablehlo.mlirbc &
cat<<EOF > T5LargeTF_Batch32.mlirbc.run_flag
--function=forward
--input=32x512xi32=0
--input=32x512xi32=0
EOF

wait
