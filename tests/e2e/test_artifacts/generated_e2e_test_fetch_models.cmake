iree_fetch_artifact(
  NAME "model-MobileNetV2_int8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224_quantized.tflite"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MobileNetV2_int8.0_224_quantized.tflite"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-EfficientNetV2STF"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/EfficientNetV2STF_2023-05-07.timestamp_1683504734.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-MiniLML12H384Uncased"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/MiniLML12H384Uncased_2023-05-07.timestamp_1683504734.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-EfficientNetV2SPT"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/EFFICIENTNET_V2_S/batch_1/linalg.mlir"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2SPT.mlir"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertForMaskedLMTF"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/BertForMaskedLMTF_2023-05-07.timestamp_1683504734.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargeTF"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual/BertLargeTF_2023-05-07.timestamp_1683504734.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-EfficientNetB7PT"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/EFFICIENTNET_B7/batch_1/linalg.mlir"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_EfficientNetB7PT.mlir"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargeTFBatch1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/BERT_LARGE_FP32_TF_384XI32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargeTFBatch32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/BERT_LARGE_FP32_TF_384XI32_BATCH32/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch32.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargeTFBatch64"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/BERT_LARGE_FP32_TF_384XI32_BATCH64/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch64.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50TFBatch1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/RESNET50_FP32_TF_224X224X3XF32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50TFBatch64"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/RESNET50_FP32_TF_224X224X3XF32_BATCH64/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch64.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50TFBatch128"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/RESNET50_FP32_TF_224X224X3XF32_BATCH128/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch128.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5LargeTFBatch1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/T5_LARGE_FP32_TF_512XI32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5LargeTFBatch16"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/T5_LARGE_FP32_TF_512XI32_BATCH16/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch16.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5LargeTFBatch32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/T5_LARGE_FP32_TF_512XI32_BATCH32/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch32.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-ClipTextSeqLen64PT"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230307.103_1678163233/SD_CLIP_TEXT_MODEL_SEQLEN64/linalg.mlir"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_ClipTextSeqLen64PT.mlir"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Unet2dPT"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230307.103_1678163233/SD_UNET_MODEL/linalg.mlir"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Unet2dPT.mlir"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargefp16PTBatch1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/BERT_LARGE_FP16_PT_384XI32_BATCH1/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-EfficientNetV2Sfp16PT"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684831160/EFFICIENTNET_V2_S_FP16/batch_1/linalg.mlir"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2Sfp16PT.mlir"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_3456x1024x2048_f16t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f16t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_3456x1024x2048_f32t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_3456x1024x2048_f32t_f32t_f32t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f32t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_2560x2560x2560_f16t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_2560x2560x2560_f16t_f16t_f16t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f16t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_2560x2560x2560_f32t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_2560x2560x2560_f32t_f32t_f32t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f32t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230525_1685058259/matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230525_1685058259/matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230525_1685058259/matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230612_1686563210/matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_128x256x8192_f16t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_128x256x8192_f16t_f16t_f16t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f16t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-matmul_128x256x8192_f32t_tile_config_default"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_128x256x8192_f32t_f32t_f32t_tile_config_default.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f32t_tile_config_default.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50TFBatch8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/RESNET50_FP32_TF_224X224X3XF32_BATCH8/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch8.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50TFBatch256"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/RESNET50_FP32_TF_224X224X3XF32_BATCH256/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch256.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargeTFBatch16"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/BERT_LARGE_FP32_TF_384XI32_BATCH16/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch16.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargeTFBatch24"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/BERT_LARGE_FP32_TF_384XI32_BATCH24/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch24.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargeTFBatch48"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/BERT_LARGE_FP32_TF_384XI32_BATCH48/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch48.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5LargeTFBatch24"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975/T5_LARGE_FP32_TF_512XI32_BATCH24/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch24.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargePTBatch1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/BERT_LARGE_FP32_PT_384XI32_BATCH1/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargePTBatch16"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/BERT_LARGE_FP32_PT_384XI32_BATCH16/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch16.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargePTBatch24"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/BERT_LARGE_FP32_PT_384XI32_BATCH24/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch24.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargePTBatch32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/BERT_LARGE_FP32_PT_384XI32_BATCH32/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch32.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargePTBatch48"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/BERT_LARGE_FP32_PT_384XI32_BATCH48/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargePTBatch48.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50PTBatch1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/RESNET50_FP32_PT_3X224X224XF32_BATCH1/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50PTBatch8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/RESNET50_FP32_PT_3X224X224XF32_BATCH8/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch8.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50PTBatch64"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/RESNET50_FP32_PT_3X224X224XF32_BATCH64/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch64.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50PTBatch128"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/RESNET50_FP32_PT_3X224X224XF32_BATCH128/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch128.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50PTBatch256"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230813.929_1692010793/RESNET50_FP32_PT_3X224X224XF32_BATCH256/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50PTBatch256.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50fp16PTBatch1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/RESNET50_FP16_PT_3X224X224XF16_BATCH1/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50fp16PTBatch8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/RESNET50_FP16_PT_3X224X224XF16_BATCH8/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch8.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50fp16PTBatch64"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/RESNET50_FP16_PT_3X224X224XF16_BATCH64/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch64.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50fp16PTBatch128"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/RESNET50_FP16_PT_3X224X224XF16_BATCH128/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch128.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-Resnet50fp16PTBatch256"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/RESNET50_FP16_PT_3X224X224XF16_BATCH256/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_Resnet50fp16PTBatch256.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargefp16PTBatch16"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/BERT_LARGE_FP16_PT_384XI32_BATCH16/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch16.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargefp16PTBatch24"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/BERT_LARGE_FP16_PT_384XI32_BATCH24/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch24.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargefp16PTBatch32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/BERT_LARGE_FP16_PT_384XI32_BATCH32/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch32.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BertLargefp16PTBatch48"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/BERT_LARGE_FP16_PT_384XI32_BATCH48/linalg.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BertLargefp16PTBatch48.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BERT_LARGE_JAX_384XI32_BATCH1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/BERT_LARGE_FP32_JAX_384XI32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BERT_LARGE_JAX_384XI32_BATCH16"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/BERT_LARGE_FP32_JAX_384XI32_BATCH16/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH16.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BERT_LARGE_JAX_384XI32_BATCH24"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/BERT_LARGE_FP32_JAX_384XI32_BATCH24/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH24.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BERT_LARGE_JAX_384XI32_BATCH32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/BERT_LARGE_FP32_JAX_384XI32_BATCH32/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH32.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-BERT_LARGE_JAX_384XI32_BATCH48"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/BERT_LARGE_FP32_JAX_384XI32_BATCH48/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_BERT_LARGE_JAX_384XI32_BATCH48.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-RESNET50_FP32_JAX_3X224X224XF32_BATCH1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/RESNET50_FP32_JAX_3X224X224XF32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-RESNET50_FP32_JAX_3X224X224XF32_BATCH8"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/RESNET50_FP32_JAX_3X224X224XF32_BATCH8/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH8.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-RESNET50_FP32_JAX_3X224X224XF32_BATCH64"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/RESNET50_FP32_JAX_3X224X224XF32_BATCH64/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH64.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-RESNET50_FP32_JAX_3X224X224XF32_BATCH128"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/RESNET50_FP32_JAX_3X224X224XF32_BATCH128/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH128.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-RESNET50_FP32_JAX_3X224X224XF32_BATCH256"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/RESNET50_FP32_JAX_3X224X224XF32_BATCH256/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_RESNET50_FP32_JAX_3X224X224XF32_BATCH256.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5_LARGE_FP32_JAX_512XI32_BATCH1"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/T5_LARGE_FP32_JAX_512XI32_BATCH1/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH1.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5_LARGE_FP32_JAX_512XI32_BATCH16"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/T5_LARGE_FP32_JAX_512XI32_BATCH16/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH16.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5_LARGE_FP32_JAX_512XI32_BATCH24"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/T5_LARGE_FP32_JAX_512XI32_BATCH24/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH24.mlirbc"
  UNPACK
)

iree_fetch_artifact(
  NAME "model-T5_LARGE_FP32_JAX_512XI32_BATCH32"
  SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.14_1691969180/T5_LARGE_FP32_JAX_512XI32_BATCH32/stablehlo.mlirbc"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/model_T5_LARGE_FP32_JAX_512XI32_BATCH32.mlirbc"
  UNPACK
)
