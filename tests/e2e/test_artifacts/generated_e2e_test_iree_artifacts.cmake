iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32.0_float.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.0_224.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
)

iree_import_tf_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v2"
    "--tf-savedmodel-exported-names=forward"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
)

iree_import_tf_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_8871f602-571c-4eb8-b94d-554cc8ceec5a_BertLargeTF"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v1"
    "--tf-savedmodel-exported-names=serving_default"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
)

iree_import_tf_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v2"
    "--tf-savedmodel-exported-names=forward"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
)

iree_import_tf_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v2"
    "--tf-savedmodel-exported-names=predict"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
)

iree_import_tf_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_c393b4fa-beb4-45d5-982a-c6328aa05d08_Resnet50TF"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v2"
    "--tf-savedmodel-exported-names=forward"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
)

iree_bytecode_module(
  NAME "iree-module-87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-bdd904cc5614ebf77609c7802a2dfc09f139aee2a247a247d10d320de72b0e28"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_bdd904cc5614ebf77609c7802a2dfc09f139aee2a247a247d10d320de72b0e28/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_bdd904cc5614ebf77609c7802a2dfc09f139aee2a247a247d10d320de72b0e28/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-45565cae821666fd34bca97be2e4cce3bd61e71308785728737d89acbb9bc9d2"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_45565cae821666fd34bca97be2e4cce3bd61e71308785728737d89acbb9bc9d2/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_45565cae821666fd34bca97be2e4cce3bd61e71308785728737d89acbb9bc9d2/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-04ca0a5077b7dd5ace66d803c9b822dff3428b24e7620a61995aff0907af9533"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_04ca0a5077b7dd5ace66d803c9b822dff3428b24e7620a61995aff0907af9533/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_04ca0a5077b7dd5ace66d803c9b822dff3428b24e7620a61995aff0907af9533/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-deafafd0926321a4b8e4dc73ed4a30b2ed9317d26488246461415be2ee857eb1"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_deafafd0926321a4b8e4dc73ed4a30b2ed9317d26488246461415be2ee857eb1/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_deafafd0926321a4b8e4dc73ed4a30b2ed9317d26488246461415be2ee857eb1/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-fd81a89e9f8773bae142040775c7e3c4774f96b64f07f8d9f66b00191864ff40"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_fd81a89e9f8773bae142040775c7e3c4774f96b64f07f8d9f66b00191864ff40/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_fd81a89e9f8773bae142040775c7e3c4774f96b64f07f8d9f66b00191864ff40/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-68f0eb37bb72d0d6605ecdf42691c64125960e122844b0beeae350871a445b1c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_68f0eb37bb72d0d6605ecdf42691c64125960e122844b0beeae350871a445b1c/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_68f0eb37bb72d0d6605ecdf42691c64125960e122844b0beeae350871a445b1c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-a7a1553d0739151f06bbc00a3ef8b67b0606463eab4b6607069aa94ea0bfd92f"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_a7a1553d0739151f06bbc00a3ef8b67b0606463eab4b6607069aa94ea0bfd92f/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_a7a1553d0739151f06bbc00a3ef8b67b0606463eab4b6607069aa94ea0bfd92f/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e80d71ed8e86c0756226b2323e27e2c7c0fff8eddde59ba69e9222d36ee3eef6"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_e80d71ed8e86c0756226b2323e27e2c7c0fff8eddde59ba69e9222d36ee3eef6/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_e80d71ed8e86c0756226b2323e27e2c7c0fff8eddde59ba69e9222d36ee3eef6/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-f59cd43a2a2d6e4b3159efa358a6fa0879e72f6f4f0a23af4c8ab550f256986a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_f59cd43a2a2d6e4b3159efa358a6fa0879e72f6f4f0a23af4c8ab550f256986a/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_f59cd43a2a2d6e4b3159efa358a6fa0879e72f6f4f0a23af4c8ab550f256986a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-14a15b9072caaee5e2a274a9bbc436a56d095611e5a8e9841f110741d34231f9"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_14a15b9072caaee5e2a274a9bbc436a56d095611e5a8e9841f110741d34231f9/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_14a15b9072caaee5e2a274a9bbc436a56d095611e5a8e9841f110741d34231f9/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e850fce2d36ddb09ccc34471641adb77418b93c0949d22ab75806d7cfc489ae3"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_e850fce2d36ddb09ccc34471641adb77418b93c0949d22ab75806d7cfc489ae3/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_e850fce2d36ddb09ccc34471641adb77418b93c0949d22ab75806d7cfc489ae3/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-f963d812114af925e0a4b110ee83aeb0e3b41d49fad19b3f449b6a9ccba43b8d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_f963d812114af925e0a4b110ee83aeb0e3b41d49fad19b3f449b6a9ccba43b8d/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv32-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv32" "--iree-llvm-target-abi=ilp32" "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_f963d812114af925e0a4b110ee83aeb0e3b41d49fad19b3f449b6a9ccba43b8d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9d909aa679e8c380ff7e93292ef28dbd3bb9e7cc62329f90aef78ce9c7efeff9"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9d909aa679e8c380ff7e93292ef28dbd3bb9e7cc62329f90aef78ce9c7efeff9/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv32-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv32" "--iree-llvm-target-abi=ilp32" "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9d909aa679e8c380ff7e93292ef28dbd3bb9e7cc62329f90aef78ce9c7efeff9/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-1ef2da238443010024d69ceb6fe6ab6fa8cf5f4ce7d424dace3a572592043e70"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_1ef2da238443010024d69ceb6fe6ab6fa8cf5f4ce7d424dace3a572592043e70/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv32-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv32" "--iree-llvm-target-abi=ilp32" "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f" "--riscv-v-fixed-length-vector-lmul-max=8"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_1ef2da238443010024d69ceb6fe6ab6fa8cf5f4ce7d424dace3a572592043e70/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-llvm-target-cpu-features=+dotprod" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-deba344af418957cbd9dc0834a100bc30ba242d7fddb4dc6ba10a87d0af32dc1"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_deba344af418957cbd9dc0834a100bc30ba242d7fddb4dc6ba10a87d0af32dc1/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_deba344af418957cbd9dc0834a100bc30ba242d7fddb4dc6ba10a87d0af32dc1/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-72feea41a0b54e4c9a761933079cba1b2c012e5a5d4b2953ffaa86faaa29a648"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_72feea41a0b54e4c9a761933079cba1b2c012e5a5d4b2953ffaa86faaa29a648/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_72feea41a0b54e4c9a761933079cba1b2c012e5a5d4b2953ffaa86faaa29a648/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-b33d5ca3311e31b99daa1c1f13ea5571713538a7889627153ea431debb9b5e2a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_b33d5ca3311e31b99daa1c1f13ea5571713538a7889627153ea431debb9b5e2a/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_b33d5ca3311e31b99daa1c1f13ea5571713538a7889627153ea431debb9b5e2a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-076a1e95f3384b58a77a672c7c36463b091e574b5a6f6eaf78841537b0d1c930"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_076a1e95f3384b58a77a672c7c36463b091e574b5a6f6eaf78841537b0d1c930/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_076a1e95f3384b58a77a672c7c36463b091e574b5a6f6eaf78841537b0d1c930/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-05e976592f58a292874d99bd7627e655b15c5460455a08b9ce67e9f7f65b6269"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_05e976592f58a292874d99bd7627e655b15c5460455a08b9ce67e9f7f65b6269/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_05e976592f58a292874d99bd7627e655b15c5460455a08b9ce67e9f7f65b6269/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-dd4366d716cccdd83b6f777ee966157b92838569f094371b325b926e73c7b1b8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_dd4366d716cccdd83b6f777ee966157b92838569f094371b325b926e73c7b1b8/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_dd4366d716cccdd83b6f777ee966157b92838569f094371b325b926e73c7b1b8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-4a1632637ce87fe991848942b028c36732b2bea00920d275ffbcaf2cd9446152"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_4a1632637ce87fe991848942b028c36732b2bea00920d275ffbcaf2cd9446152/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_4a1632637ce87fe991848942b028c36732b2bea00920d275ffbcaf2cd9446152/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7f5a7ca6eb12e8ac322d8bb0deb59630d721ab141acc3a941e168d98af507034"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7f5a7ca6eb12e8ac322d8bb0deb59630d721ab141acc3a941e168d98af507034/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7f5a7ca6eb12e8ac322d8bb0deb59630d721ab141acc3a941e168d98af507034/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e3444362e0b630df1b5f70a28089b0764d9ddc886dda852a0ef1300e369aee4d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_e3444362e0b630df1b5f70a28089b0764d9ddc886dda852a0ef1300e369aee4d/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_e3444362e0b630df1b5f70a28089b0764d9ddc886dda852a0ef1300e369aee4d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c95d82463222cf5ea385760eb1f0f0ee28a876620e29fbd59f8f4cb8a5307bc8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_c95d82463222cf5ea385760eb1f0f0ee28a876620e29fbd59f8f4cb8a5307bc8/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_c95d82463222cf5ea385760eb1f0f0ee28a876620e29fbd59f8f4cb8a5307bc8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-f73d0b9c84ec91b594b00eb3800c372884050fce3e4fb9d80eb407d7b0697412"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_f73d0b9c84ec91b594b00eb3800c372884050fce3e4fb9d80eb407d7b0697412/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_f73d0b9c84ec91b594b00eb3800c372884050fce3e4fb9d80eb407d7b0697412/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-67ad72cbb9eb9c4746249922e6232b1a17b5d6eeabd9b69ed4d527c1676c77bd"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_67ad72cbb9eb9c4746249922e6232b1a17b5d6eeabd9b69ed4d527c1676c77bd/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_67ad72cbb9eb9c4746249922e6232b1a17b5d6eeabd9b69ed4d527c1676c77bd/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ad38a059079822e7331470e086bc1caca3dbe435878b38e1355229a39d1d25d2"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_ad38a059079822e7331470e086bc1caca3dbe435878b38e1355229a39d1d25d2/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_ad38a059079822e7331470e086bc1caca3dbe435878b38e1355229a39d1d25d2/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8eb82c90485cc4281676866b62e6820b60a38ba81068e95254a7d0ddaddc59c3"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_8eb82c90485cc4281676866b62e6820b60a38ba81068e95254a7d0ddaddc59c3/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_8eb82c90485cc4281676866b62e6820b60a38ba81068e95254a7d0ddaddc59c3/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-cc9b305fb2c95f582d144f1063fc12fd996e757f84738e3de846b0197981bcc2"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_cc9b305fb2c95f582d144f1063fc12fd996e757f84738e3de846b0197981bcc2/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_cc9b305fb2c95f582d144f1063fc12fd996e757f84738e3de846b0197981bcc2/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8930c217bc20a5abf9313fd714a77bf47acefb006e95ba07b5e48caf872541b0"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_8930c217bc20a5abf9313fd714a77bf47acefb006e95ba07b5e48caf872541b0/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_8930c217bc20a5abf9313fd714a77bf47acefb006e95ba07b5e48caf872541b0/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9b6e19b8cab376fffe309f090beeecad08903c0975c9e7ffc480dd46074b97b3"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_9b6e19b8cab376fffe309f090beeecad08903c0975c9e7ffc480dd46074b97b3/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_9b6e19b8cab376fffe309f090beeecad08903c0975c9e7ffc480dd46074b97b3/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-d5ea172c189c6a6a3b61a7e83f7263b9b38cad756d2d5dec1b2db88162eece2a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_d5ea172c189c6a6a3b61a7e83f7263b9b38cad756d2d5dec1b2db88162eece2a/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_d5ea172c189c6a6a3b61a7e83f7263b9b38cad756d2d5dec1b2db88162eece2a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-bb1cb8e00fd4cee513b2481bd5faf39842ad9c57f80f2e50ddb48763fd030721"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_bb1cb8e00fd4cee513b2481bd5faf39842ad9c57f80f2e50ddb48763fd030721/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_bb1cb8e00fd4cee513b2481bd5faf39842ad9c57f80f2e50ddb48763fd030721/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-dc26eca15bb97d42bcfa019de5080da54fb3d7624809aa1c8ac731e710544e18"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_dc26eca15bb97d42bcfa019de5080da54fb3d7624809aa1c8ac731e710544e18/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_dc26eca15bb97d42bcfa019de5080da54fb3d7624809aa1c8ac731e710544e18/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-90d4cb308fc14e4832336ebd1e1387a791a20b92af633d0fec9161b807e13427"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_90d4cb308fc14e4832336ebd1e1387a791a20b92af633d0fec9161b807e13427/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_90d4cb308fc14e4832336ebd1e1387a791a20b92af633d0fec9161b807e13427/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ba322b423888fe93e86180e98668f486afa915565edf3813af3f9da6ad7c9dc9"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_ba322b423888fe93e86180e98668f486afa915565edf3813af3f9da6ad7c9dc9/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_ba322b423888fe93e86180e98668f486afa915565edf3813af3f9da6ad7c9dc9/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-42bc978e75d5fed90c10d5812a8df7a6153e311b19b9c7faf41a588bdc4da7d8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_42bc978e75d5fed90c10d5812a8df7a6153e311b19b9c7faf41a588bdc4da7d8/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_42bc978e75d5fed90c10d5812a8df7a6153e311b19b9c7faf41a588bdc4da7d8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-6819b99fa39a0e548de5fdfc59e8ef1b3c609cd7b5cbe9c0c765911e4eb50cbd"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_6819b99fa39a0e548de5fdfc59e8ef1b3c609cd7b5cbe9c0c765911e4eb50cbd/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_6819b99fa39a0e548de5fdfc59e8ef1b3c609cd7b5cbe9c0c765911e4eb50cbd/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-0732f1387e308186fd8c8d8089131799be227d5a656a446048f7e0fd0d5047ce"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_0732f1387e308186fd8c8d8089131799be227d5a656a446048f7e0fd0d5047ce/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_0732f1387e308186fd8c8d8089131799be227d5a656a446048f7e0fd0d5047ce/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-b30a25fc50be09814337382f4cf8a1fd03b8c79d8b860ac509c629f8c8a5e5f0"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_b30a25fc50be09814337382f4cf8a1fd03b8c79d8b860ac509c629f8c8a5e5f0/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_b30a25fc50be09814337382f4cf8a1fd03b8c79d8b860ac509c629f8c8a5e5f0/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-a47669d662bd18dbd323737d84fd45753c01abaa81b4761eb51b35ce04ee7491"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_a47669d662bd18dbd323737d84fd45753c01abaa81b4761eb51b35ce04ee7491/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_a47669d662bd18dbd323737d84fd45753c01abaa81b4761eb51b35ce04ee7491/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-27c12f520d68a8a754b11ad44cf8ad3ef5c1ec281fc9d84fbc64a7b573cb68b5"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_27c12f520d68a8a754b11ad44cf8ad3ef5c1ec281fc9d84fbc64a7b573cb68b5/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_27c12f520d68a8a754b11ad44cf8ad3ef5c1ec281fc9d84fbc64a7b573cb68b5/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ec14aafacf15628918390531318a4827d3ac8771b75d60a8239d901dcb4fd898"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_ec14aafacf15628918390531318a4827d3ac8771b75d60a8239d901dcb4fd898/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_ec14aafacf15628918390531318a4827d3ac8771b75d60a8239d901dcb4fd898/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-507fabae3d24c2448c8f2af676d68df0a22716a06fa7ffc3b4f865e9272ecdc8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_507fabae3d24c2448c8f2af676d68df0a22716a06fa7ffc3b4f865e9272ecdc8/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_507fabae3d24c2448c8f2af676d68df0a22716a06fa7ffc3b4f865e9272ecdc8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-4924918e121a92f802a094de2d67e9c2673c9fdc39faa6a11ac1d38b631a2914"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_4924918e121a92f802a094de2d67e9c2673c9fdc39faa6a11ac1d38b631a2914/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_4924918e121a92f802a094de2d67e9c2673c9fdc39faa6a11ac1d38b631a2914/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-383e7567e79adf2178e3d25785855905a4477b6c8daf16bc50d5f9f44d2343d9"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_383e7567e79adf2178e3d25785855905a4477b6c8daf16bc50d5f9f44d2343d9/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_383e7567e79adf2178e3d25785855905a4477b6c8daf16bc50d5f9f44d2343d9/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9faad3c50bc64c297531a36bd2ab235b680070d8233ce5ab7d964ac36c7c5563"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_9faad3c50bc64c297531a36bd2ab235b680070d8233ce5ab7d964ac36c7c5563/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_9faad3c50bc64c297531a36bd2ab235b680070d8233ce5ab7d964ac36c7c5563/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-175e3e5c14dc32cafe73af4b4b8a6f5732697a097cdf2e8699a316224afb7e31"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_175e3e5c14dc32cafe73af4b4b8a6f5732697a097cdf2e8699a316224afb7e31/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_175e3e5c14dc32cafe73af4b4b8a6f5732697a097cdf2e8699a316224afb7e31/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8964c792ed5c95032e2318b2dbac3b4f8453cde352067fa9a029d04a0c2d5fae"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_8964c792ed5c95032e2318b2dbac3b4f8453cde352067fa9a029d04a0c2d5fae/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_8964c792ed5c95032e2318b2dbac3b4f8453cde352067fa9a029d04a0c2d5fae/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c8ee4f2bcdb954fa5d5d45c0c65631ce211c0293d7a69ed6ef4644e523e919ac"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_c8ee4f2bcdb954fa5d5d45c0c65631ce211c0293d7a69ed6ef4644e523e919ac/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_c8ee4f2bcdb954fa5d5d45c0c65631ce211c0293d7a69ed6ef4644e523e919ac/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-131aa61ffcacafc6ae701ff89045cf42d2490ac0c4a1d862bc83c23edb3b92e5"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_131aa61ffcacafc6ae701ff89045cf42d2490ac0c4a1d862bc83c23edb3b92e5/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_131aa61ffcacafc6ae701ff89045cf42d2490ac0c4a1d862bc83c23edb3b92e5/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-db9784471b47ae8bf55ca7e0821e35a1686256a208df40443c114f1adcdd26f6"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_db9784471b47ae8bf55ca7e0821e35a1686256a208df40443c114f1adcdd26f6/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_db9784471b47ae8bf55ca7e0821e35a1686256a208df40443c114f1adcdd26f6/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-aada5dcefdd361b5227276129e93547a6932a05d380acd342fa33e5b672d498c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_aada5dcefdd361b5227276129e93547a6932a05d380acd342fa33e5b672d498c/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_aada5dcefdd361b5227276129e93547a6932a05d380acd342fa33e5b672d498c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-887c7a7b540f11ee5e0158143fd46a503a4851211b10b353ec0e00b8b1beb575"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_887c7a7b540f11ee5e0158143fd46a503a4851211b10b353ec0e00b8b1beb575/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_887c7a7b540f11ee5e0158143fd46a503a4851211b10b353ec0e00b8b1beb575/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-11766f32ea6a3121d7527bcdf32dead45ab7b3922d72addb46945cfdab784ec0"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_11766f32ea6a3121d7527bcdf32dead45ab7b3922d72addb46945cfdab784ec0/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_11766f32ea6a3121d7527bcdf32dead45ab7b3922d72addb46945cfdab784ec0/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-0efb794c3a385045fa4d3d086c2a593ce67c4807e9456271f05f4e28490d1c49"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0efb794c3a385045fa4d3d086c2a593ce67c4807e9456271f05f4e28490d1c49/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0efb794c3a385045fa4d3d086c2a593ce67c4807e9456271f05f4e28490d1c49/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7a3d36d7234ce4abfd833964492631bd81df823e6bde8cd3a1fadfa4faa7c787"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_7a3d36d7234ce4abfd833964492631bd81df823e6bde8cd3a1fadfa4faa7c787/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_7a3d36d7234ce4abfd833964492631bd81df823e6bde8cd3a1fadfa4faa7c787/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-77bedc82b9083aa7c270bc37d077f6ff4cecabc307a584a9b7b52e5fe18db858"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_77bedc82b9083aa7c270bc37d077f6ff4cecabc307a584a9b7b52e5fe18db858/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_77bedc82b9083aa7c270bc37d077f6ff4cecabc307a584a9b7b52e5fe18db858/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-a4a762fde005b81cec5b11ddf02fc7ac4bb919b71250ffefdd2cecda209ceeaa"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_a4a762fde005b81cec5b11ddf02fc7ac4bb919b71250ffefdd2cecda209ceeaa/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_a4a762fde005b81cec5b11ddf02fc7ac4bb919b71250ffefdd2cecda209ceeaa/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-43147a614fc5476e5a8083ddefc3ecb093c9c9ebf7864b8b6178af952540edae"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_43147a614fc5476e5a8083ddefc3ecb093c9c9ebf7864b8b6178af952540edae/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-flow-demote-f32-to-f16"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_43147a614fc5476e5a8083ddefc3ecb093c9c9ebf7864b8b6178af952540edae/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26/module.vmfb"
  FLAGS "--iree-hal-target-backends=vmvx" "--iree-input-type=tosa"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed/module.vmfb"
  FLAGS "--iree-hal-target-backends=vmvx" "--iree-input-type=tosa"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-2f2e448f73ef190ed35af1b25b6179ce15faba7ee7c12f4956730c441e9a27bd"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_2f2e448f73ef190ed35af1b25b6179ce15faba7ee7c12f4956730c441e9a27bd/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_2f2e448f73ef190ed35af1b25b6179ce15faba7ee7c12f4956730c441e9a27bd/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c370b55d34f6d3c76aa838ff0a7be520de10a4824c5feaa773e2fb73a588ad8c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_c370b55d34f6d3c76aa838ff0a7be520de10a4824c5feaa773e2fb73a588ad8c/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_c370b55d34f6d3c76aa838ff0a7be520de10a4824c5feaa773e2fb73a588ad8c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-5a4c96fc279262ad7d7f1d446d0bd3685b2ca42e06b0167df5be5737c9d42901"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_5a4c96fc279262ad7d7f1d446d0bd3685b2ca42e06b0167df5be5737c9d42901/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_5a4c96fc279262ad7d7f1d446d0bd3685b2ca42e06b0167df5be5737c9d42901/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-27bbe62536a23529b4dd0df3d4913ee18344df9b6e2a32fc834fb7d9bc520e24"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_27bbe62536a23529b4dd0df3d4913ee18344df9b6e2a32fc834fb7d9bc520e24/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_27bbe62536a23529b4dd0df3d4913ee18344df9b6e2a32fc834fb7d9bc520e24/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-78511a42a50f705b944437a040e1ee3bb5b2595a3b1d4db788586fe48f9a2453"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_78511a42a50f705b944437a040e1ee3bb5b2595a3b1d4db788586fe48f9a2453/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_78511a42a50f705b944437a040e1ee3bb5b2595a3b1d4db788586fe48f9a2453/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ef1ba1216f0f304c80b7a5b8bac545a987d04a100d9c1e5e66b75ce88636534c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_ef1ba1216f0f304c80b7a5b8bac545a987d04a100d9c1e5e66b75ce88636534c/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_ef1ba1216f0f304c80b7a5b8bac545a987d04a100d9c1e5e66b75ce88636534c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-1c4bc4b5ba3b5862efdbcbb9b3bf4a02f7ff9aa36e852e9b94dbe265d6bfaa99"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_1c4bc4b5ba3b5862efdbcbb9b3bf4a02f7ff9aa36e852e9b94dbe265d6bfaa99/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_1c4bc4b5ba3b5862efdbcbb9b3bf4a02f7ff9aa36e852e9b94dbe265d6bfaa99/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-439f7c958ce1d3200ea96935174cabde8e8fe6917a007f5e238553e9c2aa7625"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_439f7c958ce1d3200ea96935174cabde8e8fe6917a007f5e238553e9c2aa7625/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_439f7c958ce1d3200ea96935174cabde8e8fe6917a007f5e238553e9c2aa7625/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-4a3b570ba18c3c9eee458455aaff4aa29293a5c936a19862c698b4b3ddaf06e7"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_4a3b570ba18c3c9eee458455aaff4aa29293a5c936a19862c698b4b3ddaf06e7/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_4a3b570ba18c3c9eee458455aaff4aa29293a5c936a19862c698b4b3ddaf06e7/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-28e38bd436b036babc0fabe98b6e7c68ca3a7088e73dffff2c538adfa7d6af4c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_28e38bd436b036babc0fabe98b6e7c68ca3a7088e73dffff2c538adfa7d6af4c/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_28e38bd436b036babc0fabe98b6e7c68ca3a7088e73dffff2c538adfa7d6af4c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-a05a2b521a968e99411712e0e5191c3cd1d6295991f3b78acf61faca5d1cf85e"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_a05a2b521a968e99411712e0e5191c3cd1d6295991f3b78acf61faca5d1cf85e/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_a05a2b521a968e99411712e0e5191c3cd1d6295991f3b78acf61faca5d1cf85e/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ddd1657bc5433ccca5c8ce562f581626457a793670958cd8b4016c426191a9c4"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_ddd1657bc5433ccca5c8ce562f581626457a793670958cd8b4016c426191a9c4/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_ddd1657bc5433ccca5c8ce562f581626457a793670958cd8b4016c426191a9c4/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8ee3c7b136703472b53bc8a19d8d28945aca93953612ccc65e55cd1b3dfda6c8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_8ee3c7b136703472b53bc8a19d8d28945aca93953612ccc65e55cd1b3dfda6c8/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_8ee3c7b136703472b53bc8a19d8d28945aca93953612ccc65e55cd1b3dfda6c8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-01d35de2a55b9800e05151455eace0bf4493337ac1210fcc4904d630b075599a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_01d35de2a55b9800e05151455eace0bf4493337ac1210fcc4904d630b075599a/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_01d35de2a55b9800e05151455eace0bf4493337ac1210fcc4904d630b075599a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-2957930127e9b01e90ccddb7290e1c4b4abf6373cc36929809040e2c144d3fd7"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_2957930127e9b01e90ccddb7290e1c4b4abf6373cc36929809040e2c144d3fd7/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_2957930127e9b01e90ccddb7290e1c4b4abf6373cc36929809040e2c144d3fd7/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-846b19afd4c14b3e71d59087c5a2987edd65753d39db432961ce915688d457ac"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_846b19afd4c14b3e71d59087c5a2987edd65753d39db432961ce915688d457ac/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_846b19afd4c14b3e71d59087c5a2987edd65753d39db432961ce915688d457ac/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-de34105293194986d706823bd3d20ce784506ec5918c4d0efac9839020bb5fdd"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_de34105293194986d706823bd3d20ce784506ec5918c4d0efac9839020bb5fdd/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_de34105293194986d706823bd3d20ce784506ec5918c4d0efac9839020bb5fdd/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-373b890bed4c0f4828b957e37d319509bf41e39a4e47746285e27101d40f90bd"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_373b890bed4c0f4828b957e37d319509bf41e39a4e47746285e27101d40f90bd/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_373b890bed4c0f4828b957e37d319509bf41e39a4e47746285e27101d40f90bd/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-6e31f637a133e03db37c47d8a920a61306e366362e066f41c0eac0455cc6c77a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_6e31f637a133e03db37c47d8a920a61306e366362e066f41c0eac0455cc6c77a/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_6e31f637a133e03db37c47d8a920a61306e366362e066f41c0eac0455cc6c77a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e0533bdae79e15707a6eb26eb7f09c4d7dbdbfc40b993a4ad6289cf2bb1f13cb"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_e0533bdae79e15707a6eb26eb7f09c4d7dbdbfc40b993a4ad6289cf2bb1f13cb/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_e0533bdae79e15707a6eb26eb7f09c4d7dbdbfc40b993a4ad6289cf2bb1f13cb/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ad9a410e86dd9d649de58f5a7dbdc6cd2300fb6b6a363f4483e930d9944d2d07"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_ad9a410e86dd9d649de58f5a7dbdc6cd2300fb6b6a363f4483e930d9944d2d07/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_ad9a410e86dd9d649de58f5a7dbdc6cd2300fb6b6a363f4483e930d9944d2d07/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9b12e389535e365bd2c35424c5f98442e1226d73b043eb40355bf456ad0263a2"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_9b12e389535e365bd2c35424c5f98442e1226d73b043eb40355bf456ad0263a2/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_9b12e389535e365bd2c35424c5f98442e1226d73b043eb40355bf456ad0263a2/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-63d75ff4a9998a86855e0e78ab2d782f52b90b58025584f3f03ec3103a81425b"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_63d75ff4a9998a86855e0e78ab2d782f52b90b58025584f3f03ec3103a81425b/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_63d75ff4a9998a86855e0e78ab2d782f52b90b58025584f3f03ec3103a81425b/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-00a22e8ada401de8f20895beff9a153585e585c2d686983e27f9d64fdf7d39a8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_00a22e8ada401de8f20895beff9a153585e585c2d686983e27f9d64fdf7d39a8/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_00a22e8ada401de8f20895beff9a153585e585c2d686983e27f9d64fdf7d39a8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-4c74339076df00d23baa17dcb3194043e0472da9d09db4e42a23841ff7bf67b0"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_4c74339076df00d23baa17dcb3194043e0472da9d09db4e42a23841ff7bf67b0/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_4c74339076df00d23baa17dcb3194043e0472da9d09db4e42a23841ff7bf67b0/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9a1d228583ba1e56a19393f6938d16b5d582bb17f89fb5856b8b1c68e34abd45"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_9a1d228583ba1e56a19393f6938d16b5d582bb17f89fb5856b8b1c68e34abd45/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_9a1d228583ba1e56a19393f6938d16b5d582bb17f89fb5856b8b1c68e34abd45/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-152d0b6211fff7591df3418c549c979a8144fc34280c22a8b2b5ff8ea3d1b46c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_152d0b6211fff7591df3418c549c979a8144fc34280c22a8b2b5ff8ea3d1b46c/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_152d0b6211fff7591df3418c549c979a8144fc34280c22a8b2b5ff8ea3d1b46c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e16d3f99f851c11fef6be64c7f06a637b410618f2618cf16aa599b54ea8970e3"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_e16d3f99f851c11fef6be64c7f06a637b410618f2618cf16aa599b54ea8970e3/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_e16d3f99f851c11fef6be64c7f06a637b410618f2618cf16aa599b54ea8970e3/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8231a286cdc63a48f3f70a12ab5a182142c00cbebaccdc79e35ca552f02422e7"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_8231a286cdc63a48f3f70a12ab5a182142c00cbebaccdc79e35ca552f02422e7/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_8231a286cdc63a48f3f70a12ab5a182142c00cbebaccdc79e35ca552f02422e7/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c9a7c5b08db10ed782045b6810cb4ee157da9e95590456d3839c06163ee30fa7"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_c9a7c5b08db10ed782045b6810cb4ee157da9e95590456d3839c06163ee30fa7/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_c9a7c5b08db10ed782045b6810cb4ee157da9e95590456d3839c06163ee30fa7/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-838cc09b422958a332fd76cf12a6a2a95b8346c8e8d2fe7b15cb5ace4c20581e"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_838cc09b422958a332fd76cf12a6a2a95b8346c8e8d2fe7b15cb5ace4c20581e/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=mhlo" "--iree-llvm-target-triple=x86_64-unknown-linux-gnu" "--iree-llvm-target-cpu=cascadelake" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_838cc09b422958a332fd76cf12a6a2a95b8346c8e8d2fe7b15cb5ace4c20581e/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8b19868be1c797cb585551c871c4171e78817e0efc49d30d91b9d722be283de9"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_8b19868be1c797cb585551c871c4171e78817e0efc49d30d91b9d722be283de9/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_8b19868be1c797cb585551c871c4171e78817e0efc49d30d91b9d722be283de9/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c2085883b1f5c767f37508ab998a4bcd17d169fe6a5197d28e4dca8772c90253"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_c2085883b1f5c767f37508ab998a4bcd17d169fe6a5197d28e4dca8772c90253/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_c2085883b1f5c767f37508ab998a4bcd17d169fe6a5197d28e4dca8772c90253/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-25ad2815eb690276e9c2183aaafaf17a3df734bb6164071ad92dbf1e7faf7509"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_25ad2815eb690276e9c2183aaafaf17a3df734bb6164071ad92dbf1e7faf7509/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_25ad2815eb690276e9c2183aaafaf17a3df734bb6164071ad92dbf1e7faf7509/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-65586f1e5b51439dd951529c35fa9000a928f90039cc6cfb66d5c81d07a6c62b"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_65586f1e5b51439dd951529c35fa9000a928f90039cc6cfb66d5c81d07a6c62b/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_65586f1e5b51439dd951529c35fa9000a928f90039cc6cfb66d5c81d07a6c62b/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-f770b1916e0b7a9a0b4aa9480791d21a46a352002ac1e38dfcea49ec0b63ed4e"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_f770b1916e0b7a9a0b4aa9480791d21a46a352002ac1e38dfcea49ec0b63ed4e/module.vmfb"
  FLAGS "--iree-hal-target-backends=cuda" "--iree-input-type=mhlo" "--iree-hal-cuda-llvm-target-arch=sm_80" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_f770b1916e0b7a9a0b4aa9480791d21a46a352002ac1e38dfcea49ec0b63ed4e/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-16b5b80aaf1271b5ad782570340cc0c7c1c97e10b7e6c6cc6e5f3ede8393cb6c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_16b5b80aaf1271b5ad782570340cc0c7c1c97e10b7e6c6cc6e5f3ede8393cb6c/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_16b5b80aaf1271b5ad782570340cc0c7c1c97e10b7e6c6cc6e5f3ede8393cb6c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-65fa033050b916e8143d44b5081ee45db3b1946a5d77de223328a7fe92a1cc66"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_65fa033050b916e8143d44b5081ee45db3b1946a5d77de223328a7fe92a1cc66/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_65fa033050b916e8143d44b5081ee45db3b1946a5d77de223328a7fe92a1cc66/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-16ef56b6869d10b17e983fec62e9f48e6bb87e9a348ab52a0b2faabca2b03578"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_16ef56b6869d10b17e983fec62e9f48e6bb87e9a348ab52a0b2faabca2b03578/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_16ef56b6869d10b17e983fec62e9f48e6bb87e9a348ab52a0b2faabca2b03578/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-56bc9128e294585d749b8ebe34fd03060ba34d200eef185837b6002d0dcbfccb"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_56bc9128e294585d749b8ebe34fd03060ba34d200eef185837b6002d0dcbfccb/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_56bc9128e294585d749b8ebe34fd03060ba34d200eef185837b6002d0dcbfccb/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-eb1b1732e5d30ce4689b871f8ec18c50b30eedd418eb80330807fe505bb78f7e"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_eb1b1732e5d30ce4689b871f8ec18c50b30eedd418eb80330807fe505bb78f7e/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_eb1b1732e5d30ce4689b871f8ec18c50b30eedd418eb80330807fe505bb78f7e/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-bd32992875a8fc7a494c75933b1693d6d8b845fccf2b12504a8cba64d80ad110"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_bd32992875a8fc7a494c75933b1693d6d8b845fccf2b12504a8cba64d80ad110/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv64-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv64" "--iree-llvm-target-abi=lp64d" "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_bd32992875a8fc7a494c75933b1693d6d8b845fccf2b12504a8cba64d80ad110/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-ff7ed59e05efe8b9a397a179726f63da68a8a1ac3ea731924b4bd24dab491b34"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_ff7ed59e05efe8b9a397a179726f63da68a8a1ac3ea731924b4bd24dab491b34/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv32-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv32" "--iree-llvm-target-abi=ilp32" "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_ff7ed59e05efe8b9a397a179726f63da68a8a1ac3ea731924b4bd24dab491b34/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8e2d1286ad9a7e360b0c26019146a22ec9188f8bdf8ad99341eb5531cdea2417"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_8e2d1286ad9a7e360b0c26019146a22ec9188f8bdf8ad99341eb5531cdea2417/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv32-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv32" "--iree-llvm-target-abi=ilp32" "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_8e2d1286ad9a7e360b0c26019146a22ec9188f8bdf8ad99341eb5531cdea2417/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-d967e293594998e48355d30d900dbdf77dbd6eedbff768112dbe8e7ec332b9eb"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_d967e293594998e48355d30d900dbdf77dbd6eedbff768112dbe8e7ec332b9eb/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=riscv32-pc-linux-gnu" "--iree-llvm-target-cpu=generic-rv32" "--iree-llvm-target-abi=ilp32" "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f" "--riscv-v-fixed-length-vector-lmul-max=8" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_d967e293594998e48355d30d900dbdf77dbd6eedbff768112dbe8e7ec332b9eb/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-91a35228ead480e04b85998ccf3edfc891f44b5f79017b7fcab72cb66a812b07"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_91a35228ead480e04b85998ccf3edfc891f44b5f79017b7fcab72cb66a812b07/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_91a35228ead480e04b85998ccf3edfc891f44b5f79017b7fcab72cb66a812b07/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-f58c00ccab797ad4dbca3de3b50633588a68db0122aa011bdf81a9aca5ea692b"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_f58c00ccab797ad4dbca3de3b50633588a68db0122aa011bdf81a9aca5ea692b/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_f58c00ccab797ad4dbca3de3b50633588a68db0122aa011bdf81a9aca5ea692b/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-bfb6239769f044d2228f2efb5ce6aa51132455d9a8178e5a5ec8525ff5836e0d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_bfb6239769f044d2228f2efb5ce6aa51132455d9a8178e5a5ec8525ff5836e0d/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_bfb6239769f044d2228f2efb5ce6aa51132455d9a8178e5a5ec8525ff5836e0d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-469056c2ca5935d7c63d5424c635a439f94593a307e96483e4db16af1c15186e"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_469056c2ca5935d7c63d5424c635a439f94593a307e96483e4db16af1c15186e/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_469056c2ca5935d7c63d5424c635a439f94593a307e96483e4db16af1c15186e/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-3aab34d7c719c9d828a741a7900b4794302a587927c462b4ec8feec3f7d43e99"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_3aab34d7c719c9d828a741a7900b4794302a587927c462b4ec8feec3f7d43e99/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_3aab34d7c719c9d828a741a7900b4794302a587927c462b4ec8feec3f7d43e99/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-11a9de4ea6e17feff81429ed53e52a70e89c1cfeef0a73f10740c8420341b81d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_11a9de4ea6e17feff81429ed53e52a70e89c1cfeef0a73f10740c8420341b81d/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_11a9de4ea6e17feff81429ed53e52a70e89c1cfeef0a73f10740c8420341b81d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9c01136785f28f0d2c969cee8ce87bde3267d63425c5d86d39137abdf7f0f196"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9c01136785f28f0d2c969cee8ce87bde3267d63425c5d86d39137abdf7f0f196/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9c01136785f28f0d2c969cee8ce87bde3267d63425c5d86d39137abdf7f0f196/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-3160a297a2c9d3d21caeec097b6fe19150c3feae5fa872e21817af0be8a8176a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_3160a297a2c9d3d21caeec097b6fe19150c3feae5fa872e21817af0be8a8176a/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_3160a297a2c9d3d21caeec097b6fe19150c3feae5fa872e21817af0be8a8176a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-0bf641c301b26975b8919a18de98d9dfd6444d6542085dd2d8e8155ea6bc8efe"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_0bf641c301b26975b8919a18de98d9dfd6444d6542085dd2d8e8155ea6bc8efe/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_0bf641c301b26975b8919a18de98d9dfd6444d6542085dd2d8e8155ea6bc8efe/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-058ea3aae7385269d001efd9eb2303887614d138ff69150b20a703fc7b97c2c6"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_058ea3aae7385269d001efd9eb2303887614d138ff69150b20a703fc7b97c2c6/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_058ea3aae7385269d001efd9eb2303887614d138ff69150b20a703fc7b97c2c6/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-fdff4caa105318036534bd28b76a6fe34e6e2412752c1a000f50fafe7f01ef07"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_fdff4caa105318036534bd28b76a6fe34e6e2412752c1a000f50fafe7f01ef07/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_fdff4caa105318036534bd28b76a6fe34e6e2412752c1a000f50fafe7f01ef07/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-14ce4459cb4ea8aa84b5315222e9cfe00fe8a3b456b2ae75a5eb943036279d68"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_14ce4459cb4ea8aa84b5315222e9cfe00fe8a3b456b2ae75a5eb943036279d68/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_14ce4459cb4ea8aa84b5315222e9cfe00fe8a3b456b2ae75a5eb943036279d68/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-0b2b90bac148aa9f7c2ee34db723a002823dbc0d5981e47511f09cafa3693101"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0b2b90bac148aa9f7c2ee34db723a002823dbc0d5981e47511f09cafa3693101/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0b2b90bac148aa9f7c2ee34db723a002823dbc0d5981e47511f09cafa3693101/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-bd015dc23ff2f9bf5d681039cbb0f6418cd3d09d09124c0238d8c2caf01dba24"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_bd015dc23ff2f9bf5d681039cbb0f6418cd3d09d09124c0238d8c2caf01dba24/module.vmfb"
  FLAGS "--iree-hal-target-backends=llvm-cpu" "--iree-input-type=tosa" "--iree-llvm-target-triple=aarch64-none-linux-android29" "--iree-flow-enable-data-tiling" "--iree-llvm-target-cpu-features=+dotprod" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-llvmcpu-enable-pad-consumer-fusion" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_bd015dc23ff2f9bf5d681039cbb0f6418cd3d09d09124c0238d8c2caf01dba24/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e02e0460e54ee222b46c25e876f937eed5582b0823cad1b1d009fe406b160c33"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_e02e0460e54ee222b46c25e876f937eed5582b0823cad1b1d009fe406b160c33/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_e02e0460e54ee222b46c25e876f937eed5582b0823cad1b1d009fe406b160c33/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-d6590e27e94d8aac1b2bfb1e7c60e31dcddacd3a10687cdae998979fc31720fc"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_d6590e27e94d8aac1b2bfb1e7c60e31dcddacd3a10687cdae998979fc31720fc/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_d6590e27e94d8aac1b2bfb1e7c60e31dcddacd3a10687cdae998979fc31720fc/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-50567a33e0bd9aa5a32a6f61fca9ef8a70ac4d94313024f2c4ec92d9c543c599"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_50567a33e0bd9aa5a32a6f61fca9ef8a70ac4d94313024f2c4ec92d9c543c599/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_50567a33e0bd9aa5a32a6f61fca9ef8a70ac4d94313024f2c4ec92d9c543c599/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-6b9353f591f5044f661ecbbaafee502d710cf263527525d4f843b26cd43f11f7"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_6b9353f591f5044f661ecbbaafee502d710cf263527525d4f843b26cd43f11f7/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_6b9353f591f5044f661ecbbaafee502d710cf263527525d4f843b26cd43f11f7/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-5f03fee30980d1fb1074b82d329a1fa63b365858539743e672ad56c039dd732a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_5f03fee30980d1fb1074b82d329a1fa63b365858539743e672ad56c039dd732a/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_5f03fee30980d1fb1074b82d329a1fa63b365858539743e672ad56c039dd732a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-611a54141f98b17aa94abdba55d8a0487aa722bba4da6853c090f786975c5884"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_611a54141f98b17aa94abdba55d8a0487aa722bba4da6853c090f786975c5884/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_611a54141f98b17aa94abdba55d8a0487aa722bba4da6853c090f786975c5884/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-0d524f6ce80da6d1998eb8978623a2f6efd413e0b973c6f2dddf52a46b19907e"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_0d524f6ce80da6d1998eb8978623a2f6efd413e0b973c6f2dddf52a46b19907e/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_0d524f6ce80da6d1998eb8978623a2f6efd413e0b973c6f2dddf52a46b19907e/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-dd2a6a43dceabe7a807e280b43177cdf892d4ad20fdef4e3d3b6e39be7b9b09d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_dd2a6a43dceabe7a807e280b43177cdf892d4ad20fdef4e3d3b6e39be7b9b09d/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_dd2a6a43dceabe7a807e280b43177cdf892d4ad20fdef4e3d3b6e39be7b9b09d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-823ec09bcc061f124fa9229747054945cedf352e11d661a68785cb26af5f83b6"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_823ec09bcc061f124fa9229747054945cedf352e11d661a68785cb26af5f83b6/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_823ec09bcc061f124fa9229747054945cedf352e11d661a68785cb26af5f83b6/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-899c2de5e339b7e19528e80de1129a38511948ba3331932c22e23223707af4ca"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_899c2de5e339b7e19528e80de1129a38511948ba3331932c22e23223707af4ca/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_899c2de5e339b7e19528e80de1129a38511948ba3331932c22e23223707af4ca/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c3cb44c1331872dc53919d0d8b2cab4c256dcdf8b0038f9b6a692a5c874f855b"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_c3cb44c1331872dc53919d0d8b2cab4c256dcdf8b0038f9b6a692a5c874f855b/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_c3cb44c1331872dc53919d0d8b2cab4c256dcdf8b0038f9b6a692a5c874f855b/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-dc81a08fe0e5140f22f328d9dfea1828e7318d67899a2534d3b02ff36032cb61"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_dc81a08fe0e5140f22f328d9dfea1828e7318d67899a2534d3b02ff36032cb61/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_dc81a08fe0e5140f22f328d9dfea1828e7318d67899a2534d3b02ff36032cb61/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-1171fb017e88de21814d71ea2d35564de6904d3d2359ef53e0fa2c67ea6e9914"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_1171fb017e88de21814d71ea2d35564de6904d3d2359ef53e0fa2c67ea6e9914/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_1171fb017e88de21814d71ea2d35564de6904d3d2359ef53e0fa2c67ea6e9914/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-9b9a47b0a97a0bd002bd7fd1f104caaa94b8bf60cf02ffcc2b50129679e4c6f3"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_9b9a47b0a97a0bd002bd7fd1f104caaa94b8bf60cf02ffcc2b50129679e4c6f3/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_9b9a47b0a97a0bd002bd7fd1f104caaa94b8bf60cf02ffcc2b50129679e4c6f3/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-4f5ab4bfb26a82d0f83133b9e85585f0c5b97cdb00143de31675158a5a71b457"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_4f5ab4bfb26a82d0f83133b9e85585f0c5b97cdb00143de31675158a5a71b457/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_4f5ab4bfb26a82d0f83133b9e85585f0c5b97cdb00143de31675158a5a71b457/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-480b59f233af720e16db8e5da1988a8d69bd61169bf5b5899f425ff98dc0dc19"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_480b59f233af720e16db8e5da1988a8d69bd61169bf5b5899f425ff98dc0dc19/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=adreno-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_480b59f233af720e16db8e5da1988a8d69bd61169bf5b5899f425ff98dc0dc19/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7731488e1eb90da5480e76b4cd98b12c16b83d7c7011b0aa9ef3a5d6d2059a3c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_7731488e1eb90da5480e76b4cd98b12c16b83d7c7011b0aa9ef3a5d6d2059a3c/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_7731488e1eb90da5480e76b4cd98b12c16b83d7c7011b0aa9ef3a5d6d2059a3c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-61decb77e61b184a2c353fac3d60af1cd7c73abc867c23e9519f5e398265a728"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_61decb77e61b184a2c353fac3d60af1cd7c73abc867c23e9519f5e398265a728/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_61decb77e61b184a2c353fac3d60af1cd7c73abc867c23e9519f5e398265a728/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-4ec47dd2b4a43dd434d041d4d9db548076b70cfd63a2fec2971035394954f1d5"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_4ec47dd2b4a43dd434d041d4d9db548076b70cfd63a2fec2971035394954f1d5/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_4ec47dd2b4a43dd434d041d4d9db548076b70cfd63a2fec2971035394954f1d5/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-b0309994482c31c79242ee8ef3902b4cc54c1479688824734b33d2f554d6aff6"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_b0309994482c31c79242ee8ef3902b4cc54c1479688824734b33d2f554d6aff6/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_b0309994482c31c79242ee8ef3902b4cc54c1479688824734b33d2f554d6aff6/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-113994770711e5784a73ac623cbde328267c94b6341e001328b053c02b8bc08f"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_113994770711e5784a73ac623cbde328267c94b6341e001328b053c02b8bc08f/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_113994770711e5784a73ac623cbde328267c94b6341e001328b053c02b8bc08f/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-954bc3dc1fd0c22768ebfe898a67c0db3743d74e8fb776fced75eafb0421058f"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_954bc3dc1fd0c22768ebfe898a67c0db3743d74e8fb776fced75eafb0421058f/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_954bc3dc1fd0c22768ebfe898a67c0db3743d74e8fb776fced75eafb0421058f/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-6bb61b9c7107a9a30ad20c154321e7e9b14aefc407a8aeda41ac6df5eac757c4"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_6bb61b9c7107a9a30ad20c154321e7e9b14aefc407a8aeda41ac6df5eac757c4/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_6bb61b9c7107a9a30ad20c154321e7e9b14aefc407a8aeda41ac6df5eac757c4/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-24abf13f1d9be25ee353527d7f9096e5ccf53149f64eb8a90f069aea352eac21"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_24abf13f1d9be25ee353527d7f9096e5ccf53149f64eb8a90f069aea352eac21/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_24abf13f1d9be25ee353527d7f9096e5ccf53149f64eb8a90f069aea352eac21/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-cc884f3b2b1b188fad65da5462e4cea10e808adeda7f31f9657c3e3e29f876a9"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_cc884f3b2b1b188fad65da5462e4cea10e808adeda7f31f9657c3e3e29f876a9/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_cc884f3b2b1b188fad65da5462e4cea10e808adeda7f31f9657c3e3e29f876a9/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e041f40aefb2124a84ea3413d71ec072753e5098a15b4637bf2690780badb52c"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_e041f40aefb2124a84ea3413d71ec072753e5098a15b4637bf2690780badb52c/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_e041f40aefb2124a84ea3413d71ec072753e5098a15b4637bf2690780badb52c/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7bec578c7016cb7e017057c227a9b677901d14c0fff35e31c4a5cf12692db105"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_7bec578c7016cb7e017057c227a9b677901d14c0fff35e31c4a5cf12692db105/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_7bec578c7016cb7e017057c227a9b677901d14c0fff35e31c4a5cf12692db105/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-89a91c770dfce869ecb04e4b37e3b4d7da508a240da395cf240cc20ee8573857"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_89a91c770dfce869ecb04e4b37e3b4d7da508a240da395cf240cc20ee8573857/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_89a91c770dfce869ecb04e4b37e3b4d7da508a240da395cf240cc20ee8573857/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e6049d40d7925bccce4859e5408f2ad53eb68309aa38b46b8a7e47c94a2cd8a3"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_e6049d40d7925bccce4859e5408f2ad53eb68309aa38b46b8a7e47c94a2cd8a3/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_e6049d40d7925bccce4859e5408f2ad53eb68309aa38b46b8a7e47c94a2cd8a3/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-a4194c053541ebc49b4912bbdf3ca211331fdca5d157440837144e59d279bf1f"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_a4194c053541ebc49b4912bbdf3ca211331fdca5d157440837144e59d279bf1f/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_a4194c053541ebc49b4912bbdf3ca211331fdca5d157440837144e59d279bf1f/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-599701d7114956cf64777412899cff57ea5be0478f9a2331377770beaad8f923"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_599701d7114956cf64777412899cff57ea5be0478f9a2331377770beaad8f923/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_599701d7114956cf64777412899cff57ea5be0478f9a2331377770beaad8f923/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-1d26fcfdb7387659356dd99ce7e10907c8560b0925ad839334b0a6155d25167a"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_1d26fcfdb7387659356dd99ce7e10907c8560b0925ad839334b0a6155d25167a/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_1d26fcfdb7387659356dd99ce7e10907c8560b0925ad839334b0a6155d25167a/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-b74ccbdce4ec07bb65313ee96b67c1b946a6c959158714706209a9df2b93ab0d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_b74ccbdce4ec07bb65313ee96b67c1b946a6c959158714706209a9df2b93ab0d/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_b74ccbdce4ec07bb65313ee96b67c1b946a6c959158714706209a9df2b93ab0d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-69ce6a0ceae813a4fdbd4728a7d7c663ebb3a482a082a941cd1c5f43bd844b3b"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_69ce6a0ceae813a4fdbd4728a7d7c663ebb3a482a082a941cd1c5f43bd844b3b/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_69ce6a0ceae813a4fdbd4728a7d7c663ebb3a482a082a941cd1c5f43bd844b3b/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7d0c31ec6790283b9ff590485f6514f84861d58b8443e7fe70199b5c27185eb8"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_7d0c31ec6790283b9ff590485f6514f84861d58b8443e7fe70199b5c27185eb8/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_7d0c31ec6790283b9ff590485f6514f84861d58b8443e7fe70199b5c27185eb8/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c56e0b1dd1ce04dc5534024a183f090f58c9c1f27ccca3faf32f62fa55576135"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_c56e0b1dd1ce04dc5534024a183f090f58c9c1f27ccca3faf32f62fa55576135/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_c56e0b1dd1ce04dc5534024a183f090f58c9c1f27ccca3faf32f62fa55576135/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-178907b155b6322dedfa947937f9caca5158ff3af167470f2de90347dba357f4"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_178907b155b6322dedfa947937f9caca5158ff3af167470f2de90347dba357f4/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_178907b155b6322dedfa947937f9caca5158ff3af167470f2de90347dba357f4/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-7dcabfd6caa769a75657e07e7315dd42f52b3d4cbc37d75098aca446d3ff4066"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7dcabfd6caa769a75657e07e7315dd42f52b3d4cbc37d75098aca446d3ff4066/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7dcabfd6caa769a75657e07e7315dd42f52b3d4cbc37d75098aca446d3ff4066/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-247b38beca9631678d80755b0b4db2b323ddc4d95772617889a6a4bb813c6b74"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_247b38beca9631678d80755b0b4db2b323ddc4d95772617889a6a4bb813c6b74/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_247b38beca9631678d80755b0b4db2b323ddc4d95772617889a6a4bb813c6b74/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-d39340f50384e970b103694a38d7d21d5b1171d7304630d25e925c5c2486bf10"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_d39340f50384e970b103694a38d7d21d5b1171d7304630d25e925c5c2486bf10/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_d39340f50384e970b103694a38d7d21d5b1171d7304630d25e925c5c2486bf10/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-d8f22b5a700abdef68fe791ad08acdfc6d238d82e00f264367d922b99b369ff7"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_d8f22b5a700abdef68fe791ad08acdfc6d238d82e00f264367d922b99b369ff7/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_d8f22b5a700abdef68fe791ad08acdfc6d238d82e00f264367d922b99b369ff7/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c6a4903d1769d721782cf2b6e84837aca21f87fcf8759912a86ae2f572e8440d"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_c6a4903d1769d721782cf2b6e84837aca21f87fcf8759912a86ae2f572e8440d/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_c6a4903d1769d721782cf2b6e84837aca21f87fcf8759912a86ae2f572e8440d/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-3c43472d6cb0f74a1c08920e3f580b701e995a85305fd4b2e370542b4d449b18"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_3c43472d6cb0f74a1c08920e3f580b701e995a85305fd4b2e370542b4d449b18/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_3c43472d6cb0f74a1c08920e3f580b701e995a85305fd4b2e370542b4d449b18/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-92734e18b793ed29334b32490373ad4a008b4c8a47a3885c2c2dc8ed0fbce292"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_92734e18b793ed29334b32490373ad4a008b4c8a47a3885c2c2dc8ed0fbce292/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_92734e18b793ed29334b32490373ad4a008b4c8a47a3885c2c2dc8ed0fbce292/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-8408ed6dfa697c6e05e4d4c8f191c53856112e7a1b25a03f839a5046c936af37"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_8408ed6dfa697c6e05e4d4c8f191c53856112e7a1b25a03f839a5046c936af37/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_8408ed6dfa697c6e05e4d4c8f191c53856112e7a1b25a03f839a5046c936af37/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-c809128508627734ca76d422eec487a4b80d3538c0d08c3e339bebafee432f19"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_c809128508627734ca76d422eec487a4b80d3538c0d08c3e339bebafee432f19/module.vmfb"
  FLAGS "--iree-hal-target-backends=vulkan-spirv" "--iree-input-type=tosa" "--iree-vulkan-target-triple=valhall-unknown-android31" "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops" "--iree-hal-benchmark-dispatch-repeat-count=32" "--iree-flow-demote-f32-to-f16" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_c809128508627734ca76d422eec487a4b80d3538c0d08c3e339bebafee432f19/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-02b72f9538e4dfc9c789e63d722d5eab4333f3f55f8375503f433a790da119cc"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_02b72f9538e4dfc9c789e63d722d5eab4333f3f55f8375503f433a790da119cc/module.vmfb"
  FLAGS "--iree-hal-target-backends=vmvx" "--iree-input-type=tosa" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_02b72f9538e4dfc9c789e63d722d5eab4333f3f55f8375503f433a790da119cc/compilation.flagfile"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-e7bd41e564750501f39ac9690c18d1a2e77dc7999da710d0c0bf80751dda84a0"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_e7bd41e564750501f39ac9690c18d1a2e77dc7999da710d0c0bf80751dda84a0/module.vmfb"
  FLAGS "--iree-hal-target-backends=vmvx" "--iree-input-type=tosa" "--iree-vm-emit-polyglot-zip=true" "--iree-llvm-debug-symbols=false"
  DUMP_FLAGFILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_e7bd41e564750501f39ac9690c18d1a2e77dc7999da710d0c0bf80751dda84a0/compilation.flagfile"
  PUBLIC
)

set(_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x257x257x3xf32=0
)

set(_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03
--device_allocator=caching --device=local-sync
)

set(_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1
--device_allocator=caching --task_topology_group_count=1 --device=local-task
)

set(_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4
--device_allocator=caching --task_topology_group_count=4 --device=local-task
)

set(_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8
--device_allocator=caching --task_topology_group_count=8 --device=local-task
)

set(_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x224x224x3xui8=0
)

set(_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x384xi32=0 --input=1x384xi32=0 --input=1x384xi32=0
)

set(_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x384xi32=0 --input=1x384xi32=0 --input=1x384xi32=0
)

set(_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x384xi32=0 --input=1x384xi32=0 --input=1x384xi32=0
)

set(_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x224x224x3xf32=0
)

set(_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x224x224x3xf32=0
)

set(_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x224x224x3xf32=0
)

set(_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x320x320x3xf32=0
)

set(_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x96x96x1xi8=0
)

set(_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=main --input=1x353x257x3xf32=0
)

set(_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=forward --input=1x512xi32=0 --input=1x512xi32=0
)

set(_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=serving_default --input=1x384xi32=0 --input=1x384xi32=0 --input=1x384xi32=0
)

set(_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=forward --input=1x384x384x3xf32=0
)

set(_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=predict --input=1x128xi32=0 --input=1x128xi32=0 --input=1x128xi32=0
)

set(_MODEL_RUN_FLAGS_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f_8d4a034e-944d-4725-8402-d6f6e61be93c
--function=forward --input=1x224x224x3xf32=0
)

set(_EXEC_RUN_FLAGS_f7c0ec98-f028-436a-b05a-7d35cf18ce2d
--device_allocator=caching --device=cuda://${GPU_ID_PLACEHOLDER}
)

set(_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89
--device_allocator=caching --device=vulkan
)

set(_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0
--device_allocator=caching --batch_size=16 --device=vulkan
)

set(_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8
--device_allocator=caching --batch_size=32 --device=vulkan
)

set(_EXEC_RUN_FLAGS_953183e2-1e84-4a51-a43c-9b869bdc2218-4
--device_allocator=caching --task_topology_group_count=4 --device=local-task
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fcc2eb7748902acc86b82e71de537c9f38bd0baccb9ff8da2688a806278116a0-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_fcc2eb7748902acc86b82e71de537c9f38bd0baccb9ff8da2688a806278116a0/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-015af8c7c74743569726f8fecf3c5af66eb516b1e4c27b9c53444e5eb68254f9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_015af8c7c74743569726f8fecf3c5af66eb516b1e4c27b9c53444e5eb68254f9/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-3d80100b61ef99b830f9e24065147ef82b7d938788576481b0a70aadf09566a6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_3d80100b61ef99b830f9e24065147ef82b7d938788576481b0a70aadf09566a6/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-882a01b5adfe6cf932e3cacf39a21659d97c6d680a7c3aacbef5298958c13078-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_882a01b5adfe6cf932e3cacf39a21659d97c6d680a7c3aacbef5298958c13078/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2d916b25f6cf87d4d3841df2df2858ba4eaab05254ddacdc03c52c2ef66a2385-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_2d916b25f6cf87d4d3841df2df2858ba4eaab05254ddacdc03c52c2ef66a2385/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-480a2fe9ab9bd9ade098ff3c5fa0fd61a93c787c99329a1cdcecac6e5d708558-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_480a2fe9ab9bd9ade098ff3c5fa0fd61a93c787c99329a1cdcecac6e5d708558/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d1318602b2690766e3800f5c97b1d5bd754eb594f5b053e9c977705de62329c1-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_d1318602b2690766e3800f5c97b1d5bd754eb594f5b053e9c977705de62329c1/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e94f7cad9035a9a3f3f6dc8ca0fb4ecc25339cf0f4a153c842b95ec00dc66f7f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_e94f7cad9035a9a3f3f6dc8ca0fb4ecc25339cf0f4a153c842b95ec00dc66f7f/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8c1719ca3fa801fb65c375f0958c274d7fec10483a39f47419df2ab68275e359-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_8c1719ca3fa801fb65c375f0958c274d7fec10483a39f47419df2ab68275e359/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-746443fef718b98d7449c0b2d1733195479afa32e50ae726e8f695cc48611f57-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_746443fef718b98d7449c0b2d1733195479afa32e50ae726e8f695cc48611f57/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d071f553d892687709755025670f148111f85dee17950b0ecbf32192ae825421-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_d071f553d892687709755025670f148111f85dee17950b0ecbf32192ae825421/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e58daae6572fa59b33336e143c4d985e632e7b577a6b840a4fa52e4ada335e63-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_e58daae6572fa59b33336e143c4d985e632e7b577a6b840a4fa52e4ada335e63/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-104eaa83228bf7dabd693382b2402c824fca7b8dc4284f5ab626063bc839ff4c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_104eaa83228bf7dabd693382b2402c824fca7b8dc4284f5ab626063bc839ff4c/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-51473638a07429e21bf4b4fdfdb47201bbdff46edc0134cab2d589abc65a4ed6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_51473638a07429e21bf4b4fdfdb47201bbdff46edc0134cab2d589abc65a4ed6/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-82595cc516738f482ae30ff2e0441d0e4135a28e6ed68ef7ffa68f50eee4134a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_82595cc516738f482ae30ff2e0441d0e4135a28e6ed68ef7ffa68f50eee4134a/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6384be1200f350799a7bcd3cc41782341c59973d7e5343985aa135756830b1e2-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_6384be1200f350799a7bcd3cc41782341c59973d7e5343985aa135756830b1e2/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-5d93cc00896e37c8179563b6c4303d79ebaba5f93339d2873c86f9bb4462aa93-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_5d93cc00896e37c8179563b6c4303d79ebaba5f93339d2873c86f9bb4462aa93/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-5b81ba0c3d0db49f11e4c7e51f4138a723c72445c4d1b7d6d441d5a02bbf700a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_5b81ba0c3d0db49f11e4c7e51f4138a723c72445c4d1b7d6d441d5a02bbf700a/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-809ad3f979307779ec4bc1d3df2353bf843a813a295a5ca628ac4dd1f1a12164-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_809ad3f979307779ec4bc1d3df2353bf843a813a295a5ca628ac4dd1f1a12164/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c779cb79dd79861cea6f5bce3ab4dff679b45dbf4af92bc882f85d52e673204d-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_c779cb79dd79861cea6f5bce3ab4dff679b45dbf4af92bc882f85d52e673204d/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-3af1a3982de4ec043b01b93519a7d322e862cd4b8b71d4d61f2219571a66861b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_run_3af1a3982de4ec043b01b93519a7d322e862cd4b8b71d4d61f2219571a66861b/run.flagfile"
  FLAGS "--module=iree_MobileNetV1_fp32_module_02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f/module.vmfb" ${_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1622e274d5ac570e18826aaec62f223c538583eb2f76e771d24eb2f7785954aa-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_run_1622e274d5ac570e18826aaec62f223c538583eb2f76e771d24eb2f7785954aa/run.flagfile"
  FLAGS "--module=iree_MobileNetV1_fp32_module_02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f/module.vmfb" ${_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-17feba75a666cf8c6f5966637f416a835b8f5d39bc1697995db64b95d778c574-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_run_17feba75a666cf8c6f5966637f416a835b8f5d39bc1697995db64b95d778c574/run.flagfile"
  FLAGS "--module=iree_MobileNetV1_fp32_module_02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f/module.vmfb" ${_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-002cd64f66606ef48d9568103412f709d494fbea040a6879b069436ccc106733-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_run_002cd64f66606ef48d9568103412f709d494fbea040a6879b069436ccc106733/run.flagfile"
  FLAGS "--module=iree_MobileNetV1_fp32_module_02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f/module.vmfb" ${_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-dac56be634b207d7e868bc434a6299f3fd71e5e24cf683486d8b839851abf388-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_dac56be634b207d7e868bc434a6299f3fd71e5e24cf683486d8b839851abf388/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-48cac7cf7dea690dd7d8e8669fd5d6f65d1f20c0de1710dc381cf15533354bed-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_48cac7cf7dea690dd7d8e8669fd5d6f65d1f20c0de1710dc381cf15533354bed/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-058a0934550c7b633e2199e5387b14be0a564aa6512660692b6f46e464e1e991-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_058a0934550c7b633e2199e5387b14be0a564aa6512660692b6f46e464e1e991/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9accf20747a0a52c6c6b7da7433c9e9cdf68a813ec6589b781ecb7791a836e34-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_9accf20747a0a52c6c6b7da7433c9e9cdf68a813ec6589b781ecb7791a836e34/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fd46a78e4032c5fa09644bcda90d0d8b73e9196fb89e2458db2838ddf5fd4c16-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_fd46a78e4032c5fa09644bcda90d0d8b73e9196fb89e2458db2838ddf5fd4c16/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-069f6917e401e63c9e50c548c70cc699385e6f6908517eb6c79c96e597bf96d7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_069f6917e401e63c9e50c548c70cc699385e6f6908517eb6c79c96e597bf96d7/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1e0a0f881d2cefe36442136cc578bc4507bb4079146db4c435aafdb7e36daba6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_1e0a0f881d2cefe36442136cc578bc4507bb4079146db4c435aafdb7e36daba6/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1b97e25add2d6f62449e9e5c1a79cf1297e3b1bbc1bb50f28e0f962719682d17-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_1b97e25add2d6f62449e9e5c1a79cf1297e3b1bbc1bb50f28e0f962719682d17/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4f616830763f5dbc3dd47fe8bb6d181793e33051f1bd62b02160b633383e7882-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_4f616830763f5dbc3dd47fe8bb6d181793e33051f1bd62b02160b633383e7882/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0aac8a2a5c45ed0ed35dcd65338a5a414c6beefcdbb0fbb4f299b42d41b639e1-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_0aac8a2a5c45ed0ed35dcd65338a5a414c6beefcdbb0fbb4f299b42d41b639e1/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9fce5abfa2d401c58fb24687fbb7a55bb2e976bcd586db14b30d911cf16eca68-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_9fce5abfa2d401c58fb24687fbb7a55bb2e976bcd586db14b30d911cf16eca68/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4af168ed94d96166f35b8264e160ca1e85a3c6ef3faa08284f447a5613f6ce39-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_4af168ed94d96166f35b8264e160ca1e85a3c6ef3faa08284f447a5613f6ce39/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-da589d3a658ddcc4dacaab64c8c7253bab3b0b90fbd35158ba58ed883266d5dc-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_da589d3a658ddcc4dacaab64c8c7253bab3b0b90fbd35158ba58ed883266d5dc/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-77dd6dcff77b2053dbc4cbafc7ca36f8ee5aabdc138b5808830908b037014cc3-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_77dd6dcff77b2053dbc4cbafc7ca36f8ee5aabdc138b5808830908b037014cc3/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9fc0803530563719083133a4b602dba0ae57a549bcea4374bc09948d41bce849-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_9fc0803530563719083133a4b602dba0ae57a549bcea4374bc09948d41bce849/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-da282da8f4c832d72bf3eb9bc97bee39f0ba24a78e25626747f918976f19814c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_da282da8f4c832d72bf3eb9bc97bee39f0ba24a78e25626747f918976f19814c/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1b5de023c999f6523524305c2abd37a4cf24b3706f4ee9e337c87981dd3725c9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_1b5de023c999f6523524305c2abd37a4cf24b3706f4ee9e337c87981dd3725c9/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0d4e114d66ae2e078076cc40fca5e6af76232c3936effb92d33e23f76f26ede8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_0d4e114d66ae2e078076cc40fca5e6af76232c3936effb92d33e23f76f26ede8/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-150f45f52d20091945e3388e18d11a16e1e8453774c1e4b2eecf79e6f23c86fa-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_150f45f52d20091945e3388e18d11a16e1e8453774c1e4b2eecf79e6f23c86fa/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a2ebf5883d38f358868199609143debdbb2947b6e0ab6c5b03802cb813022f9f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_a2ebf5883d38f358868199609143debdbb2947b6e0ab6c5b03802cb813022f9f/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8cf9c331c7ec9e59884fe37b1a6e4326edc902e749daa2ad5f873c0b5433051b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_8cf9c331c7ec9e59884fe37b1a6e4326edc902e749daa2ad5f873c0b5433051b/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-cac4716bd8d859d6cc894c38936e1f8d22e9f218c01a94b9351c376eed985975-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_cac4716bd8d859d6cc894c38936e1f8d22e9f218c01a94b9351c376eed985975/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fe3b81659dda02e5b75d72c813626309b9c44232a5d23a7be79cc1320921add6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_fe3b81659dda02e5b75d72c813626309b9c44232a5d23a7be79cc1320921add6/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4003e2ba51615b0aca866e83846bafa2d510a5c5f4690e744f0f210cdbc592ec-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_4003e2ba51615b0aca866e83846bafa2d510a5c5f4690e744f0f210cdbc592ec/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ee47de5952e13727e6837d2f10c8979403039e27744b1b2df31cb1a9f2686538-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_ee47de5952e13727e6837d2f10c8979403039e27744b1b2df31cb1a9f2686538/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-3fcb38435a8f09de28c6b5a44b95c531b18f83d174218b923762ee9d29946543-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_3fcb38435a8f09de28c6b5a44b95c531b18f83d174218b923762ee9d29946543/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e02c3088da158fc213c051e13ac60cd39450d04e658a700b4cf67c40811e72c4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_e02c3088da158fc213c051e13ac60cd39450d04e658a700b4cf67c40811e72c4/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d7bb92092896fb847743683d802fd7bebc1b27216ae7fb5b2fb8b554c2d4f5ae-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_d7bb92092896fb847743683d802fd7bebc1b27216ae7fb5b2fb8b554c2d4f5ae/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f3a6824694bb52b8dfe735b60a86d0dd28ef151ea4a3eb6ff7706ee4aa12f9d6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_f3a6824694bb52b8dfe735b60a86d0dd28ef151ea4a3eb6ff7706ee4aa12f9d6/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d14bc72f848279de26aba8bd86bb530767acc4ca769356ab548258db49c44555-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_d14bc72f848279de26aba8bd86bb530767acc4ca769356ab548258db49c44555/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-bd39009584dcea45606af9a0f73380027510eebdb51d46c0391a527b032459ad-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_bd39009584dcea45606af9a0f73380027510eebdb51d46c0391a527b032459ad/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-974086586c080164bf56972646f76a8f116cd849d51713ca3d9b380366c8d305-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_974086586c080164bf56972646f76a8f116cd849d51713ca3d9b380366c8d305/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1d4ca3d0d6259c0d3322b04343e655cbd441759fdfad8a55f05c0dcbc13dc1ed-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_1d4ca3d0d6259c0d3322b04343e655cbd441759fdfad8a55f05c0dcbc13dc1ed/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e076babcf92c08d76f05c53bec9bcf823f3855b6280c2c74465ed25bb2bb2bd7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_e076babcf92c08d76f05c53bec9bcf823f3855b6280c2c74465ed25bb2bb2bd7/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-232dd191f613b7aebb8501342bf8c99afd519e1554b234d797e6aef3779066a7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_232dd191f613b7aebb8501342bf8c99afd519e1554b234d797e6aef3779066a7/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-bcd45d4ee5f34196aedb27bbe78f90ef2876c1908028a5d344915b67ba8294e2-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_bcd45d4ee5f34196aedb27bbe78f90ef2876c1908028a5d344915b67ba8294e2/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e496c2ea8de7fdffdb7597da63eed12cc5fb0595605d70e1f9d67a33857a299a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_run_e496c2ea8de7fdffdb7597da63eed12cc5fb0595605d70e1f9d67a33857a299a/run.flagfile"
  FLAGS "--module=iree_Resnet50TF_module_7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055/module.vmfb" ${_MODEL_RUN_FLAGS_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2cbf22689ea9dd3ac54c3a66da0fed845f3314f5ca96afef57c7e3697409536a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_run_2cbf22689ea9dd3ac54c3a66da0fed845f3314f5ca96afef57c7e3697409536a/run.flagfile"
  FLAGS "--module=iree_Resnet50TF_module_7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055/module.vmfb" ${_MODEL_RUN_FLAGS_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-58ea5e58d8226c09966c7dce3a6b0763a681bc4d2507b9d9207760c975363ab6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_run_58ea5e58d8226c09966c7dce3a6b0763a681bc4d2507b9d9207760c975363ab6/run.flagfile"
  FLAGS "--module=iree_Resnet50TF_module_7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055/module.vmfb" ${_MODEL_RUN_FLAGS_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0394a1edf412d08f215b567edcf2f1daf4a3ff5973c08590a564456232c5a171-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_run_0394a1edf412d08f215b567edcf2f1daf4a3ff5973c08590a564456232c5a171/run.flagfile"
  FLAGS "--module=iree_Resnet50TF_module_7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055/module.vmfb" ${_MODEL_RUN_FLAGS_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b1cbdfe8626953a3ab575aa1ab98ed52fb2a488c1b0ef032144961dc1ce480da-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_b1cbdfe8626953a3ab575aa1ab98ed52fb2a488c1b0ef032144961dc1ce480da/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7237c7cbf5353280472161050ccb803bd6237ac656eab0604d5cc610d73ef778-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_7237c7cbf5353280472161050ccb803bd6237ac656eab0604d5cc610d73ef778/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2387980c443202f6775c1ae692c4896612a50731639d3c51f1cb69ba0935b51a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_2387980c443202f6775c1ae692c4896612a50731639d3c51f1cb69ba0935b51a/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-60ebe003ad32386572a7515583e00883b11209d13c62d6907be645492557aa71-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_60ebe003ad32386572a7515583e00883b11209d13c62d6907be645492557aa71/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-efdc6e24cede8d60e41e9bfa18e61041c66c4dfe7ccbf731e186370055582f7a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_efdc6e24cede8d60e41e9bfa18e61041c66c4dfe7ccbf731e186370055582f7a/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-423824abc1ed6574ed1315b6c6432366edefbec9704c4b524d6daa9c7f18bf0a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_423824abc1ed6574ed1315b6c6432366edefbec9704c4b524d6daa9c7f18bf0a/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0e1f775fdc4e9c29a0c9d6fdbdb62fe04f8fef6a3d441171fc4dac4ebf0b439c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_0e1f775fdc4e9c29a0c9d6fdbdb62fe04f8fef6a3d441171fc4dac4ebf0b439c/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-579b8550840595f0dc5a89acbb574ebf022c1581132b82e56139df142953c820-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_579b8550840595f0dc5a89acbb574ebf022c1581132b82e56139df142953c820/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9a3ce9fe094e4ae324e0d11ce2407659487875b4169b1d0cfb1d7be822945cec-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_9a3ce9fe094e4ae324e0d11ce2407659487875b4169b1d0cfb1d7be822945cec/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b528e469bfd43258750e70a724bf02eeb157173782b5a5a8912ae036e3ffce58-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_b528e469bfd43258750e70a724bf02eeb157173782b5a5a8912ae036e3ffce58/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9928a2e957aea784606874bf135ba426b6688e110f8111a758274091ad1a3576-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_9928a2e957aea784606874bf135ba426b6688e110f8111a758274091ad1a3576/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-69794b071d9defd92d85bce61aa87c786ce2888bc2ce88a8cac841d6da6cac7e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_69794b071d9defd92d85bce61aa87c786ce2888bc2ce88a8cac841d6da6cac7e/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-73eb77eb75db72c97d07e8325dfc34f8c19f0bfc059a23b89027c89d0e945c31-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_73eb77eb75db72c97d07e8325dfc34f8c19f0bfc059a23b89027c89d0e945c31/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4d92c9901b7c73d8e02e63adfdcdf63ef0fb529360a908f93b888dee1c3f9c31-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_4d92c9901b7c73d8e02e63adfdcdf63ef0fb529360a908f93b888dee1c3f9c31/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-57617c1acebc961757a26d7529068227f7a75ff52f298d28f4f1dc0d6f193b19-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_57617c1acebc961757a26d7529068227f7a75ff52f298d28f4f1dc0d6f193b19/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a287c36d311b72071631c7c7743d126038ee1938e9a415d3e971f3aa7c00397b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_a287c36d311b72071631c7c7743d126038ee1938e9a415d3e971f3aa7c00397b/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ef450965157d6b8f47c7c6888eb66a8c56760986deeced8cc451877f174f2a69-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_ef450965157d6b8f47c7c6888eb66a8c56760986deeced8cc451877f174f2a69/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7001a4f2a5e52aa034f802096f625e278fc10b92cd85653335c3a7c5110492c7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_7001a4f2a5e52aa034f802096f625e278fc10b92cd85653335c3a7c5110492c7/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6fddae6f5e12a547164d4a46a7e185f0101acce8589964a668475fda218be41a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_6fddae6f5e12a547164d4a46a7e185f0101acce8589964a668475fda218be41a/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-78cc46b0727d5c5a98bcad94c2377cce22aa67fc5188cee22dc87f3c8306a722-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_78cc46b0727d5c5a98bcad94c2377cce22aa67fc5188cee22dc87f3c8306a722/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-3e6d25ceea03b4311e04345571e450c6557d8cff6e99991f822d67325d57b286-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_run_3e6d25ceea03b4311e04345571e450c6557d8cff6e99991f822d67325d57b286/run.flagfile"
  FLAGS "--module=iree_MobileNetV1_fp32_module_dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933/module.vmfb" ${_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6600e5c77f343f3727788ac55712340db67660453f0d5b2a78f8a2f00bffa9f2-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_run_6600e5c77f343f3727788ac55712340db67660453f0d5b2a78f8a2f00bffa9f2/run.flagfile"
  FLAGS "--module=iree_MobileNetV1_fp32_module_dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933/module.vmfb" ${_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-175234258abc061d624fad12b5f814c63c5031f71d59e1bafa0df468cb756f38-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_run_175234258abc061d624fad12b5f814c63c5031f71d59e1bafa0df468cb756f38/run.flagfile"
  FLAGS "--module=iree_MobileNetV1_fp32_module_dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933/module.vmfb" ${_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-14e8174454310c9b24812dca661319c7b8e78a1175003f56abe8cfa7e7bb9cb9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_run_14e8174454310c9b24812dca661319c7b8e78a1175003f56abe8cfa7e7bb9cb9/run.flagfile"
  FLAGS "--module=iree_MobileNetV1_fp32_module_dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933/module.vmfb" ${_MODEL_RUN_FLAGS_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f773cd1d5c806ca4576b9a60362058988bd80e2582f93d546f81e60ec38f451d-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_f773cd1d5c806ca4576b9a60362058988bd80e2582f93d546f81e60ec38f451d/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6272e089c33b7c5333b6188b6f61fbb15e7b6a0e9fcd9d54b3b7271cd730e0da-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_6272e089c33b7c5333b6188b6f61fbb15e7b6a0e9fcd9d54b3b7271cd730e0da/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-33ca476093f97d4862168a0feff140fbe85c5dc70fb06cbdf3e6054480575c85-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_33ca476093f97d4862168a0feff140fbe85c5dc70fb06cbdf3e6054480575c85/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ce780c2ab7c9b837611b5e1dcdbce18e7563fb9d9137e68b5a50bd917a54f83d-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_ce780c2ab7c9b837611b5e1dcdbce18e7563fb9d9137e68b5a50bd917a54f83d/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-485da7a706b6c0940ef45626ec12ab149da295cc6a3c0a2c63e5a15a952580b4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_485da7a706b6c0940ef45626ec12ab149da295cc6a3c0a2c63e5a15a952580b4/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c27738e97498c969076d1a2a693322821dd104dbcf7ba6e129ba893584bb0dfd-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_c27738e97498c969076d1a2a693322821dd104dbcf7ba6e129ba893584bb0dfd/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7cac96dd2ddb2585861de464c27b59023e8d6f44418f020821074e748f7f6388-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_7cac96dd2ddb2585861de464c27b59023e8d6f44418f020821074e748f7f6388/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d57d9beca86d9b8454422cd20afca034477dd3471577f5f4708252fffa864ebf-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_d57d9beca86d9b8454422cd20afca034477dd3471577f5f4708252fffa864ebf/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d15406ab9fa52f130a4947059f1616c6ee94c1c2b36a6078ec708d475556a9a4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_d15406ab9fa52f130a4947059f1616c6ee94c1c2b36a6078ec708d475556a9a4/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d6bfea70085e57a372f18983ddd9f7598b084dc4aac07754c80e4f4f5c4fb407-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_d6bfea70085e57a372f18983ddd9f7598b084dc4aac07754c80e4f4f5c4fb407/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-aaa50f507f6ca3e64fed593f49949d19d975c46d4986897a4dccbec89d132a3a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_aaa50f507f6ca3e64fed593f49949d19d975c46d4986897a4dccbec89d132a3a/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ec20addfc5f284c92b739d0eaf245af0027627de593635539a86709332ae5acf-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_ec20addfc5f284c92b739d0eaf245af0027627de593635539a86709332ae5acf/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-3283ddd7c21e5db8eea573c2f94ae318c5baa6bf3d9340ba157573937e7b6632-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_3283ddd7c21e5db8eea573c2f94ae318c5baa6bf3d9340ba157573937e7b6632/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8d8fd2fbd7901ece93ffa5e47c460dd793c4489b5751a15bb0c3e1b8d82073db-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_8d8fd2fbd7901ece93ffa5e47c460dd793c4489b5751a15bb0c3e1b8d82073db/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-aed9d709c26fd99b9df147829544434c944296e124a6fb575f012dbb1502da90-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_aed9d709c26fd99b9df147829544434c944296e124a6fb575f012dbb1502da90/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6e06f1507848e5fe5510a1310fb7a07919b0cd845aa419024140ace4349fb604-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_6e06f1507848e5fe5510a1310fb7a07919b0cd845aa419024140ace4349fb604/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9a309fd292fdde68ab47244e7144cda6bde73c2d4f05035cb6524f52d3323b9f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_9a309fd292fdde68ab47244e7144cda6bde73c2d4f05035cb6524f52d3323b9f/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-51181aae886260ff3c24d829e8bf9e3a892aa93305321c1012476aace79f9e65-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_51181aae886260ff3c24d829e8bf9e3a892aa93305321c1012476aace79f9e65/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fadf67f1f541c40d8b9e897a4a5cf125ffcac13ed77d6ec9f893f9cf45874192-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_fadf67f1f541c40d8b9e897a4a5cf125ffcac13ed77d6ec9f893f9cf45874192/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1e0197113e1bab228898b4e76067c7c8dcd0faf2b0cf5af9dbb227491de894e4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_1e0197113e1bab228898b4e76067c7c8dcd0faf2b0cf5af9dbb227491de894e4/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fb7e03b97045f97096e484826bf4b8f432d69d11acbadb6af721c94c4a2f9424-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_fb7e03b97045f97096e484826bf4b8f432d69d11acbadb6af721c94c4a2f9424/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9b9e37b533b9a700da481f6792656137c4591c3071401c0ad0abe792bcb5727c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_9b9e37b533b9a700da481f6792656137c4591c3071401c0ad0abe792bcb5727c/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4a5d8ec27fd956b0a83e003eec13cf3aa4ade14a6627f5ce62708d57aade9b9b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_4a5d8ec27fd956b0a83e003eec13cf3aa4ade14a6627f5ce62708d57aade9b9b/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-488a6230bbbb7f01ae76d1b32117251eb9aadf4fbe75a10540fef8b9e344f77f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_488a6230bbbb7f01ae76d1b32117251eb9aadf4fbe75a10540fef8b9e344f77f/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-19201c34fa6afcc1ea582fe6d836dfef9b93e9a2d98cb1720408fc2ad329c13a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_19201c34fa6afcc1ea582fe6d836dfef9b93e9a2d98cb1720408fc2ad329c13a/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f5718e569829b5a8af735ae68d4cdc04131cf182d496f483d643ce6beb1bc6d9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_f5718e569829b5a8af735ae68d4cdc04131cf182d496f483d643ce6beb1bc6d9/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-54f0a27002b8e2ec04cb7f783f1d178a0c360a187fa84eb1c3440403bd08d3b7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_54f0a27002b8e2ec04cb7f783f1d178a0c360a187fa84eb1c3440403bd08d3b7/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c28be7f493ec9bc790631b3f90dfe611da5a10ac37e23ddebc80530fc61df2f5-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_c28be7f493ec9bc790631b3f90dfe611da5a10ac37e23ddebc80530fc61df2f5/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d6f1a520e3361101b6a3e06f3d6d66638b9b86b88b4376da5ab2e1add311479a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_d6f1a520e3361101b6a3e06f3d6d66638b9b86b88b4376da5ab2e1add311479a/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ce7eec0c36a5fda73313a06da87ff315e0307cd6d2962d167e7e641eea50604c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_ce7eec0c36a5fda73313a06da87ff315e0307cd6d2962d167e7e641eea50604c/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-595fab3ea1ea90bbfa01d4c243295f6fd43f0e75878c75cff5a1139a0e7f4643-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_595fab3ea1ea90bbfa01d4c243295f6fd43f0e75878c75cff5a1139a0e7f4643/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-de9752b277be0f178adf7974807f310f7432bc84951aca9c7126b9752d4ffbf4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_de9752b277be0f178adf7974807f310f7432bc84951aca9c7126b9752d4ffbf4/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fc9a821b5ad5fe568b9aedc1142e9a3609685804387af4a4dddce2d7e65b9259-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_fc9a821b5ad5fe568b9aedc1142e9a3609685804387af4a4dddce2d7e65b9259/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-3da49d74eed3cd740c69a6a2a97f3ff7e54710ea66c083670042256b2648ddcf-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_3da49d74eed3cd740c69a6a2a97f3ff7e54710ea66c083670042256b2648ddcf/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-da652151b3065abd5c15dad6c2e9b88f319de04760d7f3fd8fd1f00714452a14-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_da652151b3065abd5c15dad6c2e9b88f319de04760d7f3fd8fd1f00714452a14/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9bdc58d9a9a5f3f7fed8db78bfc02d052539ece0e9eef401f52f7fe4663b7aac-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_9bdc58d9a9a5f3f7fed8db78bfc02d052539ece0e9eef401f52f7fe4663b7aac/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c4a8ace91f235fbf4ee079839016991166b1b8f01abf2f1bfccd0246323d79f0-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_run_c4a8ace91f235fbf4ee079839016991166b1b8f01abf2f1bfccd0246323d79f0/run.flagfile"
  FLAGS "--module=iree_BertForMaskedLMTF_module_bdd904cc5614ebf77609c7802a2dfc09f139aee2a247a247d10d320de72b0e28/module.vmfb" ${_MODEL_RUN_FLAGS_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_f7c0ec98-f028-436a-b05a-7d35cf18ce2d}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-335195752d56c66c4dcb075ffae3c4c0e82996f065e3ab9aacd3a3f7792174dc-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_run_335195752d56c66c4dcb075ffae3c4c0e82996f065e3ab9aacd3a3f7792174dc/run.flagfile"
  FLAGS "--module=iree_BertLargeTF_module_45565cae821666fd34bca97be2e4cce3bd61e71308785728737d89acbb9bc9d2/module.vmfb" ${_MODEL_RUN_FLAGS_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_f7c0ec98-f028-436a-b05a-7d35cf18ce2d}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f757ac686e7e26e443db2a8a1578b197d75a11ed8400b05a4476e3ac1c6713c5-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_run_f757ac686e7e26e443db2a8a1578b197d75a11ed8400b05a4476e3ac1c6713c5/run.flagfile"
  FLAGS "--module=iree_EfficientNetV2STF_module_04ca0a5077b7dd5ace66d803c9b822dff3428b24e7620a61995aff0907af9533/module.vmfb" ${_MODEL_RUN_FLAGS_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_f7c0ec98-f028-436a-b05a-7d35cf18ce2d}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1d3c521075923269cfc5bcd08c3b7638f689fe0552d897903870407e4c4b4e56-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_run_1d3c521075923269cfc5bcd08c3b7638f689fe0552d897903870407e4c4b4e56/run.flagfile"
  FLAGS "--module=iree_MiniLML12H384Uncased_module_deafafd0926321a4b8e4dc73ed4a30b2ed9317d26488246461415be2ee857eb1/module.vmfb" ${_MODEL_RUN_FLAGS_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_f7c0ec98-f028-436a-b05a-7d35cf18ce2d}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-5b33da9efba3f32f8b5761a7908c92dce139b97428585232599ca0c1a9ec426d-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_run_5b33da9efba3f32f8b5761a7908c92dce139b97428585232599ca0c1a9ec426d/run.flagfile"
  FLAGS "--module=iree_Resnet50TF_module_fd81a89e9f8773bae142040775c7e3c4774f96b64f07f8d9f66b00191864ff40/module.vmfb" ${_MODEL_RUN_FLAGS_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_f7c0ec98-f028-436a-b05a-7d35cf18ce2d}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-43205b471655d7c5d4d30801d6789b741a6318ab87adb6e0d8c4512b9ab5ee26-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_43205b471655d7c5d4d30801d6789b741a6318ab87adb6e0d8c4512b9ab5ee26/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8c9afefa68d7256a2f3d3c8cea6b7c592bef0b1afb4f1bad67d03540d22bcd80-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_8c9afefa68d7256a2f3d3c8cea6b7c592bef0b1afb4f1bad67d03540d22bcd80/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-5c523031b46b818cbef17167cdad8fbd9fc2390db00f8e48d2c7489951eae183-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_5c523031b46b818cbef17167cdad8fbd9fc2390db00f8e48d2c7489951eae183/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-95281c38b844a3b0ea1964e9634e7a8e2b40025936e3402ff2902be01dbd31b7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_95281c38b844a3b0ea1964e9634e7a8e2b40025936e3402ff2902be01dbd31b7/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-02f035862854dacda92c51e7c3b506e7b5adc3fb36818e8b171c5d670d78eff2-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_02f035862854dacda92c51e7c3b506e7b5adc3fb36818e8b171c5d670d78eff2/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8ccf5261dd5ce1229c401acee803a4d03ecce10bdba23e00e1df817e7983fdec-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_8ccf5261dd5ce1229c401acee803a4d03ecce10bdba23e00e1df817e7983fdec/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6796854808165b6275740d3692260a1349f58cdf0a8330207675b2eaf2a9735e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_6796854808165b6275740d3692260a1349f58cdf0a8330207675b2eaf2a9735e/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0705d47ed43a301d7b92b10b7a38f77703ad865f6bee8d28c2ad63f61c5c5772-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_0705d47ed43a301d7b92b10b7a38f77703ad865f6bee8d28c2ad63f61c5c5772/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1b166da68df5960cf0a58ff7fdd09f2bfdaddd5a4c3e673c712f687b4fb8ab9b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_1b166da68df5960cf0a58ff7fdd09f2bfdaddd5a4c3e673c712f687b4fb8ab9b/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2a360cb8f8f6bf23ef7aa89c5f21509f6504b6a3ba33c3529a840303dfb3c3fe-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_2a360cb8f8f6bf23ef7aa89c5f21509f6504b6a3ba33c3529a840303dfb3c3fe/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f3a4e75e14b7bf806d565fc3699afd13036cc3f14091efe2304c97b8ed81a28e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_f3a4e75e14b7bf806d565fc3699afd13036cc3f14091efe2304c97b8ed81a28e/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a56f27a4186ef4fdb9848c589a2f8f0027b4fb948ed54745b8b8d0cb79fbc213-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_a56f27a4186ef4fdb9848c589a2f8f0027b4fb948ed54745b8b8d0cb79fbc213/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b79504ca12b3835c243fb7bc186060060653cc048023c8a00c8da04eed0dabd8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_b79504ca12b3835c243fb7bc186060060653cc048023c8a00c8da04eed0dabd8/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7be38c67cfd7b39fab35c7e18be741a0ff81526243f39af14ea85deb48918885-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_7be38c67cfd7b39fab35c7e18be741a0ff81526243f39af14ea85deb48918885/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-169e82fdd31df61df4c365f1f4c5da9e178c1b491e83c633cfd706699adfa754-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_169e82fdd31df61df4c365f1f4c5da9e178c1b491e83c633cfd706699adfa754/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-cbb177a628d76a864a7ffc8112ee5a9916952d3d0b6f84b49cddecee1849b900-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_cbb177a628d76a864a7ffc8112ee5a9916952d3d0b6f84b49cddecee1849b900/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fd0020c5effd33078d74e079acbc48230001555fea77552823f8ec55b63cc44e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_fd0020c5effd33078d74e079acbc48230001555fea77552823f8ec55b63cc44e/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fa8255f23b90fc02bb1cccbc9dae74e0d7e5b7f9f3e31eb5610584d73732f1ff-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_fa8255f23b90fc02bb1cccbc9dae74e0d7e5b7f9f3e31eb5610584d73732f1ff/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e8b51276c78902bd98c7521b4c41d40b3083bb4343c7bd49cfe0d15c72331326-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_e8b51276c78902bd98c7521b4c41d40b3083bb4343c7bd49cfe0d15c72331326/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-db44d3427feb19946b43153a9690e9e830811d32fef711b8ac9cca66e50cc1c4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_db44d3427feb19946b43153a9690e9e830811d32fef711b8ac9cca66e50cc1c4/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c2885c9fa0bcddea33e9025f8a0f84b8823c3c489990a8087a309e0a39ad2566-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_c2885c9fa0bcddea33e9025f8a0f84b8823c3c489990a8087a309e0a39ad2566/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7271b79c4d00766e5a9bfe32b30993c76b2f5550b1cea07a0c2f26f51016b707-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_7271b79c4d00766e5a9bfe32b30993c76b2f5550b1cea07a0c2f26f51016b707/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9d2e9b2b3e070388dd08a932b3818d189c0a205fa3f4de3f8e0f772b31179cb7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_9d2e9b2b3e070388dd08a932b3818d189c0a205fa3f4de3f8e0f772b31179cb7/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-59642786e50eac32f2270781fedbbf1bb4db9241368ffcce6d1e9a9f454759a9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_59642786e50eac32f2270781fedbbf1bb4db9241368ffcce6d1e9a9f454759a9/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ab3a853f3d6693aaf464dcbb5f0adf9e40925cd56928fe257cd32d52b13f5599-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_ab3a853f3d6693aaf464dcbb5f0adf9e40925cd56928fe257cd32d52b13f5599/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-918149bf080db843fb731d650b77c3e2196e34dedf4b42221af7e3c9b91182d8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_918149bf080db843fb731d650b77c3e2196e34dedf4b42221af7e3c9b91182d8/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e216053a20f0e56c82bb260a93866d9edd5968941ce98a83bc01696a30760701-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_e216053a20f0e56c82bb260a93866d9edd5968941ce98a83bc01696a30760701/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c7c9eb0cfef4ef82f39f5d06a0d636e3691d62fcb8267b33143d84b246e88974-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_c7c9eb0cfef4ef82f39f5d06a0d636e3691d62fcb8267b33143d84b246e88974/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-dadde1e46eb1c96459ac710904383b6625d43b45bc7c06ce6043f3a33b3817b9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_dadde1e46eb1c96459ac710904383b6625d43b45bc7c06ce6043f3a33b3817b9/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-857b039a7c814ae00a254adf9b1a1a2c85fefb997e3bacf80510893cbb9c6f2e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_857b039a7c814ae00a254adf9b1a1a2c85fefb997e3bacf80510893cbb9c6f2e/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4b7c3c4c934d3ef254bc7e214e6c677ce6529fdcf72733521763aca5b9407e8f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_4b7c3c4c934d3ef254bc7e214e6c677ce6529fdcf72733521763aca5b9407e8f/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-aa947dc1908028becbeb0874aa7c65c9ac366f19ada505ea5155283ee86fd91b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_aa947dc1908028becbeb0874aa7c65c9ac366f19ada505ea5155283ee86fd91b/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d6dfd546eb26f91c109d344336ac213020777ac877e52e8e2e471c7d32c4a669-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_d6dfd546eb26f91c109d344336ac213020777ac877e52e8e2e471c7d32c4a669/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c065ef331c525cec65b1d9dd46fee173b55702b068d896110507a34cf55602a6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_c065ef331c525cec65b1d9dd46fee173b55702b068d896110507a34cf55602a6/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2a746c3fd6a24944a7e03b4b4bc246d0fa172d6159af4ae85cbd02ae1db2a5c0-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_2a746c3fd6a24944a7e03b4b4bc246d0fa172d6159af4ae85cbd02ae1db2a5c0/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c3ae398d16816bb66156d0aa84efd4c33ae1ff8238f437da171d8496a0ee97a8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_c3ae398d16816bb66156d0aa84efd4c33ae1ff8238f437da171d8496a0ee97a8/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9985db1af1487130c670bbc155da57c77decab98fdaff7293a079b2bcc2c7ea7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_9985db1af1487130c670bbc155da57c77decab98fdaff7293a079b2bcc2c7ea7/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6db7c3032ae546cc2102728906fbfc4f85aaba1c72fadfd55c8ad31da417ecd9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_6db7c3032ae546cc2102728906fbfc4f85aaba1c72fadfd55c8ad31da417ecd9/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2c8c3423e7567bd72e0c398a3b740233f60bc622dfdbc6023afd2dc905d5c3e3-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_2c8c3423e7567bd72e0c398a3b740233f60bc622dfdbc6023afd2dc905d5c3e3/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-dc2023c6113c87aad59f2b49214ab2995b32c7ba040b314e890ea2ec7081f90b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_dc2023c6113c87aad59f2b49214ab2995b32c7ba040b314e890ea2ec7081f90b/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-580de29df687f6838fa67a3b020e7d0b02bb6ebdbf14dd58460f09ac088b92c9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_580de29df687f6838fa67a3b020e7d0b02bb6ebdbf14dd58460f09ac088b92c9/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7b0a2b9429936d6e2c65b2474aa29baac011c9e106c49711db514dc64bb055cb-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_7b0a2b9429936d6e2c65b2474aa29baac011c9e106c49711db514dc64bb055cb/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8130cbda36a110611a415869f41370e934d21f8bf05d0c501834bbbc118d48c7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_8130cbda36a110611a415869f41370e934d21f8bf05d0c501834bbbc118d48c7/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-df3c8b2cb88a80d85abf4168ea912aadb39d3a3c3dc9dae9f3899431cea713fa-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_df3c8b2cb88a80d85abf4168ea912aadb39d3a3c3dc9dae9f3899431cea713fa/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-79d228995d626d87d488bfe58a04f3b37bbc58d6a45147c9039f4e11b19b7843-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_79d228995d626d87d488bfe58a04f3b37bbc58d6a45147c9039f4e11b19b7843/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4c331ecdd0e9e450678600deacc04cb8142b7b491c739e87bf3df7f32bb5a17e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_4c331ecdd0e9e450678600deacc04cb8142b7b491c739e87bf3df7f32bb5a17e/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1aee696bba55cf12e2f62e043c853437afbfee92a1f19bf0272b4014e633e6e5-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_1aee696bba55cf12e2f62e043c853437afbfee92a1f19bf0272b4014e633e6e5/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d61f2887ce922e7a8357640637ee69184c0ab4d84e9f90d57751e72d41cea509-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_d61f2887ce922e7a8357640637ee69184c0ab4d84e9f90d57751e72d41cea509/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-570d87aae955ec258d971849489bdec9646f0207eac47f5f8a79532ad1285f64-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_570d87aae955ec258d971849489bdec9646f0207eac47f5f8a79532ad1285f64/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d794d28f78568cf5b7b3352cae5e409e49b8e8248259cd686fb9e26c0addcee1-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_d794d28f78568cf5b7b3352cae5e409e49b8e8248259cd686fb9e26c0addcee1/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-22ac4fa00344cef241ae57b58fdcf836468b14bd86786d183d44862dcf4b862e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_22ac4fa00344cef241ae57b58fdcf836468b14bd86786d183d44862dcf4b862e/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a5fbfe5e760ae9770da3f77744c270a3caca342dd869b15b824f1bea6c853b74-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_a5fbfe5e760ae9770da3f77744c270a3caca342dd869b15b824f1bea6c853b74/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a9277ecdb3a9b16ddc617712d4a35307d39c7c272868f6e512b154c17b860e32-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_a9277ecdb3a9b16ddc617712d4a35307d39c7c272868f6e512b154c17b860e32/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fceb36e464100809c802a326aa68db53c21040c9fc2719d9c634d31741e54dfe-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_fceb36e464100809c802a326aa68db53c21040c9fc2719d9c634d31741e54dfe/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-09922a30726fd43b5532d77a806709543554eef05dfa0c59214d91639207484e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_09922a30726fd43b5532d77a806709543554eef05dfa0c59214d91639207484e/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e2184228010053416099d266bb180b71fd692000aafeed6953c4b757b5dbc0c8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_e2184228010053416099d266bb180b71fd692000aafeed6953c4b757b5dbc0c8/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0be5f52840f466497e75c2dbc83b6f15ecd6da62235e6b6101550fedcff5bdd8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_0be5f52840f466497e75c2dbc83b6f15ecd6da62235e6b6101550fedcff5bdd8/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-54dc93e0c6b0c16667a08ffd756665f0a41fec7aba3a446b0dcf84ba79b86890-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_54dc93e0c6b0c16667a08ffd756665f0a41fec7aba3a446b0dcf84ba79b86890/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f9c6d66a04485bdebd31a97572f8e17e0b46f6ff0672bbd105c9e9cea656a8b9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_f9c6d66a04485bdebd31a97572f8e17e0b46f6ff0672bbd105c9e9cea656a8b9/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f4db3a8fd7c62e69f39cd56b363540db9fe2e7b8f2f68a85028a07793e0b85f2-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_f4db3a8fd7c62e69f39cd56b363540db9fe2e7b8f2f68a85028a07793e0b85f2/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f52781039b4c7e9f9f2a8f37898d4836f4d6d355cb30469d23f5cd553874a7ee-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_f52781039b4c7e9f9f2a8f37898d4836f4d6d355cb30469d23f5cd553874a7ee/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-637aa2b40c625556a57c0197b885e8ae1a688c1bf0a18f80acd940cbef160e14-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_637aa2b40c625556a57c0197b885e8ae1a688c1bf0a18f80acd940cbef160e14/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-227bf16d1232b24440be8023b2bae4d31a1769a3bc4f9851293fd6e92f366cff-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_227bf16d1232b24440be8023b2bae4d31a1769a3bc4f9851293fd6e92f366cff/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-13e421fefdec1bea69356ec7e9b4034f73347be8b03e8e9a9fbc87dc2cb294b6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_13e421fefdec1bea69356ec7e9b4034f73347be8b03e8e9a9fbc87dc2cb294b6/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-27dd2396d5989f0bc6742f12a4fee85616ab702394f3a2c3b54b24ce7fda7e2a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_27dd2396d5989f0bc6742f12a4fee85616ab702394f3a2c3b54b24ce7fda7e2a/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c21ee1794eccb63ea6eec88b05bfc5b6436e4b84cb9a12f44b7899a0b0dea06a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_c21ee1794eccb63ea6eec88b05bfc5b6436e4b84cb9a12f44b7899a0b0dea06a/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-eaf02a02862e64d8858aeb8dca29aa925bda732bff9ff5d8e307d6202710a985-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_eaf02a02862e64d8858aeb8dca29aa925bda732bff9ff5d8e307d6202710a985/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9e56433611f5a614a3e40564b3b6d4523bac66e15ea5390749d337ab261d0ec8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_9e56433611f5a614a3e40564b3b6d4523bac66e15ea5390749d337ab261d0ec8/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ebc8a366f920012d85a63064101aebad0cc1f1bb1d51369bf313fc19a80dc05f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_ebc8a366f920012d85a63064101aebad0cc1f1bb1d51369bf313fc19a80dc05f/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8bce5fb2822f23a8e3970f57d31c7f15683723b2e1a75051994dd35981346d73-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_8bce5fb2822f23a8e3970f57d31c7f15683723b2e1a75051994dd35981346d73/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7113a412a48e3f2dd80fdb9fb53d461883d39565ca917a37ec85db0eacc40688-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_7113a412a48e3f2dd80fdb9fb53d461883d39565ca917a37ec85db0eacc40688/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f2a09db9032d52476c0d5177243db3e3a4822017b2dbd1161b524477faf9951d-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_f2a09db9032d52476c0d5177243db3e3a4822017b2dbd1161b524477faf9951d/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8d6ffece91d45840e8378898d357a433d20d1ead3266434322fc780e3bdb4e7a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_8d6ffece91d45840e8378898d357a433d20d1ead3266434322fc780e3bdb4e7a/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b1715311f2db98809942bf49519e312e143e0a745693df14ebe17e05fbdb8eda-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_b1715311f2db98809942bf49519e312e143e0a745693df14ebe17e05fbdb8eda/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ba0485d3471e2fe21fcd5fb532e00db33513d31c0321fabfd601d65576bb65d8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_ba0485d3471e2fe21fcd5fb532e00db33513d31c0321fabfd601d65576bb65d8/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-df6786c3bd20d93e1230f8b59212221a7e9de0eefdc39ac2f7192b76047d2803-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_df6786c3bd20d93e1230f8b59212221a7e9de0eefdc39ac2f7192b76047d2803/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6b7f3996f5757948b156740ec97f61bdc8e2b609f51a38cdc7ea05ed7fd1f491-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_6b7f3996f5757948b156740ec97f61bdc8e2b609f51a38cdc7ea05ed7fd1f491/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-98dd67104ca5dc13455783eefea4f0e9de90952470d48e96aaab2ff5e6ac379d-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_98dd67104ca5dc13455783eefea4f0e9de90952470d48e96aaab2ff5e6ac379d/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-611a789dc4b5ce96ef60dfa63ed0c77192d7111f29e248ee8d7931d3e0171c68-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_611a789dc4b5ce96ef60dfa63ed0c77192d7111f29e248ee8d7931d3e0171c68/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6f79f9cd5a0933a9ec2677915fb1b92c1096c75165d31ebc2c164c6e3e85ddc2-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_6f79f9cd5a0933a9ec2677915fb1b92c1096c75165d31ebc2c164c6e3e85ddc2/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-91e7cbe84292aa360181b520bf882b94abc99dcd23ff8657729e23fff98dbf84-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_91e7cbe84292aa360181b520bf882b94abc99dcd23ff8657729e23fff98dbf84/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b6a4aa6d970860b1ef463cac3ea42b2fa446d2af02cda319727ea559c3b219a0-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_b6a4aa6d970860b1ef463cac3ea42b2fa446d2af02cda319727ea559c3b219a0/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-bafbc5b2763a01f9a20a1e6ce4bc612e95775cde9becabb8ed29667e07acf77b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_bafbc5b2763a01f9a20a1e6ce4bc612e95775cde9becabb8ed29667e07acf77b/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ebe2cabf50990753ec25f8ee5aa01d9c6b1fff4cd7dba9b970b27bf6a2a919ee-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_ebe2cabf50990753ec25f8ee5aa01d9c6b1fff4cd7dba9b970b27bf6a2a919ee/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-79fe23a6b5f014529a46504ef8a2e54edb801dacbbe32cbf0b8d4e0c1d26f813-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_79fe23a6b5f014529a46504ef8a2e54edb801dacbbe32cbf0b8d4e0c1d26f813/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-abe2879afa276d0b16ce493283655e600385c4aa613cb56f0ff6548846b33b4f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_abe2879afa276d0b16ce493283655e600385c4aa613cb56f0ff6548846b33b4f/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9477b1088e1a7d8e7dfa9b086a464ec02a9d4e3c25dbe3f0c4dcd97fc41b13e5-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_9477b1088e1a7d8e7dfa9b086a464ec02a9d4e3c25dbe3f0c4dcd97fc41b13e5/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f17944b7339d0d84be14cd71d31c10b495df98114d5af917259df75540551fa4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_f17944b7339d0d84be14cd71d31c10b495df98114d5af917259df75540551fa4/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-400fb0da6b26d5ad0dada81386e3509ceee892450bb3bb07bfb818655fe6388a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_400fb0da6b26d5ad0dada81386e3509ceee892450bb3bb07bfb818655fe6388a/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-51805cbc48c47b1698237121609d7c5585f3d3780ba12dbe02f1ae74fe20f8b8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_51805cbc48c47b1698237121609d7c5585f3d3780ba12dbe02f1ae74fe20f8b8/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-443869906d5762513b4d5591d0c48cfc5ea5a21a1bd42e192bd16430b5887707-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_443869906d5762513b4d5591d0c48cfc5ea5a21a1bd42e192bd16430b5887707/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-bf1285e742efe7824381c9cb3df2b455f70f996597ab7a0ce8344c8248cd6d8c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_bf1285e742efe7824381c9cb3df2b455f70f996597ab7a0ce8344c8248cd6d8c/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f26daa540292860d856cbdaa8d971c5d72419e96da82f61e94ff5a3e94d56e2f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_f26daa540292860d856cbdaa8d971c5d72419e96da82f61e94ff5a3e94d56e2f/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0f1f53582997baad9aa36456f0268590ba384be42726f6ad1fdd6bbcfc626c08-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_0f1f53582997baad9aa36456f0268590ba384be42726f6ad1fdd6bbcfc626c08/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7f373fb8e7ee89bb77c43a376ceb5bd22e4a88b2f5ada6914b49b35449105263-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_7f373fb8e7ee89bb77c43a376ceb5bd22e4a88b2f5ada6914b49b35449105263/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-84ca03417474c9a2acf48d260d1c053759f95b6589e9e4f45809ef56e7e9ce46-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_84ca03417474c9a2acf48d260d1c053759f95b6589e9e4f45809ef56e7e9ce46/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b09dcf7831947fc2c23cc2d570d548a398cc1db47ed5940401e41f5019878917-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_b09dcf7831947fc2c23cc2d570d548a398cc1db47ed5940401e41f5019878917/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c94f38cbeae26e9eb7f539b144bef30f5f43c9a5ed292614a07fea745701543c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_c94f38cbeae26e9eb7f539b144bef30f5f43c9a5ed292614a07fea745701543c/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0ca018bd91eb89e4ffad42f3b5e7aa6e30faf11310eeb93e800cd696cdfb652e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_0ca018bd91eb89e4ffad42f3b5e7aa6e30faf11310eeb93e800cd696cdfb652e/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d4572856894af9013e311991e4371c81498ee30b1fc90ee840632d1a3a512193-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_d4572856894af9013e311991e4371c81498ee30b1fc90ee840632d1a3a512193/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d4e65beafd0c9ae94ef7803c610f825fee8845ef9431742c5c11714ed2544ae2-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_d4e65beafd0c9ae94ef7803c610f825fee8845ef9431742c5c11714ed2544ae2/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-486d294efbdab2b8b9a254695a0f228c54fe75fe2df6adcc71c520d4e1976eac-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_486d294efbdab2b8b9a254695a0f228c54fe75fe2df6adcc71c520d4e1976eac/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-154a4c5d9380e2a084bb8747db72990255ac96e6161d10cae80daa514df78c0f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_154a4c5d9380e2a084bb8747db72990255ac96e6161d10cae80daa514df78c0f/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2db48318a3bbf367e6df3414a58b3b5d9c4e0bd5a688ce7308dd8df127baf37f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_2db48318a3bbf367e6df3414a58b3b5d9c4e0bd5a688ce7308dd8df127baf37f/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-528453ea57613e45540f349b229866c680a17eff84af54642571acabe799c354-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_528453ea57613e45540f349b229866c680a17eff84af54642571acabe799c354/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-23f07cde725d495d43e3ab8ef3d7e2768ec81a0c98d05d2cddded7bcf3be5022-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_23f07cde725d495d43e3ab8ef3d7e2768ec81a0c98d05d2cddded7bcf3be5022/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d67c9b75cf43f3c46d9a23caf6921b2030d4efa1b160be4bcb5f80abcba8abf2-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_d67c9b75cf43f3c46d9a23caf6921b2030d4efa1b160be4bcb5f80abcba8abf2/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a56ddac5571fa0e1c36411e4202663641fa9737a6d22b8051bb3cbeb19207f24-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_a56ddac5571fa0e1c36411e4202663641fa9737a6d22b8051bb3cbeb19207f24/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9cd79e8d48d8f6306a6ea47e3239918ef393a09e6ad18895b0d365c1332502e8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_9cd79e8d48d8f6306a6ea47e3239918ef393a09e6ad18895b0d365c1332502e8/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-108aa19ab15996026054223cb4ae1da6d952682ba17b71fcd0e09686d13a3f70-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_108aa19ab15996026054223cb4ae1da6d952682ba17b71fcd0e09686d13a3f70/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-98140fab01ba25d74943e6bcc2acbee688b8356d2925e22330cfbb7a1a23c075-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_98140fab01ba25d74943e6bcc2acbee688b8356d2925e22330cfbb7a1a23c075/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-60b0f8b20b74d3ef1d937a368e98d1cfb0a31a9c5b6d81d34872b9a6310dcc07-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_60b0f8b20b74d3ef1d937a368e98d1cfb0a31a9c5b6d81d34872b9a6310dcc07/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_13fc65a9-e5dc-4cbb-9c09-25b0b08f4c03}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-53a27f3e75e7d285d0dfe635dc13c2f72aae257eafc0d3e2fcfcc4040b056487-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_53a27f3e75e7d285d0dfe635dc13c2f72aae257eafc0d3e2fcfcc4040b056487/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-49317ac60d1d798e1f8fe89c255a2028b1642d6e65ecdcf6bf720270ce12b960-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_49317ac60d1d798e1f8fe89c255a2028b1642d6e65ecdcf6bf720270ce12b960/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7aff468d197ea449f8cda6cb6719cde1cb3d214f47298b92c03eb10b8da7014b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_7aff468d197ea449f8cda6cb6719cde1cb3d214f47298b92c03eb10b8da7014b/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a10ec940197f56cbac77926b9573a7d3b04dc5012e3edd5df5289d371ea74428-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_a10ec940197f56cbac77926b9573a7d3b04dc5012e3edd5df5289d371ea74428/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a3275b0fbf3ee381927da85ac6794ff29e54ef10a59031fb639ef8170a00e963-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_a3275b0fbf3ee381927da85ac6794ff29e54ef10a59031fb639ef8170a00e963/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0a7bd61c612b2d143d7c4293f82bb570ca9ee3978f43cd656f71179805bed7ef-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_0a7bd61c612b2d143d7c4293f82bb570ca9ee3978f43cd656f71179805bed7ef/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ce3fff17fd34b163da9513e5009262e520421ec594c33b9d5c16bc3ac511f950-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_ce3fff17fd34b163da9513e5009262e520421ec594c33b9d5c16bc3ac511f950/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-65465ca77dcf48e6bf1a786fc858f46b2b4fa802a0558a252c56984ed33e41be-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_65465ca77dcf48e6bf1a786fc858f46b2b4fa802a0558a252c56984ed33e41be/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-73f3ece7417f1ce8eb05da3450362b4e52a651f9a4313894c5ed6cbfde9f61ab-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_73f3ece7417f1ce8eb05da3450362b4e52a651f9a4313894c5ed6cbfde9f61ab/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-72d30a598499416c4f4bc949f9e4e9dcadf8fd02ebf5f594352227664276e13d-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_72d30a598499416c4f4bc949f9e4e9dcadf8fd02ebf5f594352227664276e13d/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6f107299a90bc6e2d9a962d28236eb99f1589d049e53d6f05cedf5a2d6820466-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_6f107299a90bc6e2d9a962d28236eb99f1589d049e53d6f05cedf5a2d6820466/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2e1ec1c72cd4eeab43d5a5e5f3468b751c732af97a4d04af969589004b6b1f15-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_2e1ec1c72cd4eeab43d5a5e5f3468b751c732af97a4d04af969589004b6b1f15/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ecbbd1f41ef3fee6317a6845d86e3103f7c9c6019e117f3023386b2c43c05f41-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_ecbbd1f41ef3fee6317a6845d86e3103f7c9c6019e117f3023386b2c43c05f41/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-dd0407527936cadd32b2d76e572a6b24c3fa0aebc71100227038538f9027ca89-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_dd0407527936cadd32b2d76e572a6b24c3fa0aebc71100227038538f9027ca89/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-cae5ec5b02228b25142bd564d9d45cd61a3011b09f8e4444f456913bb7911978-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_cae5ec5b02228b25142bd564d9d45cd61a3011b09f8e4444f456913bb7911978/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6b62e2878a4aa7abf23fbf3585f8b5d5317dd7487807d4253906aa7b5815f5eb-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_6b62e2878a4aa7abf23fbf3585f8b5d5317dd7487807d4253906aa7b5815f5eb/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-20d82f50c7675c5aa0592b3f0fdae6712d83c0cdc2d10a579e7b32b1e4560455-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_20d82f50c7675c5aa0592b3f0fdae6712d83c0cdc2d10a579e7b32b1e4560455/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-62847ad95dc56b400bd0471d62da88904d0bec489ef9de82b84b2179e70eecbd-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_62847ad95dc56b400bd0471d62da88904d0bec489ef9de82b84b2179e70eecbd/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-3665d5c55e91e2410b877d72d28460742791ff7a173c2163485a97154376468b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_3665d5c55e91e2410b877d72d28460742791ff7a173c2163485a97154376468b/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-17f368591afda1e361c872f855576202e7bff1c71e432ae0d8a5027b4ebf6b24-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_17f368591afda1e361c872f855576202e7bff1c71e432ae0d8a5027b4ebf6b24/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1d38ee595dd7aed8bac66c4bb54f252e16a37bd7efca5784c5cf5251edf20660-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_1d38ee595dd7aed8bac66c4bb54f252e16a37bd7efca5784c5cf5251edf20660/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e547285f102de03190bf6fbee132d5b251eb632b8d94a9084bf67255640ecab5-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_e547285f102de03190bf6fbee132d5b251eb632b8d94a9084bf67255640ecab5/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fc1859fb976b75f059fb7ebc5e75032a4a540461b6f37be57c5b85993f0fa18b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_fc1859fb976b75f059fb7ebc5e75032a4a540461b6f37be57c5b85993f0fa18b/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6a6ca63f3da8f6ef340e3c380d49e0f09535f05ca917d2842f0f76fc9f34a570-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_6a6ca63f3da8f6ef340e3c380d49e0f09535f05ca917d2842f0f76fc9f34a570/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ea8d5529de4316b9bec5e80d604a37a4c2304c653984f7ed04d479c7da15b1b7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_ea8d5529de4316b9bec5e80d604a37a4c2304c653984f7ed04d479c7da15b1b7/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-411c292e140d21cf2a4270627e277c8d31dd18f64112dec9162cd01e31d1364e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_411c292e140d21cf2a4270627e277c8d31dd18f64112dec9162cd01e31d1364e/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-1}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9301b78b64f71dbcd7e18c9f3e74c72100b94f84b857ebbf0cfe85d2f88ade01-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_9301b78b64f71dbcd7e18c9f3e74c72100b94f84b857ebbf0cfe85d2f88ade01/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-11a7307088b796963f5249c11d8dcf99cdb35fba9c68115d1510ab3d8951efd6-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_11a7307088b796963f5249c11d8dcf99cdb35fba9c68115d1510ab3d8951efd6/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c7c4a15e-b20c-4898-bb4a-864f34ff34b2-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f60dff2e9fb2a0bdeb0c6d704174d67e4fdb6d4165647b463c9ff5ecff124008-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_f60dff2e9fb2a0bdeb0c6d704174d67e4fdb6d4165647b463c9ff5ecff124008/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_deba344af418957cbd9dc0834a100bc30ba242d7fddb4dc6ba10a87d0af32dc1/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4465cfacd1bd870fc3c4a1ca3baf450e65a2269b531449fc39088d646166aca7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_4465cfacd1bd870fc3c4a1ca3baf450e65a2269b531449fc39088d646166aca7/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_deba344af418957cbd9dc0834a100bc30ba242d7fddb4dc6ba10a87d0af32dc1/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-cab84e227e3ea301e7215d2660ef1abc95c316fad1e790cd6b0ff76708884af8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_cab84e227e3ea301e7215d2660ef1abc95c316fad1e790cd6b0ff76708884af8/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_72feea41a0b54e4c9a761933079cba1b2c012e5a5d4b2953ffaa86faaa29a648/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ea196f677da0ee2484c509ea2d1e6422e3f4abbaf9b3694af960942d697213d0-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_ea196f677da0ee2484c509ea2d1e6422e3f4abbaf9b3694af960942d697213d0/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_72feea41a0b54e4c9a761933079cba1b2c012e5a5d4b2953ffaa86faaa29a648/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-291d0d979338966adc92b269dca4ced52382b7771a6b91f9bc71e440952e9bca-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_291d0d979338966adc92b269dca4ced52382b7771a6b91f9bc71e440952e9bca/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_b33d5ca3311e31b99daa1c1f13ea5571713538a7889627153ea431debb9b5e2a/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-41319614792667b0df14d9df5dce27730745ece78b4500bbbb683cb1d83d08fd-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_41319614792667b0df14d9df5dce27730745ece78b4500bbbb683cb1d83d08fd/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_b33d5ca3311e31b99daa1c1f13ea5571713538a7889627153ea431debb9b5e2a/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-5340ad727cf93b663444abdc3d3e1b71b410b4d1cafdd20d86390e3a0d316c81-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_5340ad727cf93b663444abdc3d3e1b71b410b4d1cafdd20d86390e3a0d316c81/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_076a1e95f3384b58a77a672c7c36463b091e574b5a6f6eaf78841537b0d1c930/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f7430d394f574c3b65b2c4a8503f4d6fe59d06d028c63b3dbe9d8ca3147918b3-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_f7430d394f574c3b65b2c4a8503f4d6fe59d06d028c63b3dbe9d8ca3147918b3/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_076a1e95f3384b58a77a672c7c36463b091e574b5a6f6eaf78841537b0d1c930/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-08522902c67de8ac584fa34c92fa460abd51e07ea6b7cbfbe511f6c49d75c901-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_08522902c67de8ac584fa34c92fa460abd51e07ea6b7cbfbe511f6c49d75c901/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_05e976592f58a292874d99bd7627e655b15c5460455a08b9ce67e9f7f65b6269/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-86aec1b2ed49dcb4581bddcb95097dafc4e04332fee4e3f429fa6622c0ae7423-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_86aec1b2ed49dcb4581bddcb95097dafc4e04332fee4e3f429fa6622c0ae7423/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_05e976592f58a292874d99bd7627e655b15c5460455a08b9ce67e9f7f65b6269/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-89583c9d8173f86736e142bad1d53806b2e14598ac8e5c5b57a3d4d6350825c4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_89583c9d8173f86736e142bad1d53806b2e14598ac8e5c5b57a3d4d6350825c4/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_dd4366d716cccdd83b6f777ee966157b92838569f094371b325b926e73c7b1b8/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a9be3cb6d9f42925eab55ecb13110d675538a3ab5e51edf1ebe93e45ac7498b7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_a9be3cb6d9f42925eab55ecb13110d675538a3ab5e51edf1ebe93e45ac7498b7/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_dd4366d716cccdd83b6f777ee966157b92838569f094371b325b926e73c7b1b8/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c89ee60ba03610e4be991f6b5244044c96be3c52ac01c745c466bc7acc5129e8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_c89ee60ba03610e4be991f6b5244044c96be3c52ac01c745c466bc7acc5129e8/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_4a1632637ce87fe991848942b028c36732b2bea00920d275ffbcaf2cd9446152/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b11ab96441a0143f6d94da043e24f1d5a6953d41ed59e8494eddf8fc4b504eaa-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_b11ab96441a0143f6d94da043e24f1d5a6953d41ed59e8494eddf8fc4b504eaa/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_4a1632637ce87fe991848942b028c36732b2bea00920d275ffbcaf2cd9446152/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e7177ef165288b2cfdd95e00844c78f7b7aaa2002107cb6d365ec42c43da8d1e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_e7177ef165288b2cfdd95e00844c78f7b7aaa2002107cb6d365ec42c43da8d1e/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7f5a7ca6eb12e8ac322d8bb0deb59630d721ab141acc3a941e168d98af507034/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-608e9c811acd31af1333a967ae7fbfc6e108f107a9eeb8e65695612c33a27a09-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_608e9c811acd31af1333a967ae7fbfc6e108f107a9eeb8e65695612c33a27a09/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_7f5a7ca6eb12e8ac322d8bb0deb59630d721ab141acc3a941e168d98af507034/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-562f5fc48bd3d23bc95ef5a2b950cbbb88708ca270992fa6323343dae7969403-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_562f5fc48bd3d23bc95ef5a2b950cbbb88708ca270992fa6323343dae7969403/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_e3444362e0b630df1b5f70a28089b0764d9ddc886dda852a0ef1300e369aee4d/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-faa528c855571f526c505e6b63af0be02a669dbceb95c001d3078896aea2449c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_faa528c855571f526c505e6b63af0be02a669dbceb95c001d3078896aea2449c/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_e3444362e0b630df1b5f70a28089b0764d9ddc886dda852a0ef1300e369aee4d/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9c4d667f951f20083f816432aa572776e3135655fd8e33f43b74ebf582f307e9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_9c4d667f951f20083f816432aa572776e3135655fd8e33f43b74ebf582f307e9/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_c95d82463222cf5ea385760eb1f0f0ee28a876620e29fbd59f8f4cb8a5307bc8/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1610e5fda0bed864f7723bcd2fc94b948d7acbdd1bb296c7c09dde7060ad5448-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_1610e5fda0bed864f7723bcd2fc94b948d7acbdd1bb296c7c09dde7060ad5448/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_c95d82463222cf5ea385760eb1f0f0ee28a876620e29fbd59f8f4cb8a5307bc8/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6af3f661193e72ea38cbc498414a38294e53d6c201eef9555b2888c1adbdae7f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_6af3f661193e72ea38cbc498414a38294e53d6c201eef9555b2888c1adbdae7f/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_f73d0b9c84ec91b594b00eb3800c372884050fce3e4fb9d80eb407d7b0697412/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-09fbc68cb81435106363e102bb163cfe97bd851610b0daa65e4af37586242878-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_09fbc68cb81435106363e102bb163cfe97bd851610b0daa65e4af37586242878/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_f73d0b9c84ec91b594b00eb3800c372884050fce3e4fb9d80eb407d7b0697412/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b402994b2e254a233bc4c102d1be16cb81d56b23722c12fa504f8af15ec8760a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_b402994b2e254a233bc4c102d1be16cb81d56b23722c12fa504f8af15ec8760a/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_67ad72cbb9eb9c4746249922e6232b1a17b5d6eeabd9b69ed4d527c1676c77bd/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c7b1f6d4fe7a1767d1d039434869344bff311a88039f3f36c954f97f21b3ebf3-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_c7b1f6d4fe7a1767d1d039434869344bff311a88039f3f36c954f97f21b3ebf3/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_67ad72cbb9eb9c4746249922e6232b1a17b5d6eeabd9b69ed4d527c1676c77bd/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-060a4950ed5780ba74bc100a05ad36b4baaec0cde8f8315e79d470c0d6f6ed83-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_060a4950ed5780ba74bc100a05ad36b4baaec0cde8f8315e79d470c0d6f6ed83/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_ad38a059079822e7331470e086bc1caca3dbe435878b38e1355229a39d1d25d2/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-64cc76a84e86dfce248344e2d5cce3846cb7bd59d9f510b345e8b8f6f8045259-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_64cc76a84e86dfce248344e2d5cce3846cb7bd59d9f510b345e8b8f6f8045259/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_ad38a059079822e7331470e086bc1caca3dbe435878b38e1355229a39d1d25d2/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-bc7afc6c841b4749167d612ee8874d809e93668e81a653de4066368b4a48e8cf-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_bc7afc6c841b4749167d612ee8874d809e93668e81a653de4066368b4a48e8cf/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8eb82c90485cc4281676866b62e6820b60a38ba81068e95254a7d0ddaddc59c3/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-70e3c62e269dbc27ac0124ce039255b47690342dc2cc8647ac5e9a419c24ee5b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_70e3c62e269dbc27ac0124ce039255b47690342dc2cc8647ac5e9a419c24ee5b/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_8eb82c90485cc4281676866b62e6820b60a38ba81068e95254a7d0ddaddc59c3/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-de0f4257cda0c2ad674beabe91a443a5cee25adddbadbd72232d20daf4a348e0-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_de0f4257cda0c2ad674beabe91a443a5cee25adddbadbd72232d20daf4a348e0/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_cc9b305fb2c95f582d144f1063fc12fd996e757f84738e3de846b0197981bcc2/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b0068e24d909cb9085f10f77233b1e8e2b648b12ad26906bb532ce442f3452fb-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_b0068e24d909cb9085f10f77233b1e8e2b648b12ad26906bb532ce442f3452fb/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_cc9b305fb2c95f582d144f1063fc12fd996e757f84738e3de846b0197981bcc2/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9302f4f7f2e4b8ca6513a742549b18e98519a60aeb16e0156491aa754ab86fcc-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_9302f4f7f2e4b8ca6513a742549b18e98519a60aeb16e0156491aa754ab86fcc/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_8930c217bc20a5abf9313fd714a77bf47acefb006e95ba07b5e48caf872541b0/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-448c132659a1369a42e70e95b614ab2644947dfe9d4549dea39bdaf55ae26374-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_448c132659a1369a42e70e95b614ab2644947dfe9d4549dea39bdaf55ae26374/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_8930c217bc20a5abf9313fd714a77bf47acefb006e95ba07b5e48caf872541b0/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_b10737a8-5da4-4052-9b7a-5b07f21e02d0}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d0916289a64c89db14f4b09459b6404cd24668d7f2d2ff0668631174beb5bc19-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_d0916289a64c89db14f4b09459b6404cd24668d7f2d2ff0668631174beb5bc19/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_9b6e19b8cab376fffe309f090beeecad08903c0975c9e7ffc480dd46074b97b3/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-c986bfe95c3405f1665c12b4333122cf61fabbf943e0afdb38d4b8929bc01ac4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_c986bfe95c3405f1665c12b4333122cf61fabbf943e0afdb38d4b8929bc01ac4/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_d5ea172c189c6a6a3b61a7e83f7263b9b38cad756d2d5dec1b2db88162eece2a/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d361ce2137e89c76be8f487cbb8807e1ade9589bf5e356fcbc2540f59aa56171-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_d361ce2137e89c76be8f487cbb8807e1ade9589bf5e356fcbc2540f59aa56171/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_bb1cb8e00fd4cee513b2481bd5faf39842ad9c57f80f2e50ddb48763fd030721/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-fe9f27c6710225035e741ab8b669dc159f4c016ff421b7292e9cf52b410f113e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_fe9f27c6710225035e741ab8b669dc159f4c016ff421b7292e9cf52b410f113e/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_dc26eca15bb97d42bcfa019de5080da54fb3d7624809aa1c8ac731e710544e18/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e39a11cb267f85b5a16655190d1d6395791ba1afa3d1050fc7de6106899c5197-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_e39a11cb267f85b5a16655190d1d6395791ba1afa3d1050fc7de6106899c5197/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_90d4cb308fc14e4832336ebd1e1387a791a20b92af633d0fec9161b807e13427/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-da1823a15ae91422ed25bee8390afea31d539a3ce5087d50d746d910f6d9f2f8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_da1823a15ae91422ed25bee8390afea31d539a3ce5087d50d746d910f6d9f2f8/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_ba322b423888fe93e86180e98668f486afa915565edf3813af3f9da6ad7c9dc9/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-7d644106168eda8dfe40b1a79cfe07cc4ad69f40df2fb13858fe58728af084bd-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_7d644106168eda8dfe40b1a79cfe07cc4ad69f40df2fb13858fe58728af084bd/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_42bc978e75d5fed90c10d5812a8df7a6153e311b19b9c7faf41a588bdc4da7d8/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-49bb980c861bf0db681172f8610e4663b9cf2dc5f83c5a8d49ad645873d20a62-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_49bb980c861bf0db681172f8610e4663b9cf2dc5f83c5a8d49ad645873d20a62/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_6819b99fa39a0e548de5fdfc59e8ef1b3c609cd7b5cbe9c0c765911e4eb50cbd/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-5f722e7441428249a620a10a1bad788ddf96c13cebda3be9b54bc021605341b8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_5f722e7441428249a620a10a1bad788ddf96c13cebda3be9b54bc021605341b8/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_0732f1387e308186fd8c8d8089131799be227d5a656a446048f7e0fd0d5047ce/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-868ceaddb4382e723150b21ae1ce04f431e1f04b40212c5ca930c8ed26671afc-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_868ceaddb4382e723150b21ae1ce04f431e1f04b40212c5ca930c8ed26671afc/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_b30a25fc50be09814337382f4cf8a1fd03b8c79d8b860ac509c629f8c8a5e5f0/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9292f811797ee8e56233f46f6d28d946d57773e9f2a6abf609301c5dd0709cf1-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_9292f811797ee8e56233f46f6d28d946d57773e9f2a6abf609301c5dd0709cf1/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_a47669d662bd18dbd323737d84fd45753c01abaa81b4761eb51b35ce04ee7491/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0113327cba37fa3fe8730981f5943b3689e3b93906b0d66051145fbd6dc52c12-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_0113327cba37fa3fe8730981f5943b3689e3b93906b0d66051145fbd6dc52c12/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_27c12f520d68a8a754b11ad44cf8ad3ef5c1ec281fc9d84fbc64a7b573cb68b5/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f565c9c78849e1761f634aeeeb6d6ee35f89af1554f4afe50b02d3d7233b5536-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_f565c9c78849e1761f634aeeeb6d6ee35f89af1554f4afe50b02d3d7233b5536/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_ec14aafacf15628918390531318a4827d3ac8771b75d60a8239d901dcb4fd898/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-9eee607453cd1462cfc2fdf7dfdcab2016fe0701329e078cc1d901818ffa3ce4-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_9eee607453cd1462cfc2fdf7dfdcab2016fe0701329e078cc1d901818ffa3ce4/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_507fabae3d24c2448c8f2af676d68df0a22716a06fa7ffc3b4f865e9272ecdc8/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0f149a7ff9e4aec7858d3eebbcd26e137bc086f01a132057e701a304d713cda7-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_0f149a7ff9e4aec7858d3eebbcd26e137bc086f01a132057e701a304d713cda7/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_4924918e121a92f802a094de2d67e9c2673c9fdc39faa6a11ac1d38b631a2914/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-d9e181508dc318aa3aec6db6d6f10c48b0f36edc4bc3e95027d199e77da19793-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_d9e181508dc318aa3aec6db6d6f10c48b0f36edc4bc3e95027d199e77da19793/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_383e7567e79adf2178e3d25785855905a4477b6c8daf16bc50d5f9f44d2343d9/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6a0da04aae3f1342a4f6aaa310b1fcdcb6c5382187f7610d90dffb94884e74d9-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_6a0da04aae3f1342a4f6aaa310b1fcdcb6c5382187f7610d90dffb94884e74d9/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_9faad3c50bc64c297531a36bd2ab235b680070d8233ce5ab7d964ac36c7c5563/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-f7c135234a821cdf109ab2e997d43310a4ee51e5865eea7ebbaafe76d71b732c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_f7c135234a821cdf109ab2e997d43310a4ee51e5865eea7ebbaafe76d71b732c/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_175e3e5c14dc32cafe73af4b4b8a6f5732697a097cdf2e8699a316224afb7e31/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-6b79f25650f49dee7c19bcd7753f79e7348fb4c332dd3d67e4b06a6150c8f45e-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_6b79f25650f49dee7c19bcd7753f79e7348fb4c332dd3d67e4b06a6150c8f45e/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_8964c792ed5c95032e2318b2dbac3b4f8453cde352067fa9a029d04a0c2d5fae/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-e23e6d368629ebe9f1dd8dba224f5a8b729608b939606f7c2a9edd1ed9ee8c6f-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_e23e6d368629ebe9f1dd8dba224f5a8b729608b939606f7c2a9edd1ed9ee8c6f/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_c8ee4f2bcdb954fa5d5d45c0c65631ce211c0293d7a69ed6ef4644e523e919ac/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_34ae13f0-d6d9-43f7-befb-15d024e88e89}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-8a94184fd7b5022ad836488e8948f2fbd8ac818e40cb92f49f1bc51d2a6d2647-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_run_8a94184fd7b5022ad836488e8948f2fbd8ac818e40cb92f49f1bc51d2a6d2647/run.flagfile"
  FLAGS "--module=iree_DeepLabV3_fp32_module_131aa61ffcacafc6ae701ff89045cf42d2490ac0c4a1d862bc83c23edb3b92e5/module.vmfb" ${_MODEL_RUN_FLAGS_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-a9fd069a0dc5dbd3f65aadf383cc7ead04cd8f16bb7ee58b473cfe692a7933f8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_run_a9fd069a0dc5dbd3f65aadf383cc7ead04cd8f16bb7ee58b473cfe692a7933f8/run.flagfile"
  FLAGS "--module=iree_MobileSSD_fp32_module_db9784471b47ae8bf55ca7e0821e35a1686256a208df40443c114f1adcdd26f6/module.vmfb" ${_MODEL_RUN_FLAGS_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-dd39a2d0e6a198ff1ff6eeac105fe39892cc90965249e7e8ecb453e6b9a0e5fe-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_run_dd39a2d0e6a198ff1ff6eeac105fe39892cc90965249e7e8ecb453e6b9a0e5fe/run.flagfile"
  FLAGS "--module=iree_PoseNet_fp32_module_aada5dcefdd361b5227276129e93547a6932a05d380acd342fa33e5b672d498c/module.vmfb" ${_MODEL_RUN_FLAGS_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b6caf787b60262f27b833f67aa87a394b5298b1965a0690a980bf01b9eb6e3cf-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_run_b6caf787b60262f27b833f67aa87a394b5298b1965a0690a980bf01b9eb6e3cf/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp32_module_887c7a7b540f11ee5e0158143fd46a503a4851211b10b353ec0e00b8b1beb575/module.vmfb" ${_MODEL_RUN_FLAGS_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-1a1883b71e25f7c0845d81504620d4e3f40903e25a3dcd9f4a75080ba5964eda-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_1a1883b71e25f7c0845d81504620d4e3f40903e25a3dcd9f4a75080ba5964eda/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_11766f32ea6a3121d7527bcdf32dead45ab7b3922d72addb46945cfdab784ec0/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b4788176d092953bb8490e61a09828fa371815977615a78e68a38180c6b507fd-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_b4788176d092953bb8490e61a09828fa371815977615a78e68a38180c6b507fd/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_0efb794c3a385045fa4d3d086c2a593ce67c4807e9456271f05f4e28490d1c49/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ed8f6bfbd572877cbc548a986d386f60a8e9060a3e3bfdd478680a41655382f3-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_run_ed8f6bfbd572877cbc548a986d386f60a8e9060a3e3bfdd478680a41655382f3/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_fp16_module_7a3d36d7234ce4abfd833964492631bd81df823e6bde8cd3a1fadfa4faa7c787/module.vmfb" ${_MODEL_RUN_FLAGS_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-2a60a6a2462142cb66816ee12062c6f4efa3def8fea9adc4df46aaacd9c136bc-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_run_2a60a6a2462142cb66816ee12062c6f4efa3def8fea9adc4df46aaacd9c136bc/run.flagfile"
  FLAGS "--module=iree_MobileBertSquad_int8_module_77bedc82b9083aa7c270bc37d077f6ff4cecabc307a584a9b7b52e5fe18db858/module.vmfb" ${_MODEL_RUN_FLAGS_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-426fe139b2f7c190b0656c35f5efdc59475286c401654f2ffd80f860f08cf563-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_run_426fe139b2f7c190b0656c35f5efdc59475286c401654f2ffd80f860f08cf563/run.flagfile"
  FLAGS "--module=iree_EfficientNet_int8_module_a4a762fde005b81cec5b11ddf02fc7ac4bb919b71250ffefdd2cecda209ceeaa/module.vmfb" ${_MODEL_RUN_FLAGS_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4aa5488ba3a15a67fc054c6e7c38d7589ea6780501ceae91f7f730b4e0a696d8-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_run_4aa5488ba3a15a67fc054c6e7c38d7589ea6780501ceae91f7f730b4e0a696d8/run.flagfile"
  FLAGS "--module=iree_PersonDetect_int8_module_43147a614fc5476e5a8083ddefc3ecb093c9c9ebf7864b8b6178af952540edae/module.vmfb" ${_MODEL_RUN_FLAGS_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_c59f6ed8-ef78-4ddd-93ea-f173c5e4d6b8}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-0d167e8f1a24edc7620fa7ccd5c9f96a697ba3d0381e2118828af79ccdd06ca5-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_0d167e8f1a24edc7620fa7ccd5c9f96a697ba3d0381e2118828af79ccdd06ca5/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_953183e2-1e84-4a51-a43c-9b869bdc2218-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-ec6627de9282f1e49e0aa3d5bab632a950b310b4d401b6f3c7d12ce969abef2b-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_run_ec6627de9282f1e49e0aa3d5bab632a950b310b4d401b6f3c7d12ce969abef2b/run.flagfile"
  FLAGS "--module=iree_MobileNetV2_fp32_module_86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26/module.vmfb" ${_MODEL_RUN_FLAGS_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_953183e2-1e84-4a51-a43c-9b869bdc2218-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-4366620da66f5f7219af4e0882d29c1a5422ad1740f729b8b85609f120b4932a-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_4366620da66f5f7219af4e0882d29c1a5422ad1740f729b8b85609f120b4932a/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_953183e2-1e84-4a51-a43c-9b869bdc2218-4}
)

iree_dump_flagfile(
  TARGET_NAME "${PACKAGE_NAME}_iree-run-b400f170bd47a674f4d360fe0ec2c79a22b475689ea25a21d7eb38d168b6ad0c-flagfile"
  OUTPUT "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_run_b400f170bd47a674f4d360fe0ec2c79a22b475689ea25a21d7eb38d168b6ad0c/run.flagfile"
  FLAGS "--module=iree_MobileNetV3Small_fp32_module_bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed/module.vmfb" ${_MODEL_RUN_FLAGS_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337_8d4a034e-944d-4725-8402-d6f6e61be93c} ${_EXEC_RUN_FLAGS_953183e2-1e84-4a51-a43c-9b869bdc2218-4}
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d
  ${PACKAGE_NAME}_iree-imported-model-3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025
  ${PACKAGE_NAME}_iree-imported-model-4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59
  ${PACKAGE_NAME}_iree-imported-model-af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882
  ${PACKAGE_NAME}_iree-imported-model-3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6
  ${PACKAGE_NAME}_iree-imported-model-a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee
  ${PACKAGE_NAME}_iree-imported-model-a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded
  ${PACKAGE_NAME}_iree-imported-model-394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337
  ${PACKAGE_NAME}_iree-imported-model-2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e
  ${PACKAGE_NAME}_iree-imported-model-93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850
  ${PACKAGE_NAME}_iree-imported-model-3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4
  ${PACKAGE_NAME}_iree-imported-model-a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225
  ${PACKAGE_NAME}_iree-imported-model-2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697
  ${PACKAGE_NAME}_iree-imported-model-213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0
  ${PACKAGE_NAME}_iree-imported-model-d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960
  ${PACKAGE_NAME}_iree-imported-model-a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6
  ${PACKAGE_NAME}_iree-module-3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d
  ${PACKAGE_NAME}_iree-module-472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d
  ${PACKAGE_NAME}_iree-module-95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f
  ${PACKAGE_NAME}_iree-module-78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a
  ${PACKAGE_NAME}_iree-module-02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f
  ${PACKAGE_NAME}_iree-module-429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d
  ${PACKAGE_NAME}_iree-module-baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49
  ${PACKAGE_NAME}_iree-module-737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742
  ${PACKAGE_NAME}_iree-module-eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed
  ${PACKAGE_NAME}_iree-module-92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a
  ${PACKAGE_NAME}_iree-module-1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732
  ${PACKAGE_NAME}_iree-module-9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04
  ${PACKAGE_NAME}_iree-module-c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84
  ${PACKAGE_NAME}_iree-module-a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252
  ${PACKAGE_NAME}_iree-module-7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055
  ${PACKAGE_NAME}_iree-module-3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90
  ${PACKAGE_NAME}_iree-module-531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc
  ${PACKAGE_NAME}_iree-module-c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b
  ${PACKAGE_NAME}_iree-module-8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8
  ${PACKAGE_NAME}_iree-module-9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3
  ${PACKAGE_NAME}_iree-module-dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933
  ${PACKAGE_NAME}_iree-module-7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d
  ${PACKAGE_NAME}_iree-module-c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4
  ${PACKAGE_NAME}_iree-module-3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c
  ${PACKAGE_NAME}_iree-module-711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1
  ${PACKAGE_NAME}_iree-module-4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152
  ${PACKAGE_NAME}_iree-module-9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38
  ${PACKAGE_NAME}_iree-module-f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe
  ${PACKAGE_NAME}_iree-module-aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76
  ${PACKAGE_NAME}_iree-module-80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757
  ${PACKAGE_NAME}_iree-module-bdd904cc5614ebf77609c7802a2dfc09f139aee2a247a247d10d320de72b0e28
  ${PACKAGE_NAME}_iree-module-45565cae821666fd34bca97be2e4cce3bd61e71308785728737d89acbb9bc9d2
  ${PACKAGE_NAME}_iree-module-04ca0a5077b7dd5ace66d803c9b822dff3428b24e7620a61995aff0907af9533
  ${PACKAGE_NAME}_iree-module-deafafd0926321a4b8e4dc73ed4a30b2ed9317d26488246461415be2ee857eb1
  ${PACKAGE_NAME}_iree-module-fd81a89e9f8773bae142040775c7e3c4774f96b64f07f8d9f66b00191864ff40
  ${PACKAGE_NAME}_iree-module-68f0eb37bb72d0d6605ecdf42691c64125960e122844b0beeae350871a445b1c
  ${PACKAGE_NAME}_iree-module-a7a1553d0739151f06bbc00a3ef8b67b0606463eab4b6607069aa94ea0bfd92f
  ${PACKAGE_NAME}_iree-module-e80d71ed8e86c0756226b2323e27e2c7c0fff8eddde59ba69e9222d36ee3eef6
  ${PACKAGE_NAME}_iree-module-f59cd43a2a2d6e4b3159efa358a6fa0879e72f6f4f0a23af4c8ab550f256986a
  ${PACKAGE_NAME}_iree-module-14a15b9072caaee5e2a274a9bbc436a56d095611e5a8e9841f110741d34231f9
  ${PACKAGE_NAME}_iree-module-e850fce2d36ddb09ccc34471641adb77418b93c0949d22ab75806d7cfc489ae3
  ${PACKAGE_NAME}_iree-module-f963d812114af925e0a4b110ee83aeb0e3b41d49fad19b3f449b6a9ccba43b8d
  ${PACKAGE_NAME}_iree-module-9d909aa679e8c380ff7e93292ef28dbd3bb9e7cc62329f90aef78ce9c7efeff9
  ${PACKAGE_NAME}_iree-module-1ef2da238443010024d69ceb6fe6ab6fa8cf5f4ce7d424dace3a572592043e70
  ${PACKAGE_NAME}_iree-module-f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb
  ${PACKAGE_NAME}_iree-module-91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8
  ${PACKAGE_NAME}_iree-module-8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0
  ${PACKAGE_NAME}_iree-module-83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae
  ${PACKAGE_NAME}_iree-module-244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f
  ${PACKAGE_NAME}_iree-module-5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad
  ${PACKAGE_NAME}_iree-module-012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b
  ${PACKAGE_NAME}_iree-module-5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a
  ${PACKAGE_NAME}_iree-module-7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719
  ${PACKAGE_NAME}_iree-module-0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115
  ${PACKAGE_NAME}_iree-module-2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c
  ${PACKAGE_NAME}_iree-module-78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620
  ${PACKAGE_NAME}_iree-module-0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef
  ${PACKAGE_NAME}_iree-module-69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0
  ${PACKAGE_NAME}_iree-module-deba344af418957cbd9dc0834a100bc30ba242d7fddb4dc6ba10a87d0af32dc1
  ${PACKAGE_NAME}_iree-module-72feea41a0b54e4c9a761933079cba1b2c012e5a5d4b2953ffaa86faaa29a648
  ${PACKAGE_NAME}_iree-module-b33d5ca3311e31b99daa1c1f13ea5571713538a7889627153ea431debb9b5e2a
  ${PACKAGE_NAME}_iree-module-076a1e95f3384b58a77a672c7c36463b091e574b5a6f6eaf78841537b0d1c930
  ${PACKAGE_NAME}_iree-module-05e976592f58a292874d99bd7627e655b15c5460455a08b9ce67e9f7f65b6269
  ${PACKAGE_NAME}_iree-module-dd4366d716cccdd83b6f777ee966157b92838569f094371b325b926e73c7b1b8
  ${PACKAGE_NAME}_iree-module-4a1632637ce87fe991848942b028c36732b2bea00920d275ffbcaf2cd9446152
  ${PACKAGE_NAME}_iree-module-7f5a7ca6eb12e8ac322d8bb0deb59630d721ab141acc3a941e168d98af507034
  ${PACKAGE_NAME}_iree-module-e3444362e0b630df1b5f70a28089b0764d9ddc886dda852a0ef1300e369aee4d
  ${PACKAGE_NAME}_iree-module-c95d82463222cf5ea385760eb1f0f0ee28a876620e29fbd59f8f4cb8a5307bc8
  ${PACKAGE_NAME}_iree-module-f73d0b9c84ec91b594b00eb3800c372884050fce3e4fb9d80eb407d7b0697412
  ${PACKAGE_NAME}_iree-module-67ad72cbb9eb9c4746249922e6232b1a17b5d6eeabd9b69ed4d527c1676c77bd
  ${PACKAGE_NAME}_iree-module-ad38a059079822e7331470e086bc1caca3dbe435878b38e1355229a39d1d25d2
  ${PACKAGE_NAME}_iree-module-8eb82c90485cc4281676866b62e6820b60a38ba81068e95254a7d0ddaddc59c3
  ${PACKAGE_NAME}_iree-module-cc9b305fb2c95f582d144f1063fc12fd996e757f84738e3de846b0197981bcc2
  ${PACKAGE_NAME}_iree-module-8930c217bc20a5abf9313fd714a77bf47acefb006e95ba07b5e48caf872541b0
  ${PACKAGE_NAME}_iree-module-9b6e19b8cab376fffe309f090beeecad08903c0975c9e7ffc480dd46074b97b3
  ${PACKAGE_NAME}_iree-module-d5ea172c189c6a6a3b61a7e83f7263b9b38cad756d2d5dec1b2db88162eece2a
  ${PACKAGE_NAME}_iree-module-bb1cb8e00fd4cee513b2481bd5faf39842ad9c57f80f2e50ddb48763fd030721
  ${PACKAGE_NAME}_iree-module-dc26eca15bb97d42bcfa019de5080da54fb3d7624809aa1c8ac731e710544e18
  ${PACKAGE_NAME}_iree-module-90d4cb308fc14e4832336ebd1e1387a791a20b92af633d0fec9161b807e13427
  ${PACKAGE_NAME}_iree-module-ba322b423888fe93e86180e98668f486afa915565edf3813af3f9da6ad7c9dc9
  ${PACKAGE_NAME}_iree-module-42bc978e75d5fed90c10d5812a8df7a6153e311b19b9c7faf41a588bdc4da7d8
  ${PACKAGE_NAME}_iree-module-6819b99fa39a0e548de5fdfc59e8ef1b3c609cd7b5cbe9c0c765911e4eb50cbd
  ${PACKAGE_NAME}_iree-module-0732f1387e308186fd8c8d8089131799be227d5a656a446048f7e0fd0d5047ce
  ${PACKAGE_NAME}_iree-module-b30a25fc50be09814337382f4cf8a1fd03b8c79d8b860ac509c629f8c8a5e5f0
  ${PACKAGE_NAME}_iree-module-a47669d662bd18dbd323737d84fd45753c01abaa81b4761eb51b35ce04ee7491
  ${PACKAGE_NAME}_iree-module-27c12f520d68a8a754b11ad44cf8ad3ef5c1ec281fc9d84fbc64a7b573cb68b5
  ${PACKAGE_NAME}_iree-module-ec14aafacf15628918390531318a4827d3ac8771b75d60a8239d901dcb4fd898
  ${PACKAGE_NAME}_iree-module-507fabae3d24c2448c8f2af676d68df0a22716a06fa7ffc3b4f865e9272ecdc8
  ${PACKAGE_NAME}_iree-module-4924918e121a92f802a094de2d67e9c2673c9fdc39faa6a11ac1d38b631a2914
  ${PACKAGE_NAME}_iree-module-383e7567e79adf2178e3d25785855905a4477b6c8daf16bc50d5f9f44d2343d9
  ${PACKAGE_NAME}_iree-module-9faad3c50bc64c297531a36bd2ab235b680070d8233ce5ab7d964ac36c7c5563
  ${PACKAGE_NAME}_iree-module-175e3e5c14dc32cafe73af4b4b8a6f5732697a097cdf2e8699a316224afb7e31
  ${PACKAGE_NAME}_iree-module-8964c792ed5c95032e2318b2dbac3b4f8453cde352067fa9a029d04a0c2d5fae
  ${PACKAGE_NAME}_iree-module-c8ee4f2bcdb954fa5d5d45c0c65631ce211c0293d7a69ed6ef4644e523e919ac
  ${PACKAGE_NAME}_iree-module-131aa61ffcacafc6ae701ff89045cf42d2490ac0c4a1d862bc83c23edb3b92e5
  ${PACKAGE_NAME}_iree-module-db9784471b47ae8bf55ca7e0821e35a1686256a208df40443c114f1adcdd26f6
  ${PACKAGE_NAME}_iree-module-aada5dcefdd361b5227276129e93547a6932a05d380acd342fa33e5b672d498c
  ${PACKAGE_NAME}_iree-module-887c7a7b540f11ee5e0158143fd46a503a4851211b10b353ec0e00b8b1beb575
  ${PACKAGE_NAME}_iree-module-11766f32ea6a3121d7527bcdf32dead45ab7b3922d72addb46945cfdab784ec0
  ${PACKAGE_NAME}_iree-module-0efb794c3a385045fa4d3d086c2a593ce67c4807e9456271f05f4e28490d1c49
  ${PACKAGE_NAME}_iree-module-7a3d36d7234ce4abfd833964492631bd81df823e6bde8cd3a1fadfa4faa7c787
  ${PACKAGE_NAME}_iree-module-77bedc82b9083aa7c270bc37d077f6ff4cecabc307a584a9b7b52e5fe18db858
  ${PACKAGE_NAME}_iree-module-a4a762fde005b81cec5b11ddf02fc7ac4bb919b71250ffefdd2cecda209ceeaa
  ${PACKAGE_NAME}_iree-module-43147a614fc5476e5a8083ddefc3ecb093c9c9ebf7864b8b6178af952540edae
  ${PACKAGE_NAME}_iree-module-86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26
  ${PACKAGE_NAME}_iree-module-bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed
  ${PACKAGE_NAME}_iree-run-fcc2eb7748902acc86b82e71de537c9f38bd0baccb9ff8da2688a806278116a0-flagfile
  ${PACKAGE_NAME}_iree-run-015af8c7c74743569726f8fecf3c5af66eb516b1e4c27b9c53444e5eb68254f9-flagfile
  ${PACKAGE_NAME}_iree-run-3d80100b61ef99b830f9e24065147ef82b7d938788576481b0a70aadf09566a6-flagfile
  ${PACKAGE_NAME}_iree-run-882a01b5adfe6cf932e3cacf39a21659d97c6d680a7c3aacbef5298958c13078-flagfile
  ${PACKAGE_NAME}_iree-run-2d916b25f6cf87d4d3841df2df2858ba4eaab05254ddacdc03c52c2ef66a2385-flagfile
  ${PACKAGE_NAME}_iree-run-480a2fe9ab9bd9ade098ff3c5fa0fd61a93c787c99329a1cdcecac6e5d708558-flagfile
  ${PACKAGE_NAME}_iree-run-d1318602b2690766e3800f5c97b1d5bd754eb594f5b053e9c977705de62329c1-flagfile
  ${PACKAGE_NAME}_iree-run-e94f7cad9035a9a3f3f6dc8ca0fb4ecc25339cf0f4a153c842b95ec00dc66f7f-flagfile
  ${PACKAGE_NAME}_iree-run-8c1719ca3fa801fb65c375f0958c274d7fec10483a39f47419df2ab68275e359-flagfile
  ${PACKAGE_NAME}_iree-run-746443fef718b98d7449c0b2d1733195479afa32e50ae726e8f695cc48611f57-flagfile
  ${PACKAGE_NAME}_iree-run-d071f553d892687709755025670f148111f85dee17950b0ecbf32192ae825421-flagfile
  ${PACKAGE_NAME}_iree-run-e58daae6572fa59b33336e143c4d985e632e7b577a6b840a4fa52e4ada335e63-flagfile
  ${PACKAGE_NAME}_iree-run-104eaa83228bf7dabd693382b2402c824fca7b8dc4284f5ab626063bc839ff4c-flagfile
  ${PACKAGE_NAME}_iree-run-51473638a07429e21bf4b4fdfdb47201bbdff46edc0134cab2d589abc65a4ed6-flagfile
  ${PACKAGE_NAME}_iree-run-82595cc516738f482ae30ff2e0441d0e4135a28e6ed68ef7ffa68f50eee4134a-flagfile
  ${PACKAGE_NAME}_iree-run-6384be1200f350799a7bcd3cc41782341c59973d7e5343985aa135756830b1e2-flagfile
  ${PACKAGE_NAME}_iree-run-5d93cc00896e37c8179563b6c4303d79ebaba5f93339d2873c86f9bb4462aa93-flagfile
  ${PACKAGE_NAME}_iree-run-5b81ba0c3d0db49f11e4c7e51f4138a723c72445c4d1b7d6d441d5a02bbf700a-flagfile
  ${PACKAGE_NAME}_iree-run-809ad3f979307779ec4bc1d3df2353bf843a813a295a5ca628ac4dd1f1a12164-flagfile
  ${PACKAGE_NAME}_iree-run-c779cb79dd79861cea6f5bce3ab4dff679b45dbf4af92bc882f85d52e673204d-flagfile
  ${PACKAGE_NAME}_iree-run-3af1a3982de4ec043b01b93519a7d322e862cd4b8b71d4d61f2219571a66861b-flagfile
  ${PACKAGE_NAME}_iree-run-1622e274d5ac570e18826aaec62f223c538583eb2f76e771d24eb2f7785954aa-flagfile
  ${PACKAGE_NAME}_iree-run-17feba75a666cf8c6f5966637f416a835b8f5d39bc1697995db64b95d778c574-flagfile
  ${PACKAGE_NAME}_iree-run-002cd64f66606ef48d9568103412f709d494fbea040a6879b069436ccc106733-flagfile
  ${PACKAGE_NAME}_iree-run-dac56be634b207d7e868bc434a6299f3fd71e5e24cf683486d8b839851abf388-flagfile
  ${PACKAGE_NAME}_iree-run-48cac7cf7dea690dd7d8e8669fd5d6f65d1f20c0de1710dc381cf15533354bed-flagfile
  ${PACKAGE_NAME}_iree-run-058a0934550c7b633e2199e5387b14be0a564aa6512660692b6f46e464e1e991-flagfile
  ${PACKAGE_NAME}_iree-run-9accf20747a0a52c6c6b7da7433c9e9cdf68a813ec6589b781ecb7791a836e34-flagfile
  ${PACKAGE_NAME}_iree-run-fd46a78e4032c5fa09644bcda90d0d8b73e9196fb89e2458db2838ddf5fd4c16-flagfile
  ${PACKAGE_NAME}_iree-run-069f6917e401e63c9e50c548c70cc699385e6f6908517eb6c79c96e597bf96d7-flagfile
  ${PACKAGE_NAME}_iree-run-1e0a0f881d2cefe36442136cc578bc4507bb4079146db4c435aafdb7e36daba6-flagfile
  ${PACKAGE_NAME}_iree-run-1b97e25add2d6f62449e9e5c1a79cf1297e3b1bbc1bb50f28e0f962719682d17-flagfile
  ${PACKAGE_NAME}_iree-run-4f616830763f5dbc3dd47fe8bb6d181793e33051f1bd62b02160b633383e7882-flagfile
  ${PACKAGE_NAME}_iree-run-0aac8a2a5c45ed0ed35dcd65338a5a414c6beefcdbb0fbb4f299b42d41b639e1-flagfile
  ${PACKAGE_NAME}_iree-run-9fce5abfa2d401c58fb24687fbb7a55bb2e976bcd586db14b30d911cf16eca68-flagfile
  ${PACKAGE_NAME}_iree-run-4af168ed94d96166f35b8264e160ca1e85a3c6ef3faa08284f447a5613f6ce39-flagfile
  ${PACKAGE_NAME}_iree-run-da589d3a658ddcc4dacaab64c8c7253bab3b0b90fbd35158ba58ed883266d5dc-flagfile
  ${PACKAGE_NAME}_iree-run-77dd6dcff77b2053dbc4cbafc7ca36f8ee5aabdc138b5808830908b037014cc3-flagfile
  ${PACKAGE_NAME}_iree-run-9fc0803530563719083133a4b602dba0ae57a549bcea4374bc09948d41bce849-flagfile
  ${PACKAGE_NAME}_iree-run-da282da8f4c832d72bf3eb9bc97bee39f0ba24a78e25626747f918976f19814c-flagfile
  ${PACKAGE_NAME}_iree-run-1b5de023c999f6523524305c2abd37a4cf24b3706f4ee9e337c87981dd3725c9-flagfile
  ${PACKAGE_NAME}_iree-run-0d4e114d66ae2e078076cc40fca5e6af76232c3936effb92d33e23f76f26ede8-flagfile
  ${PACKAGE_NAME}_iree-run-150f45f52d20091945e3388e18d11a16e1e8453774c1e4b2eecf79e6f23c86fa-flagfile
  ${PACKAGE_NAME}_iree-run-a2ebf5883d38f358868199609143debdbb2947b6e0ab6c5b03802cb813022f9f-flagfile
  ${PACKAGE_NAME}_iree-run-8cf9c331c7ec9e59884fe37b1a6e4326edc902e749daa2ad5f873c0b5433051b-flagfile
  ${PACKAGE_NAME}_iree-run-cac4716bd8d859d6cc894c38936e1f8d22e9f218c01a94b9351c376eed985975-flagfile
  ${PACKAGE_NAME}_iree-run-fe3b81659dda02e5b75d72c813626309b9c44232a5d23a7be79cc1320921add6-flagfile
  ${PACKAGE_NAME}_iree-run-4003e2ba51615b0aca866e83846bafa2d510a5c5f4690e744f0f210cdbc592ec-flagfile
  ${PACKAGE_NAME}_iree-run-ee47de5952e13727e6837d2f10c8979403039e27744b1b2df31cb1a9f2686538-flagfile
  ${PACKAGE_NAME}_iree-run-3fcb38435a8f09de28c6b5a44b95c531b18f83d174218b923762ee9d29946543-flagfile
  ${PACKAGE_NAME}_iree-run-e02c3088da158fc213c051e13ac60cd39450d04e658a700b4cf67c40811e72c4-flagfile
  ${PACKAGE_NAME}_iree-run-d7bb92092896fb847743683d802fd7bebc1b27216ae7fb5b2fb8b554c2d4f5ae-flagfile
  ${PACKAGE_NAME}_iree-run-f3a6824694bb52b8dfe735b60a86d0dd28ef151ea4a3eb6ff7706ee4aa12f9d6-flagfile
  ${PACKAGE_NAME}_iree-run-d14bc72f848279de26aba8bd86bb530767acc4ca769356ab548258db49c44555-flagfile
  ${PACKAGE_NAME}_iree-run-bd39009584dcea45606af9a0f73380027510eebdb51d46c0391a527b032459ad-flagfile
  ${PACKAGE_NAME}_iree-run-974086586c080164bf56972646f76a8f116cd849d51713ca3d9b380366c8d305-flagfile
  ${PACKAGE_NAME}_iree-run-1d4ca3d0d6259c0d3322b04343e655cbd441759fdfad8a55f05c0dcbc13dc1ed-flagfile
  ${PACKAGE_NAME}_iree-run-e076babcf92c08d76f05c53bec9bcf823f3855b6280c2c74465ed25bb2bb2bd7-flagfile
  ${PACKAGE_NAME}_iree-run-232dd191f613b7aebb8501342bf8c99afd519e1554b234d797e6aef3779066a7-flagfile
  ${PACKAGE_NAME}_iree-run-bcd45d4ee5f34196aedb27bbe78f90ef2876c1908028a5d344915b67ba8294e2-flagfile
  ${PACKAGE_NAME}_iree-run-e496c2ea8de7fdffdb7597da63eed12cc5fb0595605d70e1f9d67a33857a299a-flagfile
  ${PACKAGE_NAME}_iree-run-2cbf22689ea9dd3ac54c3a66da0fed845f3314f5ca96afef57c7e3697409536a-flagfile
  ${PACKAGE_NAME}_iree-run-58ea5e58d8226c09966c7dce3a6b0763a681bc4d2507b9d9207760c975363ab6-flagfile
  ${PACKAGE_NAME}_iree-run-0394a1edf412d08f215b567edcf2f1daf4a3ff5973c08590a564456232c5a171-flagfile
  ${PACKAGE_NAME}_iree-run-b1cbdfe8626953a3ab575aa1ab98ed52fb2a488c1b0ef032144961dc1ce480da-flagfile
  ${PACKAGE_NAME}_iree-run-7237c7cbf5353280472161050ccb803bd6237ac656eab0604d5cc610d73ef778-flagfile
  ${PACKAGE_NAME}_iree-run-2387980c443202f6775c1ae692c4896612a50731639d3c51f1cb69ba0935b51a-flagfile
  ${PACKAGE_NAME}_iree-run-60ebe003ad32386572a7515583e00883b11209d13c62d6907be645492557aa71-flagfile
  ${PACKAGE_NAME}_iree-run-efdc6e24cede8d60e41e9bfa18e61041c66c4dfe7ccbf731e186370055582f7a-flagfile
  ${PACKAGE_NAME}_iree-run-423824abc1ed6574ed1315b6c6432366edefbec9704c4b524d6daa9c7f18bf0a-flagfile
  ${PACKAGE_NAME}_iree-run-0e1f775fdc4e9c29a0c9d6fdbdb62fe04f8fef6a3d441171fc4dac4ebf0b439c-flagfile
  ${PACKAGE_NAME}_iree-run-579b8550840595f0dc5a89acbb574ebf022c1581132b82e56139df142953c820-flagfile
  ${PACKAGE_NAME}_iree-run-9a3ce9fe094e4ae324e0d11ce2407659487875b4169b1d0cfb1d7be822945cec-flagfile
  ${PACKAGE_NAME}_iree-run-b528e469bfd43258750e70a724bf02eeb157173782b5a5a8912ae036e3ffce58-flagfile
  ${PACKAGE_NAME}_iree-run-9928a2e957aea784606874bf135ba426b6688e110f8111a758274091ad1a3576-flagfile
  ${PACKAGE_NAME}_iree-run-69794b071d9defd92d85bce61aa87c786ce2888bc2ce88a8cac841d6da6cac7e-flagfile
  ${PACKAGE_NAME}_iree-run-73eb77eb75db72c97d07e8325dfc34f8c19f0bfc059a23b89027c89d0e945c31-flagfile
  ${PACKAGE_NAME}_iree-run-4d92c9901b7c73d8e02e63adfdcdf63ef0fb529360a908f93b888dee1c3f9c31-flagfile
  ${PACKAGE_NAME}_iree-run-57617c1acebc961757a26d7529068227f7a75ff52f298d28f4f1dc0d6f193b19-flagfile
  ${PACKAGE_NAME}_iree-run-a287c36d311b72071631c7c7743d126038ee1938e9a415d3e971f3aa7c00397b-flagfile
  ${PACKAGE_NAME}_iree-run-ef450965157d6b8f47c7c6888eb66a8c56760986deeced8cc451877f174f2a69-flagfile
  ${PACKAGE_NAME}_iree-run-7001a4f2a5e52aa034f802096f625e278fc10b92cd85653335c3a7c5110492c7-flagfile
  ${PACKAGE_NAME}_iree-run-6fddae6f5e12a547164d4a46a7e185f0101acce8589964a668475fda218be41a-flagfile
  ${PACKAGE_NAME}_iree-run-78cc46b0727d5c5a98bcad94c2377cce22aa67fc5188cee22dc87f3c8306a722-flagfile
  ${PACKAGE_NAME}_iree-run-3e6d25ceea03b4311e04345571e450c6557d8cff6e99991f822d67325d57b286-flagfile
  ${PACKAGE_NAME}_iree-run-6600e5c77f343f3727788ac55712340db67660453f0d5b2a78f8a2f00bffa9f2-flagfile
  ${PACKAGE_NAME}_iree-run-175234258abc061d624fad12b5f814c63c5031f71d59e1bafa0df468cb756f38-flagfile
  ${PACKAGE_NAME}_iree-run-14e8174454310c9b24812dca661319c7b8e78a1175003f56abe8cfa7e7bb9cb9-flagfile
  ${PACKAGE_NAME}_iree-run-f773cd1d5c806ca4576b9a60362058988bd80e2582f93d546f81e60ec38f451d-flagfile
  ${PACKAGE_NAME}_iree-run-6272e089c33b7c5333b6188b6f61fbb15e7b6a0e9fcd9d54b3b7271cd730e0da-flagfile
  ${PACKAGE_NAME}_iree-run-33ca476093f97d4862168a0feff140fbe85c5dc70fb06cbdf3e6054480575c85-flagfile
  ${PACKAGE_NAME}_iree-run-ce780c2ab7c9b837611b5e1dcdbce18e7563fb9d9137e68b5a50bd917a54f83d-flagfile
  ${PACKAGE_NAME}_iree-run-485da7a706b6c0940ef45626ec12ab149da295cc6a3c0a2c63e5a15a952580b4-flagfile
  ${PACKAGE_NAME}_iree-run-c27738e97498c969076d1a2a693322821dd104dbcf7ba6e129ba893584bb0dfd-flagfile
  ${PACKAGE_NAME}_iree-run-7cac96dd2ddb2585861de464c27b59023e8d6f44418f020821074e748f7f6388-flagfile
  ${PACKAGE_NAME}_iree-run-d57d9beca86d9b8454422cd20afca034477dd3471577f5f4708252fffa864ebf-flagfile
  ${PACKAGE_NAME}_iree-run-d15406ab9fa52f130a4947059f1616c6ee94c1c2b36a6078ec708d475556a9a4-flagfile
  ${PACKAGE_NAME}_iree-run-d6bfea70085e57a372f18983ddd9f7598b084dc4aac07754c80e4f4f5c4fb407-flagfile
  ${PACKAGE_NAME}_iree-run-aaa50f507f6ca3e64fed593f49949d19d975c46d4986897a4dccbec89d132a3a-flagfile
  ${PACKAGE_NAME}_iree-run-ec20addfc5f284c92b739d0eaf245af0027627de593635539a86709332ae5acf-flagfile
  ${PACKAGE_NAME}_iree-run-3283ddd7c21e5db8eea573c2f94ae318c5baa6bf3d9340ba157573937e7b6632-flagfile
  ${PACKAGE_NAME}_iree-run-8d8fd2fbd7901ece93ffa5e47c460dd793c4489b5751a15bb0c3e1b8d82073db-flagfile
  ${PACKAGE_NAME}_iree-run-aed9d709c26fd99b9df147829544434c944296e124a6fb575f012dbb1502da90-flagfile
  ${PACKAGE_NAME}_iree-run-6e06f1507848e5fe5510a1310fb7a07919b0cd845aa419024140ace4349fb604-flagfile
  ${PACKAGE_NAME}_iree-run-9a309fd292fdde68ab47244e7144cda6bde73c2d4f05035cb6524f52d3323b9f-flagfile
  ${PACKAGE_NAME}_iree-run-51181aae886260ff3c24d829e8bf9e3a892aa93305321c1012476aace79f9e65-flagfile
  ${PACKAGE_NAME}_iree-run-fadf67f1f541c40d8b9e897a4a5cf125ffcac13ed77d6ec9f893f9cf45874192-flagfile
  ${PACKAGE_NAME}_iree-run-1e0197113e1bab228898b4e76067c7c8dcd0faf2b0cf5af9dbb227491de894e4-flagfile
  ${PACKAGE_NAME}_iree-run-fb7e03b97045f97096e484826bf4b8f432d69d11acbadb6af721c94c4a2f9424-flagfile
  ${PACKAGE_NAME}_iree-run-9b9e37b533b9a700da481f6792656137c4591c3071401c0ad0abe792bcb5727c-flagfile
  ${PACKAGE_NAME}_iree-run-4a5d8ec27fd956b0a83e003eec13cf3aa4ade14a6627f5ce62708d57aade9b9b-flagfile
  ${PACKAGE_NAME}_iree-run-488a6230bbbb7f01ae76d1b32117251eb9aadf4fbe75a10540fef8b9e344f77f-flagfile
  ${PACKAGE_NAME}_iree-run-19201c34fa6afcc1ea582fe6d836dfef9b93e9a2d98cb1720408fc2ad329c13a-flagfile
  ${PACKAGE_NAME}_iree-run-f5718e569829b5a8af735ae68d4cdc04131cf182d496f483d643ce6beb1bc6d9-flagfile
  ${PACKAGE_NAME}_iree-run-54f0a27002b8e2ec04cb7f783f1d178a0c360a187fa84eb1c3440403bd08d3b7-flagfile
  ${PACKAGE_NAME}_iree-run-c28be7f493ec9bc790631b3f90dfe611da5a10ac37e23ddebc80530fc61df2f5-flagfile
  ${PACKAGE_NAME}_iree-run-d6f1a520e3361101b6a3e06f3d6d66638b9b86b88b4376da5ab2e1add311479a-flagfile
  ${PACKAGE_NAME}_iree-run-ce7eec0c36a5fda73313a06da87ff315e0307cd6d2962d167e7e641eea50604c-flagfile
  ${PACKAGE_NAME}_iree-run-595fab3ea1ea90bbfa01d4c243295f6fd43f0e75878c75cff5a1139a0e7f4643-flagfile
  ${PACKAGE_NAME}_iree-run-de9752b277be0f178adf7974807f310f7432bc84951aca9c7126b9752d4ffbf4-flagfile
  ${PACKAGE_NAME}_iree-run-fc9a821b5ad5fe568b9aedc1142e9a3609685804387af4a4dddce2d7e65b9259-flagfile
  ${PACKAGE_NAME}_iree-run-3da49d74eed3cd740c69a6a2a97f3ff7e54710ea66c083670042256b2648ddcf-flagfile
  ${PACKAGE_NAME}_iree-run-da652151b3065abd5c15dad6c2e9b88f319de04760d7f3fd8fd1f00714452a14-flagfile
  ${PACKAGE_NAME}_iree-run-9bdc58d9a9a5f3f7fed8db78bfc02d052539ece0e9eef401f52f7fe4663b7aac-flagfile
  ${PACKAGE_NAME}_iree-run-c4a8ace91f235fbf4ee079839016991166b1b8f01abf2f1bfccd0246323d79f0-flagfile
  ${PACKAGE_NAME}_iree-run-335195752d56c66c4dcb075ffae3c4c0e82996f065e3ab9aacd3a3f7792174dc-flagfile
  ${PACKAGE_NAME}_iree-run-f757ac686e7e26e443db2a8a1578b197d75a11ed8400b05a4476e3ac1c6713c5-flagfile
  ${PACKAGE_NAME}_iree-run-1d3c521075923269cfc5bcd08c3b7638f689fe0552d897903870407e4c4b4e56-flagfile
  ${PACKAGE_NAME}_iree-run-5b33da9efba3f32f8b5761a7908c92dce139b97428585232599ca0c1a9ec426d-flagfile
  ${PACKAGE_NAME}_iree-run-43205b471655d7c5d4d30801d6789b741a6318ab87adb6e0d8c4512b9ab5ee26-flagfile
  ${PACKAGE_NAME}_iree-run-8c9afefa68d7256a2f3d3c8cea6b7c592bef0b1afb4f1bad67d03540d22bcd80-flagfile
  ${PACKAGE_NAME}_iree-run-5c523031b46b818cbef17167cdad8fbd9fc2390db00f8e48d2c7489951eae183-flagfile
  ${PACKAGE_NAME}_iree-run-95281c38b844a3b0ea1964e9634e7a8e2b40025936e3402ff2902be01dbd31b7-flagfile
  ${PACKAGE_NAME}_iree-run-02f035862854dacda92c51e7c3b506e7b5adc3fb36818e8b171c5d670d78eff2-flagfile
  ${PACKAGE_NAME}_iree-run-8ccf5261dd5ce1229c401acee803a4d03ecce10bdba23e00e1df817e7983fdec-flagfile
  ${PACKAGE_NAME}_iree-run-6796854808165b6275740d3692260a1349f58cdf0a8330207675b2eaf2a9735e-flagfile
  ${PACKAGE_NAME}_iree-run-0705d47ed43a301d7b92b10b7a38f77703ad865f6bee8d28c2ad63f61c5c5772-flagfile
  ${PACKAGE_NAME}_iree-run-1b166da68df5960cf0a58ff7fdd09f2bfdaddd5a4c3e673c712f687b4fb8ab9b-flagfile
  ${PACKAGE_NAME}_iree-run-2a360cb8f8f6bf23ef7aa89c5f21509f6504b6a3ba33c3529a840303dfb3c3fe-flagfile
  ${PACKAGE_NAME}_iree-run-f3a4e75e14b7bf806d565fc3699afd13036cc3f14091efe2304c97b8ed81a28e-flagfile
  ${PACKAGE_NAME}_iree-run-a56f27a4186ef4fdb9848c589a2f8f0027b4fb948ed54745b8b8d0cb79fbc213-flagfile
  ${PACKAGE_NAME}_iree-run-b79504ca12b3835c243fb7bc186060060653cc048023c8a00c8da04eed0dabd8-flagfile
  ${PACKAGE_NAME}_iree-run-7be38c67cfd7b39fab35c7e18be741a0ff81526243f39af14ea85deb48918885-flagfile
  ${PACKAGE_NAME}_iree-run-169e82fdd31df61df4c365f1f4c5da9e178c1b491e83c633cfd706699adfa754-flagfile
  ${PACKAGE_NAME}_iree-run-cbb177a628d76a864a7ffc8112ee5a9916952d3d0b6f84b49cddecee1849b900-flagfile
  ${PACKAGE_NAME}_iree-run-fd0020c5effd33078d74e079acbc48230001555fea77552823f8ec55b63cc44e-flagfile
  ${PACKAGE_NAME}_iree-run-fa8255f23b90fc02bb1cccbc9dae74e0d7e5b7f9f3e31eb5610584d73732f1ff-flagfile
  ${PACKAGE_NAME}_iree-run-e8b51276c78902bd98c7521b4c41d40b3083bb4343c7bd49cfe0d15c72331326-flagfile
  ${PACKAGE_NAME}_iree-run-db44d3427feb19946b43153a9690e9e830811d32fef711b8ac9cca66e50cc1c4-flagfile
  ${PACKAGE_NAME}_iree-run-c2885c9fa0bcddea33e9025f8a0f84b8823c3c489990a8087a309e0a39ad2566-flagfile
  ${PACKAGE_NAME}_iree-run-7271b79c4d00766e5a9bfe32b30993c76b2f5550b1cea07a0c2f26f51016b707-flagfile
  ${PACKAGE_NAME}_iree-run-9d2e9b2b3e070388dd08a932b3818d189c0a205fa3f4de3f8e0f772b31179cb7-flagfile
  ${PACKAGE_NAME}_iree-run-59642786e50eac32f2270781fedbbf1bb4db9241368ffcce6d1e9a9f454759a9-flagfile
  ${PACKAGE_NAME}_iree-run-ab3a853f3d6693aaf464dcbb5f0adf9e40925cd56928fe257cd32d52b13f5599-flagfile
  ${PACKAGE_NAME}_iree-run-918149bf080db843fb731d650b77c3e2196e34dedf4b42221af7e3c9b91182d8-flagfile
  ${PACKAGE_NAME}_iree-run-e216053a20f0e56c82bb260a93866d9edd5968941ce98a83bc01696a30760701-flagfile
  ${PACKAGE_NAME}_iree-run-c7c9eb0cfef4ef82f39f5d06a0d636e3691d62fcb8267b33143d84b246e88974-flagfile
  ${PACKAGE_NAME}_iree-run-dadde1e46eb1c96459ac710904383b6625d43b45bc7c06ce6043f3a33b3817b9-flagfile
  ${PACKAGE_NAME}_iree-run-857b039a7c814ae00a254adf9b1a1a2c85fefb997e3bacf80510893cbb9c6f2e-flagfile
  ${PACKAGE_NAME}_iree-run-4b7c3c4c934d3ef254bc7e214e6c677ce6529fdcf72733521763aca5b9407e8f-flagfile
  ${PACKAGE_NAME}_iree-run-aa947dc1908028becbeb0874aa7c65c9ac366f19ada505ea5155283ee86fd91b-flagfile
  ${PACKAGE_NAME}_iree-run-d6dfd546eb26f91c109d344336ac213020777ac877e52e8e2e471c7d32c4a669-flagfile
  ${PACKAGE_NAME}_iree-run-c065ef331c525cec65b1d9dd46fee173b55702b068d896110507a34cf55602a6-flagfile
  ${PACKAGE_NAME}_iree-run-2a746c3fd6a24944a7e03b4b4bc246d0fa172d6159af4ae85cbd02ae1db2a5c0-flagfile
  ${PACKAGE_NAME}_iree-run-c3ae398d16816bb66156d0aa84efd4c33ae1ff8238f437da171d8496a0ee97a8-flagfile
  ${PACKAGE_NAME}_iree-run-9985db1af1487130c670bbc155da57c77decab98fdaff7293a079b2bcc2c7ea7-flagfile
  ${PACKAGE_NAME}_iree-run-6db7c3032ae546cc2102728906fbfc4f85aaba1c72fadfd55c8ad31da417ecd9-flagfile
  ${PACKAGE_NAME}_iree-run-2c8c3423e7567bd72e0c398a3b740233f60bc622dfdbc6023afd2dc905d5c3e3-flagfile
  ${PACKAGE_NAME}_iree-run-dc2023c6113c87aad59f2b49214ab2995b32c7ba040b314e890ea2ec7081f90b-flagfile
  ${PACKAGE_NAME}_iree-run-580de29df687f6838fa67a3b020e7d0b02bb6ebdbf14dd58460f09ac088b92c9-flagfile
  ${PACKAGE_NAME}_iree-run-7b0a2b9429936d6e2c65b2474aa29baac011c9e106c49711db514dc64bb055cb-flagfile
  ${PACKAGE_NAME}_iree-run-8130cbda36a110611a415869f41370e934d21f8bf05d0c501834bbbc118d48c7-flagfile
  ${PACKAGE_NAME}_iree-run-df3c8b2cb88a80d85abf4168ea912aadb39d3a3c3dc9dae9f3899431cea713fa-flagfile
  ${PACKAGE_NAME}_iree-run-79d228995d626d87d488bfe58a04f3b37bbc58d6a45147c9039f4e11b19b7843-flagfile
  ${PACKAGE_NAME}_iree-run-4c331ecdd0e9e450678600deacc04cb8142b7b491c739e87bf3df7f32bb5a17e-flagfile
  ${PACKAGE_NAME}_iree-run-1aee696bba55cf12e2f62e043c853437afbfee92a1f19bf0272b4014e633e6e5-flagfile
  ${PACKAGE_NAME}_iree-run-d61f2887ce922e7a8357640637ee69184c0ab4d84e9f90d57751e72d41cea509-flagfile
  ${PACKAGE_NAME}_iree-run-570d87aae955ec258d971849489bdec9646f0207eac47f5f8a79532ad1285f64-flagfile
  ${PACKAGE_NAME}_iree-run-d794d28f78568cf5b7b3352cae5e409e49b8e8248259cd686fb9e26c0addcee1-flagfile
  ${PACKAGE_NAME}_iree-run-22ac4fa00344cef241ae57b58fdcf836468b14bd86786d183d44862dcf4b862e-flagfile
  ${PACKAGE_NAME}_iree-run-a5fbfe5e760ae9770da3f77744c270a3caca342dd869b15b824f1bea6c853b74-flagfile
  ${PACKAGE_NAME}_iree-run-a9277ecdb3a9b16ddc617712d4a35307d39c7c272868f6e512b154c17b860e32-flagfile
  ${PACKAGE_NAME}_iree-run-fceb36e464100809c802a326aa68db53c21040c9fc2719d9c634d31741e54dfe-flagfile
  ${PACKAGE_NAME}_iree-run-09922a30726fd43b5532d77a806709543554eef05dfa0c59214d91639207484e-flagfile
  ${PACKAGE_NAME}_iree-run-e2184228010053416099d266bb180b71fd692000aafeed6953c4b757b5dbc0c8-flagfile
  ${PACKAGE_NAME}_iree-run-0be5f52840f466497e75c2dbc83b6f15ecd6da62235e6b6101550fedcff5bdd8-flagfile
  ${PACKAGE_NAME}_iree-run-54dc93e0c6b0c16667a08ffd756665f0a41fec7aba3a446b0dcf84ba79b86890-flagfile
  ${PACKAGE_NAME}_iree-run-f9c6d66a04485bdebd31a97572f8e17e0b46f6ff0672bbd105c9e9cea656a8b9-flagfile
  ${PACKAGE_NAME}_iree-run-f4db3a8fd7c62e69f39cd56b363540db9fe2e7b8f2f68a85028a07793e0b85f2-flagfile
  ${PACKAGE_NAME}_iree-run-f52781039b4c7e9f9f2a8f37898d4836f4d6d355cb30469d23f5cd553874a7ee-flagfile
  ${PACKAGE_NAME}_iree-run-637aa2b40c625556a57c0197b885e8ae1a688c1bf0a18f80acd940cbef160e14-flagfile
  ${PACKAGE_NAME}_iree-run-227bf16d1232b24440be8023b2bae4d31a1769a3bc4f9851293fd6e92f366cff-flagfile
  ${PACKAGE_NAME}_iree-run-13e421fefdec1bea69356ec7e9b4034f73347be8b03e8e9a9fbc87dc2cb294b6-flagfile
  ${PACKAGE_NAME}_iree-run-27dd2396d5989f0bc6742f12a4fee85616ab702394f3a2c3b54b24ce7fda7e2a-flagfile
  ${PACKAGE_NAME}_iree-run-c21ee1794eccb63ea6eec88b05bfc5b6436e4b84cb9a12f44b7899a0b0dea06a-flagfile
  ${PACKAGE_NAME}_iree-run-eaf02a02862e64d8858aeb8dca29aa925bda732bff9ff5d8e307d6202710a985-flagfile
  ${PACKAGE_NAME}_iree-run-9e56433611f5a614a3e40564b3b6d4523bac66e15ea5390749d337ab261d0ec8-flagfile
  ${PACKAGE_NAME}_iree-run-ebc8a366f920012d85a63064101aebad0cc1f1bb1d51369bf313fc19a80dc05f-flagfile
  ${PACKAGE_NAME}_iree-run-8bce5fb2822f23a8e3970f57d31c7f15683723b2e1a75051994dd35981346d73-flagfile
  ${PACKAGE_NAME}_iree-run-7113a412a48e3f2dd80fdb9fb53d461883d39565ca917a37ec85db0eacc40688-flagfile
  ${PACKAGE_NAME}_iree-run-f2a09db9032d52476c0d5177243db3e3a4822017b2dbd1161b524477faf9951d-flagfile
  ${PACKAGE_NAME}_iree-run-8d6ffece91d45840e8378898d357a433d20d1ead3266434322fc780e3bdb4e7a-flagfile
  ${PACKAGE_NAME}_iree-run-b1715311f2db98809942bf49519e312e143e0a745693df14ebe17e05fbdb8eda-flagfile
  ${PACKAGE_NAME}_iree-run-ba0485d3471e2fe21fcd5fb532e00db33513d31c0321fabfd601d65576bb65d8-flagfile
  ${PACKAGE_NAME}_iree-run-df6786c3bd20d93e1230f8b59212221a7e9de0eefdc39ac2f7192b76047d2803-flagfile
  ${PACKAGE_NAME}_iree-run-6b7f3996f5757948b156740ec97f61bdc8e2b609f51a38cdc7ea05ed7fd1f491-flagfile
  ${PACKAGE_NAME}_iree-run-98dd67104ca5dc13455783eefea4f0e9de90952470d48e96aaab2ff5e6ac379d-flagfile
  ${PACKAGE_NAME}_iree-run-611a789dc4b5ce96ef60dfa63ed0c77192d7111f29e248ee8d7931d3e0171c68-flagfile
  ${PACKAGE_NAME}_iree-run-6f79f9cd5a0933a9ec2677915fb1b92c1096c75165d31ebc2c164c6e3e85ddc2-flagfile
  ${PACKAGE_NAME}_iree-run-91e7cbe84292aa360181b520bf882b94abc99dcd23ff8657729e23fff98dbf84-flagfile
  ${PACKAGE_NAME}_iree-run-b6a4aa6d970860b1ef463cac3ea42b2fa446d2af02cda319727ea559c3b219a0-flagfile
  ${PACKAGE_NAME}_iree-run-bafbc5b2763a01f9a20a1e6ce4bc612e95775cde9becabb8ed29667e07acf77b-flagfile
  ${PACKAGE_NAME}_iree-run-ebe2cabf50990753ec25f8ee5aa01d9c6b1fff4cd7dba9b970b27bf6a2a919ee-flagfile
  ${PACKAGE_NAME}_iree-run-79fe23a6b5f014529a46504ef8a2e54edb801dacbbe32cbf0b8d4e0c1d26f813-flagfile
  ${PACKAGE_NAME}_iree-run-abe2879afa276d0b16ce493283655e600385c4aa613cb56f0ff6548846b33b4f-flagfile
  ${PACKAGE_NAME}_iree-run-9477b1088e1a7d8e7dfa9b086a464ec02a9d4e3c25dbe3f0c4dcd97fc41b13e5-flagfile
  ${PACKAGE_NAME}_iree-run-f17944b7339d0d84be14cd71d31c10b495df98114d5af917259df75540551fa4-flagfile
  ${PACKAGE_NAME}_iree-run-400fb0da6b26d5ad0dada81386e3509ceee892450bb3bb07bfb818655fe6388a-flagfile
  ${PACKAGE_NAME}_iree-run-51805cbc48c47b1698237121609d7c5585f3d3780ba12dbe02f1ae74fe20f8b8-flagfile
  ${PACKAGE_NAME}_iree-run-443869906d5762513b4d5591d0c48cfc5ea5a21a1bd42e192bd16430b5887707-flagfile
  ${PACKAGE_NAME}_iree-run-bf1285e742efe7824381c9cb3df2b455f70f996597ab7a0ce8344c8248cd6d8c-flagfile
  ${PACKAGE_NAME}_iree-run-f26daa540292860d856cbdaa8d971c5d72419e96da82f61e94ff5a3e94d56e2f-flagfile
  ${PACKAGE_NAME}_iree-run-0f1f53582997baad9aa36456f0268590ba384be42726f6ad1fdd6bbcfc626c08-flagfile
  ${PACKAGE_NAME}_iree-run-7f373fb8e7ee89bb77c43a376ceb5bd22e4a88b2f5ada6914b49b35449105263-flagfile
  ${PACKAGE_NAME}_iree-run-84ca03417474c9a2acf48d260d1c053759f95b6589e9e4f45809ef56e7e9ce46-flagfile
  ${PACKAGE_NAME}_iree-run-b09dcf7831947fc2c23cc2d570d548a398cc1db47ed5940401e41f5019878917-flagfile
  ${PACKAGE_NAME}_iree-run-c94f38cbeae26e9eb7f539b144bef30f5f43c9a5ed292614a07fea745701543c-flagfile
  ${PACKAGE_NAME}_iree-run-0ca018bd91eb89e4ffad42f3b5e7aa6e30faf11310eeb93e800cd696cdfb652e-flagfile
  ${PACKAGE_NAME}_iree-run-d4572856894af9013e311991e4371c81498ee30b1fc90ee840632d1a3a512193-flagfile
  ${PACKAGE_NAME}_iree-run-d4e65beafd0c9ae94ef7803c610f825fee8845ef9431742c5c11714ed2544ae2-flagfile
  ${PACKAGE_NAME}_iree-run-486d294efbdab2b8b9a254695a0f228c54fe75fe2df6adcc71c520d4e1976eac-flagfile
  ${PACKAGE_NAME}_iree-run-154a4c5d9380e2a084bb8747db72990255ac96e6161d10cae80daa514df78c0f-flagfile
  ${PACKAGE_NAME}_iree-run-2db48318a3bbf367e6df3414a58b3b5d9c4e0bd5a688ce7308dd8df127baf37f-flagfile
  ${PACKAGE_NAME}_iree-run-528453ea57613e45540f349b229866c680a17eff84af54642571acabe799c354-flagfile
  ${PACKAGE_NAME}_iree-run-23f07cde725d495d43e3ab8ef3d7e2768ec81a0c98d05d2cddded7bcf3be5022-flagfile
  ${PACKAGE_NAME}_iree-run-d67c9b75cf43f3c46d9a23caf6921b2030d4efa1b160be4bcb5f80abcba8abf2-flagfile
  ${PACKAGE_NAME}_iree-run-a56ddac5571fa0e1c36411e4202663641fa9737a6d22b8051bb3cbeb19207f24-flagfile
  ${PACKAGE_NAME}_iree-run-9cd79e8d48d8f6306a6ea47e3239918ef393a09e6ad18895b0d365c1332502e8-flagfile
  ${PACKAGE_NAME}_iree-run-108aa19ab15996026054223cb4ae1da6d952682ba17b71fcd0e09686d13a3f70-flagfile
  ${PACKAGE_NAME}_iree-run-98140fab01ba25d74943e6bcc2acbee688b8356d2925e22330cfbb7a1a23c075-flagfile
  ${PACKAGE_NAME}_iree-run-60b0f8b20b74d3ef1d937a368e98d1cfb0a31a9c5b6d81d34872b9a6310dcc07-flagfile
  ${PACKAGE_NAME}_iree-run-53a27f3e75e7d285d0dfe635dc13c2f72aae257eafc0d3e2fcfcc4040b056487-flagfile
  ${PACKAGE_NAME}_iree-run-49317ac60d1d798e1f8fe89c255a2028b1642d6e65ecdcf6bf720270ce12b960-flagfile
  ${PACKAGE_NAME}_iree-run-7aff468d197ea449f8cda6cb6719cde1cb3d214f47298b92c03eb10b8da7014b-flagfile
  ${PACKAGE_NAME}_iree-run-a10ec940197f56cbac77926b9573a7d3b04dc5012e3edd5df5289d371ea74428-flagfile
  ${PACKAGE_NAME}_iree-run-a3275b0fbf3ee381927da85ac6794ff29e54ef10a59031fb639ef8170a00e963-flagfile
  ${PACKAGE_NAME}_iree-run-0a7bd61c612b2d143d7c4293f82bb570ca9ee3978f43cd656f71179805bed7ef-flagfile
  ${PACKAGE_NAME}_iree-run-ce3fff17fd34b163da9513e5009262e520421ec594c33b9d5c16bc3ac511f950-flagfile
  ${PACKAGE_NAME}_iree-run-65465ca77dcf48e6bf1a786fc858f46b2b4fa802a0558a252c56984ed33e41be-flagfile
  ${PACKAGE_NAME}_iree-run-73f3ece7417f1ce8eb05da3450362b4e52a651f9a4313894c5ed6cbfde9f61ab-flagfile
  ${PACKAGE_NAME}_iree-run-72d30a598499416c4f4bc949f9e4e9dcadf8fd02ebf5f594352227664276e13d-flagfile
  ${PACKAGE_NAME}_iree-run-6f107299a90bc6e2d9a962d28236eb99f1589d049e53d6f05cedf5a2d6820466-flagfile
  ${PACKAGE_NAME}_iree-run-2e1ec1c72cd4eeab43d5a5e5f3468b751c732af97a4d04af969589004b6b1f15-flagfile
  ${PACKAGE_NAME}_iree-run-ecbbd1f41ef3fee6317a6845d86e3103f7c9c6019e117f3023386b2c43c05f41-flagfile
  ${PACKAGE_NAME}_iree-run-dd0407527936cadd32b2d76e572a6b24c3fa0aebc71100227038538f9027ca89-flagfile
  ${PACKAGE_NAME}_iree-run-cae5ec5b02228b25142bd564d9d45cd61a3011b09f8e4444f456913bb7911978-flagfile
  ${PACKAGE_NAME}_iree-run-6b62e2878a4aa7abf23fbf3585f8b5d5317dd7487807d4253906aa7b5815f5eb-flagfile
  ${PACKAGE_NAME}_iree-run-20d82f50c7675c5aa0592b3f0fdae6712d83c0cdc2d10a579e7b32b1e4560455-flagfile
  ${PACKAGE_NAME}_iree-run-62847ad95dc56b400bd0471d62da88904d0bec489ef9de82b84b2179e70eecbd-flagfile
  ${PACKAGE_NAME}_iree-run-3665d5c55e91e2410b877d72d28460742791ff7a173c2163485a97154376468b-flagfile
  ${PACKAGE_NAME}_iree-run-17f368591afda1e361c872f855576202e7bff1c71e432ae0d8a5027b4ebf6b24-flagfile
  ${PACKAGE_NAME}_iree-run-1d38ee595dd7aed8bac66c4bb54f252e16a37bd7efca5784c5cf5251edf20660-flagfile
  ${PACKAGE_NAME}_iree-run-e547285f102de03190bf6fbee132d5b251eb632b8d94a9084bf67255640ecab5-flagfile
  ${PACKAGE_NAME}_iree-run-fc1859fb976b75f059fb7ebc5e75032a4a540461b6f37be57c5b85993f0fa18b-flagfile
  ${PACKAGE_NAME}_iree-run-6a6ca63f3da8f6ef340e3c380d49e0f09535f05ca917d2842f0f76fc9f34a570-flagfile
  ${PACKAGE_NAME}_iree-run-ea8d5529de4316b9bec5e80d604a37a4c2304c653984f7ed04d479c7da15b1b7-flagfile
  ${PACKAGE_NAME}_iree-run-411c292e140d21cf2a4270627e277c8d31dd18f64112dec9162cd01e31d1364e-flagfile
  ${PACKAGE_NAME}_iree-run-9301b78b64f71dbcd7e18c9f3e74c72100b94f84b857ebbf0cfe85d2f88ade01-flagfile
  ${PACKAGE_NAME}_iree-run-11a7307088b796963f5249c11d8dcf99cdb35fba9c68115d1510ab3d8951efd6-flagfile
  ${PACKAGE_NAME}_iree-run-f60dff2e9fb2a0bdeb0c6d704174d67e4fdb6d4165647b463c9ff5ecff124008-flagfile
  ${PACKAGE_NAME}_iree-run-4465cfacd1bd870fc3c4a1ca3baf450e65a2269b531449fc39088d646166aca7-flagfile
  ${PACKAGE_NAME}_iree-run-cab84e227e3ea301e7215d2660ef1abc95c316fad1e790cd6b0ff76708884af8-flagfile
  ${PACKAGE_NAME}_iree-run-ea196f677da0ee2484c509ea2d1e6422e3f4abbaf9b3694af960942d697213d0-flagfile
  ${PACKAGE_NAME}_iree-run-291d0d979338966adc92b269dca4ced52382b7771a6b91f9bc71e440952e9bca-flagfile
  ${PACKAGE_NAME}_iree-run-41319614792667b0df14d9df5dce27730745ece78b4500bbbb683cb1d83d08fd-flagfile
  ${PACKAGE_NAME}_iree-run-5340ad727cf93b663444abdc3d3e1b71b410b4d1cafdd20d86390e3a0d316c81-flagfile
  ${PACKAGE_NAME}_iree-run-f7430d394f574c3b65b2c4a8503f4d6fe59d06d028c63b3dbe9d8ca3147918b3-flagfile
  ${PACKAGE_NAME}_iree-run-08522902c67de8ac584fa34c92fa460abd51e07ea6b7cbfbe511f6c49d75c901-flagfile
  ${PACKAGE_NAME}_iree-run-86aec1b2ed49dcb4581bddcb95097dafc4e04332fee4e3f429fa6622c0ae7423-flagfile
  ${PACKAGE_NAME}_iree-run-89583c9d8173f86736e142bad1d53806b2e14598ac8e5c5b57a3d4d6350825c4-flagfile
  ${PACKAGE_NAME}_iree-run-a9be3cb6d9f42925eab55ecb13110d675538a3ab5e51edf1ebe93e45ac7498b7-flagfile
  ${PACKAGE_NAME}_iree-run-c89ee60ba03610e4be991f6b5244044c96be3c52ac01c745c466bc7acc5129e8-flagfile
  ${PACKAGE_NAME}_iree-run-b11ab96441a0143f6d94da043e24f1d5a6953d41ed59e8494eddf8fc4b504eaa-flagfile
  ${PACKAGE_NAME}_iree-run-e7177ef165288b2cfdd95e00844c78f7b7aaa2002107cb6d365ec42c43da8d1e-flagfile
  ${PACKAGE_NAME}_iree-run-608e9c811acd31af1333a967ae7fbfc6e108f107a9eeb8e65695612c33a27a09-flagfile
  ${PACKAGE_NAME}_iree-run-562f5fc48bd3d23bc95ef5a2b950cbbb88708ca270992fa6323343dae7969403-flagfile
  ${PACKAGE_NAME}_iree-run-faa528c855571f526c505e6b63af0be02a669dbceb95c001d3078896aea2449c-flagfile
  ${PACKAGE_NAME}_iree-run-9c4d667f951f20083f816432aa572776e3135655fd8e33f43b74ebf582f307e9-flagfile
  ${PACKAGE_NAME}_iree-run-1610e5fda0bed864f7723bcd2fc94b948d7acbdd1bb296c7c09dde7060ad5448-flagfile
  ${PACKAGE_NAME}_iree-run-6af3f661193e72ea38cbc498414a38294e53d6c201eef9555b2888c1adbdae7f-flagfile
  ${PACKAGE_NAME}_iree-run-09fbc68cb81435106363e102bb163cfe97bd851610b0daa65e4af37586242878-flagfile
  ${PACKAGE_NAME}_iree-run-b402994b2e254a233bc4c102d1be16cb81d56b23722c12fa504f8af15ec8760a-flagfile
  ${PACKAGE_NAME}_iree-run-c7b1f6d4fe7a1767d1d039434869344bff311a88039f3f36c954f97f21b3ebf3-flagfile
  ${PACKAGE_NAME}_iree-run-060a4950ed5780ba74bc100a05ad36b4baaec0cde8f8315e79d470c0d6f6ed83-flagfile
  ${PACKAGE_NAME}_iree-run-64cc76a84e86dfce248344e2d5cce3846cb7bd59d9f510b345e8b8f6f8045259-flagfile
  ${PACKAGE_NAME}_iree-run-bc7afc6c841b4749167d612ee8874d809e93668e81a653de4066368b4a48e8cf-flagfile
  ${PACKAGE_NAME}_iree-run-70e3c62e269dbc27ac0124ce039255b47690342dc2cc8647ac5e9a419c24ee5b-flagfile
  ${PACKAGE_NAME}_iree-run-de0f4257cda0c2ad674beabe91a443a5cee25adddbadbd72232d20daf4a348e0-flagfile
  ${PACKAGE_NAME}_iree-run-b0068e24d909cb9085f10f77233b1e8e2b648b12ad26906bb532ce442f3452fb-flagfile
  ${PACKAGE_NAME}_iree-run-9302f4f7f2e4b8ca6513a742549b18e98519a60aeb16e0156491aa754ab86fcc-flagfile
  ${PACKAGE_NAME}_iree-run-448c132659a1369a42e70e95b614ab2644947dfe9d4549dea39bdaf55ae26374-flagfile
  ${PACKAGE_NAME}_iree-run-d0916289a64c89db14f4b09459b6404cd24668d7f2d2ff0668631174beb5bc19-flagfile
  ${PACKAGE_NAME}_iree-run-c986bfe95c3405f1665c12b4333122cf61fabbf943e0afdb38d4b8929bc01ac4-flagfile
  ${PACKAGE_NAME}_iree-run-d361ce2137e89c76be8f487cbb8807e1ade9589bf5e356fcbc2540f59aa56171-flagfile
  ${PACKAGE_NAME}_iree-run-fe9f27c6710225035e741ab8b669dc159f4c016ff421b7292e9cf52b410f113e-flagfile
  ${PACKAGE_NAME}_iree-run-e39a11cb267f85b5a16655190d1d6395791ba1afa3d1050fc7de6106899c5197-flagfile
  ${PACKAGE_NAME}_iree-run-da1823a15ae91422ed25bee8390afea31d539a3ce5087d50d746d910f6d9f2f8-flagfile
  ${PACKAGE_NAME}_iree-run-7d644106168eda8dfe40b1a79cfe07cc4ad69f40df2fb13858fe58728af084bd-flagfile
  ${PACKAGE_NAME}_iree-run-49bb980c861bf0db681172f8610e4663b9cf2dc5f83c5a8d49ad645873d20a62-flagfile
  ${PACKAGE_NAME}_iree-run-5f722e7441428249a620a10a1bad788ddf96c13cebda3be9b54bc021605341b8-flagfile
  ${PACKAGE_NAME}_iree-run-868ceaddb4382e723150b21ae1ce04f431e1f04b40212c5ca930c8ed26671afc-flagfile
  ${PACKAGE_NAME}_iree-run-9292f811797ee8e56233f46f6d28d946d57773e9f2a6abf609301c5dd0709cf1-flagfile
  ${PACKAGE_NAME}_iree-run-0113327cba37fa3fe8730981f5943b3689e3b93906b0d66051145fbd6dc52c12-flagfile
  ${PACKAGE_NAME}_iree-run-f565c9c78849e1761f634aeeeb6d6ee35f89af1554f4afe50b02d3d7233b5536-flagfile
  ${PACKAGE_NAME}_iree-run-9eee607453cd1462cfc2fdf7dfdcab2016fe0701329e078cc1d901818ffa3ce4-flagfile
  ${PACKAGE_NAME}_iree-run-0f149a7ff9e4aec7858d3eebbcd26e137bc086f01a132057e701a304d713cda7-flagfile
  ${PACKAGE_NAME}_iree-run-d9e181508dc318aa3aec6db6d6f10c48b0f36edc4bc3e95027d199e77da19793-flagfile
  ${PACKAGE_NAME}_iree-run-6a0da04aae3f1342a4f6aaa310b1fcdcb6c5382187f7610d90dffb94884e74d9-flagfile
  ${PACKAGE_NAME}_iree-run-f7c135234a821cdf109ab2e997d43310a4ee51e5865eea7ebbaafe76d71b732c-flagfile
  ${PACKAGE_NAME}_iree-run-6b79f25650f49dee7c19bcd7753f79e7348fb4c332dd3d67e4b06a6150c8f45e-flagfile
  ${PACKAGE_NAME}_iree-run-e23e6d368629ebe9f1dd8dba224f5a8b729608b939606f7c2a9edd1ed9ee8c6f-flagfile
  ${PACKAGE_NAME}_iree-run-8a94184fd7b5022ad836488e8948f2fbd8ac818e40cb92f49f1bc51d2a6d2647-flagfile
  ${PACKAGE_NAME}_iree-run-a9fd069a0dc5dbd3f65aadf383cc7ead04cd8f16bb7ee58b473cfe692a7933f8-flagfile
  ${PACKAGE_NAME}_iree-run-dd39a2d0e6a198ff1ff6eeac105fe39892cc90965249e7e8ecb453e6b9a0e5fe-flagfile
  ${PACKAGE_NAME}_iree-run-b6caf787b60262f27b833f67aa87a394b5298b1965a0690a980bf01b9eb6e3cf-flagfile
  ${PACKAGE_NAME}_iree-run-1a1883b71e25f7c0845d81504620d4e3f40903e25a3dcd9f4a75080ba5964eda-flagfile
  ${PACKAGE_NAME}_iree-run-b4788176d092953bb8490e61a09828fa371815977615a78e68a38180c6b507fd-flagfile
  ${PACKAGE_NAME}_iree-run-ed8f6bfbd572877cbc548a986d386f60a8e9060a3e3bfdd478680a41655382f3-flagfile
  ${PACKAGE_NAME}_iree-run-2a60a6a2462142cb66816ee12062c6f4efa3def8fea9adc4df46aaacd9c136bc-flagfile
  ${PACKAGE_NAME}_iree-run-426fe139b2f7c190b0656c35f5efdc59475286c401654f2ffd80f860f08cf563-flagfile
  ${PACKAGE_NAME}_iree-run-4aa5488ba3a15a67fc054c6e7c38d7589ea6780501ceae91f7f730b4e0a696d8-flagfile
  ${PACKAGE_NAME}_iree-run-0d167e8f1a24edc7620fa7ccd5c9f96a697ba3d0381e2118828af79ccdd06ca5-flagfile
  ${PACKAGE_NAME}_iree-run-ec6627de9282f1e49e0aa3d5bab632a950b310b4d401b6f3c7d12ce969abef2b-flagfile
  ${PACKAGE_NAME}_iree-run-4366620da66f5f7219af4e0882d29c1a5422ad1740f729b8b85609f120b4932a-flagfile
  ${PACKAGE_NAME}_iree-run-b400f170bd47a674f4d360fe0ec2c79a22b475689ea25a21d7eb38d168b6ad0c-flagfile
)

add_dependencies(iree-e2e-compile-stats-suites
  ${PACKAGE_NAME}_iree-module-2f2e448f73ef190ed35af1b25b6179ce15faba7ee7c12f4956730c441e9a27bd
  ${PACKAGE_NAME}_iree-module-c370b55d34f6d3c76aa838ff0a7be520de10a4824c5feaa773e2fb73a588ad8c
  ${PACKAGE_NAME}_iree-module-5a4c96fc279262ad7d7f1d446d0bd3685b2ca42e06b0167df5be5737c9d42901
  ${PACKAGE_NAME}_iree-module-27bbe62536a23529b4dd0df3d4913ee18344df9b6e2a32fc834fb7d9bc520e24
  ${PACKAGE_NAME}_iree-module-78511a42a50f705b944437a040e1ee3bb5b2595a3b1d4db788586fe48f9a2453
  ${PACKAGE_NAME}_iree-module-ef1ba1216f0f304c80b7a5b8bac545a987d04a100d9c1e5e66b75ce88636534c
  ${PACKAGE_NAME}_iree-module-1c4bc4b5ba3b5862efdbcbb9b3bf4a02f7ff9aa36e852e9b94dbe265d6bfaa99
  ${PACKAGE_NAME}_iree-module-439f7c958ce1d3200ea96935174cabde8e8fe6917a007f5e238553e9c2aa7625
  ${PACKAGE_NAME}_iree-module-4a3b570ba18c3c9eee458455aaff4aa29293a5c936a19862c698b4b3ddaf06e7
  ${PACKAGE_NAME}_iree-module-28e38bd436b036babc0fabe98b6e7c68ca3a7088e73dffff2c538adfa7d6af4c
  ${PACKAGE_NAME}_iree-module-a05a2b521a968e99411712e0e5191c3cd1d6295991f3b78acf61faca5d1cf85e
  ${PACKAGE_NAME}_iree-module-ddd1657bc5433ccca5c8ce562f581626457a793670958cd8b4016c426191a9c4
  ${PACKAGE_NAME}_iree-module-8ee3c7b136703472b53bc8a19d8d28945aca93953612ccc65e55cd1b3dfda6c8
  ${PACKAGE_NAME}_iree-module-01d35de2a55b9800e05151455eace0bf4493337ac1210fcc4904d630b075599a
  ${PACKAGE_NAME}_iree-module-2957930127e9b01e90ccddb7290e1c4b4abf6373cc36929809040e2c144d3fd7
  ${PACKAGE_NAME}_iree-module-846b19afd4c14b3e71d59087c5a2987edd65753d39db432961ce915688d457ac
  ${PACKAGE_NAME}_iree-module-de34105293194986d706823bd3d20ce784506ec5918c4d0efac9839020bb5fdd
  ${PACKAGE_NAME}_iree-module-373b890bed4c0f4828b957e37d319509bf41e39a4e47746285e27101d40f90bd
  ${PACKAGE_NAME}_iree-module-6e31f637a133e03db37c47d8a920a61306e366362e066f41c0eac0455cc6c77a
  ${PACKAGE_NAME}_iree-module-e0533bdae79e15707a6eb26eb7f09c4d7dbdbfc40b993a4ad6289cf2bb1f13cb
  ${PACKAGE_NAME}_iree-module-ad9a410e86dd9d649de58f5a7dbdc6cd2300fb6b6a363f4483e930d9944d2d07
  ${PACKAGE_NAME}_iree-module-9b12e389535e365bd2c35424c5f98442e1226d73b043eb40355bf456ad0263a2
  ${PACKAGE_NAME}_iree-module-63d75ff4a9998a86855e0e78ab2d782f52b90b58025584f3f03ec3103a81425b
  ${PACKAGE_NAME}_iree-module-00a22e8ada401de8f20895beff9a153585e585c2d686983e27f9d64fdf7d39a8
  ${PACKAGE_NAME}_iree-module-4c74339076df00d23baa17dcb3194043e0472da9d09db4e42a23841ff7bf67b0
  ${PACKAGE_NAME}_iree-module-9a1d228583ba1e56a19393f6938d16b5d582bb17f89fb5856b8b1c68e34abd45
  ${PACKAGE_NAME}_iree-module-152d0b6211fff7591df3418c549c979a8144fc34280c22a8b2b5ff8ea3d1b46c
  ${PACKAGE_NAME}_iree-module-e16d3f99f851c11fef6be64c7f06a637b410618f2618cf16aa599b54ea8970e3
  ${PACKAGE_NAME}_iree-module-8231a286cdc63a48f3f70a12ab5a182142c00cbebaccdc79e35ca552f02422e7
  ${PACKAGE_NAME}_iree-module-c9a7c5b08db10ed782045b6810cb4ee157da9e95590456d3839c06163ee30fa7
  ${PACKAGE_NAME}_iree-module-838cc09b422958a332fd76cf12a6a2a95b8346c8e8d2fe7b15cb5ace4c20581e
  ${PACKAGE_NAME}_iree-module-8b19868be1c797cb585551c871c4171e78817e0efc49d30d91b9d722be283de9
  ${PACKAGE_NAME}_iree-module-c2085883b1f5c767f37508ab998a4bcd17d169fe6a5197d28e4dca8772c90253
  ${PACKAGE_NAME}_iree-module-25ad2815eb690276e9c2183aaafaf17a3df734bb6164071ad92dbf1e7faf7509
  ${PACKAGE_NAME}_iree-module-65586f1e5b51439dd951529c35fa9000a928f90039cc6cfb66d5c81d07a6c62b
  ${PACKAGE_NAME}_iree-module-f770b1916e0b7a9a0b4aa9480791d21a46a352002ac1e38dfcea49ec0b63ed4e
  ${PACKAGE_NAME}_iree-module-16b5b80aaf1271b5ad782570340cc0c7c1c97e10b7e6c6cc6e5f3ede8393cb6c
  ${PACKAGE_NAME}_iree-module-65fa033050b916e8143d44b5081ee45db3b1946a5d77de223328a7fe92a1cc66
  ${PACKAGE_NAME}_iree-module-16ef56b6869d10b17e983fec62e9f48e6bb87e9a348ab52a0b2faabca2b03578
  ${PACKAGE_NAME}_iree-module-56bc9128e294585d749b8ebe34fd03060ba34d200eef185837b6002d0dcbfccb
  ${PACKAGE_NAME}_iree-module-eb1b1732e5d30ce4689b871f8ec18c50b30eedd418eb80330807fe505bb78f7e
  ${PACKAGE_NAME}_iree-module-bd32992875a8fc7a494c75933b1693d6d8b845fccf2b12504a8cba64d80ad110
  ${PACKAGE_NAME}_iree-module-ff7ed59e05efe8b9a397a179726f63da68a8a1ac3ea731924b4bd24dab491b34
  ${PACKAGE_NAME}_iree-module-8e2d1286ad9a7e360b0c26019146a22ec9188f8bdf8ad99341eb5531cdea2417
  ${PACKAGE_NAME}_iree-module-d967e293594998e48355d30d900dbdf77dbd6eedbff768112dbe8e7ec332b9eb
  ${PACKAGE_NAME}_iree-module-91a35228ead480e04b85998ccf3edfc891f44b5f79017b7fcab72cb66a812b07
  ${PACKAGE_NAME}_iree-module-f58c00ccab797ad4dbca3de3b50633588a68db0122aa011bdf81a9aca5ea692b
  ${PACKAGE_NAME}_iree-module-bfb6239769f044d2228f2efb5ce6aa51132455d9a8178e5a5ec8525ff5836e0d
  ${PACKAGE_NAME}_iree-module-469056c2ca5935d7c63d5424c635a439f94593a307e96483e4db16af1c15186e
  ${PACKAGE_NAME}_iree-module-3aab34d7c719c9d828a741a7900b4794302a587927c462b4ec8feec3f7d43e99
  ${PACKAGE_NAME}_iree-module-11a9de4ea6e17feff81429ed53e52a70e89c1cfeef0a73f10740c8420341b81d
  ${PACKAGE_NAME}_iree-module-9c01136785f28f0d2c969cee8ce87bde3267d63425c5d86d39137abdf7f0f196
  ${PACKAGE_NAME}_iree-module-3160a297a2c9d3d21caeec097b6fe19150c3feae5fa872e21817af0be8a8176a
  ${PACKAGE_NAME}_iree-module-0bf641c301b26975b8919a18de98d9dfd6444d6542085dd2d8e8155ea6bc8efe
  ${PACKAGE_NAME}_iree-module-058ea3aae7385269d001efd9eb2303887614d138ff69150b20a703fc7b97c2c6
  ${PACKAGE_NAME}_iree-module-fdff4caa105318036534bd28b76a6fe34e6e2412752c1a000f50fafe7f01ef07
  ${PACKAGE_NAME}_iree-module-14ce4459cb4ea8aa84b5315222e9cfe00fe8a3b456b2ae75a5eb943036279d68
  ${PACKAGE_NAME}_iree-module-0b2b90bac148aa9f7c2ee34db723a002823dbc0d5981e47511f09cafa3693101
  ${PACKAGE_NAME}_iree-module-bd015dc23ff2f9bf5d681039cbb0f6418cd3d09d09124c0238d8c2caf01dba24
  ${PACKAGE_NAME}_iree-module-e02e0460e54ee222b46c25e876f937eed5582b0823cad1b1d009fe406b160c33
  ${PACKAGE_NAME}_iree-module-d6590e27e94d8aac1b2bfb1e7c60e31dcddacd3a10687cdae998979fc31720fc
  ${PACKAGE_NAME}_iree-module-50567a33e0bd9aa5a32a6f61fca9ef8a70ac4d94313024f2c4ec92d9c543c599
  ${PACKAGE_NAME}_iree-module-6b9353f591f5044f661ecbbaafee502d710cf263527525d4f843b26cd43f11f7
  ${PACKAGE_NAME}_iree-module-5f03fee30980d1fb1074b82d329a1fa63b365858539743e672ad56c039dd732a
  ${PACKAGE_NAME}_iree-module-611a54141f98b17aa94abdba55d8a0487aa722bba4da6853c090f786975c5884
  ${PACKAGE_NAME}_iree-module-0d524f6ce80da6d1998eb8978623a2f6efd413e0b973c6f2dddf52a46b19907e
  ${PACKAGE_NAME}_iree-module-dd2a6a43dceabe7a807e280b43177cdf892d4ad20fdef4e3d3b6e39be7b9b09d
  ${PACKAGE_NAME}_iree-module-823ec09bcc061f124fa9229747054945cedf352e11d661a68785cb26af5f83b6
  ${PACKAGE_NAME}_iree-module-899c2de5e339b7e19528e80de1129a38511948ba3331932c22e23223707af4ca
  ${PACKAGE_NAME}_iree-module-c3cb44c1331872dc53919d0d8b2cab4c256dcdf8b0038f9b6a692a5c874f855b
  ${PACKAGE_NAME}_iree-module-dc81a08fe0e5140f22f328d9dfea1828e7318d67899a2534d3b02ff36032cb61
  ${PACKAGE_NAME}_iree-module-1171fb017e88de21814d71ea2d35564de6904d3d2359ef53e0fa2c67ea6e9914
  ${PACKAGE_NAME}_iree-module-9b9a47b0a97a0bd002bd7fd1f104caaa94b8bf60cf02ffcc2b50129679e4c6f3
  ${PACKAGE_NAME}_iree-module-4f5ab4bfb26a82d0f83133b9e85585f0c5b97cdb00143de31675158a5a71b457
  ${PACKAGE_NAME}_iree-module-480b59f233af720e16db8e5da1988a8d69bd61169bf5b5899f425ff98dc0dc19
  ${PACKAGE_NAME}_iree-module-7731488e1eb90da5480e76b4cd98b12c16b83d7c7011b0aa9ef3a5d6d2059a3c
  ${PACKAGE_NAME}_iree-module-61decb77e61b184a2c353fac3d60af1cd7c73abc867c23e9519f5e398265a728
  ${PACKAGE_NAME}_iree-module-4ec47dd2b4a43dd434d041d4d9db548076b70cfd63a2fec2971035394954f1d5
  ${PACKAGE_NAME}_iree-module-b0309994482c31c79242ee8ef3902b4cc54c1479688824734b33d2f554d6aff6
  ${PACKAGE_NAME}_iree-module-113994770711e5784a73ac623cbde328267c94b6341e001328b053c02b8bc08f
  ${PACKAGE_NAME}_iree-module-954bc3dc1fd0c22768ebfe898a67c0db3743d74e8fb776fced75eafb0421058f
  ${PACKAGE_NAME}_iree-module-6bb61b9c7107a9a30ad20c154321e7e9b14aefc407a8aeda41ac6df5eac757c4
  ${PACKAGE_NAME}_iree-module-24abf13f1d9be25ee353527d7f9096e5ccf53149f64eb8a90f069aea352eac21
  ${PACKAGE_NAME}_iree-module-cc884f3b2b1b188fad65da5462e4cea10e808adeda7f31f9657c3e3e29f876a9
  ${PACKAGE_NAME}_iree-module-e041f40aefb2124a84ea3413d71ec072753e5098a15b4637bf2690780badb52c
  ${PACKAGE_NAME}_iree-module-7bec578c7016cb7e017057c227a9b677901d14c0fff35e31c4a5cf12692db105
  ${PACKAGE_NAME}_iree-module-89a91c770dfce869ecb04e4b37e3b4d7da508a240da395cf240cc20ee8573857
  ${PACKAGE_NAME}_iree-module-e6049d40d7925bccce4859e5408f2ad53eb68309aa38b46b8a7e47c94a2cd8a3
  ${PACKAGE_NAME}_iree-module-a4194c053541ebc49b4912bbdf3ca211331fdca5d157440837144e59d279bf1f
  ${PACKAGE_NAME}_iree-module-599701d7114956cf64777412899cff57ea5be0478f9a2331377770beaad8f923
  ${PACKAGE_NAME}_iree-module-1d26fcfdb7387659356dd99ce7e10907c8560b0925ad839334b0a6155d25167a
  ${PACKAGE_NAME}_iree-module-b74ccbdce4ec07bb65313ee96b67c1b946a6c959158714706209a9df2b93ab0d
  ${PACKAGE_NAME}_iree-module-69ce6a0ceae813a4fdbd4728a7d7c663ebb3a482a082a941cd1c5f43bd844b3b
  ${PACKAGE_NAME}_iree-module-7d0c31ec6790283b9ff590485f6514f84861d58b8443e7fe70199b5c27185eb8
  ${PACKAGE_NAME}_iree-module-c56e0b1dd1ce04dc5534024a183f090f58c9c1f27ccca3faf32f62fa55576135
  ${PACKAGE_NAME}_iree-module-178907b155b6322dedfa947937f9caca5158ff3af167470f2de90347dba357f4
  ${PACKAGE_NAME}_iree-module-7dcabfd6caa769a75657e07e7315dd42f52b3d4cbc37d75098aca446d3ff4066
  ${PACKAGE_NAME}_iree-module-247b38beca9631678d80755b0b4db2b323ddc4d95772617889a6a4bb813c6b74
  ${PACKAGE_NAME}_iree-module-d39340f50384e970b103694a38d7d21d5b1171d7304630d25e925c5c2486bf10
  ${PACKAGE_NAME}_iree-module-d8f22b5a700abdef68fe791ad08acdfc6d238d82e00f264367d922b99b369ff7
  ${PACKAGE_NAME}_iree-module-c6a4903d1769d721782cf2b6e84837aca21f87fcf8759912a86ae2f572e8440d
  ${PACKAGE_NAME}_iree-module-3c43472d6cb0f74a1c08920e3f580b701e995a85305fd4b2e370542b4d449b18
  ${PACKAGE_NAME}_iree-module-92734e18b793ed29334b32490373ad4a008b4c8a47a3885c2c2dc8ed0fbce292
  ${PACKAGE_NAME}_iree-module-8408ed6dfa697c6e05e4d4c8f191c53856112e7a1b25a03f839a5046c936af37
  ${PACKAGE_NAME}_iree-module-c809128508627734ca76d422eec487a4b80d3538c0d08c3e339bebafee432f19
  ${PACKAGE_NAME}_iree-module-02b72f9538e4dfc9c789e63d722d5eab4333f3f55f8375503f433a790da119cc
  ${PACKAGE_NAME}_iree-module-e7bd41e564750501f39ac9690c18d1a2e77dc7999da710d0c0bf80751dda84a0
)
