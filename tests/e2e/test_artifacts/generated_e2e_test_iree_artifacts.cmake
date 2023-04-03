iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32.0_float.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.0_224.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_3dd5a95e-92a9-4486-9062-9a33224f28db_MobileNetV2_int8.0_224_quantized.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.tflite"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v2"
    "--tf-savedmodel-exported-names=forward"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v2"
    "--tf-savedmodel-exported-names=predict"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v2"
    "--tf-savedmodel-exported-names=forward"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_8871f602-571c-4eb8-b94d-554cc8ceec5a_BertLargeTF"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v1"
    "--tf-savedmodel-exported-names=serving_default"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_c393b4fa-beb4-45d5-982a-c6328aa05d08_Resnet50TF"
  IMPORT_FLAGS
    "--output-format=mlir-bytecode"
    "--tf-import-type=savedmodel_v2"
    "--tf-savedmodel-exported-names=forward"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
)

iree_bytecode_module(
  NAME
    "iree-module-87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-50084c6a5ef433943ae46fbbf852d266d20557be110fff9dea1e4e298040c7c6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_module_50084c6a5ef433943ae46fbbf852d266d20557be110fff9dea1e4e298040c7c6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "EfficientNetV2STF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "MiniLML12H384Uncased(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-047e75c462648b5fe1133f4ffbc3d1c7bdda154081d3eaa3be0b5445725b272b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cc474102-7d2f-4ec1-92ae-84e83ba0f390_EfficientNetV2SPT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2SPT_module_047e75c462648b5fe1133f4ffbc3d1c7bdda154081d3eaa3be0b5445725b272b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "EfficientNetV2SPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "BertForMaskedLMTF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "BertLargeTF(tf_v1) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "Resnet50TF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-2e9ba5db29e643ec5b7c3fb5a2e3cd4a8fd5ab2c13d66af4bd938d17f5f5c682"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_9a9515c7-cb68-4c34-b1d2-0e8c0a3620b8_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ClipTextSeqLen64PT_module_2e9ba5db29e643ec5b7c3fb5a2e3cd4a8fd5ab2c13d66af4bd938d17f5f5c682/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "ClipTextSeqLen64PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-4f70b75b4c1d77495814987264c5df3fc81a87b49574065458be603578f6b00d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_340553d1-e6fe-41b6-b2c7-687c74ccec56_Unet2dPT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Unet2dPT_module_4f70b75b4c1d77495814987264c5df3fc81a87b49574065458be603578f6b00d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "Unet2dPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-fd5cf4cd75cc1734be3eec2f4396491e9ebb7edd6b9c889cdc29ef72b47d6233"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_68caa96e-b8bb-48a2-bb08-a3044981a370_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetB7PT_module_fd5cf4cd75cc1734be3eec2f4396491e9ebb7edd6b9c889cdc29ef72b47d6233/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
  FRIENDLY_NAME
    "EfficientNetB7PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-759fdb2deb885ca5568741853e3497ab3d7037a42a4fdff0af590c803681ec60"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_module_759fdb2deb885ca5568741853e3497ab3d7037a42a4fdff0af590c803681ec60/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "EfficientNetV2STF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MiniLML12H384Uncased(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "BertForMaskedLMTF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "BertLargeTF(tf_v1) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-abe8d72fb1ef1302b96470e94c129843ebf76fdde715860b32cf73a24837fd21"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_9a9515c7-cb68-4c34-b1d2-0e8c0a3620b8_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ClipTextSeqLen64PT_module_abe8d72fb1ef1302b96470e94c129843ebf76fdde715860b32cf73a24837fd21/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "ClipTextSeqLen64PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-d3a776d4bb360ebce4148472f14fa242dd57baa59621172abece40b611497ef7"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_340553d1-e6fe-41b6-b2c7-687c74ccec56_Unet2dPT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Unet2dPT_module_d3a776d4bb360ebce4148472f14fa242dd57baa59621172abece40b611497ef7/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "Unet2dPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-bdd904cc5614ebf77609c7802a2dfc09f139aee2a247a247d10d320de72b0e28"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_bdd904cc5614ebf77609c7802a2dfc09f139aee2a247a247d10d320de72b0e28/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "BertForMaskedLMTF(tf_v2) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-45565cae821666fd34bca97be2e4cce3bd61e71308785728737d89acbb9bc9d2"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_45565cae821666fd34bca97be2e4cce3bd61e71308785728737d89acbb9bc9d2/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "BertLargeTF(tf_v1) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-fd81a89e9f8773bae142040775c7e3c4774f96b64f07f8d9f66b00191864ff40"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_fd81a89e9f8773bae142040775c7e3c4774f96b64f07f8d9f66b00191864ff40/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "Resnet50TF(tf_v2) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c4b43b31944dbd567e48efacfdb33f707eb248538cf70fa7dcf1085c6c7dbd3f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_9a9515c7-cb68-4c34-b1d2-0e8c0a3620b8_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ClipTextSeqLen64PT_module_c4b43b31944dbd567e48efacfdb33f707eb248538cf70fa7dcf1085c6c7dbd3f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "ClipTextSeqLen64PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c8ec2db5ee884e0af17814e61b13d7f7f1f2d4f7028e8c1920d0d968c27de2bb"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_340553d1-e6fe-41b6-b2c7-687c74ccec56_Unet2dPT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Unet2dPT_module_c8ec2db5ee884e0af17814e61b13d7f7f1f2d4f7028e8c1920d0d968c27de2bb/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "Unet2dPT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9470c46965ea67794da45496454c82eade29b5a519d8037b1314738621e02260"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_68caa96e-b8bb-48a2-bb08-a3044981a370_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetB7PT_module_9470c46965ea67794da45496454c82eade29b5a519d8037b1314738621e02260/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "EfficientNetB7PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-154de838dc7742304c9e27a9f315645a915493bc4f84160e29b15b7fc6dc475e"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-1_BertLargePTBatch1.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch1_module_154de838dc7742304c9e27a9f315645a915493bc4f84160e29b15b7fc6dc475e/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "BertLargePTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-21a83414dc1feb2a3b5fb6afadc36c1022a0ab747380291dbb309637d5f32eab"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-16_BertLargePTBatch16.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch16_module_21a83414dc1feb2a3b5fb6afadc36c1022a0ab747380291dbb309637d5f32eab/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "BertLargePTBatch16(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7d93a3e9b342045ed2546960ee3d08f60be237fd8974fe35447a41234f322148"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-24_BertLargePTBatch24.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch24_module_7d93a3e9b342045ed2546960ee3d08f60be237fd8974fe35447a41234f322148/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "BertLargePTBatch24(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-995595b7b80370d9c484413e9d06c2de928db32777101c91d3bac0d3797058ec"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-32_BertLargePTBatch32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch32_module_995595b7b80370d9c484413e9d06c2de928db32777101c91d3bac0d3797058ec/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "BertLargePTBatch32(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-45ed9ca2efb9bb645316e856a5c464b76e56902e930ff1d4e54fdef38043b33f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-48_BertLargePTBatch48.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch48_module_45ed9ca2efb9bb645316e856a5c464b76e56902e930ff1d4e54fdef38043b33f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "BertLargePTBatch48(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8c26886533593b666597f8698e33c0e9c98b349bdf8ac01c3d122cc20b741def"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-64_BertLargePTBatch64.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch64_module_8c26886533593b666597f8698e33c0e9c98b349bdf8ac01c3d122cc20b741def/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "BertLargePTBatch64(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-222490941c581c67f5ce710e9ea141482cd8074294a43a5dc67a01a127037cd4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_fd05da43-5e37-4fa0-88f8-3ceec1682345-batch-1_Resnet50PTBatch1.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50PTBatch1_module_222490941c581c67f5ce710e9ea141482cd8074294a43a5dc67a01a127037cd4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "Resnet50PTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-39556c12d84502be71243197b99b0f8c22949093c561c55c01ca906812a86288"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_fd05da43-5e37-4fa0-88f8-3ceec1682345-batch-8_Resnet50PTBatch8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50PTBatch8_module_39556c12d84502be71243197b99b0f8c22949093c561c55c01ca906812a86288/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "Resnet50PTBatch8(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-04ca0a5077b7dd5ace66d803c9b822dff3428b24e7620a61995aff0907af9533"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_04ca0a5077b7dd5ace66d803c9b822dff3428b24e7620a61995aff0907af9533/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "EfficientNetV2STF(tf_v2) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-deafafd0926321a4b8e4dc73ed4a30b2ed9317d26488246461415be2ee857eb1"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_deafafd0926321a4b8e4dc73ed4a30b2ed9317d26488246461415be2ee857eb1/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME
    "MiniLML12H384Uncased(tf_v2) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-68f0eb37bb72d0d6605ecdf42691c64125960e122844b0beeae350871a445b1c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_68f0eb37bb72d0d6605ecdf42691c64125960e122844b0beeae350871a445b1c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-a7a1553d0739151f06bbc00a3ef8b67b0606463eab4b6607069aa94ea0bfd92f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_a7a1553d0739151f06bbc00a3ef8b67b0606463eab4b6607069aa94ea0bfd92f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e80d71ed8e86c0756226b2323e27e2c7c0fff8eddde59ba69e9222d36ee3eef6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_e80d71ed8e86c0756226b2323e27e2c7c0fff8eddde59ba69e9222d36ee3eef6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "MobileNetV1_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-f59cd43a2a2d6e4b3159efa358a6fa0879e72f6f4f0a23af4c8ab550f256986a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_f59cd43a2a2d6e4b3159efa358a6fa0879e72f6f4f0a23af4c8ab550f256986a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-14a15b9072caaee5e2a274a9bbc436a56d095611e5a8e9841f110741d34231f9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_14a15b9072caaee5e2a274a9bbc436a56d095611e5a8e9841f110741d34231f9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e850fce2d36ddb09ccc34471641adb77418b93c0949d22ab75806d7cfc489ae3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_e850fce2d36ddb09ccc34471641adb77418b93c0949d22ab75806d7cfc489ae3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-93034e879f88475eeeff56dd38297a1d1183f123fbbf5925a07b7c934e5f5d48"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_module_93034e879f88475eeeff56dd38297a1d1183f123fbbf5925a07b7c934e5f5d48/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "MobileNetV2_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-f963d812114af925e0a4b110ee83aeb0e3b41d49fad19b3f449b6a9ccba43b8d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_f963d812114af925e0a4b110ee83aeb0e3b41d49fad19b3f449b6a9ccba43b8d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9d909aa679e8c380ff7e93292ef28dbd3bb9e7cc62329f90aef78ce9c7efeff9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9d909aa679e8c380ff7e93292ef28dbd3bb9e7cc62329f90aef78ce9c7efeff9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-1ef2da238443010024d69ceb6fe6ab6fa8cf5f4ce7d424dace3a572592043e70"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_1ef2da238443010024d69ceb6fe6ab6fa8cf5f4ce7d424dace3a572592043e70/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-d399d2540161867967b0590a659d0a86206e145d884bd7afd9abd818713aacd2"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_module_d399d2540161867967b0590a659d0a86206e145d884bd7afd9abd818713aacd2/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME
    "MobileNetV2_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_f06fff09f8cebc27d1674045712aaa60afe7aef388c4bc505897f55c3a0d8abb/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_91469dce5ace1877a717a8ca5f53c1c8a3ac2f2eb8d116382eb4a5f5ace3a9c8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_8b6e3f5ddee79e020f1b25ce676d3772dda8cf14f619a3f0649a9f14e0159cd0/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_83ef2a35d14b4acf5d569798accc3b5bb693394633d444bdd5c9d55461f549ae/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_244092c4ecd63533f225b0bc16b308e11265ca6b10696f8975866844793e2c4f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_5f46d5bcca6c809578ba7b32829ee4417a67eaa5033b5b58b81e3a8b0b3433ad/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_012cc9d71edd7a345ef45d52c630d53ed04ee93523524b417330c644d8f6ce1b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_5eeff62c8ab391e247ede57b226b4be03b81067b8831b13bfa47b62e2dac275a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7d44f561ce87a08f4ed53264b6b95ddd7926263fef781e5135cc696d5a4d3719/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_0d6fed1f6845ce0d70bf81ac2522e9806b4538917cb7ee02a1b6e6a60d6b7115/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_2c042cd2ded71abef51933f0466cf58a52de2c67e1a23fde98b1fad1b33e6d7c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_78db72d7b7a1f871a3752c8165cbebc678047f2df5d36b10dd82ee870873b620/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0689c824551e3a3650463db3807664420f1033ee1b290fa25e63d04cf1a0a6ef/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_69f971c48d4b181ed4a4ec22790dc62fa997aa28ad16e0e176642955ccbbc2f0/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d,dotprod]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-deba344af418957cbd9dc0834a100bc30ba242d7fddb4dc6ba10a87d0af32dc1"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_deba344af418957cbd9dc0834a100bc30ba242d7fddb4dc6ba10a87d0af32dc1/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-72feea41a0b54e4c9a761933079cba1b2c012e5a5d4b2953ffaa86faaa29a648"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_72feea41a0b54e4c9a761933079cba1b2c012e5a5d4b2953ffaa86faaa29a648/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-b33d5ca3311e31b99daa1c1f13ea5571713538a7889627153ea431debb9b5e2a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_b33d5ca3311e31b99daa1c1f13ea5571713538a7889627153ea431debb9b5e2a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-076a1e95f3384b58a77a672c7c36463b091e574b5a6f6eaf78841537b0d1c930"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_076a1e95f3384b58a77a672c7c36463b091e574b5a6f6eaf78841537b0d1c930/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-05e976592f58a292874d99bd7627e655b15c5460455a08b9ce67e9f7f65b6269"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_05e976592f58a292874d99bd7627e655b15c5460455a08b9ce67e9f7f65b6269/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-dd4366d716cccdd83b6f777ee966157b92838569f094371b325b926e73c7b1b8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_dd4366d716cccdd83b6f777ee966157b92838569f094371b325b926e73c7b1b8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-4a1632637ce87fe991848942b028c36732b2bea00920d275ffbcaf2cd9446152"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_4a1632637ce87fe991848942b028c36732b2bea00920d275ffbcaf2cd9446152/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7f5a7ca6eb12e8ac322d8bb0deb59630d721ab141acc3a941e168d98af507034"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7f5a7ca6eb12e8ac322d8bb0deb59630d721ab141acc3a941e168d98af507034/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e3444362e0b630df1b5f70a28089b0764d9ddc886dda852a0ef1300e369aee4d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_e3444362e0b630df1b5f70a28089b0764d9ddc886dda852a0ef1300e369aee4d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c95d82463222cf5ea385760eb1f0f0ee28a876620e29fbd59f8f4cb8a5307bc8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_c95d82463222cf5ea385760eb1f0f0ee28a876620e29fbd59f8f4cb8a5307bc8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-f73d0b9c84ec91b594b00eb3800c372884050fce3e4fb9d80eb407d7b0697412"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_f73d0b9c84ec91b594b00eb3800c372884050fce3e4fb9d80eb407d7b0697412/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-67ad72cbb9eb9c4746249922e6232b1a17b5d6eeabd9b69ed4d527c1676c77bd"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_67ad72cbb9eb9c4746249922e6232b1a17b5d6eeabd9b69ed4d527c1676c77bd/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-ad38a059079822e7331470e086bc1caca3dbe435878b38e1355229a39d1d25d2"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_ad38a059079822e7331470e086bc1caca3dbe435878b38e1355229a39d1d25d2/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8eb82c90485cc4281676866b62e6820b60a38ba81068e95254a7d0ddaddc59c3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_8eb82c90485cc4281676866b62e6820b60a38ba81068e95254a7d0ddaddc59c3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-cc9b305fb2c95f582d144f1063fc12fd996e757f84738e3de846b0197981bcc2"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_cc9b305fb2c95f582d144f1063fc12fd996e757f84738e3de846b0197981bcc2/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8930c217bc20a5abf9313fd714a77bf47acefb006e95ba07b5e48caf872541b0"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_8930c217bc20a5abf9313fd714a77bf47acefb006e95ba07b5e48caf872541b0/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9b6e19b8cab376fffe309f090beeecad08903c0975c9e7ffc480dd46074b97b3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_9b6e19b8cab376fffe309f090beeecad08903c0975c9e7ffc480dd46074b97b3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-d5ea172c189c6a6a3b61a7e83f7263b9b38cad756d2d5dec1b2db88162eece2a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_d5ea172c189c6a6a3b61a7e83f7263b9b38cad756d2d5dec1b2db88162eece2a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-bb1cb8e00fd4cee513b2481bd5faf39842ad9c57f80f2e50ddb48763fd030721"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_bb1cb8e00fd4cee513b2481bd5faf39842ad9c57f80f2e50ddb48763fd030721/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-dc26eca15bb97d42bcfa019de5080da54fb3d7624809aa1c8ac731e710544e18"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_dc26eca15bb97d42bcfa019de5080da54fb3d7624809aa1c8ac731e710544e18/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-90d4cb308fc14e4832336ebd1e1387a791a20b92af633d0fec9161b807e13427"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_90d4cb308fc14e4832336ebd1e1387a791a20b92af633d0fec9161b807e13427/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-ba322b423888fe93e86180e98668f486afa915565edf3813af3f9da6ad7c9dc9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_ba322b423888fe93e86180e98668f486afa915565edf3813af3f9da6ad7c9dc9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-42bc978e75d5fed90c10d5812a8df7a6153e311b19b9c7faf41a588bdc4da7d8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_42bc978e75d5fed90c10d5812a8df7a6153e311b19b9c7faf41a588bdc4da7d8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-demote-f32-to-f16"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,demote-f32-to-f16]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-894671e6d7319df243fa8b386b14d5e648fe2ffd66cb95e5a82ad58d7273ada8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_894671e6d7319df243fa8b386b14d5e648fe2ffd66cb95e5a82ad58d7273ada8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-0af4e7fa84d361d2faa267336c3c090e337f7feff47f0e559337521c07626ba7"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_0af4e7fa84d361d2faa267336c3c090e337f7feff47f0e559337521c07626ba7/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-138aacc154449b519698bd7ddb4dc9f407cbc1fd86fdbbe901be974058b70319"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_138aacc154449b519698bd7ddb4dc9f407cbc1fd86fdbbe901be974058b70319/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-a47669d662bd18dbd323737d84fd45753c01abaa81b4761eb51b35ce04ee7491"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_a47669d662bd18dbd323737d84fd45753c01abaa81b4761eb51b35ce04ee7491/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-27c12f520d68a8a754b11ad44cf8ad3ef5c1ec281fc9d84fbc64a7b573cb68b5"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_27c12f520d68a8a754b11ad44cf8ad3ef5c1ec281fc9d84fbc64a7b573cb68b5/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-ec14aafacf15628918390531318a4827d3ac8771b75d60a8239d901dcb4fd898"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_ec14aafacf15628918390531318a4827d3ac8771b75d60a8239d901dcb4fd898/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-507fabae3d24c2448c8f2af676d68df0a22716a06fa7ffc3b4f865e9272ecdc8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_507fabae3d24c2448c8f2af676d68df0a22716a06fa7ffc3b4f865e9272ecdc8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-4924918e121a92f802a094de2d67e9c2673c9fdc39faa6a11ac1d38b631a2914"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_4924918e121a92f802a094de2d67e9c2673c9fdc39faa6a11ac1d38b631a2914/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-383e7567e79adf2178e3d25785855905a4477b6c8daf16bc50d5f9f44d2343d9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_383e7567e79adf2178e3d25785855905a4477b6c8daf16bc50d5f9f44d2343d9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9faad3c50bc64c297531a36bd2ab235b680070d8233ce5ab7d964ac36c7c5563"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_9faad3c50bc64c297531a36bd2ab235b680070d8233ce5ab7d964ac36c7c5563/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-flow-demote-f32-to-f16"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,demote-f32-to-f16]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-90eb882b36d0293eb5b22ffb64f7c395cefa0360b37d4b84854d8b2c52a76358"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_90eb882b36d0293eb5b22ffb64f7c395cefa0360b37d4b84854d8b2c52a76358/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e1143fcb8d1d262a3b2ac689cf21a1943cc59b2de06192fef185a9bf84994b86"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_e1143fcb8d1d262a3b2ac689cf21a1943cc59b2de06192fef185a9bf84994b86/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-65aa6dabd0c24dbe48a090db641d82b437d4abd428a6ca0e7b360d3f087a16c8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_65aa6dabd0c24dbe48a090db641d82b437d4abd428a6ca0e7b360d3f087a16c8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-131aa61ffcacafc6ae701ff89045cf42d2490ac0c4a1d862bc83c23edb3b92e5"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_131aa61ffcacafc6ae701ff89045cf42d2490ac0c4a1d862bc83c23edb3b92e5/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-db9784471b47ae8bf55ca7e0821e35a1686256a208df40443c114f1adcdd26f6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_db9784471b47ae8bf55ca7e0821e35a1686256a208df40443c114f1adcdd26f6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-aada5dcefdd361b5227276129e93547a6932a05d380acd342fa33e5b672d498c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_aada5dcefdd361b5227276129e93547a6932a05d380acd342fa33e5b672d498c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-887c7a7b540f11ee5e0158143fd46a503a4851211b10b353ec0e00b8b1beb575"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_887c7a7b540f11ee5e0158143fd46a503a4851211b10b353ec0e00b8b1beb575/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-11766f32ea6a3121d7527bcdf32dead45ab7b3922d72addb46945cfdab784ec0"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_11766f32ea6a3121d7527bcdf32dead45ab7b3922d72addb46945cfdab784ec0/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-0efb794c3a385045fa4d3d086c2a593ce67c4807e9456271f05f4e28490d1c49"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0efb794c3a385045fa4d3d086c2a593ce67c4807e9456271f05f4e28490d1c49/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7a3d36d7234ce4abfd833964492631bd81df823e6bde8cd3a1fadfa4faa7c787"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_7a3d36d7234ce4abfd833964492631bd81df823e6bde8cd3a1fadfa4faa7c787/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-flow-demote-f32-to-f16"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,demote-f32-to-f16]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e52ef218d6b5b16f35c420fca42ff31c5cf733bc075ed235ce57cf4662824606"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_e52ef218d6b5b16f35c420fca42ff31c5cf733bc075ed235ce57cf4662824606/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-1202e44fbbde0a9581c641d124452363273f0d4395445dc06e60c8e49a27357b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_1202e44fbbde0a9581c641d124452363273f0d4395445dc06e60c8e49a27357b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3c1fbaa12bbd37bf2a17d8a605cdeca75c0e8cc9c6f7e95359d13e47a4270dc3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_3c1fbaa12bbd37bf2a17d8a605cdeca75c0e8cc9c6f7e95359d13e47a4270dc3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [vmvx-generic-vmvx-vmvx][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [vmvx-generic-vmvx-vmvx][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-2f2e448f73ef190ed35af1b25b6179ce15faba7ee7c12f4956730c441e9a27bd"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_2f2e448f73ef190ed35af1b25b6179ce15faba7ee7c12f4956730c441e9a27bd/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c370b55d34f6d3c76aa838ff0a7be520de10a4824c5feaa773e2fb73a588ad8c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_c370b55d34f6d3c76aa838ff0a7be520de10a4824c5feaa773e2fb73a588ad8c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-5a4c96fc279262ad7d7f1d446d0bd3685b2ca42e06b0167df5be5737c9d42901"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_5a4c96fc279262ad7d7f1d446d0bd3685b2ca42e06b0167df5be5737c9d42901/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-27bbe62536a23529b4dd0df3d4913ee18344df9b6e2a32fc834fb7d9bc520e24"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_27bbe62536a23529b4dd0df3d4913ee18344df9b6e2a32fc834fb7d9bc520e24/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-78511a42a50f705b944437a040e1ee3bb5b2595a3b1d4db788586fe48f9a2453"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_78511a42a50f705b944437a040e1ee3bb5b2595a3b1d4db788586fe48f9a2453/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-ef1ba1216f0f304c80b7a5b8bac545a987d04a100d9c1e5e66b75ce88636534c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_ef1ba1216f0f304c80b7a5b8bac545a987d04a100d9c1e5e66b75ce88636534c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-1c4bc4b5ba3b5862efdbcbb9b3bf4a02f7ff9aa36e852e9b94dbe265d6bfaa99"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_1c4bc4b5ba3b5862efdbcbb9b3bf4a02f7ff9aa36e852e9b94dbe265d6bfaa99/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-2cb62cec021aeb2abd869ba5276996e362bae130db7cc2d79389cccee6e372b9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_module_2cb62cec021aeb2abd869ba5276996e362bae130db7cc2d79389cccee6e372b9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-439f7c958ce1d3200ea96935174cabde8e8fe6917a007f5e238553e9c2aa7625"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_439f7c958ce1d3200ea96935174cabde8e8fe6917a007f5e238553e9c2aa7625/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-4a3b570ba18c3c9eee458455aaff4aa29293a5c936a19862c698b4b3ddaf06e7"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_4a3b570ba18c3c9eee458455aaff4aa29293a5c936a19862c698b4b3ddaf06e7/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-28e38bd436b036babc0fabe98b6e7c68ca3a7088e73dffff2c538adfa7d6af4c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_28e38bd436b036babc0fabe98b6e7c68ca3a7088e73dffff2c538adfa7d6af4c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-a05a2b521a968e99411712e0e5191c3cd1d6295991f3b78acf61faca5d1cf85e"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_a05a2b521a968e99411712e0e5191c3cd1d6295991f3b78acf61faca5d1cf85e/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-01d35de2a55b9800e05151455eace0bf4493337ac1210fcc4904d630b075599a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_01d35de2a55b9800e05151455eace0bf4493337ac1210fcc4904d630b075599a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNetV2STF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-2957930127e9b01e90ccddb7290e1c4b4abf6373cc36929809040e2c144d3fd7"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_2957930127e9b01e90ccddb7290e1c4b4abf6373cc36929809040e2c144d3fd7/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MiniLML12H384Uncased(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-b5078b9d2031b69ec5ce9b775c8701cef73add8ebfb786d9189ca3fb6474cf73"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cc474102-7d2f-4ec1-92ae-84e83ba0f390_EfficientNetV2SPT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2SPT_module_b5078b9d2031b69ec5ce9b775c8701cef73add8ebfb786d9189ca3fb6474cf73/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNetV2SPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-ddd1657bc5433ccca5c8ce562f581626457a793670958cd8b4016c426191a9c4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_ddd1657bc5433ccca5c8ce562f581626457a793670958cd8b4016c426191a9c4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertForMaskedLMTF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8ee3c7b136703472b53bc8a19d8d28945aca93953612ccc65e55cd1b3dfda6c8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_8ee3c7b136703472b53bc8a19d8d28945aca93953612ccc65e55cd1b3dfda6c8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargeTF(tf_v1) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-846b19afd4c14b3e71d59087c5a2987edd65753d39db432961ce915688d457ac"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_846b19afd4c14b3e71d59087c5a2987edd65753d39db432961ce915688d457ac/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "Resnet50TF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-f13d9b8b55decd1684c262d230ad42c95c5def21d9ee605ebc656181d1c54a0a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_9a9515c7-cb68-4c34-b1d2-0e8c0a3620b8_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ClipTextSeqLen64PT_module_f13d9b8b55decd1684c262d230ad42c95c5def21d9ee605ebc656181d1c54a0a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "ClipTextSeqLen64PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-48b14091499ede62cbd8622a11fff448d9a0354662ad87fae832256fc1610f13"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_340553d1-e6fe-41b6-b2c7-687c74ccec56_Unet2dPT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Unet2dPT_module_48b14091499ede62cbd8622a11fff448d9a0354662ad87fae832256fc1610f13/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "Unet2dPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-63a2a9acdb2dafb66a49487fd7eecb79e49d870ec9b9ef17a1e8e9d50fe5c9ba"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_68caa96e-b8bb-48a2-bb08-a3044981a370_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetB7PT_module_63a2a9acdb2dafb66a49487fd7eecb79e49d870ec9b9ef17a1e8e9d50fe5c9ba/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNetB7PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-de34105293194986d706823bd3d20ce784506ec5918c4d0efac9839020bb5fdd"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_de34105293194986d706823bd3d20ce784506ec5918c4d0efac9839020bb5fdd/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-373b890bed4c0f4828b957e37d319509bf41e39a4e47746285e27101d40f90bd"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_373b890bed4c0f4828b957e37d319509bf41e39a4e47746285e27101d40f90bd/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-6e31f637a133e03db37c47d8a920a61306e366362e066f41c0eac0455cc6c77a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_6e31f637a133e03db37c47d8a920a61306e366362e066f41c0eac0455cc6c77a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e0533bdae79e15707a6eb26eb7f09c4d7dbdbfc40b993a4ad6289cf2bb1f13cb"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_e0533bdae79e15707a6eb26eb7f09c4d7dbdbfc40b993a4ad6289cf2bb1f13cb/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-ad9a410e86dd9d649de58f5a7dbdc6cd2300fb6b6a363f4483e930d9944d2d07"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_ad9a410e86dd9d649de58f5a7dbdc6cd2300fb6b6a363f4483e930d9944d2d07/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9b12e389535e365bd2c35424c5f98442e1226d73b043eb40355bf456ad0263a2"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_9b12e389535e365bd2c35424c5f98442e1226d73b043eb40355bf456ad0263a2/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-63d75ff4a9998a86855e0e78ab2d782f52b90b58025584f3f03ec3103a81425b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_63d75ff4a9998a86855e0e78ab2d782f52b90b58025584f3f03ec3103a81425b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-fdce9dd1dfd20592880fce8969f91ac6abe84eca6922b4df4cbffe512abbfcb6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_module_fdce9dd1dfd20592880fce8969f91ac6abe84eca6922b4df4cbffe512abbfcb6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-00a22e8ada401de8f20895beff9a153585e585c2d686983e27f9d64fdf7d39a8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_00a22e8ada401de8f20895beff9a153585e585c2d686983e27f9d64fdf7d39a8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-4c74339076df00d23baa17dcb3194043e0472da9d09db4e42a23841ff7bf67b0"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_4c74339076df00d23baa17dcb3194043e0472da9d09db4e42a23841ff7bf67b0/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9a1d228583ba1e56a19393f6938d16b5d582bb17f89fb5856b8b1c68e34abd45"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_9a1d228583ba1e56a19393f6938d16b5d582bb17f89fb5856b8b1c68e34abd45/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-152d0b6211fff7591df3418c549c979a8144fc34280c22a8b2b5ff8ea3d1b46c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_152d0b6211fff7591df3418c549c979a8144fc34280c22a8b2b5ff8ea3d1b46c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c9a7c5b08db10ed782045b6810cb4ee157da9e95590456d3839c06163ee30fa7"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_c9a7c5b08db10ed782045b6810cb4ee157da9e95590456d3839c06163ee30fa7/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNetV2STF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-838cc09b422958a332fd76cf12a6a2a95b8346c8e8d2fe7b15cb5ace4c20581e"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_838cc09b422958a332fd76cf12a6a2a95b8346c8e8d2fe7b15cb5ace4c20581e/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MiniLML12H384Uncased(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e16d3f99f851c11fef6be64c7f06a637b410618f2618cf16aa599b54ea8970e3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_e16d3f99f851c11fef6be64c7f06a637b410618f2618cf16aa599b54ea8970e3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertForMaskedLMTF(tf_v2) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8231a286cdc63a48f3f70a12ab5a182142c00cbebaccdc79e35ca552f02422e7"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_8231a286cdc63a48f3f70a12ab5a182142c00cbebaccdc79e35ca552f02422e7/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargeTF(tf_v1) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-23a395a0b9ec63ecaeffe98988633de34fc32c267c35bc2584651d639a15af6e"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_9a9515c7-cb68-4c34-b1d2-0e8c0a3620b8_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ClipTextSeqLen64PT_module_23a395a0b9ec63ecaeffe98988633de34fc32c267c35bc2584651d639a15af6e/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "ClipTextSeqLen64PT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-eb1a0a23dc0205fd44ae3b5ca385c651de811b10b6d98fa8b0fb9ef54002c728"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_340553d1-e6fe-41b6-b2c7-687c74ccec56_Unet2dPT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Unet2dPT_module_eb1a0a23dc0205fd44ae3b5ca385c651de811b10b6d98fa8b0fb9ef54002c728/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "Unet2dPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8b19868be1c797cb585551c871c4171e78817e0efc49d30d91b9d722be283de9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertForMaskedLMTF_module_8b19868be1c797cb585551c871c4171e78817e0efc49d30d91b9d722be283de9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertForMaskedLMTF(tf_v2) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c2085883b1f5c767f37508ab998a4bcd17d169fe6a5197d28e4dca8772c90253"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargeTF_module_c2085883b1f5c767f37508ab998a4bcd17d169fe6a5197d28e4dca8772c90253/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargeTF(tf_v1) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-f770b1916e0b7a9a0b4aa9480791d21a46a352002ac1e38dfcea49ec0b63ed4e"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50TF_module_f770b1916e0b7a9a0b4aa9480791d21a46a352002ac1e38dfcea49ec0b63ed4e/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "Resnet50TF(tf_v2) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-88b6b5f712cd2f40d07a136e7f911c05b976c390e07f104c970292dee9a77e9a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_9a9515c7-cb68-4c34-b1d2-0e8c0a3620b8_ClipTextSeqLen64PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ClipTextSeqLen64PT_module_88b6b5f712cd2f40d07a136e7f911c05b976c390e07f104c970292dee9a77e9a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "ClipTextSeqLen64PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-925cdb19f2aa31a1907c81b5a9e179d91280c77b08a039c1cbf146f71683dde9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_340553d1-e6fe-41b6-b2c7-687c74ccec56_Unet2dPT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Unet2dPT_module_925cdb19f2aa31a1907c81b5a9e179d91280c77b08a039c1cbf146f71683dde9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "Unet2dPT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3c94ab45ad76bd8b2083729b65340b987da3247c854faf7d06431cb05a3b0a23"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_68caa96e-b8bb-48a2-bb08-a3044981a370_EfficientNetB7PT.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetB7PT_module_3c94ab45ad76bd8b2083729b65340b987da3247c854faf7d06431cb05a3b0a23/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNetB7PT(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-999f2edcdf9fbc84e0969923f8605e9069810a63849973a5b74488f83d14a2fe"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-1_BertLargePTBatch1.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch1_module_999f2edcdf9fbc84e0969923f8605e9069810a63849973a5b74488f83d14a2fe/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargePTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-5176d0f5fa331fa047395186a866870fe4210f637472ef4bb0a3f1ba3a62a749"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-16_BertLargePTBatch16.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch16_module_5176d0f5fa331fa047395186a866870fe4210f637472ef4bb0a3f1ba3a62a749/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargePTBatch16(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e67816e321544bb61b63ed8cd8b7faa8ff7d35cca15832b5fbc117f4693b3e78"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-24_BertLargePTBatch24.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch24_module_e67816e321544bb61b63ed8cd8b7faa8ff7d35cca15832b5fbc117f4693b3e78/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargePTBatch24(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-d746ecf3d747f18306b6dea4cb6b9e9dbf987fe7fd4d0b27b39a57a213e75dd9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-32_BertLargePTBatch32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch32_module_d746ecf3d747f18306b6dea4cb6b9e9dbf987fe7fd4d0b27b39a57a213e75dd9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargePTBatch32(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-439dab16cd6df449fc83eb3e1603fa86ad811e749bcad2c3e3176976c56848e5"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-48_BertLargePTBatch48.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch48_module_439dab16cd6df449fc83eb3e1603fa86ad811e749bcad2c3e3176976c56848e5/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargePTBatch48(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-815e44d1ac31402e86d0fef72e79474a25dfad3a5c15b8cdd3642e101274342d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-64_BertLargePTBatch64.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_BertLargePTBatch64_module_815e44d1ac31402e86d0fef72e79474a25dfad3a5c15b8cdd3642e101274342d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "BertLargePTBatch64(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-681f456f27dbb79e7d8d0266bf835866d9f29f87eafad7e867ac13c84602742f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_fd05da43-5e37-4fa0-88f8-3ceec1682345-batch-1_Resnet50PTBatch1.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50PTBatch1_module_681f456f27dbb79e7d8d0266bf835866d9f29f87eafad7e867ac13c84602742f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "Resnet50PTBatch1(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3b0ae1403ef444d812f0c7b37fda7311e2cc4ea407850ee7b91e6984b9c86100"
  SRC
    "${ROOT_ARTIFACTS_DIR}/model_fd05da43-5e37-4fa0-88f8-3ceec1682345-batch-8_Resnet50PTBatch8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_Resnet50PTBatch8_module_3b0ae1403ef444d812f0c7b37fda7311e2cc4ea407850ee7b91e6984b9c86100/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "Resnet50PTBatch8(linalg) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-25ad2815eb690276e9c2183aaafaf17a3df734bb6164071ad92dbf1e7faf7509"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNetV2STF_module_25ad2815eb690276e9c2183aaafaf17a3df734bb6164071ad92dbf1e7faf7509/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNetV2STF(tf_v2) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-65586f1e5b51439dd951529c35fa9000a928f90039cc6cfb66d5c81d07a6c62b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MiniLML12H384Uncased_module_65586f1e5b51439dd951529c35fa9000a928f90039cc6cfb66d5c81d07a6c62b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MiniLML12H384Uncased(tf_v2) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-16b5b80aaf1271b5ad782570340cc0c7c1c97e10b7e6c6cc6e5f3ede8393cb6c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_16b5b80aaf1271b5ad782570340cc0c7c1c97e10b7e6c6cc6e5f3ede8393cb6c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-65fa033050b916e8143d44b5081ee45db3b1946a5d77de223328a7fe92a1cc66"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_65fa033050b916e8143d44b5081ee45db3b1946a5d77de223328a7fe92a1cc66/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-16ef56b6869d10b17e983fec62e9f48e6bb87e9a348ab52a0b2faabca2b03578"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_module_16ef56b6869d10b17e983fec62e9f48e6bb87e9a348ab52a0b2faabca2b03578/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV1_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-56bc9128e294585d749b8ebe34fd03060ba34d200eef185837b6002d0dcbfccb"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_56bc9128e294585d749b8ebe34fd03060ba34d200eef185837b6002d0dcbfccb/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-eb1b1732e5d30ce4689b871f8ec18c50b30eedd418eb80330807fe505bb78f7e"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_eb1b1732e5d30ce4689b871f8ec18c50b30eedd418eb80330807fe505bb78f7e/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-bd32992875a8fc7a494c75933b1693d6d8b845fccf2b12504a8cba64d80ad110"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_bd32992875a8fc7a494c75933b1693d6d8b845fccf2b12504a8cba64d80ad110/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-5e966f45b952e5406271a1a23edd5d9ffab75524b450b7cf4ee35263bb0830f3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_module_5e966f45b952e5406271a1a23edd5d9ffab75524b450b7cf4ee35263bb0830f3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-ff7ed59e05efe8b9a397a179726f63da68a8a1ac3ea731924b4bd24dab491b34"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_ff7ed59e05efe8b9a397a179726f63da68a8a1ac3ea731924b4bd24dab491b34/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-8e2d1286ad9a7e360b0c26019146a22ec9188f8bdf8ad99341eb5531cdea2417"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_8e2d1286ad9a7e360b0c26019146a22ec9188f8bdf8ad99341eb5531cdea2417/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-d967e293594998e48355d30d900dbdf77dbd6eedbff768112dbe8e7ec332b9eb"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_d967e293594998e48355d30d900dbdf77dbd6eedbff768112dbe8e7ec332b9eb/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-94a5cba0f45ffc2dc028a5f9fa5780fa61ed91b27382c86c4d96de0e2cd002f3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_module_94a5cba0f45ffc2dc028a5f9fa5780fa61ed91b27382c86c4d96de0e2cd002f3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-91a35228ead480e04b85998ccf3edfc891f44b5f79017b7fcab72cb66a812b07"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_91a35228ead480e04b85998ccf3edfc891f44b5f79017b7fcab72cb66a812b07/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-f58c00ccab797ad4dbca3de3b50633588a68db0122aa011bdf81a9aca5ea692b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_f58c00ccab797ad4dbca3de3b50633588a68db0122aa011bdf81a9aca5ea692b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-bfb6239769f044d2228f2efb5ce6aa51132455d9a8178e5a5ec8525ff5836e0d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_bfb6239769f044d2228f2efb5ce6aa51132455d9a8178e5a5ec8525ff5836e0d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-469056c2ca5935d7c63d5424c635a439f94593a307e96483e4db16af1c15186e"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_469056c2ca5935d7c63d5424c635a439f94593a307e96483e4db16af1c15186e/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3aab34d7c719c9d828a741a7900b4794302a587927c462b4ec8feec3f7d43e99"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_3aab34d7c719c9d828a741a7900b4794302a587927c462b4ec8feec3f7d43e99/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-11a9de4ea6e17feff81429ed53e52a70e89c1cfeef0a73f10740c8420341b81d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_11a9de4ea6e17feff81429ed53e52a70e89c1cfeef0a73f10740c8420341b81d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9c01136785f28f0d2c969cee8ce87bde3267d63425c5d86d39137abdf7f0f196"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_9c01136785f28f0d2c969cee8ce87bde3267d63425c5d86d39137abdf7f0f196/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3160a297a2c9d3d21caeec097b6fe19150c3feae5fa872e21817af0be8a8176a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_3160a297a2c9d3d21caeec097b6fe19150c3feae5fa872e21817af0be8a8176a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-0bf641c301b26975b8919a18de98d9dfd6444d6542085dd2d8e8155ea6bc8efe"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_0bf641c301b26975b8919a18de98d9dfd6444d6542085dd2d8e8155ea6bc8efe/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-058ea3aae7385269d001efd9eb2303887614d138ff69150b20a703fc7b97c2c6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_058ea3aae7385269d001efd9eb2303887614d138ff69150b20a703fc7b97c2c6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-fdff4caa105318036534bd28b76a6fe34e6e2412752c1a000f50fafe7f01ef07"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_fdff4caa105318036534bd28b76a6fe34e6e2412752c1a000f50fafe7f01ef07/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-14ce4459cb4ea8aa84b5315222e9cfe00fe8a3b456b2ae75a5eb943036279d68"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_14ce4459cb4ea8aa84b5315222e9cfe00fe8a3b456b2ae75a5eb943036279d68/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-0b2b90bac148aa9f7c2ee34db723a002823dbc0d5981e47511f09cafa3693101"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_0b2b90bac148aa9f7c2ee34db723a002823dbc0d5981e47511f09cafa3693101/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-bd015dc23ff2f9bf5d681039cbb0f6418cd3d09d09124c0238d8c2caf01dba24"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_bd015dc23ff2f9bf5d681039cbb0f6418cd3d09d09124c0238d8c2caf01dba24/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,mmt4d,dotprod,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e02e0460e54ee222b46c25e876f937eed5582b0823cad1b1d009fe406b160c33"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_e02e0460e54ee222b46c25e876f937eed5582b0823cad1b1d009fe406b160c33/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-d6590e27e94d8aac1b2bfb1e7c60e31dcddacd3a10687cdae998979fc31720fc"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_d6590e27e94d8aac1b2bfb1e7c60e31dcddacd3a10687cdae998979fc31720fc/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-50567a33e0bd9aa5a32a6f61fca9ef8a70ac4d94313024f2c4ec92d9c543c599"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_50567a33e0bd9aa5a32a6f61fca9ef8a70ac4d94313024f2c4ec92d9c543c599/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-6b9353f591f5044f661ecbbaafee502d710cf263527525d4f843b26cd43f11f7"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_6b9353f591f5044f661ecbbaafee502d710cf263527525d4f843b26cd43f11f7/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-5f03fee30980d1fb1074b82d329a1fa63b365858539743e672ad56c039dd732a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_5f03fee30980d1fb1074b82d329a1fa63b365858539743e672ad56c039dd732a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-611a54141f98b17aa94abdba55d8a0487aa722bba4da6853c090f786975c5884"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_611a54141f98b17aa94abdba55d8a0487aa722bba4da6853c090f786975c5884/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-0d524f6ce80da6d1998eb8978623a2f6efd413e0b973c6f2dddf52a46b19907e"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_0d524f6ce80da6d1998eb8978623a2f6efd413e0b973c6f2dddf52a46b19907e/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-dd2a6a43dceabe7a807e280b43177cdf892d4ad20fdef4e3d3b6e39be7b9b09d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_dd2a6a43dceabe7a807e280b43177cdf892d4ad20fdef4e3d3b6e39be7b9b09d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-823ec09bcc061f124fa9229747054945cedf352e11d661a68785cb26af5f83b6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_823ec09bcc061f124fa9229747054945cedf352e11d661a68785cb26af5f83b6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-899c2de5e339b7e19528e80de1129a38511948ba3331932c22e23223707af4ca"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_899c2de5e339b7e19528e80de1129a38511948ba3331932c22e23223707af4ca/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c3cb44c1331872dc53919d0d8b2cab4c256dcdf8b0038f9b6a692a5c874f855b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_c3cb44c1331872dc53919d0d8b2cab4c256dcdf8b0038f9b6a692a5c874f855b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-dc81a08fe0e5140f22f328d9dfea1828e7318d67899a2534d3b02ff36032cb61"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_dc81a08fe0e5140f22f328d9dfea1828e7318d67899a2534d3b02ff36032cb61/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-1171fb017e88de21814d71ea2d35564de6904d3d2359ef53e0fa2c67ea6e9914"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_1171fb017e88de21814d71ea2d35564de6904d3d2359ef53e0fa2c67ea6e9914/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-9b9a47b0a97a0bd002bd7fd1f104caaa94b8bf60cf02ffcc2b50129679e4c6f3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_9b9a47b0a97a0bd002bd7fd1f104caaa94b8bf60cf02ffcc2b50129679e4c6f3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-4f5ab4bfb26a82d0f83133b9e85585f0c5b97cdb00143de31675158a5a71b457"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_4f5ab4bfb26a82d0f83133b9e85585f0c5b97cdb00143de31675158a5a71b457/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-480b59f233af720e16db8e5da1988a8d69bd61169bf5b5899f425ff98dc0dc19"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_480b59f233af720e16db8e5da1988a8d69bd61169bf5b5899f425ff98dc0dc19/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [adreno-generic-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7731488e1eb90da5480e76b4cd98b12c16b83d7c7011b0aa9ef3a5d6d2059a3c"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_7731488e1eb90da5480e76b4cd98b12c16b83d7c7011b0aa9ef3a5d6d2059a3c/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-61decb77e61b184a2c353fac3d60af1cd7c73abc867c23e9519f5e398265a728"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_61decb77e61b184a2c353fac3d60af1cd7c73abc867c23e9519f5e398265a728/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-4ec47dd2b4a43dd434d041d4d9db548076b70cfd63a2fec2971035394954f1d5"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_4ec47dd2b4a43dd434d041d4d9db548076b70cfd63a2fec2971035394954f1d5/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-b0309994482c31c79242ee8ef3902b4cc54c1479688824734b33d2f554d6aff6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_b0309994482c31c79242ee8ef3902b4cc54c1479688824734b33d2f554d6aff6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-113994770711e5784a73ac623cbde328267c94b6341e001328b053c02b8bc08f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_113994770711e5784a73ac623cbde328267c94b6341e001328b053c02b8bc08f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-954bc3dc1fd0c22768ebfe898a67c0db3743d74e8fb776fced75eafb0421058f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_954bc3dc1fd0c22768ebfe898a67c0db3743d74e8fb776fced75eafb0421058f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-6bb61b9c7107a9a30ad20c154321e7e9b14aefc407a8aeda41ac6df5eac757c4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_6bb61b9c7107a9a30ad20c154321e7e9b14aefc407a8aeda41ac6df5eac757c4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,demote-f32-to-f16,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-2a38d4c2e9ce2c3f9a78f5dce9e889145a1d4ec821d5776f15d6d6e679901474"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_2a38d4c2e9ce2c3f9a78f5dce9e889145a1d4ec821d5776f15d6d6e679901474/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-660bd6b0861f368e51c6ecac52d6dce2998d863bc0a4bbd9dbdd77508ea761d4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_660bd6b0861f368e51c6ecac52d6dce2998d863bc0a4bbd9dbdd77508ea761d4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-224bd124c8446c300caa77db186ca926e71e79cf187980db5dea593d6f29d434"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_224bd124c8446c300caa77db186ca926e71e79cf187980db5dea593d6f29d434/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7bec578c7016cb7e017057c227a9b677901d14c0fff35e31c4a5cf12692db105"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_7bec578c7016cb7e017057c227a9b677901d14c0fff35e31c4a5cf12692db105/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-89a91c770dfce869ecb04e4b37e3b4d7da508a240da395cf240cc20ee8573857"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_89a91c770dfce869ecb04e4b37e3b4d7da508a240da395cf240cc20ee8573857/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e6049d40d7925bccce4859e5408f2ad53eb68309aa38b46b8a7e47c94a2cd8a3"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_e6049d40d7925bccce4859e5408f2ad53eb68309aa38b46b8a7e47c94a2cd8a3/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-a4194c053541ebc49b4912bbdf3ca211331fdca5d157440837144e59d279bf1f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_a4194c053541ebc49b4912bbdf3ca211331fdca5d157440837144e59d279bf1f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-599701d7114956cf64777412899cff57ea5be0478f9a2331377770beaad8f923"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_599701d7114956cf64777412899cff57ea5be0478f9a2331377770beaad8f923/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-1d26fcfdb7387659356dd99ce7e10907c8560b0925ad839334b0a6155d25167a"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_1d26fcfdb7387659356dd99ce7e10907c8560b0925ad839334b0a6155d25167a/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-b74ccbdce4ec07bb65313ee96b67c1b946a6c959158714706209a9df2b93ab0d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_b74ccbdce4ec07bb65313ee96b67c1b946a6c959158714706209a9df2b93ab0d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,demote-f32-to-f16,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-b8ee6e3d12e6662f0c78b851876a0759da23967e71ae1a2b5569d0ec3101b43b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_b8ee6e3d12e6662f0c78b851876a0759da23967e71ae1a2b5569d0ec3101b43b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-96e2ebf26260627484f375448ef478c35608ada2d2a0f81c0bec697db9ea3105"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_96e2ebf26260627484f375448ef478c35608ada2d2a0f81c0bec697db9ea3105/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-0ee6c4e8f636f98e5443974526d95e057d5a62bb311bb6dbadd87d50b395a373"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_0ee6c4e8f636f98e5443974526d95e057d5a62bb311bb6dbadd87d50b395a373/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-178907b155b6322dedfa947937f9caca5158ff3af167470f2de90347dba357f4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_module_178907b155b6322dedfa947937f9caca5158ff3af167470f2de90347dba357f4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "DeepLabV3_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-7dcabfd6caa769a75657e07e7315dd42f52b3d4cbc37d75098aca446d3ff4066"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_module_7dcabfd6caa769a75657e07e7315dd42f52b3d4cbc37d75098aca446d3ff4066/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileSSD_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-247b38beca9631678d80755b0b4db2b323ddc4d95772617889a6a4bb813c6b74"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_module_247b38beca9631678d80755b0b4db2b323ddc4d95772617889a6a4bb813c6b74/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PoseNet_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-d39340f50384e970b103694a38d7d21d5b1171d7304630d25e925c5c2486bf10"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_module_d39340f50384e970b103694a38d7d21d5b1171d7304630d25e925c5c2486bf10/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-d8f22b5a700abdef68fe791ad08acdfc6d238d82e00f264367d922b99b369ff7"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_d8f22b5a700abdef68fe791ad08acdfc6d238d82e00f264367d922b99b369ff7/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-c6a4903d1769d721782cf2b6e84837aca21f87fcf8759912a86ae2f572e8440d"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_c6a4903d1769d721782cf2b6e84837aca21f87fcf8759912a86ae2f572e8440d/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-3c43472d6cb0f74a1c08920e3f580b701e995a85305fd4b2e370542b4d449b18"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_module_3c43472d6cb0f74a1c08920e3f580b701e995a85305fd4b2e370542b4d449b18/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_fp16(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,demote-f32-to-f16,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-6e207112c3da58908537a07d168c78e7d166fe6803659d4b9f07848c968d6d12"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_module_6e207112c3da58908537a07d168c78e7d166fe6803659d4b9f07848c968d6d12/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileBertSquad_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-75259bfdbf7eb331691860ffb18e04a146168a72b7b10cf070d5db7f67dd2378"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_module_75259bfdbf7eb331691860ffb18e04a146168a72b7b10cf070d5db7f67dd2378/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "EfficientNet_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-5ef6657fb694df545b7f87fbb78dfa99af891778790ac0924c08a87d335c1bf9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_module_5ef6657fb694df545b7f87fbb78dfa99af891778790ac0924c08a87d335c1bf9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "PersonDetect_int8(tflite) [valhall-mali-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-02b72f9538e4dfc9c789e63d722d5eab4333f3f55f8375503f433a790da119cc"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_module_02b72f9538e4dfc9c789e63d722d5eab4333f3f55f8375503f433a790da119cc/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV2_fp32(tflite) [vmvx-generic-vmvx-vmvx][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME
    "iree-module-e7bd41e564750501f39ac9690c18d1a2e77dc7999da710d0c0bf80751dda84a0"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_module_e7bd41e564750501f39ac9690c18d1a2e77dc7999da710d0c0bf80751dda84a0/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
  FRIENDLY_NAME
    "MobileNetV3Small_fp32(tflite) [vmvx-generic-vmvx-vmvx][default-flags,compile-stats]"
  PUBLIC
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-05c50f54ffea1fce722d07588e7de026ce10324eccc5d83d1eac2c5a9f5d639d
  ${PACKAGE_NAME}_iree-imported-model-3f492fde47abecd3640f1e04c14f2bfc24940f1cf11f66e72128c590bc711025
  ${PACKAGE_NAME}_iree-imported-model-4afb465f351db6dbf3d06a03d98f0e77c9cc1e284d89f6db6367969793e44a59
  ${PACKAGE_NAME}_iree-imported-model-af95be67ed750381753ca4fe4341853e7e2884d4c4bd1dd9a17d8ceab3ee8882
  ${PACKAGE_NAME}_iree-imported-model-3bda9f3a5eb6a0fd3adc80187495d7ab840e409f379c70e3fd687934fafdd3b6
  ${PACKAGE_NAME}_iree-imported-model-a359c8f1e19b6f476e843bdc5566f9554e329cbfb3b4437995ad9cccfb381aee
  ${PACKAGE_NAME}_iree-imported-model-a56388db344366834963ce4295c7695bd3f75b6840962c0b4aec857f34575ded
  ${PACKAGE_NAME}_iree-imported-model-7f3f1c653a612b1a7c13af9e9e6a3b0d0d12ae159e5e4e619d00717b0686bc9a
  ${PACKAGE_NAME}_iree-imported-model-394878992fb35f2ed531b7f0442c05bde693346932f049cbb3614e06b3c82337
  ${PACKAGE_NAME}_iree-imported-model-2b9d79769b2b1ece7dd4a4bccd146a99fc3c38c4427a179d37308146adfc5c0e
  ${PACKAGE_NAME}_iree-imported-model-93c0f75188363af77647cb2f1deb41446575c9cd084d86119287156eb181d850
  ${PACKAGE_NAME}_iree-imported-model-3ea5b376aec708e6c6827b0a9da7135fc50f20400dc0d55f16a3378a08fa5cf4
  ${PACKAGE_NAME}_iree-imported-model-213fe9a8738a01f2b02b6f0614a40a31c83a2603ca3e3ae0aeab8090fedbe3a0
  ${PACKAGE_NAME}_iree-imported-model-d4a10c6d3e8a11d808baf398822ea8b61be07673517ff9be30fbe199b7fdd960
  ${PACKAGE_NAME}_model-cc474102-7d2f-4ec1-92ae-84e83ba0f390
  ${PACKAGE_NAME}_iree-imported-model-a3a701aaac95a47e7e0c1875793fbe88c976864cac611ccdf7d373d43d670225
  ${PACKAGE_NAME}_iree-imported-model-2494ed4b5c065c4a78b03d46161d4c9cccef27edf9568170c7dd2158281fe697
  ${PACKAGE_NAME}_iree-imported-model-a122dabcac56c201a4c98d3474265f15adba14bff88353f421b1a11cadcdea1f
  ${PACKAGE_NAME}_model-9a9515c7-cb68-4c34-b1d2-0e8c0a3620b8
  ${PACKAGE_NAME}_model-340553d1-e6fe-41b6-b2c7-687c74ccec56
  ${PACKAGE_NAME}_model-68caa96e-b8bb-48a2-bb08-a3044981a370
  ${PACKAGE_NAME}_model-cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-1
  ${PACKAGE_NAME}_model-cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-16
  ${PACKAGE_NAME}_model-cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-24
  ${PACKAGE_NAME}_model-cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-32
  ${PACKAGE_NAME}_model-cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-48
  ${PACKAGE_NAME}_model-cbc5e400-7c93-4844-aca8-bce8f1bf9948-batch-64
  ${PACKAGE_NAME}_model-fd05da43-5e37-4fa0-88f8-3ceec1682345-batch-1
  ${PACKAGE_NAME}_model-fd05da43-5e37-4fa0-88f8-3ceec1682345-batch-8
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-87aead729018ce5f114501cecefb6315086eb2a21ae1b30984b1794f619871c6
  ${PACKAGE_NAME}_iree-module-3926415c1504dfc277674fee17bdfbd68090634b8b52620d8d5755082a89a16d
  ${PACKAGE_NAME}_iree-module-472e6a18344f13d47e89f87670b37eff583ae610c2a1d15ac1cca307ccfc2f4d
  ${PACKAGE_NAME}_iree-module-95dee09d7f3f9ee36d6c70645585b44b347ea001a1ab9a04b150ca2dc052255f
  ${PACKAGE_NAME}_iree-module-78154d58dddac432100d656b22fa9bcb45e4207a9ea2bc371bf089a68bad397a
  ${PACKAGE_NAME}_iree-module-02cebfbec13685725c5b3c805c6c620ea3f885027bfbb14d17425798e391486f
  ${PACKAGE_NAME}_iree-module-429d7055b93250cc19866107e305bef1dc0b7e49e579ef19ea1eeb70eb2fb86d
  ${PACKAGE_NAME}_iree-module-50084c6a5ef433943ae46fbbf852d266d20557be110fff9dea1e4e298040c7c6
  ${PACKAGE_NAME}_iree-module-baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49
  ${PACKAGE_NAME}_iree-module-737d273c18eb7537e2dde47c4a46391e8657c38f9650032c39bc67fa5f132742
  ${PACKAGE_NAME}_iree-module-eb56e91246a131fa41bd335c1c072ffb6e7ffe651ecf65f4eeb171b12848b0ed
  ${PACKAGE_NAME}_iree-module-92dd923f493f67509a6b54007416f16ac8e6f2023e88f79b3017ea2260ee561a
  ${PACKAGE_NAME}_iree-module-c8949024e2472bec7b18c4e3757412715c248273005ca6f8d5769656ed425a84
  ${PACKAGE_NAME}_iree-module-a30b64a3d7850881ee9db94e8f75c661af3f76f48d10b3342a6912e1c8879252
  ${PACKAGE_NAME}_iree-module-047e75c462648b5fe1133f4ffbc3d1c7bdda154081d3eaa3be0b5445725b272b
  ${PACKAGE_NAME}_iree-module-1c7402f88ba881ec6abb39204faa4b5fedb2ffff4a6066555fcff0c7c4b74732
  ${PACKAGE_NAME}_iree-module-9c849d0ccfc89c0bca0740949572db8735832012a43c4c9f15c3a8ef0d9cca04
  ${PACKAGE_NAME}_iree-module-7a0add4835462bc66025022cdb6e87569da79cf103825a809863b8bd57a49055
  ${PACKAGE_NAME}_iree-module-2e9ba5db29e643ec5b7c3fb5a2e3cd4a8fd5ab2c13d66af4bd938d17f5f5c682
  ${PACKAGE_NAME}_iree-module-4f70b75b4c1d77495814987264c5df3fc81a87b49574065458be603578f6b00d
  ${PACKAGE_NAME}_iree-module-fd5cf4cd75cc1734be3eec2f4396491e9ebb7edd6b9c889cdc29ef72b47d6233
  ${PACKAGE_NAME}_iree-module-3ba167fc59b5959be276b331951c759391d6e158572fc05603981dcc4bd3fc90
  ${PACKAGE_NAME}_iree-module-531f99d8bc669343f967598c13ca787c62b01b4dfcd7c4b4ad04a163063a1ddc
  ${PACKAGE_NAME}_iree-module-c30da024a2f99f8a88e5a4d40637c2a00dd754223c532202e10f524e6b28089b
  ${PACKAGE_NAME}_iree-module-8d15dde1c8c2ed90009698c357455dfc94fd96ae877f4892954bef0ec4361de8
  ${PACKAGE_NAME}_iree-module-9a2128d69c5c5a51402e01c4a848e90ec369fa601a3d9fc1ab69dab8db47e6d3
  ${PACKAGE_NAME}_iree-module-dc6bba0cdf2a0368e020384e72a19dfbceded8021e38aa0320cef78025c72933
  ${PACKAGE_NAME}_iree-module-7e0380f1df059cf6040c12934af4f1c88a469c716a294f383200d5dd5cc69b1d
  ${PACKAGE_NAME}_iree-module-759fdb2deb885ca5568741853e3497ab3d7037a42a4fdff0af590c803681ec60
  ${PACKAGE_NAME}_iree-module-c37bfe2bb995b2703170a890bd81e372b687bed57087b1d8a6d8bb16b91c5ad4
  ${PACKAGE_NAME}_iree-module-3dbf200159f328bb69c6c5bc79cce408a4ba49d2d07dfb3939786557e63d035c
  ${PACKAGE_NAME}_iree-module-711cb4e615cc4e032fe9f198b89e32f4f85c94cab6e9101eb8202b22c97a37b1
  ${PACKAGE_NAME}_iree-module-4b61f532e6cf4e175a7e90cd9418e3c2614176b5253875951f99e52d2621c152
  ${PACKAGE_NAME}_iree-module-aed1ca2855056bcd5b7e51063741685b4a387d9d0574b343e8ddc383b49afc76
  ${PACKAGE_NAME}_iree-module-80a2368e148d9605d98060027b9198dea46efbf050a383784ec5df5e85904757
  ${PACKAGE_NAME}_iree-module-9645062212c891963bc5ee32750ebdd3d3485354c1bc7bff0f584ad967cd7a38
  ${PACKAGE_NAME}_iree-module-f146632eb124afeb899eeae8aaf5ab6cd9efae22ee9ffb26d34fe1da10049fbe
  ${PACKAGE_NAME}_iree-module-abe8d72fb1ef1302b96470e94c129843ebf76fdde715860b32cf73a24837fd21
  ${PACKAGE_NAME}_iree-module-d3a776d4bb360ebce4148472f14fa242dd57baa59621172abece40b611497ef7
  ${PACKAGE_NAME}_iree-module-bdd904cc5614ebf77609c7802a2dfc09f139aee2a247a247d10d320de72b0e28
  ${PACKAGE_NAME}_iree-module-45565cae821666fd34bca97be2e4cce3bd61e71308785728737d89acbb9bc9d2
  ${PACKAGE_NAME}_iree-module-fd81a89e9f8773bae142040775c7e3c4774f96b64f07f8d9f66b00191864ff40
  ${PACKAGE_NAME}_iree-module-c4b43b31944dbd567e48efacfdb33f707eb248538cf70fa7dcf1085c6c7dbd3f
  ${PACKAGE_NAME}_iree-module-c8ec2db5ee884e0af17814e61b13d7f7f1f2d4f7028e8c1920d0d968c27de2bb
  ${PACKAGE_NAME}_iree-module-9470c46965ea67794da45496454c82eade29b5a519d8037b1314738621e02260
  ${PACKAGE_NAME}_iree-module-154de838dc7742304c9e27a9f315645a915493bc4f84160e29b15b7fc6dc475e
  ${PACKAGE_NAME}_iree-module-21a83414dc1feb2a3b5fb6afadc36c1022a0ab747380291dbb309637d5f32eab
  ${PACKAGE_NAME}_iree-module-7d93a3e9b342045ed2546960ee3d08f60be237fd8974fe35447a41234f322148
  ${PACKAGE_NAME}_iree-module-995595b7b80370d9c484413e9d06c2de928db32777101c91d3bac0d3797058ec
  ${PACKAGE_NAME}_iree-module-45ed9ca2efb9bb645316e856a5c464b76e56902e930ff1d4e54fdef38043b33f
  ${PACKAGE_NAME}_iree-module-8c26886533593b666597f8698e33c0e9c98b349bdf8ac01c3d122cc20b741def
  ${PACKAGE_NAME}_iree-module-222490941c581c67f5ce710e9ea141482cd8074294a43a5dc67a01a127037cd4
  ${PACKAGE_NAME}_iree-module-39556c12d84502be71243197b99b0f8c22949093c561c55c01ca906812a86288
  ${PACKAGE_NAME}_iree-module-04ca0a5077b7dd5ace66d803c9b822dff3428b24e7620a61995aff0907af9533
  ${PACKAGE_NAME}_iree-module-deafafd0926321a4b8e4dc73ed4a30b2ed9317d26488246461415be2ee857eb1
  ${PACKAGE_NAME}_iree-module-68f0eb37bb72d0d6605ecdf42691c64125960e122844b0beeae350871a445b1c
  ${PACKAGE_NAME}_iree-module-a7a1553d0739151f06bbc00a3ef8b67b0606463eab4b6607069aa94ea0bfd92f
  ${PACKAGE_NAME}_iree-module-e80d71ed8e86c0756226b2323e27e2c7c0fff8eddde59ba69e9222d36ee3eef6
  ${PACKAGE_NAME}_iree-module-f59cd43a2a2d6e4b3159efa358a6fa0879e72f6f4f0a23af4c8ab550f256986a
  ${PACKAGE_NAME}_iree-module-14a15b9072caaee5e2a274a9bbc436a56d095611e5a8e9841f110741d34231f9
  ${PACKAGE_NAME}_iree-module-e850fce2d36ddb09ccc34471641adb77418b93c0949d22ab75806d7cfc489ae3
  ${PACKAGE_NAME}_iree-module-93034e879f88475eeeff56dd38297a1d1183f123fbbf5925a07b7c934e5f5d48
  ${PACKAGE_NAME}_iree-module-f963d812114af925e0a4b110ee83aeb0e3b41d49fad19b3f449b6a9ccba43b8d
  ${PACKAGE_NAME}_iree-module-9d909aa679e8c380ff7e93292ef28dbd3bb9e7cc62329f90aef78ce9c7efeff9
  ${PACKAGE_NAME}_iree-module-1ef2da238443010024d69ceb6fe6ab6fa8cf5f4ce7d424dace3a572592043e70
  ${PACKAGE_NAME}_iree-module-d399d2540161867967b0590a659d0a86206e145d884bd7afd9abd818713aacd2
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
  ${PACKAGE_NAME}_iree-module-894671e6d7319df243fa8b386b14d5e648fe2ffd66cb95e5a82ad58d7273ada8
  ${PACKAGE_NAME}_iree-module-0af4e7fa84d361d2faa267336c3c090e337f7feff47f0e559337521c07626ba7
  ${PACKAGE_NAME}_iree-module-138aacc154449b519698bd7ddb4dc9f407cbc1fd86fdbbe901be974058b70319
  ${PACKAGE_NAME}_iree-module-a47669d662bd18dbd323737d84fd45753c01abaa81b4761eb51b35ce04ee7491
  ${PACKAGE_NAME}_iree-module-27c12f520d68a8a754b11ad44cf8ad3ef5c1ec281fc9d84fbc64a7b573cb68b5
  ${PACKAGE_NAME}_iree-module-ec14aafacf15628918390531318a4827d3ac8771b75d60a8239d901dcb4fd898
  ${PACKAGE_NAME}_iree-module-507fabae3d24c2448c8f2af676d68df0a22716a06fa7ffc3b4f865e9272ecdc8
  ${PACKAGE_NAME}_iree-module-4924918e121a92f802a094de2d67e9c2673c9fdc39faa6a11ac1d38b631a2914
  ${PACKAGE_NAME}_iree-module-383e7567e79adf2178e3d25785855905a4477b6c8daf16bc50d5f9f44d2343d9
  ${PACKAGE_NAME}_iree-module-9faad3c50bc64c297531a36bd2ab235b680070d8233ce5ab7d964ac36c7c5563
  ${PACKAGE_NAME}_iree-module-90eb882b36d0293eb5b22ffb64f7c395cefa0360b37d4b84854d8b2c52a76358
  ${PACKAGE_NAME}_iree-module-e1143fcb8d1d262a3b2ac689cf21a1943cc59b2de06192fef185a9bf84994b86
  ${PACKAGE_NAME}_iree-module-65aa6dabd0c24dbe48a090db641d82b437d4abd428a6ca0e7b360d3f087a16c8
  ${PACKAGE_NAME}_iree-module-131aa61ffcacafc6ae701ff89045cf42d2490ac0c4a1d862bc83c23edb3b92e5
  ${PACKAGE_NAME}_iree-module-db9784471b47ae8bf55ca7e0821e35a1686256a208df40443c114f1adcdd26f6
  ${PACKAGE_NAME}_iree-module-aada5dcefdd361b5227276129e93547a6932a05d380acd342fa33e5b672d498c
  ${PACKAGE_NAME}_iree-module-887c7a7b540f11ee5e0158143fd46a503a4851211b10b353ec0e00b8b1beb575
  ${PACKAGE_NAME}_iree-module-11766f32ea6a3121d7527bcdf32dead45ab7b3922d72addb46945cfdab784ec0
  ${PACKAGE_NAME}_iree-module-0efb794c3a385045fa4d3d086c2a593ce67c4807e9456271f05f4e28490d1c49
  ${PACKAGE_NAME}_iree-module-7a3d36d7234ce4abfd833964492631bd81df823e6bde8cd3a1fadfa4faa7c787
  ${PACKAGE_NAME}_iree-module-e52ef218d6b5b16f35c420fca42ff31c5cf733bc075ed235ce57cf4662824606
  ${PACKAGE_NAME}_iree-module-1202e44fbbde0a9581c641d124452363273f0d4395445dc06e60c8e49a27357b
  ${PACKAGE_NAME}_iree-module-3c1fbaa12bbd37bf2a17d8a605cdeca75c0e8cc9c6f7e95359d13e47a4270dc3
  ${PACKAGE_NAME}_iree-module-86ce8dfe2979d777a7f9eb0d3b6f8dcd4b594f46e9d610ad8a73edc89a006b26
  ${PACKAGE_NAME}_iree-module-bd0ea10065a27dea6875ceb70f769e7e0f67a08e857920ca0d0322593441e9ed
)

add_dependencies(iree-e2e-compile-stats-suites
  ${PACKAGE_NAME}_iree-module-2f2e448f73ef190ed35af1b25b6179ce15faba7ee7c12f4956730c441e9a27bd
  ${PACKAGE_NAME}_iree-module-c370b55d34f6d3c76aa838ff0a7be520de10a4824c5feaa773e2fb73a588ad8c
  ${PACKAGE_NAME}_iree-module-5a4c96fc279262ad7d7f1d446d0bd3685b2ca42e06b0167df5be5737c9d42901
  ${PACKAGE_NAME}_iree-module-27bbe62536a23529b4dd0df3d4913ee18344df9b6e2a32fc834fb7d9bc520e24
  ${PACKAGE_NAME}_iree-module-78511a42a50f705b944437a040e1ee3bb5b2595a3b1d4db788586fe48f9a2453
  ${PACKAGE_NAME}_iree-module-ef1ba1216f0f304c80b7a5b8bac545a987d04a100d9c1e5e66b75ce88636534c
  ${PACKAGE_NAME}_iree-module-1c4bc4b5ba3b5862efdbcbb9b3bf4a02f7ff9aa36e852e9b94dbe265d6bfaa99
  ${PACKAGE_NAME}_iree-module-2cb62cec021aeb2abd869ba5276996e362bae130db7cc2d79389cccee6e372b9
  ${PACKAGE_NAME}_iree-module-439f7c958ce1d3200ea96935174cabde8e8fe6917a007f5e238553e9c2aa7625
  ${PACKAGE_NAME}_iree-module-4a3b570ba18c3c9eee458455aaff4aa29293a5c936a19862c698b4b3ddaf06e7
  ${PACKAGE_NAME}_iree-module-28e38bd436b036babc0fabe98b6e7c68ca3a7088e73dffff2c538adfa7d6af4c
  ${PACKAGE_NAME}_iree-module-a05a2b521a968e99411712e0e5191c3cd1d6295991f3b78acf61faca5d1cf85e
  ${PACKAGE_NAME}_iree-module-01d35de2a55b9800e05151455eace0bf4493337ac1210fcc4904d630b075599a
  ${PACKAGE_NAME}_iree-module-2957930127e9b01e90ccddb7290e1c4b4abf6373cc36929809040e2c144d3fd7
  ${PACKAGE_NAME}_iree-module-b5078b9d2031b69ec5ce9b775c8701cef73add8ebfb786d9189ca3fb6474cf73
  ${PACKAGE_NAME}_iree-module-ddd1657bc5433ccca5c8ce562f581626457a793670958cd8b4016c426191a9c4
  ${PACKAGE_NAME}_iree-module-8ee3c7b136703472b53bc8a19d8d28945aca93953612ccc65e55cd1b3dfda6c8
  ${PACKAGE_NAME}_iree-module-846b19afd4c14b3e71d59087c5a2987edd65753d39db432961ce915688d457ac
  ${PACKAGE_NAME}_iree-module-f13d9b8b55decd1684c262d230ad42c95c5def21d9ee605ebc656181d1c54a0a
  ${PACKAGE_NAME}_iree-module-48b14091499ede62cbd8622a11fff448d9a0354662ad87fae832256fc1610f13
  ${PACKAGE_NAME}_iree-module-63a2a9acdb2dafb66a49487fd7eecb79e49d870ec9b9ef17a1e8e9d50fe5c9ba
  ${PACKAGE_NAME}_iree-module-de34105293194986d706823bd3d20ce784506ec5918c4d0efac9839020bb5fdd
  ${PACKAGE_NAME}_iree-module-373b890bed4c0f4828b957e37d319509bf41e39a4e47746285e27101d40f90bd
  ${PACKAGE_NAME}_iree-module-6e31f637a133e03db37c47d8a920a61306e366362e066f41c0eac0455cc6c77a
  ${PACKAGE_NAME}_iree-module-e0533bdae79e15707a6eb26eb7f09c4d7dbdbfc40b993a4ad6289cf2bb1f13cb
  ${PACKAGE_NAME}_iree-module-ad9a410e86dd9d649de58f5a7dbdc6cd2300fb6b6a363f4483e930d9944d2d07
  ${PACKAGE_NAME}_iree-module-9b12e389535e365bd2c35424c5f98442e1226d73b043eb40355bf456ad0263a2
  ${PACKAGE_NAME}_iree-module-63d75ff4a9998a86855e0e78ab2d782f52b90b58025584f3f03ec3103a81425b
  ${PACKAGE_NAME}_iree-module-fdce9dd1dfd20592880fce8969f91ac6abe84eca6922b4df4cbffe512abbfcb6
  ${PACKAGE_NAME}_iree-module-00a22e8ada401de8f20895beff9a153585e585c2d686983e27f9d64fdf7d39a8
  ${PACKAGE_NAME}_iree-module-4c74339076df00d23baa17dcb3194043e0472da9d09db4e42a23841ff7bf67b0
  ${PACKAGE_NAME}_iree-module-9a1d228583ba1e56a19393f6938d16b5d582bb17f89fb5856b8b1c68e34abd45
  ${PACKAGE_NAME}_iree-module-152d0b6211fff7591df3418c549c979a8144fc34280c22a8b2b5ff8ea3d1b46c
  ${PACKAGE_NAME}_iree-module-c9a7c5b08db10ed782045b6810cb4ee157da9e95590456d3839c06163ee30fa7
  ${PACKAGE_NAME}_iree-module-838cc09b422958a332fd76cf12a6a2a95b8346c8e8d2fe7b15cb5ace4c20581e
  ${PACKAGE_NAME}_iree-module-e16d3f99f851c11fef6be64c7f06a637b410618f2618cf16aa599b54ea8970e3
  ${PACKAGE_NAME}_iree-module-8231a286cdc63a48f3f70a12ab5a182142c00cbebaccdc79e35ca552f02422e7
  ${PACKAGE_NAME}_iree-module-23a395a0b9ec63ecaeffe98988633de34fc32c267c35bc2584651d639a15af6e
  ${PACKAGE_NAME}_iree-module-eb1a0a23dc0205fd44ae3b5ca385c651de811b10b6d98fa8b0fb9ef54002c728
  ${PACKAGE_NAME}_iree-module-8b19868be1c797cb585551c871c4171e78817e0efc49d30d91b9d722be283de9
  ${PACKAGE_NAME}_iree-module-c2085883b1f5c767f37508ab998a4bcd17d169fe6a5197d28e4dca8772c90253
  ${PACKAGE_NAME}_iree-module-f770b1916e0b7a9a0b4aa9480791d21a46a352002ac1e38dfcea49ec0b63ed4e
  ${PACKAGE_NAME}_iree-module-88b6b5f712cd2f40d07a136e7f911c05b976c390e07f104c970292dee9a77e9a
  ${PACKAGE_NAME}_iree-module-925cdb19f2aa31a1907c81b5a9e179d91280c77b08a039c1cbf146f71683dde9
  ${PACKAGE_NAME}_iree-module-3c94ab45ad76bd8b2083729b65340b987da3247c854faf7d06431cb05a3b0a23
  ${PACKAGE_NAME}_iree-module-999f2edcdf9fbc84e0969923f8605e9069810a63849973a5b74488f83d14a2fe
  ${PACKAGE_NAME}_iree-module-5176d0f5fa331fa047395186a866870fe4210f637472ef4bb0a3f1ba3a62a749
  ${PACKAGE_NAME}_iree-module-e67816e321544bb61b63ed8cd8b7faa8ff7d35cca15832b5fbc117f4693b3e78
  ${PACKAGE_NAME}_iree-module-d746ecf3d747f18306b6dea4cb6b9e9dbf987fe7fd4d0b27b39a57a213e75dd9
  ${PACKAGE_NAME}_iree-module-439dab16cd6df449fc83eb3e1603fa86ad811e749bcad2c3e3176976c56848e5
  ${PACKAGE_NAME}_iree-module-815e44d1ac31402e86d0fef72e79474a25dfad3a5c15b8cdd3642e101274342d
  ${PACKAGE_NAME}_iree-module-681f456f27dbb79e7d8d0266bf835866d9f29f87eafad7e867ac13c84602742f
  ${PACKAGE_NAME}_iree-module-3b0ae1403ef444d812f0c7b37fda7311e2cc4ea407850ee7b91e6984b9c86100
  ${PACKAGE_NAME}_iree-module-25ad2815eb690276e9c2183aaafaf17a3df734bb6164071ad92dbf1e7faf7509
  ${PACKAGE_NAME}_iree-module-65586f1e5b51439dd951529c35fa9000a928f90039cc6cfb66d5c81d07a6c62b
  ${PACKAGE_NAME}_iree-module-16b5b80aaf1271b5ad782570340cc0c7c1c97e10b7e6c6cc6e5f3ede8393cb6c
  ${PACKAGE_NAME}_iree-module-65fa033050b916e8143d44b5081ee45db3b1946a5d77de223328a7fe92a1cc66
  ${PACKAGE_NAME}_iree-module-16ef56b6869d10b17e983fec62e9f48e6bb87e9a348ab52a0b2faabca2b03578
  ${PACKAGE_NAME}_iree-module-56bc9128e294585d749b8ebe34fd03060ba34d200eef185837b6002d0dcbfccb
  ${PACKAGE_NAME}_iree-module-eb1b1732e5d30ce4689b871f8ec18c50b30eedd418eb80330807fe505bb78f7e
  ${PACKAGE_NAME}_iree-module-bd32992875a8fc7a494c75933b1693d6d8b845fccf2b12504a8cba64d80ad110
  ${PACKAGE_NAME}_iree-module-5e966f45b952e5406271a1a23edd5d9ffab75524b450b7cf4ee35263bb0830f3
  ${PACKAGE_NAME}_iree-module-ff7ed59e05efe8b9a397a179726f63da68a8a1ac3ea731924b4bd24dab491b34
  ${PACKAGE_NAME}_iree-module-8e2d1286ad9a7e360b0c26019146a22ec9188f8bdf8ad99341eb5531cdea2417
  ${PACKAGE_NAME}_iree-module-d967e293594998e48355d30d900dbdf77dbd6eedbff768112dbe8e7ec332b9eb
  ${PACKAGE_NAME}_iree-module-94a5cba0f45ffc2dc028a5f9fa5780fa61ed91b27382c86c4d96de0e2cd002f3
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
  ${PACKAGE_NAME}_iree-module-2a38d4c2e9ce2c3f9a78f5dce9e889145a1d4ec821d5776f15d6d6e679901474
  ${PACKAGE_NAME}_iree-module-660bd6b0861f368e51c6ecac52d6dce2998d863bc0a4bbd9dbdd77508ea761d4
  ${PACKAGE_NAME}_iree-module-224bd124c8446c300caa77db186ca926e71e79cf187980db5dea593d6f29d434
  ${PACKAGE_NAME}_iree-module-7bec578c7016cb7e017057c227a9b677901d14c0fff35e31c4a5cf12692db105
  ${PACKAGE_NAME}_iree-module-89a91c770dfce869ecb04e4b37e3b4d7da508a240da395cf240cc20ee8573857
  ${PACKAGE_NAME}_iree-module-e6049d40d7925bccce4859e5408f2ad53eb68309aa38b46b8a7e47c94a2cd8a3
  ${PACKAGE_NAME}_iree-module-a4194c053541ebc49b4912bbdf3ca211331fdca5d157440837144e59d279bf1f
  ${PACKAGE_NAME}_iree-module-599701d7114956cf64777412899cff57ea5be0478f9a2331377770beaad8f923
  ${PACKAGE_NAME}_iree-module-1d26fcfdb7387659356dd99ce7e10907c8560b0925ad839334b0a6155d25167a
  ${PACKAGE_NAME}_iree-module-b74ccbdce4ec07bb65313ee96b67c1b946a6c959158714706209a9df2b93ab0d
  ${PACKAGE_NAME}_iree-module-b8ee6e3d12e6662f0c78b851876a0759da23967e71ae1a2b5569d0ec3101b43b
  ${PACKAGE_NAME}_iree-module-96e2ebf26260627484f375448ef478c35608ada2d2a0f81c0bec697db9ea3105
  ${PACKAGE_NAME}_iree-module-0ee6c4e8f636f98e5443974526d95e057d5a62bb311bb6dbadd87d50b395a373
  ${PACKAGE_NAME}_iree-module-178907b155b6322dedfa947937f9caca5158ff3af167470f2de90347dba357f4
  ${PACKAGE_NAME}_iree-module-7dcabfd6caa769a75657e07e7315dd42f52b3d4cbc37d75098aca446d3ff4066
  ${PACKAGE_NAME}_iree-module-247b38beca9631678d80755b0b4db2b323ddc4d95772617889a6a4bb813c6b74
  ${PACKAGE_NAME}_iree-module-d39340f50384e970b103694a38d7d21d5b1171d7304630d25e925c5c2486bf10
  ${PACKAGE_NAME}_iree-module-d8f22b5a700abdef68fe791ad08acdfc6d238d82e00f264367d922b99b369ff7
  ${PACKAGE_NAME}_iree-module-c6a4903d1769d721782cf2b6e84837aca21f87fcf8759912a86ae2f572e8440d
  ${PACKAGE_NAME}_iree-module-3c43472d6cb0f74a1c08920e3f580b701e995a85305fd4b2e370542b4d449b18
  ${PACKAGE_NAME}_iree-module-6e207112c3da58908537a07d168c78e7d166fe6803659d4b9f07848c968d6d12
  ${PACKAGE_NAME}_iree-module-75259bfdbf7eb331691860ffb18e04a146168a72b7b10cf070d5db7f67dd2378
  ${PACKAGE_NAME}_iree-module-5ef6657fb694df545b7f87fbb78dfa99af891778790ac0924c08a87d335c1bf9
  ${PACKAGE_NAME}_iree-module-02b72f9538e4dfc9c789e63d722d5eab4333f3f55f8375503f433a790da119cc
  ${PACKAGE_NAME}_iree-module-e7bd41e564750501f39ac9690c18d1a2e77dc7999da710d0c0bf80751dda84a0
)
