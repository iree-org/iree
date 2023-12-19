iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-PersonDetect_int8_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_PersonDetect_int8.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileNetV3Small_fp32_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileNetV3Small_fp32.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-DeepLabV3_fp32_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_DeepLabV3_fp32.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-EfficientNet_int8_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_EfficientNet_int8.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileNetV1_fp32_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileNetV1_fp32.0_float.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileNetV2_fp32_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileNetV2_fp32.0_224.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileNetV2_int8_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileNetV2_int8.0_224_quantized.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileSSD_fp32_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileSSD_fp32.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-PoseNet_fp32_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_PoseNet_fp32.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileBertSquad_fp16_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_fp16.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileBertSquad_fp32_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_fp32.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-MobileBertSquad_int8_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_MobileBertSquad_int8.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
)

iree_import_tflite_model(
  TARGET_NAME "${PACKAGE_NAME}_iree-imported-model-Vit_int8_tflite_"
  SOURCE "${ROOT_ARTIFACTS_DIR}/model_Vit_int8.tflite"
  OUTPUT_MLIR_FILE "${ROOT_ARTIFACTS_DIR}/iree_Vit_int8_tflite_.mlir"
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bGptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Falcon7bGptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bInt4GptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
  FRIENDLY_NAME "Falcon7bInt4GptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bGptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Falcon7bGptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bInt4GptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
  FRIENDLY_NAME "Falcon7bInt4GptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bGptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Falcon7bGptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bInt4GptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
  FRIENDLY_NAME "Falcon7bInt4GptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertLargeTFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertLargeTFBatch32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "BertLargeTFBatch64(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Resnet50TFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Resnet50TFBatch64(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Resnet50TFBatch128(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "T5LargeTFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "T5LargeTFBatch16(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "T5LargeTFBatch32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_3456x1024x2048_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_3456x1024x2048_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2560x2560x2560_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2560x2560x2560_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
  FRIENDLY_NAME "matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-flow-split-matmul-reduction=4"
    "--iree-codegen-llvmgpu-use-wmma"
  FRIENDLY_NAME "matmul_128x256x8192_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,splitk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-flow-split-matmul-reduction=4"
    "--iree-codegen-llvmgpu-use-wmma"
  FRIENDLY_NAME "matmul_128x256x8192_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,splitk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV1_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MobileNetV1_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv32"
    "--iree-llvmcpu-target-abi=ilp32"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Vit_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Vit_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Vit_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Vit_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Vit_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
  FRIENDLY_NAME "Vit_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [qualcomm-adreno-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [qualcomm-adreno-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_demote-f32-to-f16_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_demote-f32-to-f16_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-opt-demote-f32-to-f16"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][default-flags,demote-f32-to-f16]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][default-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_demote-f32-to-f16_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_demote-f32-to-f16_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-opt-demote-f32-to-f16"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,demote-f32-to-f16]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_demote-f32-to-f16_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_demote-f32-to-f16_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-opt-demote-f32-to-f16"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,repeated-kernel,demote-f32-to-f16]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,repeated-kernel]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
  FRIENDLY_NAME "MobileNetV2_fp32(tflite) [vmvx-generic-vmvx-vmvx][experimental-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV3Small_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
  FRIENDLY_NAME "MobileNetV3Small_fp32(tflite) [vmvx-generic-vmvx-vmvx][experimental-flags]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bGptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Falcon7bGptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bInt4GptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Falcon7bInt4GptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bGptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Falcon7bGptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bInt4GptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Falcon7bInt4GptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV1_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileSSD_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileSSD_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PoseNet_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "PoseNet_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bGptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Falcon7bGptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Falcon7bInt4GptqPT.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=none"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Falcon7bInt4GptqPT(linalg) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTFBatch64(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch64.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch64(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_Resnet50TFBatch128.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Resnet50TFBatch128(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch1.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch1(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch16.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch16(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_T5LargeTFBatch32.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvmcpu-target-cpu=cascadelake"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "T5LargeTFBatch32(stablehlo) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_EfficientNetV2STF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNetV2STF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertForMaskedLMTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertForMaskedLMTF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_BertLargeTF.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=stablehlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "BertLargeTF(stablehlo) [cuda-sm_80-linux_gnu-cuda][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_3456x1024x2048_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_3456x1024x2048_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_3456x1024x2048_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2560x2560x2560_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2560x2560x2560_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2560x2560x2560_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f16t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-flow-split-matmul-reduction=4"
    "--iree-codegen-llvmgpu-use-wmma"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_128x256x8192_f16t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,splitk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_matmul_128x256x8192_f32t_tile_config_default.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=none"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
    "--iree-hal-benchmark-dispatch-repeat-count=100"
    "--iree-flow-split-matmul-reduction=4"
    "--iree-codegen-llvmgpu-use-wmma"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "matmul_128x256x8192_f32t_tile_config_default(linalg) [cuda-sm_80-linux_gnu-cuda][ukernel,matmul,splitk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MiniLML12H384Uncased_stablehlo___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_MiniLML12H384Uncased.timestamp_1683504734.mlirbc"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvmcpu-target-cpu=generic-rv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MiniLML12H384Uncased_stablehlo___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MiniLML12H384Uncased(stablehlo) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV1_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV1_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV1_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV1_fp32(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [riscv_64-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-EfficientNet_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_EfficientNet_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_EfficientNet_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "EfficientNet_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-PersonDetect_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_PersonDetect_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_PersonDetect_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "PersonDetect_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/module.vmfb"
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
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_int8(tflite) [riscv_32-generic-linux_gnu-llvm_cpu][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Vit_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=false"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Vit_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,no-dt,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Vit_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Vit_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][default-flags,dt-uk,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_DeepLabV3_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "DeepLabV3_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X4XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X4XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/model_GPT2_117M_TF_1X1XI32.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=stablehlo"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "GPT2_117M_TF_1X1XI32(stablehlo) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_Vit_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvmcpu-target-triple=aarch64-none-linux-android29"
    "--iree-opt-data-tiling=true"
    "--iree-llvmcpu-enable-ukernels=none"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-global-opt-propagate-transposes=false"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "Vit_int8(tflite) [armv8.2-a-generic-linux_android29-llvm_cpu][experimental-flags,dt-only,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [qualcomm-adreno-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [qualcomm-adreno-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_demote-f32-to-f16_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_demote-f32-to-f16_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-opt-demote-f32-to-f16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_demote-f32-to-f16_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][default-flags,demote-f32-to-f16,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][default-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_demote-f32-to-f16_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_demote-f32-to-f16_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-opt-demote-f32-to-f16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_demote-f32-to-f16_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,demote-f32-to-f16,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp32(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_demote-f32-to-f16_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_fp16_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_demote-f32-to-f16_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-opt-demote-f32-to-f16"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_demote-f32-to-f16_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_fp16(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,repeated-kernel,demote-f32-to-f16,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileBertSquad_int8_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-stream-partitioning-favor=max-concurrency"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileBertSquad_int8(tflite) [arm-valhall-vulkan_android31-vulkan_spirv][experimental-flags,fuse-padding,max-concurrency,repeated-kernel,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV2_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV2_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV2_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV2_fp32(tflite) [vmvx-generic-vmvx-vmvx][experimental-flags,compile-stats]"
  PUBLIC
)

iree_bytecode_module(
  NAME "iree-module-MobileNetV3Small_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_compile-stats_"
  SRC "${ROOT_ARTIFACTS_DIR}/iree_MobileNetV3Small_fp32_tflite_.mlir"
  MODULE_FILE_NAME "${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_compile-stats_/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
    "--iree-vm-emit-polyglot-zip=true"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-scheduling-dump-statistics-format=json"
    "--iree-scheduling-dump-statistics-file=${ROOT_ARTIFACTS_DIR}/iree_module_MobileNetV3Small_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_compile-stats_/scheduling_stats.json"
  FRIENDLY_NAME "MobileNetV3Small_fp32(tflite) [vmvx-generic-vmvx-vmvx][experimental-flags,compile-stats]"
  PUBLIC
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-DeepLabV3_fp32_tflite_
  ${PACKAGE_NAME}_iree-imported-model-EfficientNet_int8_tflite_
  ${PACKAGE_NAME}_iree-imported-model-MobileBertSquad_fp16_tflite_
  ${PACKAGE_NAME}_iree-imported-model-MobileBertSquad_fp32_tflite_
  ${PACKAGE_NAME}_iree-imported-model-MobileBertSquad_int8_tflite_
  ${PACKAGE_NAME}_iree-imported-model-MobileNetV1_fp32_tflite_
  ${PACKAGE_NAME}_iree-imported-model-MobileNetV2_fp32_tflite_
  ${PACKAGE_NAME}_iree-imported-model-MobileNetV2_int8_tflite_
  ${PACKAGE_NAME}_iree-imported-model-MobileNetV3Small_fp32_tflite_
  ${PACKAGE_NAME}_iree-imported-model-MobileSSD_fp32_tflite_
  ${PACKAGE_NAME}_iree-imported-model-PersonDetect_int8_tflite_
  ${PACKAGE_NAME}_iree-imported-model-PoseNet_fp32_tflite_
  ${PACKAGE_NAME}_iree-imported-model-Vit_int8_tflite_
  ${PACKAGE_NAME}_model-BertForMaskedLMTF
  ${PACKAGE_NAME}_model-BertLargeTF
  ${PACKAGE_NAME}_model-EfficientNetV2STF
  ${PACKAGE_NAME}_model-Falcon7bGptqPT
  ${PACKAGE_NAME}_model-Falcon7bInt4GptqPT
  ${PACKAGE_NAME}_model-GPT2_117M_TF_1X1XI32
  ${PACKAGE_NAME}_model-GPT2_117M_TF_1X4XI32
  ${PACKAGE_NAME}_model-MiniLML12H384Uncased
  ${PACKAGE_NAME}_model-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_128x256x8192_f16t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_128x256x8192_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2560x2560x2560_f16t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2560x2560x2560_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_3456x1024x2048_f16t_tile_config_default
  ${PACKAGE_NAME}_model-matmul_3456x1024x2048_f32t_tile_config_default
)

add_dependencies(iree-benchmark-import-models-large
  ${PACKAGE_NAME}_model-BertLargeTFBatch1
  ${PACKAGE_NAME}_model-BertLargeTFBatch32
  ${PACKAGE_NAME}_model-BertLargeTFBatch64
  ${PACKAGE_NAME}_model-Resnet50TFBatch1
  ${PACKAGE_NAME}_model-Resnet50TFBatch128
  ${PACKAGE_NAME}_model-Resnet50TFBatch64
  ${PACKAGE_NAME}_model-T5LargeTFBatch1
  ${PACKAGE_NAME}_model-T5LargeTFBatch16
  ${PACKAGE_NAME}_model-T5LargeTFBatch32
)

add_dependencies(iree-benchmark-suites-android-cpu
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
)

add_dependencies(iree-benchmark-suites-android-cpu-dt-only
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
)

add_dependencies(iree-benchmark-suites-android-gpu
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_demote-f32-to-f16_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_demote-f32-to-f16_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_demote-f32-to-f16_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_
)

add_dependencies(iree-benchmark-suites-comp-stats
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_demote-f32-to-f16_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_demote-f32-to-f16_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_demote-f32-to-f16_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_compile-stats_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_compile-stats_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_compile-stats_
)

add_dependencies(iree-benchmark-suites-comp-stats-large
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_compile-stats_
)

add_dependencies(iree-benchmark-suites-cuda
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
)

add_dependencies(iree-benchmark-suites-default
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___cuda-sm_80-linux_gnu-cuda__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_demote-f32-to-f16_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_demote-f32-to-f16_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_demote-f32-to-f16_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___qualcomm-adreno-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___arm-valhall-vulkan_android31-vulkan_spirv__experimental-flags_fuse-padding_max-concurrency_repeated-kernel_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___vmvx-generic-vmvx-vmvx__experimental-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-Vit_int8_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-matmul_123x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_
  ${PACKAGE_NAME}_iree-module-matmul_128x256x8192_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_splitk_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2560x2560x2560_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2561x2561_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2562x2564x2562_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_2564x2564x2564_f32t_f32t_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f16t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
  ${PACKAGE_NAME}_iree-module-matmul_3456x1024x2048_f32t_tile_config_default_linalg___cuda-sm_80-linux_gnu-cuda__ukernel_matmul_
)

add_dependencies(iree-benchmark-suites-large
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
)

add_dependencies(iree-benchmark-suites-riscv
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___riscv_32-generic-linux_gnu-llvm_cpu__default-flags_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___riscv_64-generic-linux_gnu-llvm_cpu__default-flags_
)

add_dependencies(iree-benchmark-suites-x86_64
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_no-dt_
)

add_dependencies(iree-benchmark-suites-x86_64-dt-only
  ${PACKAGE_NAME}_iree-module-BertForMaskedLMTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-BertLargeTF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-DeepLabV3_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-EfficientNetV2STF_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-EfficientNet_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-Falcon7bGptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-Falcon7bInt4GptqPT_linalg___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X1XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-GPT2_117M_TF_1X4XI32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MiniLML12H384Uncased_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp16_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileBertSquad_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileNetV1_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileNetV2_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileNetV3Small_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-MobileSSD_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-PersonDetect_int8_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
  ${PACKAGE_NAME}_iree-module-PoseNet_fp32_tflite___x86_64-cascadelake-linux_gnu-llvm_cpu__experimental-flags_dt-only_
)

add_dependencies(iree-benchmark-suites-x86_64-large
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-BertLargeTFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch128_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-Resnet50TFBatch64_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch16_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch1_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
  ${PACKAGE_NAME}_iree-module-T5LargeTFBatch32_stablehlo___x86_64-cascadelake-linux_gnu-llvm_cpu__default-flags_dt-uk_
)
