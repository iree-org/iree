# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models from TFLite                                                 #
#                                                                              #
# Each module specification should be a list containing alternating keys and   #
# values. The fields are: NAME, TAGS, SOURCE, ENTRY_FUNCTION, and              #
# FUNCTION_INPUTS. See the iree_benchmark_suite definition for details         #
# about these fields. Note that these must be quoted when used as arguments.   #
#                                                                              #
################################################################################

set(DEEPLABV3_FP32_MODULE
  NAME
    "DeepLabV3"
  TAGS
    "fp32"
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1
    "https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x257x257x3xf32"
)

set(MOBILESSD_FP32_MODULE
  NAME
    "MobileSSD"
  TAGS
    "fp32"
  SOURCE
    # Mirror of https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite
    "https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x320x320x3xf32"
)

set(POSENET_FP32_MODULE
  NAME
    "PoseNet"
  TAGS
    "fp32"
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1
    "https://storage.googleapis.com/iree-model-artifacts/posenet.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x353x257x3xf32"
)

set(MOBILEBERT_FP32_MODULE
  NAME
    "MobileBertSquad"
  TAGS
    "fp32"
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1
    "https://storage.googleapis.com/iree-model-artifacts/mobilebertsquad.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x384xi32,1x384xi32,1x384xi32"
)

set(MOBILEBERT_FP16_MODULE
  NAME
    "MobileBertSquad"
  TAGS
    "fp16"
  # This uses the same input MLIR source as fp32 to save download time.
  # It requires users to have "--iree-flow-demote-f32-to-f16".
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1
    "https://storage.googleapis.com/iree-model-artifacts/mobilebertsquad.tflite"
  ENTRY_FUNCTION
    "main"
  # The conversion done by "--iree-flow-demote-f32-to-f16" won't change the
  # original input signature.
  FUNCTION_INPUTS
    "1x384xi32,1x384xi32,1x384xi32"
)

set(MOBILENET_V2_MODULE
  NAME
    "MobileNetV2"
  TAGS
    "fp32,imagenet"
  SOURCE
    # Mirror https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/mobilenet_v2_1.0_224.tflite
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x224x224x3xf32"
)

set(MOBILENET_V3SMALL_MODULE
  NAME
    "MobileNetV3Small"
  TAGS
    "fp32,imagenet"
  SOURCE
    # https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5
    # Manually exported to tflite with static batch dimension
    "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3SmallStaticBatch.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x224x224x3xf32"
)


################################################################################
#                                                                              #
# Default benchmark configurations                                             #
#                                                                              #
# Each suite benchmarks a list of modules with configurations specifying a     #
# target architecture and runtime characteristics (e.g. threads/cores). These  #
# benchmarks only configure IREE translation and runtime flags for the target  #
# architecture and do *not* include any non-default flags. No non-default      #
# flags should be added here.                                                  #
#                                                                              #
################################################################################

set(ANDROID_CPU_TRANSLATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvm-target-triple=aarch64-none-linux-android29")
set(ANDROID_ADRENO_GPU_TRANSLATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-vulkan-target-triple=adreno-unknown-android11"
)
set(ANDROID_MALI_GPU_TRANSLATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-vulkan-target-triple=valhall-unknown-android11"
)

# CPU, Dylib-Sync, big/little-core, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference,default-flags"
    "little-core,full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    ${ANDROID_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1 through 4 threads, big/little-core, full-inference.
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference,default-flags"
    "1-thread,little-core,full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    ${ANDROID_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# TODO(#7792): Re-enable these when we are able to run different benchmarks
# depending on use-case (presubmit, postsubmit, nightly, etc.)
# iree_benchmark_suite(
#   MODULES
#     "${DEEPLABV3_FP32_MODULE}"
#     "${MOBILESSD_FP32_MODULE}"
#     "${POSENET_FP32_MODULE}"
#     "${MOBILEBERT_FP32_MODULE}"
#     "${MOBILENET_V2_MODULE}"
#     "${MOBILENET_V3SMALL_MODULE}"

#   BENCHMARK_MODES
#     "2-thread,big-core,full-inference,default-flags"
#     "2-thread,little-core,full-inference,default-flags"
#   TARGET_BACKEND
#     "dylib-llvm-aot"
#   TARGET_ARCHITECTURE
#     "CPU-ARM64-v8A"
#   TRANSLATION_FLAGS
#     ${ANDROID_CPU_TRANSLATION_FLAGS}
#   BENCHMARK_TOOL
#     iree-benchmark-module
#   DRIVER
#     "dylib"
#   RUNTIME_FLAGS
#     "--task_topology_group_count=2"
# )

# iree_benchmark_suite(
#   MODULES
#     "${DEEPLABV3_FP32_MODULE}"
#     "${MOBILESSD_FP32_MODULE}"
#     "${POSENET_FP32_MODULE}"
#     "${MOBILEBERT_FP32_MODULE}"
#     "${MOBILENET_V2_MODULE}"
#     "${MOBILENET_V3SMALL_MODULE}"

#   BENCHMARK_MODES
#     "3-thread,big-core,full-inference,default-flags"
#     "3-thread,little-core,full-inference,default-flags"
#   TARGET_BACKEND
#     "dylib-llvm-aot"
#   TARGET_ARCHITECTURE
#     "CPU-ARM64-v8A"
#   TRANSLATION_FLAGS
#     ${ANDROID_CPU_TRANSLATION_FLAGS}
#   BENCHMARK_TOOL
#     iree-benchmark-module
#   DRIVER
#     "dylib"
#   RUNTIME_FLAGS
#     "--task_topology_group_count=3"
# )

iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "4-thread,big-core,full-inference,default-flags"
    "4-thread,little-core,full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    ${ANDROID_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)

# GPU, Vulkan, Adreno, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  TRANSLATION_FLAGS
    ${ANDROID_ADRENO_GPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    ${ANDROID_MALI_GPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, full-inference
iree_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP16_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    ${ANDROID_MALI_GPU_TRANSLATION_FLAGS}
    # This isn't a special optimization flag. It's so we can reuse the same f32
    # model file. See comments on MOBILEBERT_FP16_MODULE
    "--iree-flow-demote-f32-to-f16"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
)

################################################################################

################################################################################
#                                                                              #
# Specialized benchmark configurations                                         #
#                                                                              #
# Each suite benchmarks one or more module with configurations that can vary   #
# on model or architecture characteristics. These are intended for providing   #
# continuous benchmarks of experimental features that cannot be turned on by   #
# default yet. It is primarily intended for whoever is actively investigating  #
# optimizations for a feature exemplified in a specific model or architecture. #
# Due to our current benchmark setup, there can only be one experimental       #
# configuration per model and other benchmark mode.                            #
#                                                                              #
################################################################################

# CPU, Dylib-Sync, big/little-core, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference,experimental-flags"
    "little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    ${ANDROID_CPU_TRANSLATION_FLAGS}
    "--iree-llvm-loop-unrolling=true"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib-sync"
)

# TODO(#7792): Consider re-enabling little-core experimental-flags if we start
# optimizing for little cores or we can just run them occasionally

# CPU, Dylib, 1 through 4 threads, big/little-core, full-inference.
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference,experimental-flags"
    # "1-thread,little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    ${ANDROID_CPU_TRANSLATION_FLAGS}
    "--iree-llvm-loop-unrolling=true"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# TODO(#7792): Re-enable these when we are able to run different benchmarks
# depending on use-case (presubmit, postsubmit, nightly, etc.)
# iree_benchmark_suite(
#   MODULES
#     "${DEEPLABV3_FP32_MODULE}"
#     "${MOBILESSD_FP32_MODULE}"
#     "${POSENET_FP32_MODULE}"
#     "${MOBILEBERT_FP32_MODULE}"
#     "${MOBILENET_V2_MODULE}"
#     "${MOBILENET_V3SMALL_MODULE}"

#   BENCHMARK_MODES
#     "2-thread,big-core,full-inference,experimental-flags"
#     "2-thread,little-core,full-inference,experimental-flags"
#   TARGET_BACKEND
#     "dylib-llvm-aot"
#   TARGET_ARCHITECTURE
#     "CPU-ARM64-v8A"
#   TRANSLATION_FLAGS
#     ${ANDROID_CPU_TRANSLATION_FLAGS}
#     "--iree-llvm-loop-unrolling=true"
#   BENCHMARK_TOOL
#     iree-benchmark-module
#   DRIVER
#     "dylib"
#   RUNTIME_FLAGS
#     "--task_topology_group_count=2"
# )

# iree_benchmark_suite(
#   MODULES
#   "${DEEPLABV3_FP32_MODULE}"
#   "${MOBILESSD_FP32_MODULE}"
#   "${POSENET_FP32_MODULE}"
#   "${MOBILEBERT_FP32_MODULE}"
#   "${MOBILENET_V2_MODULE}"
#   "${MOBILENET_V3SMALL_MODULE}"

#   BENCHMARK_MODES
#     "3-thread,big-core,full-inference,experimental-flags"
#     "3-thread,little-core,full-inference,experimental-flags"
#   TARGET_BACKEND
#     "dylib-llvm-aot"
#   TARGET_ARCHITECTURE
#     "CPU-ARM64-v8A"
#   TRANSLATION_FLAGS
#     ${ANDROID_CPU_TRANSLATION_FLAGS}
#     "--iree-llvm-loop-unrolling=true"
#   BENCHMARK_TOOL
#     iree-benchmark-module
#   DRIVER
#     "dylib"
#   RUNTIME_FLAGS
#     "--task_topology_group_count=3"
# )

iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "4-thread,big-core,full-inference,experimental-flags"
    # "4-thread,little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    ${ANDROID_CPU_TRANSLATION_FLAGS}
    "--iree-llvm-loop-unrolling=true"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)


# CPU, VMVX, 4-thread, big-core, full-inference
# VMVX is slow and we're not optimizing perf yet. Leaving in a single max-thread
# benchmark because it's useful to keep an eye on and helps disambiguate where a
# performance change may be coming from (e.g. if it's in vmvx as well, it's
# probably not a codegen issue).
iree_benchmark_suite(
  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "4-thread,big-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "vmvx"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=tosa"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vmvx"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)


# GPU, Vulkan, Adreno, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  TRANSLATION_FLAGS
    ${ANDROID_ADRENO_GPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    ${ANDROID_MALI_GPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
)

iree_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP16_MODULE}"

  BENCHMARK_MODES
    "full-inference,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
)

# kernel-execution

# Note that for kernel-execution benchmarks batch_size/repeat-count need to be
# low enough that the whole dispatch completes within an OS-specific timeout.
# Otherwise you'll get error like:
# ```
# INTERNAL; VK_ERROR_DEVICE_LOST; vkQueueSubmit; while invoking native function
# hal.ex.submit_and_wait; while calling import;
# ```
# With current kernel performance and timeouts on Adreno Pixel 4, this means we
# have no kernel benchmark for the DeepLabV3 and MobileBert models
# TODO: Add kernel-execution config for DEEPLABV3_FP32_MODULE and
# MOBILEBERT_FP32_MODULE when they can run with at least 8 repetitions.

# GPU, Vulkan, Adreno, kernel-execution
iree_benchmark_suite(
  MODULES
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "kernel-execution,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  TRANSLATION_FLAGS
    ${ANDROID_ADRENO_GPU_TRANSLATION_FLAGS}
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=16"
)

# GPU, Vulkan, Mali, kernel-execution
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "kernel-execution,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    ${ANDROID_MALI_GPU_TRANSLATION_FLAGS}
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)

iree_benchmark_suite(
  MODULES
    "${MOBILEBERT_FP16_MODULE}"

  BENCHMARK_MODES
    "kernel-execution,experimental-flags"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=tosa"
    "--iree-flow-demote-f32-to-f16"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)
