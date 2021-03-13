# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test coverage across backends for e2e tests is defined directly in the BUILD
# files. Coverage tables generated from this file can be viewed here:
#   https://google.github.io/iree/tensorflow-coverage/vision-coverage
# Updates made to test suite names should also be reflected here:
#   https://github.com/google/iree/blob/main/scripts/update_e2e_coverage.py

load(
    "//e2e:iree_e2e_cartesian_product_test_suite.bzl",
    "iree_e2e_cartesian_product_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# Create binaries for all test srcs to allow them to be run manually.
py_binary(
    name = "slim_vision_model_test_manual",
    srcs = ["slim_vision_model_test.py"],
    args = ["--tf_hub_url=https://tfhub.dev/google/imagenet/"],
    main = "slim_vision_model_test.py",
    python_version = "PY3",
)

iree_e2e_cartesian_product_test_suite(
    name = "slim_vision_tests",
    size = "enormous",
    failing_configurations = [
        {
            # SavedModelV2 (classification/4) not available.
            "model": "amoebanet_a_n18_f448",
        },
        {
            # Failing on vmla with negative inputs.
            "model": [
                "nasnet_large",
                "nasnet_mobile",
            ],
            "target_backends": "iree_vmla",
        },
        {
            # Failing llvmjit and vulkan:
            "model": [
                "nasnet_mobile",
                "nasnet_large",
                "pnasnet_large",
                "resnet_v2_50",
                "resnet_v2_101",
                "resnet_v2_152",
            ],
            "target_backends": [
                "iree_vulkan",
            ],
        },
        {
            # Failing vulkan:
            "model": [
                # [ERROR]: cannot separate Linalg/Parallel ops into multiple kernels
                "inception_v1",
                "inception_v2",
                "inception_v3",
                "inception_resnet_v2",
            ],
            "target_backends": [
                "iree_vulkan",
            ],
        },
    ],
    matrix = {
        "src": "slim_vision_model_test.py",
        "reference_backend": "tf",
        "tf_hub_url": ["https://tfhub.dev/google/imagenet/"],
        "model": [
            "amoebanet_a_n18_f448",
            "inception_resnet_v2",
            "inception_v1",
            "inception_v2",
            "inception_v3",
            # MobileNetV1
            "mobilenet_v1_025_128",
            "mobilenet_v1_025_160",
            "mobilenet_v1_025_192",
            "mobilenet_v1_025_224",
            "mobilenet_v1_050_128",
            "mobilenet_v1_050_160",
            "mobilenet_v1_050_192",
            "mobilenet_v1_050_224",
            "mobilenet_v1_075_128",
            "mobilenet_v1_075_160",
            "mobilenet_v1_075_192",
            "mobilenet_v1_075_224",
            "mobilenet_v1_100_128",
            "mobilenet_v1_100_160",
            "mobilenet_v1_100_192",
            "mobilenet_v1_100_224",
            # MobileNetV2:
            "mobilenet_v2_035_96",
            "mobilenet_v2_035_128",
            "mobilenet_v2_035_160",
            "mobilenet_v2_035_192",
            "mobilenet_v2_035_224",
            "mobilenet_v2_050_96",
            "mobilenet_v2_050_128",
            "mobilenet_v2_050_160",
            "mobilenet_v2_050_192",
            "mobilenet_v2_050_224",
            "mobilenet_v2_075_96",
            "mobilenet_v2_075_128",
            "mobilenet_v2_075_160",
            "mobilenet_v2_075_192",
            "mobilenet_v2_075_224",
            "mobilenet_v2_100_96",
            "mobilenet_v2_100_128",
            "mobilenet_v2_100_160",
            "mobilenet_v2_100_192",
            "mobilenet_v2_100_224",
            "mobilenet_v2_130_224",
            "mobilenet_v2_140_224",
            "nasnet_mobile",
            "nasnet_large",
            "pnasnet_large",
            # ResNetV1
            "resnet_v1_50",
            "resnet_v1_101",
            "resnet_v1_152",
            # ResNetV2
            "resnet_v2_50",
            "resnet_v2_101",
            "resnet_v2_152",
        ],
        "target_backends": [
            "tf",
            "tflite",
            "iree_vmla",
            "iree_vulkan",
        ],
    },
    tags = [
        "external",
        "guitar",
        "manual",
        "no-remote",
        "nokokoro",
        "notap",
    ],
)
