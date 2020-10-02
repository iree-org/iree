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
# files. A coverage table generated from this file can be viewed here:
#   https://google.github.io/iree/tf-e2e-coverage
# Updates made to test suite names should also be reflected here:
#   https://github.com/google/iree/blob/main/scripts/update_e2e_coverage.py

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "iree_py_binary",
)
load(
    "//integrations/tensorflow/e2e/slim_vision_models:iree_slim_vision_test_suite.bzl",
    "iree_slim_vision_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# Create binaries for all test srcs to allow them to be run manually.
[
    iree_py_binary(
        name = src.replace(".py", "_manual"),
        srcs = [src],
        main = src,
        python_version = "PY3",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for src in glob(["*_test.py"])
]

iree_slim_vision_test_suite(
    name = "slim_vision_tests",
    backends = [
        "tf",
        "tflite",
        "iree_vmla",
        "iree_llvmjit",
        "iree_vulkan",
    ],
    failing_configurations = [
        {
            # Failing all but tf and vmla:
            "models": [
                "inception_resnet_v2",
                # tflite: RuntimeError: tensorflow/lite/kernels/reshape.cc:66 num_input_elements != num_output_elements (38400 != -1571481807)Node number 333 (RESHAPE) failed to prepare.
                # llvmjit: *** Received signal 6 *** (mangled stack trace)
                # vulkan: Floating point difference between ref and tar was too large. Max abs diff: 1.3961234, atol: 2e-05, max relative diff: 0.27304956, rtol: 1e-06
                "resnet_v2_101",
                # tflite: RuntimeError: tensorflow/lite/core/subgraph.cc BytesRequired number of elements overflowed.
                # llvmjit: Floating point difference between ref and tar was too large. Max abs diff: 11.668068, atol: 2e-05, max relative diff: 0.93950737, rtol: 1e-06
                # vulkan: Floating point difference between ref and tar was too large. Max abs diff: 11.668067, atol: 2e-05, max relative diff: 0.93950737, rtol: 1e-06
                "resnet_v2_152",
                # tflite: RuntimeError: tensorflow/lite/core/subgraph.cc BytesRequired number of elements overflowed.
                # llvmjit: Floating point difference between ref and tar was too large. Max abs diff: 7.080696, atol: 2e-05, max relative diff: 0.97750616, rtol: 1e-06
                # vulkan: Floating point difference between ref and tar was too large. Max abs diff: 7.08069, atol: 2e-05, max relative diff: 0.97750485, rtol: 1e-06
            ],
            "backends": [
                "tflite",
                "iree_llvmjit",
                "iree_vulkan",
            ],
        },
        {
            # Failing llvmjit and vulkan:
            "models": [
                "inception_v2",
                # llvmjit: double free or corruption (!prev); *** Received signal 6 ***
                # vulkan: Floating point difference between ref and tar was too large. Max abs diff: 1.0769763, atol: 2e-05, max relative diff: 0.19576924, rtol: 1e-06
                "inception_v3",
                # llvmjit: double free or corruption (!prev); *** Received signal 6 ***
                # vulkan: Floating point difference between ref and tar was too large. Max abs diff: 2.5201874, atol: 2e-05, max relative diff: 0.53700095, rtol: 1e-06
                "nasnet_mobile",
                # llvmjit: corrupted size vs. prev_size; *** Received signal 6 ***
                # vulkan: *** Received signal 11 ***
                "resnet_v2_50",
                # llvmjit: Floating point difference between ref and tar was too large. Max abs diff: 5.8187943, atol: 2e-05, max relative diff: 0.7946711, rtol: 1e-06
                # vulkan: Floating point difference between ref and tar was too large. Max abs diff: 5.8187933, atol: 2e-05, max relative diff: 0.79467094, rtol: 1e-06
            ],
            "backends": [
                "iree_llvmjit",
                "iree_vulkan",
            ],
        },
    ],
    models = [
        "inception_resnet_v2",
        "inception_v1",
        "inception_v2",
        "inception_v3",
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
        "mobilenet_v2_035_224",
        "mobilenet_v2_050_224",
        "mobilenet_v2_075_224",
        "mobilenet_v2_100_224",
        "mobilenet_v2_130_224",
        "mobilenet_v2_140_224",
        "nasnet_mobile",
        "resnet_v1_101",
        "resnet_v1_152",
        "resnet_v1_50",
        "resnet_v2_101",
        "resnet_v2_152",
        "resnet_v2_50",
    ],
    reference_backend = "tf",
    tags = [
        "external",
        "guitar",
        "manual",
        "no-remote",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

# TODO(meadowlark): Get these working on tf.
iree_slim_vision_test_suite(
    name = "slim_vision_ill_configured_tests",
    backends = [
        "tf",
        "tflite",
        "iree_vmla",
        "iree_llvmjit",
        "iree_vulkan",
    ],
    failing_configurations = [
        {
            # Failing on all backends:
            "models": [
                "amoebanet_a_n18_f448",
                "mobilenet_v2_035_128",
                "mobilenet_v2_035_160",
                "mobilenet_v2_035_192",
                "mobilenet_v2_035_96",
                "mobilenet_v2_050_128",
                "mobilenet_v2_050_160",
                "mobilenet_v2_050_192",
                "mobilenet_v2_050_96",
                "mobilenet_v2_075_128",
                "mobilenet_v2_075_160",
                "mobilenet_v2_075_192",
                "mobilenet_v2_075_96",
                "mobilenet_v2_100_128",
                "mobilenet_v2_100_160",
                "mobilenet_v2_100_192",
                "mobilenet_v2_100_96",
                "nasnet_large",
                "pnasnet_large",
            ],
            "backends": [
                "tf",
                "tflite",
                "iree_vmla",
                "iree_llvmjit",
                "iree_vulkan",
            ],
        },
    ],
    models = [
        "amoebanet_a_n18_f448",
        "mobilenet_v2_035_128",
        "mobilenet_v2_035_160",
        "mobilenet_v2_035_192",
        "mobilenet_v2_035_96",
        "mobilenet_v2_050_128",
        "mobilenet_v2_050_160",
        "mobilenet_v2_050_192",
        "mobilenet_v2_050_96",
        "mobilenet_v2_075_128",
        "mobilenet_v2_075_160",
        "mobilenet_v2_075_192",
        "mobilenet_v2_075_96",
        "mobilenet_v2_100_128",
        "mobilenet_v2_100_160",
        "mobilenet_v2_100_192",
        "mobilenet_v2_100_96",
        "nasnet_large",
        "pnasnet_large",
    ],
    reference_backend = "tf",
    tags = [
        "external",
        "guitar",
        "manual",
        "no-remote",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
