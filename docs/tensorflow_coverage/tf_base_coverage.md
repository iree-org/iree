---
layout: default
permalink: tensorflow-coverage/tf-base-coverage
title: "TensorFlow Base APIs"
nav_order: 1
parent: TensorFlow Coverage
js_files: 
- js/add_classes.js
---

# TensorFlow Base APIs
{: .no_toc }

Tests of the `tf`, `tf.math`, `tf.nn`, `tf.signal` and `tf.strings` APIs.

IREE has three backend
[targets](https://github.com/google/iree/tree/main/iree/compiler/Dialect/HAL/Target):
`vmla`, `llvm-ir` and `vulkan-spirv`. We also test TFLite in our infrastructure
for benchmarking purposes. The coverage tables below are automatically generated
from IREE's test suites.

## End to end TensorFlow tests

target | tflite | vmla | llvm-ir | vulkan-spirv
:-: | :-: | :-: | :-: | :-:
[batch_norm_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/batch_norm_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[batch_to_space_nd_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/batch_to_space_nd_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[bool_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/bool_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span>
[broadcast_to_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/broadcast_to_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[broadcasting_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/broadcasting_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[complex_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/complex_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[concat_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/concat_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[control_flow_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/control_flow_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[conv_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/conv_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[conv_transpose_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/conv_transpose_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[depth_conv_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/depth_conv_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[dynamic_mlp_relu_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/dynamic_mlp_relu_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[dynamic_mlp_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/dynamic_mlp_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[einsum_dynamic_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/einsum_dynamic_test.py) | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[einsum_static_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/einsum_static_test.py) | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[einsum_vector_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/einsum_vector_test.py) | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[fft_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/fft_test.py) | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[fill_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/fill_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[finite_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/finite_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[gather_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/gather_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[linspace_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/linspace_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[logical_ops_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/logical_ops_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[mandelbrot_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/mandelbrot_test.py) | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[math_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/math_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[matrix_ops_dynamic_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/matrix_ops_dynamic_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[matrix_ops_static_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/matrix_ops_static_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[range_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/range_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[resource_ops_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/resource_ops_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[ring_buffer_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/ring_buffer_test.py) | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[scatter_update_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/scatter_update_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[simple_arithmetic_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/simple_arithmetic_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[simple_stateful_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/simple_stateful_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[sliding_window_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/sliding_window_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[space_to_batch_nd_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/space_to_batch_nd_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[strings_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/strings_test.py) | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[tensorlist_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/tensorlist_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>