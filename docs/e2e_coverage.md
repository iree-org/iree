---
layout: default
permalink: TensorFlowE2ECoverage
title: TensorFlow E2E Coverage
nav_order: 7
js_files: 
- js/add_classes.js
---

# TensorFlow End to End Coverage
{: .no_toc }
There are three backend [targets](https://github.com/google/iree/tree/main/iree/compiler/Dialect/HAL/Target) in IREE:

- vmla
- llvm-ir
- vulkan-spirv

The table shows the supported TensorFlow functions and models on each backend.

## End to end TensorFlow tests

target | tensorflow | vmla | llvm-ir | vulkan-spirv
:-: | :-: | :-: | :-: | :-:
[batch_norm_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/batch_norm_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[broadcasting_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/broadcasting_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[control_flow_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/control_flow_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[conv_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/conv_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[depth_conv_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/depth_conv_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[dynamic_mlp_relu_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/dynamic_mlp_relu_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[dynamic_mlp_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/dynamic_mlp_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[exported_names_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/exported_names_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[fill_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/fill_test.py) | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[gather_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/gather_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[mandelbrot_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/mandelbrot_test.py) | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[math_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/math_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[matrix_ops_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/matrix_ops_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[resource_ops_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/resource_ops_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[ring_buffer_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/ring_buffer_test.py) | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[simple_arithmetic_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/simple_arithmetic_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[simple_stateful_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/simple_stateful_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[sliding_window_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/sliding_window_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[strings_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/strings_test.py) | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[tensorlist_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/tensorlist_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span>

## End to end tests written using tf.keras

target | tensorflow | vmla | llvm-ir | vulkan-spirv
:-: | :-: | :-: | :-: | :-:
[lstm_static_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/lstm_static_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[lstm_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/lstm_test.py) | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>

## End to end tests of tf.keras.applications vision models

target | tensorflow | vmla | llvm-ir | vulkan-spirv
:-: | :-: | :-: | :-: | :-:
[MobileNetV2_cifar10](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[MobileNetV2_imagenet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[MobileNet_cifar10](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[MobileNet_imagenet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[ResNet50_cifar10](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[ResNet50_imagenet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>