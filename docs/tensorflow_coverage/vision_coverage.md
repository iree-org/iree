---
layout: default
permalink: tensorflow-coverage/vision-coverage
title: "Vision Models"
nav_order: 4
parent: TensorFlow Coverage
js_files: 
- js/add_classes.js
---

# Vision Models
{: .no_toc }

Tests of Keras and Slim vision models.

IREE has three main backend
[targets](https://github.com/google/iree/tree/main/iree/compiler/Dialect/HAL/Target):
`vmla` , `llvm` and `vulkan-spirv`. We also test TFLite in our infrastructure
for benchmarking purposes.

*Last Updated: 2020/12/8*

## End to end tests of tf.keras.applications vision models on Imagenet

|                                                               target                                                              |                    tflite                    |                     vmla                     |                 vulkan-spirv                 |
|:---------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
|    [DenseNet121](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [DenseNet169](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [DenseNet201](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [EfficientNetB0](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)  | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [EfficientNetB1](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)  | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [EfficientNetB2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)  | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [EfficientNetB3](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)  | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [EfficientNetB4](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)  | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [EfficientNetB5](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)  | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [EfficientNetB6](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)  | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [EfficientNetB7](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)  | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
| [InceptionResNetV2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|    [InceptionV3](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|     [MobileNet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)     | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [MobileNetV2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|  [MobileNetV3Large](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|  [MobileNetV3Small](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [NASNetLarge](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> |
|    [NASNetMobile](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)   | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> |
|     [ResNet101](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)     | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [ResNet101V2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|     [ResNet152](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)     | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [ResNet152V2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|      [ResNet50](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)     | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|     [ResNet50V2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|       [VGG16](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)       | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|       [VGG19](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/applications/vision_model_test.py)       | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |

## End to end tests of TensorFlow slim vision models

|                                                                  target                                                                  |                    tflite                    |                     vmla                     |                 vulkan-spirv                 |
|:----------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
| [inception_resnet_v2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|     [inception_v1](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|     [inception_v2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|     [inception_v3](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|     [nasnet_large](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> |
|    [nasnet_mobile](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> |
|    [pnasnet_large](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|    [resnet_v1_101](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [resnet_v1_152](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|     [resnet_v1_50](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [resnet_v2_101](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|    [resnet_v2_152](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |
|     [resnet_v2_50](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> |