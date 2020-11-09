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

IREE has three backend
[targets](https://github.com/google/iree/tree/main/iree/compiler/Dialect/HAL/Target):
`vmla`, `llvm-ir` and `vulkan-spirv`. We also test TFLite in our infrastructure
for benchmarking purposes. The coverage tables below are automatically generated
from IREE's test suites.

## End to end tests of tf.keras.applications vision models on Imagenet

target | tflite | vmla | llvm-ir | vulkan-spirv
:-: | :-: | :-: | :-: | :-:
[DenseNet121](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[DenseNet169](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[DenseNet201](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[InceptionResNetV2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span>
[InceptionV3](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span>
[MobileNet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[MobileNetV2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[NASNetLarge](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[NASNetMobile](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[ResNet101](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[ResNet101V2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[ResNet152](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[ResNet152V2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[ResNet50](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[ResNet50V2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[VGG16](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[VGG19](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[Xception](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>

## End to end tests of TensorFlow slim vision models

target | tflite | vmla | llvm-ir | vulkan-spirv
:-: | :-: | :-: | :-: | :-:
[inception_resnet_v2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span>
[inception_v1](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span>
[inception_v2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span>
[inception_v3](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span>
[nasnet_large](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[nasnet_mobile](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[pnasnet_large](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[resnet_v1_101](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[resnet_v1_152](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[resnet_v1_50](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span>
[resnet_v2_101](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[resnet_v2_152](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>
[resnet_v2_50](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/slim_vision_models/slim_vision_model_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span>