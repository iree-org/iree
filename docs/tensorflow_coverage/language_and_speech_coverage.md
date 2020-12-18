---
layout: default
permalink: tensorflow-coverage/language-and-speech-coverage
title: "Language and Speech Models"
nav_order: 3
parent: TensorFlow Coverage
js_files: 
- js/add_classes.js
---

# Language and Speech Models
{: .no_toc }

Tests of MobileBert and streamable Keyword Spotting models.

IREE has three main backend
[targets](https://github.com/google/iree/tree/main/iree/compiler/Dialect/HAL/Target):
`vmla` , `llvm` and `vulkan-spirv`. We also test TFLite in our infrastructure
for benchmarking purposes.

*Last Updated: 2020/12/8*

## End to end test of MobileBert on SQuAD

|                                                          target                                                          |                    tflite                    |                     vmla                     |                 vulkan-spirv                 |
|:------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
| [mobile_bert_squad_test](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/mobile_bert_squad_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |

## End to end tests of [Keyword Spotting Streaming](https://github.com/google-research/google-research/tree/master/kws_streaming) models

|                                                               target                                                              |                    tflite                    |                     vmla                     |                 vulkan-spirv                 |
|:---------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
|    [att_mh_rnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)    | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> |
|      [att_rnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)     | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="failure-table-element">✗</span> |
|        [cnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)       | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [cnn_stride](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|       [crnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)       | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> |
|        [dnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)       | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|      [ds_cnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)      | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [ds_tc_resnet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)   | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|        [gru](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)       | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> |
|     [inception](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
| [inception_resnet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py) | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|       [lstm](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)       | <span class="success-table-element">✓</span> | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> |
|     [mobilenet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|   [mobilenet_v2](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)   | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|       [svdf](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)       | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|    [svdf_resnet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)   | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|     [tc_resnet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)    | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|     [xception](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)     | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |

## End to end tests of [Keyword Spotting Streaming](https://github.com/google-research/google-research/tree/master/kws_streaming) models in internal streaming mode

|                                                             target                                                            |                    tflite                    |                     vmla                     |                 vulkan-spirv                 |
|:-----------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
|      [cnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)     | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|  [cnn_stride](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)  | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|     [crnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)     | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|      [dnn](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)     | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
| [ds_tc_resnet](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py) | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|      [gru](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)     | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|     [lstm](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)     | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |
|     [svdf](https://github.com/google/iree/tree/main/integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py)     | <span class="failure-table-element">✗</span> | <span class="success-table-element">✓</span> | <span class="success-table-element">✓</span> |