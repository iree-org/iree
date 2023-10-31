---
date: 2021-07-19
authors:
  - rsuderman
  - jennik
categories:
  - Frontends
tags:
  - TensorFlow
---

# TFLite support via TOSA

IREE can now execute [TensorFlow Lite](https://www.tensorflow.org/lite)
(TFLite) models through the use of
[TOSA](https://developer.mlplatform.org/w/tosa/), an open standard of common
tensor operations, and a part of [MLIR](https://mlir.llvm.org/) core. TOSA’s
high-level representation of tensor operations provides a common front-end for
ingesting models from different frameworks. In this case we ingest a TFLite
FlatBuffer and compile it to TOSA IR, which IREE takes as an input format to
compile to its various backends.

<!-- more -->

![Compilation diagram](./tflite-tosa-compilation-diagram.png){align=left}

Using TFLite as a frontend for IREE provides an alternative ingestion method for
already existing models that could benefit from IREE’s design. This enables
models already designed for on-device inference to have an alternative path for
execution without requiring any additional porting, while benefiting from
IREE’s improvements in buffer management, work dispatch system, and compact
binary format. With continued improvements to IREE/MLIR’s compilation
performance, more optimized versions can be compiled and distributed to target
devices without an update to the clientside environment.

Today, we have validated floating point support for a variety of models,
including
[mobilenet](https://tfhub.dev/s?deployment-format=lite&network-architecture=mobilenet,mobilenet-v2,mobilenet-v3,mobilenet-v1&q=mobilenet)
(v1, v2, and v3) and
[mobilebert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1).
More work is in progress to support fully quantized models, and TFLite’s hybrid
quantization, along with dynamic shape support.

## Examples

TFLite with IREE is available in Python and Java.  We have a
[colab notebook](https://colab.research.google.com/github/openxla/iree/blob/main/samples/colab/tflite_text_classification.ipynb)
that shows how to use IREE’s python bindings and TFLite compiler tools to
compile a pre-trained TFLite model from a FlatBuffer and run using IREE.  We
also have an
[Android Java app](https://github.com/not-jenni/iree-android-tflite-demo) that
was forked from an existing TFLite demo app, swapping out the TFLite library
for our own AAR.  More information on IREE’s TFLite frontend is available
[here](../../../guides/ml-frameworks/tflite.md).
