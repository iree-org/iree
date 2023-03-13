# IREE Performance Dashboard

This documentation explains IREE's performance dashboard (https://perf.iree.dev).
A [Buildkite pipeline](https://buildkite.com/iree/iree-benchmark) runs on each
commit to the `main` branch and posts those results to the dashboard.

## Benchmarking philosophy

Benchmarking and interpreting results properly is a delicate task. We can record
metrics from various parts of a system, but depending on what we are trying to
evaluate, those numbers may or may not be relevant. For example, for somebody
working solely on better kernel code generation, the end-to-end model reference
latency is unlikely meaningful given it also includes runtime overhead. The
environment could also vary per benchmark run in uncontrollable ways, causing
instability in the results. This is especially true for mobile and embedded
systems, where a tight compromise between performance and thermal/battery limits
is made. Too many aspects can affect the benchmarking results. So before going
into details, it's worth nothing the general guideline to IREE benchmarking as
context.

The overarching goal for benchmarking here is to track IREE's performance
progress and guard against regression. So the benchmarks are meant to understand
the performance of IREE _itself_, not the absolute capability of the exercised
hardware. In order to fulfill the above goal, we have the following guidelines
for benchmarking:

* We choose representative real-world models with varying characteristics.
* We cover different IREE backends and different modes for each backend so that
  folks working on different components can find the metrics they need.

## Model benchmark specification

Each benchmark in IREE has a unique identifier with the following format:

```
<model-name> `[` <model-tag>.. `]` `(` <model-source> `)` <benchmark-mode>..
`with` <iree-driver>
`@` <device-name> `(` <target-architecture> `)`
```

The following subsections explain possible choices in each field.

### Model source

This field specifies the original model source:

* `TFLite`: Models originally in TensorFlow Lite Flatbuffer format.

### Model name

This field specifies the input model:

* `DeepLabV3` [[source](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1)]:
  Vision model for semantic image segmentation.
  Characteristics: convolution, feedforward NN.
* `MobileBERT` [[source](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1)]:
  NLP for Q&A.
  Characteristics: matmul, attention, feedforward NN.
* `MobileNetV2` [[source](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/mobilenet_v2_1.0_224.tflite)]:
  Vision model for image classification.
  Characteristics: convolution, feedforward NN
* `MobileNetV3Small` [[source](https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5)]:
  Vision model for image classification.
  Characteristics: convolution, feedforward NN.
* `MobileSSD` [[source](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)]:
  Vision model for object detection.
  Characteristics: convolution, feedforward NN.
* `PoseNet` [[source](https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1)]:
  Vision model for pose estimation.
  Characteristics: convolution, feedforward NN.

### Model tag

This field specifies the model variant. It depends on the model, but here are
some examples:

* `f32`: the model is working on float types.
* `imagenet`: the model takes ImageNet-sized inputs (224x224x3).

### IREE driver

This field specifies the IREE HAL driver:

* [`local-task`](https://openxla.github.io/iree/deployment-configurations/cpu/):
  For CPU via the local task system. Kernels contain CPU native instructions AOT
  compiled using LLVM. This driver issues workloads to the CPU asynchronously
  and supports multithreading.
* [`local-sync`](https://openxla.github.io/iree/deployment-configurations/cpu/):
  For CPU via the local 'sync' device. Kernels contain contain CPU native
  instructions AOT compiled using LLVM. This driver issues workloads to the CPU
  synchronously.
* [`Vulkan`](https://openxla.github.io/iree/deployment-configurations/gpu-vulkan/):
  For GPU via Vulkan. Kernels contain SPIR-V. This driver issues workload to
  the GPU via the Vulkan API.

### Device name and target architecture

These two fields are tightly coupled. They specify the device and hardware
target for executing the benchmark.

Right now there are two Android devices:

* `Pixel-4`: Google Pixel 4 running Android 11. The SoC is
  [Snapdragon 855](https://www.qualcomm.com/products/snapdragon-855-plus-and-860-mobile-platform),
  with 1+3+4 ARMv8.2 CPU cores and Adreno 640 GPU.
* `Pixel-6`: Google Pixel 6 running Android 12. The SoC is
  [Google Tensor](https://blog.google/products/pixel/introducing-google-tensor/),
  with 2+2+4 ARMv8 CPU cores and Mali G78 GPU.
* `SM-G980F`: Samsung Galaxy S20 running Android 11. The SoC is
  [Exynos 990](https://www.samsung.com/semiconductor/minisite/exynos/products/mobileprocessor/exynos-990/),
  with 2+2+4 ARMv8.2 CPU cores and Mali G77 MP11 GPU.

Therefore the target architectures are:

* `CPU-CPU-ARMv8.2-A`: can benchmark all CPU-based IREE backends and drivers.
* `GPU-Adreno-640`: can benchmark IREE Vulkan with Adreno target triples.
* `GPU-Mali-G77`: can benchmark IREE Vulkan with Mali target triples.
* `GPU-Mali-G78`: can benchmark IREE Vulkan with Mali target triples.

### Benchmark mode

This field is to further specify the benchmark variant, given the same input
model and target architecture. It controls important aspects like:

* `*-core`: specifies the core flavor for CPU.
* `*-thread`: specifies the number of threads for CPU.
* `full-inference`: measures the latency for one full inference. Note that this
  does not include the IREE system initialization time.
* `kernel-execution`: measures only kernel execution latency for GPU. Note that
  this is only possible for feedforward NN models that can be put into one
  command buffer.

`*-core` and `*-thread` together determines the `taskset` mask used for
benchmarking IREE backends and drivers on CPU. For example,

* `1-thread,big-core` would mean `taskset 80`.
* `1-thread,little-core` would mean `taskset 08`.
* `3-thread,big-core` would mean `taskset f0`.
* `3-thread,little-core` would mean `taskset 0f`.
