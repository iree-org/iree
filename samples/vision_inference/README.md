# Vision Inference Sample

This sample demonstrates how to run a MNIST handwritten digit detection vision
model on an image using IREE's command line tools.

A similar sample is implemented in C code over in the iree-samples repository
at https://github.com/iree-org/iree-samples/tree/main/cpp/vision_inference

* This version of the sample uses a Python script to convert an image into the
  expected format then runs the compiled MNIST program through IREE's command
  line tools
* The other version uses a C library to decode and pre-process an image then
  uses IREE's C API to load the compiled program and run it on the image

## Instructions

From this directory:

```bash
# Compile the MNIST program.
iree-compile \
    ../models/mnist.mlir \
    --iree-input-type=mhlo_legacy \
    --iree-hal-target-backends=llvm-cpu \
    -o /tmp/mnist_cpu.vmfb

# Convert the test image to the 1x28x28x1xf32 buffer format the program expects.
cat mnist_test.png | python3 convert_image.py > /tmp/mnist_test.bin

# Run the program, passing the path to the binary file as a function input.
iree-run-module \
  --module=/tmp/mnist_cpu.vmfb \
  --function=predict \
  --input=1x28x28x1xf32=@/tmp/mnist_test.bin

# Observe the results - a list of prediction confidence scores for each digit.
```

<!-- TODO(scotttodd): lit test for that ^ (requires python in lit.cfg.py) -->
