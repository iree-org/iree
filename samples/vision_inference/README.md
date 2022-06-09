# Vision Inference Sample

This sample demonstrates how to run a MNIST handwritten digit detection vision
model on an image using IREE's command line tools.

## Instructions

From this directory:

```bash
# Compile the MNIST program.
iree-compile \
    ../models/mnist.mlir \
    --iree-input-type=mhlo \
    --iree-hal-target-backends=cpu \
    -o /tmp/mnist_cpu.vmfb

# Convert the test image to the 1x28x28x1xf32 buffer format the program expects.
cat mnist_test.png | python3 convert_to_float_grayscale.py > /tmp/mnist_test.bin

# Run the program, passing the path to the binary file as a function input.
iree-run-module \
  /tmp/mnist_test.bin \
  --function_input=1x28x28x1xf32=@/tmp/mnist_test.bin

# Observe the results - a list of prediction confidence scores for each digit.
```

<!-- TODO(scotttodd): lit test for that ^ (requires python in lit.cfg.py) -->
