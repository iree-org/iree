/*
 * Copyright 2021 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package org.tensorflow.lite;

import androidx.annotation.NonNull;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * Main driver class for the IREE Java compatibility shim. Provides model
 * creation and inference for IREE compatible TFLite models.
 *
 * <p>This shim aims to mimic the functionality of Tensorflow Lite's
 * Interpeter.java class, however, there are a few notable features IREE doesn't
 * support:
 *
 * <ul>
 *  <li> Delegates and the NNAPI
 *  <li> Advanced interpreter options
 *  <li> Interrupting or canceling inference before it's complete
 *  <li> Tensor method signatures
 *  <li> Dynamic shapes
 *  <li> String input/output isn't supported
 * <ul>
 *
 * <p>In addition, this compatibility shim only accepts {@link java.nio.Buffer}
 * type input. Consumers with scalar and array inputs can see conversion methods
 * below.
 *
 * <p>Example using the Interpreter with a model with a single input/output Tensor:
 *
 * <pre>{@code
 *  // Load model/initialize interpreter:
 *  ByteBuffer modelByteBuffer = ... load model here ....
 *  Interpreter interpreter = new Interpreter(modelByteBuffer);
 *  interpreter.allocateTensors();
 *
 *  // Prepare inputs:
 *  float[] input = {1, 3};
 *  float[] output = new float[2];
 *
 *  int bytesInFloat = 4;
 *  FloatBuffer inputBuffer = ByteBuffer.allocateDirect(bytesInFloat * input.length)
 *    .order(ByteOrder.nativeOrder())
 *    .asFloatBuffer();
 *  FloatBuffer outputBuffer = ByteBuffer.allocateDirect(bytesInFloat * output.length)
 *    .order(ByteOrder.nativeOrder())
 *    .asFloatBuffer();
 *
 *  // Run inference:
 *  inputBuffer.put(input);
 *  interpreter.run(inputBuffer, outputBuffer);
 *  outputBuffer.get(output);
 *  ... process output ...
 *
 *  // Cleanup:
 *  interpreter.close();
 * }</pre>
 *
 * <p>If a model takes multiple inputs, use
 *   {@link #runForMultipleInputsOutputs(Buffer[], Map)}:
 *
 * <pre>{@code
 *  // Load model/initialize interpreter same as above.
 *
 *  // Prepare inputs:
 *  Buffer[] inputs = {input0, input1, ...};
 *  Map<Integer, Buffer> indexToOutput = new HashMap<>();
 *  FloatBuffer ithOutput = ... allocate (native) and populate buffer ...
 *  indexToOutput.put(i, ithOutput);
 *
 *  // Run inference:
 *  interpreter.runForMultipleInputsOutputs(inputs, indexToOutput);
 *
 *  // Cleanup same as above.
 * }</pre>
 *
 * <p>Orders of inputs and outputs are determined when converting TensorFlow
 * model to TensorFlow Lite model with TOCO, as are the default shapes of the
 * inputs.
 *
 * <p><b>WARNING:</b>Instances of {@link Interpreter} are <b>not</b> thread-safe.
 * A {@link Interpreter} owns resources that <b>must</b> be explicitly freed by
 * invoking {@link #close()}
 */
public final class Interpreter implements AutoCloseable {
  private static final String TAG = Interpreter.class.getCanonicalName();

  /**
   * Options class for controlling runtime interpreter behavior.
   *
   * <p>Currently only {@link #setNumThreads(int)} is supported.
   */
  public static class Options {
    int numThreads = -1;

    public Options() {}

    /**
     * Sets the number of threads to be used for ops that support multi-threading.
     *
     * <p>{@code numThreads} should be >= -1. Setting {@code numThreads} to 0 has the effect to
     * disable multithreading, which is equivalent to setting {@code numThreads} to 1. If
     * unspecified, or set to the value -1, the number of threads used will be
     * implementation-defined and platform-dependent.
     */
    public Options setNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }
  }

  private final int inputTensorCount;
  private final int outputTensorCount;
  private final Tensor[] inputTensors;
  private final Tensor[] outputTensors;
  private final long nativeAddress;

  private long inferenceDurationNanoseconds;
  private boolean tensorsAllocated;

  /**
   * Initializes the Interpreter with model and options.
   *
   * @param modelByteBuffer: a directly allocated, native {@link java.nio.ByteOrder} byte buffer of
   *     an IREE compatible TFLite model.
   * @param options: options for the interpreter, or null (to use defaults).
   * @throws IllegalArgumentException if the model cannot be initialized in the Interpreter.
   */
  public Interpreter(@NonNull ByteBuffer modelByteBuffer, Options options)
      throws IllegalArgumentException {
    TensorFlowLite.init();
    if (options == null) {
      options = new Options();
      options.setNumThreads(2);
    }
    nativeAddress = nativeNew(modelByteBuffer, options.numThreads);

    if (nativeAddress == 0) {
      throw new IllegalArgumentException("Could not create Interpreter");
    }

    inputTensorCount = nativeInputTensorCount();
    outputTensorCount = nativeOutputTensorCount();
    inputTensors = new Tensor[inputTensorCount];
    outputTensors = new Tensor[outputTensorCount];
  }

  /**
   * Runs model inference with a single input/output pair.
   *
   * <p>This API is compatible with the following buffer types:
   *
   * <ul>
   *   <li>{@link ByteBuffer} - compatible with any underlying primitive Tensor type.
   *   <li>{@link java.nio.FloatBuffer} - compatible with float Tensors.
   *   <li>{@link java.nio.IntBuffer} - compatible with int32 Tensors.
   *   <li>{@link java.nio.LongBuffer} - compatible with int64 Tensors.
   * </ul>
   *
   * String input is currently not supported.
   *
   * <p>The API is most efficient with direct byte buffers. All buffers must be in native {@link
   * java.nio.ByteOrder}.
   *
   * <p>Scalar input should be wrapped in the appropriate buffer of capacity 1. Ex: {@code
   * inputFloatBuffer.put(1.0f);}.
   *
   * <pre>{@code
   * FloatBuffer inputBuffer = ... create float buffer of capacity 1 ...
   * }</pre>
   *
   * <p>Single dimensional arrays/tensor input should be wrapped directly in a buffer with matching
   * capacity. Ex: {@code inputIntBuffer.put(new int[4] {1, 2, 3, 4})};
   *
   * <p>Multidimensional array/tensor input should be flattened, iterating over each dimension up to
   * the nth dimension, writing values in order at the nth dimension. Thus, the following
   * multidimensional array:
   *
   * <pre>{@code
   * int[] = new int[3][2] { [1, 2],
   *                         [3, 4]
   *                         [5, 6] };
   *
   * }</pre>
   *
   * Is encoded with the following order: [ 1, 2, 3, 4, 5, 6].
   *
   * <p>The API will output single and multidimensional tensors in buffers with the encoding noted
   * above.
   *
   * <p>Note: that the number of elements in single/multi arrays should match the tensor's {@link
   * Tensor#numElements()} output, else inference will fail.
   *
   * @param input a {@link java.nio.Buffer} with correct capacity populated with tensor input.
   * @param output an empty {@link java.nio.Buffer} to be filled with tensor output. The caller must
   *     ensure that it is set to the appropriate write position and remaining capacity.
   * @throws IllegalArgumentException if {@code input} or {@code output} is null or empty (for input
   *     only), in the wrong format, or capacity.
   * @throws IllegalStateException if inference fails.
   */
  public void run(Buffer input, Buffer output) {
    Buffer[] inputs = {input};
    Map<Integer, Buffer> outputs = new HashMap<>();
    outputs.put(0, output);
    runForMultipleInputsOutputs(inputs, outputs);
  }

  /**
   * Runs model inference with multiple inputs/outputs.
   *
   * <p>See {@link #run(Buffer, Buffer)} for buffer encoding format.
   *
   * <p>Runs model inference if the model takes multiple inputs, or returns multiple outputs.
   *
   * @param inputs an array of input {@link java.nio.Buffer}s. The inputs should be in the same
   *     order as inputs of the model.
   * @param outputs a map mapping output indices to {@link java.nio.Buffer}s. The caller must ensure
   *     they are set to the appropriate write position and remaining capacity.
   * @throws IllegalArgumentException if {@code inputs} or {@code outputs} is null or empty, in the
   *     wrong format, or capacity.
   * @throws IllegalStateException if inference fails.
   */
  public void runForMultipleInputsOutputs(
      @NonNull Buffer[] inputs, @NonNull Map<Integer, Buffer> outputs) {
    if (inputs == null || inputs.length == 0) {
      throw new IllegalArgumentException("Input error: Inputs should not be null or empty.");
    }
    if (outputs == null || outputs.isEmpty()) {
      throw new IllegalArgumentException("Input error: Outputs should not be null or empty.");
    }

    if (!tensorsAllocated) {
      allocateTensors();
    }

    for (int i = 0; i < inputs.length; ++i) {
      getInputTensor(i).copyFromBuffer(inputs[i]);
    }

    long inferenceStartNanos = System.nanoTime();
    int status = nativeInvoke();
    inferenceDurationNanoseconds = System.nanoTime() - inferenceStartNanos;

    if (status != 0) {
      throw new IllegalStateException(
          String.format("Failed to run Interpreter. Returned status code: %d", status));
    }

    for (Map.Entry<Integer, Buffer> output : outputs.entrySet()) {
      getOutputTensor(output.getKey()).copyToBuffer(output.getValue());
    }
  }

  /**
   * Explicitly updates allocations for all tensors, if necessary.
   *
   * <p>This will propagate shapes and memory allocations for all dependent tensors using the input
   * tensor shape(s) as given.
   *
   * <p>Note: This call is *purely optional*. Tensor allocation will occur automatically during
   * execution if any input tensors have been resized. This call is most useful in determining the
   * shapes for any output tensors before executing the graph, e.g.,
   *
   * <pre>{@code
   * interpreter.resizeInput(0, new int[]{1, 4, 4, 3}));
   * interpreter.allocateTensors();
   * FloatBuffer input = FloatBuffer.allocate(interpreter.getInputTensor(0),numElements());
   * // Populate inputs...
   * FloatBuffer output = FloatBuffer.allocate(interpreter.getOutputTensor(0).numElements());
   * interpreter.run(input, output)
   * // Process outputs...
   * }</pre>
   *
   * @throws IllegalStateException if the graph's tensors could not be successfully allocated.
   */
  public void allocateTensors() {
    if (nativeAllocateTensors() != 0) {
      throw new IllegalStateException("Failed to allocate Tensors.");
    }
    tensorsAllocated = true;
  }

  /**
   * Resizes the specified input of the native model to the given dims.
   *
   * @param inputIndex index of input to resize
   * @param dims array specifying new shape
   * @throws IllegalArgumentException if {@code inputIndex} is negative or is not smaller than the
   *     number of model inputs; or if error occurs when resizing the specified input.
   */
  public void resizeInput(int inputIndex, @NonNull int[] dims) {
    if (nativeResizeInputTensor(inputIndex, dims) != 0) {
      throw new IllegalArgumentException("Unable to resize to input tensor.");
    }
    tensorsAllocated = false;
  }

  /** Gets the number of input tensors. */
  public int getInputTensorCount() {
    return inputTensorCount;
  }

  /**
   * Gets index of an input given the op name of the input.
   *
   * @throws IllegalArgumentException if {@code opName} does not match any input in the model used
   *     to initialize the {@link Interpreter}.
   */
  public int getInputIndex(String opName) {
    for (int i = 0; i < getInputTensorCount(); ++i) {
      if (getInputTensor(i).name().equals(opName)) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Gets the Tensor associated with the provdied input index.
   *
   * @throws IllegalArgumentException if {@code inputIndex} is negtive or is not smaller than the
   *     number of model inputs.
   */
  public Tensor getInputTensor(int index) {
    if (index < 0 || index >= inputTensors.length) {
      throw new IllegalArgumentException(String.format("Invalid input Tensor index: %d", index));
    }
    if (inputTensors[index] == null) {
      inputTensors[index] = Tensor.inputFromIndex(nativeAddress, index);
    }
    return inputTensors[index];
  }

  /** Gets the number of output Tensors. */
  public int getOutputTensorCount() {
    return outputTensorCount;
  }

  /** Gets index of an output given the op name of the output or -1 if not found. */
  public int getOutputIndex(String opName) {
    for (int i = 0; i < getOutputTensorCount(); ++i) {
      if (getOutputTensor(i).name().equals(opName)) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Gets the Tensor associated with the provdied output index.
   *
   * <p>Note: Output tensor details (e.g., shape) may not be fully populated until after inference
   * is executed. If you need updated details *before* running inference (e.g., after resizing an
   * input tensor, which may invalidate output tensor shapes), use {@link #allocateTensors()} to
   * explicitly trigger allocation and shape propagation. Note that, for graphs with output shapes
   * that are dependent on input *values*, the output shape may not be fully determined until
   * running inference.
   *
   * @throws IllegalArgumentException if {@code outputIndex} is negative or is not smaller than the
   *     number of model outputs.
   */
  public Tensor getOutputTensor(int index) {
    if (index < 0 || index >= outputTensors.length) {
      throw new IllegalArgumentException(String.format("Invalid output Tensor index: %d", index));
    }
    if (outputTensors[index] == null) {
      outputTensors[index] = Tensor.outputFromIndex(nativeAddress, index);
    }
    return outputTensors[index];
  }

  /** Returns native inference timing, or -1 if inference isn't complete yet. */
  public Long getLastNativeInferenceDurationNanoseconds() {
    return inferenceDurationNanoseconds;
  }

  /** Release resources associated with the {@code Interpreter}. */
  @Override
  public void close() {
    nativeFree();
  }

  private native long nativeNew(ByteBuffer modelByteBuffer, int numThreads);

  private native void nativeFree();

  private native int nativeInputTensorCount();

  private native int nativeOutputTensorCount();

  private native int nativeAllocateTensors();

  private native int nativeResizeInputTensor(int inputIndex, int[] dims);

  private native int nativeInvoke();
}
