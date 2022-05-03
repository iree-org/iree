/*
 * Copyright 2021 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package org.tensorflow.lite;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

/**
 * A typed multi-dimensional array used in the IREE Java comptability shim.
 *
 * <p>The native handle of a tensor is managed by {@link Interpreter}, and does not needed to be
 * closed by the client. However, once the {@link Interpreter} has been closed, the tensor will be
 * invalid.
 */
public final class Tensor {
  private static final String TAG = Tensor.class.getCanonicalName();

  static Tensor inputFromIndex(long nativeInterpreterHandle, int tensorIndex) {
    long nativeAddress = nativeCreateInput(nativeInterpreterHandle, tensorIndex);
    if (nativeAddress == 0) {
      throw new RuntimeException(String.format("Failed to create input tensor %d", tensorIndex));
    }
    return new Tensor(nativeAddress, tensorIndex);
  }

  static Tensor outputFromIndex(long nativeInterpreterHandle, int tensorIndex) {
    long nativeAddress = nativeCreateOutput(nativeInterpreterHandle, tensorIndex);
    if (nativeAddress == 0) {
      throw new RuntimeException(String.format("Failed to create output tensor %d", tensorIndex));
    }
    return new Tensor(nativeAddress, tensorIndex);
  }

  /**
   * Quantization parameters that corresponds to the table, {@code QuantizationParameters}, in the
   * <a
   * href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs">TFLite
   * Model schema file.</a>
   *
   * <p>Since per-channel quantization does not apply to input and output tensors, {@code scale} and
   * {@code zero_point} are both single values instead of arrays.
   *
   * <p>For tensor that are not quantized, the values of scale and zero_point are both 0.
   *
   * <p>Given a quantized value q, the corresponding float value f should be: <br>
   * f = scale * (q - zero_point) <br>
   */
  public static class QuantizationParams {
    /** The scale value used in quantization. */
    private final float scale;
    /** The zero point value used in quantization. */
    private final int zeroPoint;

    /**
     * Creates a {@link QuantizationParams} with {@code scale} and {@code zero_point}.
     *
     * @param scale The scale value used in quantization.
     * @param zeroPoint The zero point value used in quantization.
     */
    QuantizationParams(final float scale, final int zeroPoint) {
      this.scale = scale;
      this.zeroPoint = zeroPoint;
    }

    /** Returns the scale value. */
    public float getScale() {
      return scale;
    }

    /** Returns the zero point value. */
    public int getZeroPoint() {
      return zeroPoint;
    }
  }

  /** Returns the {@link DataType} of elements stored in the Tensor. */
  public DataType dataType() {
    return DataType.fromC(nativeType());
  }

  /**
   * Returns the number of dimensions (sometimes referred to as <a
   * href="https://www.tensorflow.org/resources/dims_types.html#rank">rank</a>) of the tensor.
   *
   * <p>Will be 0 for a scalar, 1 for a vector, 2 for a matrix, 3 for a 3-dimensional tensor etc.
   */
  public int numDimensions() {
    return nativeNumDims();
  }

  /** Returns the size, in bytes, of the tensor data. */
  public int numBytes() {
    return nativeBytesSize();
  }

  /** Returns the number of elements in a flattened (1-D) view of the tensor. */
  public int numElements() {
    int[] shape = shape();
    int n = 1;
    for (int i = 0; i < shape.length; ++i) {
      n *= shape[i];
    }
    return n;
  }

  /**
   * Returns the <a href="https://www.tensorflow.org/resources/dims_types.html#shape">shape</a> of
   * the Tensor, i.e., the sizes of each dimension.
   *
   * @return an array where the i-th element is the size of the i-th dimension of the tensor.
   */
  public int[] shape() {
    int[] shape = new int[nativeNumDims()];
    for (int i = 0; i < shape.length; ++i) {
      shape[i] = nativeDim(i);
    }
    return shape;
  }

  /**
   * Returns the original <a
   * href="https://www.tensorflow.org/resources/dims_types.html#shape">shape</a> of the Tensor,
   * i.e., the sizes of each dimension - before any resizing was performed. Unknown dimensions are
   * designated with a value of -1.
   *
   * @return an array where the i-th element is the size of the i-th dimension of the tensor.
   */
  public int[] shapeSignature() {
    return shapeSignature;
  }

  /**
   * Returns the index of the tensor within the owning {@link Interpreter}. Note: both input and
   * output tensors indexed starting at 0.
   */
  public int index() {
    return tensorIndex;
  }

  /** Returns the name of the tensor within the owning {@link Interpreter}. */
  public String name() {
    return nativeName();
  }

  /**
   * Returns the quantization parameters of the tensor.
   *
   * <p>Only quantized tensors have valid {@code QuantizationParameters}. For tensor that are not
   * quantized, the values of scale and zero_point are both 0.
   */
  public QuantizationParams quantizationParams() {
    return quantizationParams;
  }

  void copyFromBuffer(Buffer inputBuffer) {
    checkBufferCapacity(inputBuffer);
    if (isDirectBuffer(inputBuffer)) {
      copyFromDirectBuffer(inputBuffer);
    } else {
      if (inputBuffer instanceof ByteBuffer) {
        ((ByteBuffer) inputBuffer).put(getNativeBuffer());
      } else if (inputBuffer instanceof FloatBuffer) {
        ((FloatBuffer) inputBuffer).put(getNativeBuffer().asFloatBuffer());
      } else if (inputBuffer instanceof IntBuffer) {
        ((IntBuffer) inputBuffer).put(getNativeBuffer().asIntBuffer());
      } else if (inputBuffer instanceof LongBuffer) {
        ((LongBuffer) inputBuffer).put(getNativeBuffer().asLongBuffer());
      } else {
        throw new IllegalArgumentException(
            "Unexpected input buffer type: " + inputBuffer.getClass());
      }
    }
  }

  void copyToBuffer(Buffer outputBuffer) {
    checkBufferCapacity(outputBuffer);
    if (isDirectBuffer(outputBuffer)) {
      copyToDirectBuffer(outputBuffer);
    } else {
      if (outputBuffer instanceof ByteBuffer) {
        getNativeBuffer().put((ByteBuffer) outputBuffer);
      } else if (outputBuffer instanceof FloatBuffer) {
        getNativeBuffer().asFloatBuffer().put((FloatBuffer) outputBuffer);
      } else if (outputBuffer instanceof IntBuffer) {
        getNativeBuffer().asIntBuffer().put((IntBuffer) outputBuffer);
      } else if (outputBuffer instanceof LongBuffer) {
        getNativeBuffer().asLongBuffer().put((LongBuffer) outputBuffer);
      } else {
        throw new IllegalArgumentException(
            "Unexpected output buffer type: " + outputBuffer.getClass());
      }
    }
  }

  private boolean isDirectBuffer(Buffer object) {
    if (object instanceof ByteBuffer) {
      ByteBuffer buffer = (ByteBuffer) object;
      return buffer.isDirect();
    }
    if (object instanceof LongBuffer) {
      LongBuffer buffer = (LongBuffer) object;
      return buffer.isDirect();
    }
    if (object instanceof FloatBuffer) {
      FloatBuffer buffer = (FloatBuffer) object;
      return buffer.isDirect();
    }
    if (object instanceof IntBuffer) {
      IntBuffer buffer = (IntBuffer) object;
      return buffer.isDirect();
    }
    return false;
  }

  private void checkBufferCapacity(Buffer otherBuffer) {
    int numBytes = numBytes();
    int otherBytes = otherBuffer.capacity();
    // Non ByteBuffers report capacity based on the number of elements rather raw bytes.
    if (otherBuffer instanceof LongBuffer) {
      otherBytes *= 8;
    }
    if (otherBuffer instanceof FloatBuffer) {
      otherBytes *= 4;
    }
    if (otherBuffer instanceof IntBuffer) {
      otherBytes *= 4;
    }

    if (numBytes != otherBytes) {
      throw new IllegalArgumentException(String.format(
          "Capacity of buffer does not match Tensor(%d). Expected %d bytes, got %d bytes",
          tensorIndex, numBytes, otherBytes));
    }
  }

  private void copyFromDirectBuffer(Buffer inputBuffer) {
    int statusCode = nativeCopyFromDirectBuffer(inputBuffer);
    if (statusCode != 0) {
      throw new IllegalArgumentException(
          String.format("Unable to write buffer data for input tensor(%d). Return code: %d",
              tensorIndex, statusCode));
    }
  }

  private void copyToDirectBuffer(Buffer outputBuffer) {
    int statusCode = nativeCopyToDirectBuffer(outputBuffer);
    if (statusCode != 0) {
      throw new IllegalArgumentException(
          String.format("Unable to write buffer data for output tensor(%d). Return code: %d",
              tensorIndex, statusCode));
    }
  }

  private ByteBuffer getNativeBuffer() {
    return nativeGetByteBuffer().order(ByteOrder.nativeOrder());
  }

  private final long nativeAddress;
  private final int tensorIndex;
  private final QuantizationParams quantizationParams;
  private final int shapeSignature[];

  private Tensor(long nativeAddress, int tensorIndex) {
    this.nativeAddress = nativeAddress;
    this.tensorIndex = tensorIndex;
    this.quantizationParams =
        new QuantizationParams(nativeQuantizationScale(), nativeQuantizationZeroPoint());
    this.shapeSignature = shape();
  }

  private static native long nativeCreateInput(long interpreterAddress, int inputIndex);

  private static native long nativeCreateOutput(long interpreterAddress, int outputIndex);

  private native int nativeType();

  private native int nativeNumDims();

  private native int nativeDim(int dimIndex);

  private native int nativeBytesSize();

  private native String nativeName();

  private native float nativeQuantizationScale();

  private native int nativeQuantizationZeroPoint();

  private native int nativeCopyFromDirectBuffer(Buffer inputByteBuffer);

  private native int nativeCopyToDirectBuffer(Buffer outputByteBuffer);

  private native ByteBuffer nativeGetByteBuffer();
}
