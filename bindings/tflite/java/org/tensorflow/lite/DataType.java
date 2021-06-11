/*
 * Copyright 2021 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package org.tensorflow.lite;

/** Represents the type of elements in a TensorFlow Lite {@link Tensor} as an enum. */
public enum DataType {
  /** 32-bit single precision floating point. */
  FLOAT32(1),

  /** 32-bit signed integer. */
  INT32(2),

  /** 8-bit unsigned integer. */
  UINT8(3),

  /** 64-bit signed integer. */
  INT64(4),

  /** Strings. */
  STRING(5),

  /** Bool. */
  BOOL(6),

  /** 8-bit signed integer. */
  INT8(9);

  private final int value;

  DataType(int value) {
    this.value = value;
  }

  /** Returns the size of an element of this type, in bytes, or -1 if element size is variable. */
  public int byteSize() {
    switch (this) {
      case FLOAT32:
      case INT32:
        return 4;
      case INT8:
      case UINT8:
        return 1;
      case INT64:
        return 8;
      case BOOL:
        // Boolean size is JVM-dependent.
        return -1;
      case STRING:
        return -1;
    }
    throw new IllegalArgumentException(
        "DataType error: DataType " + this + " is not supported yet");
  }

  /** Converts a C TfLiteType enum value to the corresponding type. */
  static DataType fromC(int c) {
    for (DataType dataType : DataType.values()) {
      if (dataType.value == c) {
        return dataType;
      }
    }
    throw new IllegalArgumentException("DataType error: DataType " + c
        + " is not recognized in Java (version " + TensorFlowLite.runtimeVersion() + ")");
  }
}
