/*
 * Copyright 2020 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package com.google.iree;

import android.util.Log;
import java.nio.ByteBuffer;

/** A VM module. */
// TODO(jennik): Add BytecodeModule and HALModule classes.
final class Module {
  private static final String TAG = Module.class.getCanonicalName();

  private static class Signature {
    int importFunctionCount;
    int exportFunctionCount;
    int internalFunctionCount;

    public Signature(int importFunctionCount, int exportFunctionCount, int internalFunctionCount) {
      this.importFunctionCount = importFunctionCount;
      this.exportFunctionCount = exportFunctionCount;
      this.internalFunctionCount = internalFunctionCount;
    }
  }

  /**
   * Creates a VM module from a flatbuffer. The input ByteBuffer must be direct, and acceptable
   * schemas for the flatbuffer are available at iree/schemas
   */
  public Module(ByteBuffer flatbufferData) throws Exception {
    nativeAddress = nativeNew();
    Status status = Status.fromCode(nativeCreate(flatbufferData));

    if (!status.isOk()) {
      throw status.toException("Could not create Module");
    }
  }

  public long getNativeAddress() {
    return nativeAddress;
  }

  public String getName() {
    return nativeGetName();
  }

  public Signature getSignature() {
    return nativeGetSignature();
  }

  public String getDebugString() {
    String name = getName();
    Signature signature = getSignature();
    String debugString = String.format("Module debug string\n"
            + "-Name: %s\n"
            + "-Import function count: %d\n"
            + "-Export function count: %d\n"
            + "-Internal function count: %d",
        name, signature.importFunctionCount, signature.exportFunctionCount,
        signature.internalFunctionCount);
    return debugString;
  }

  public void printDebugString() {
    Log.d(TAG, getDebugString());
  }

  public void free() {
    nativeFree();
  }

  private final long nativeAddress;

  private native long nativeNew();

  private native int nativeCreate(ByteBuffer flatbufferData);

  private native String nativeGetName();

  private native Signature nativeGetSignature();

  private native void nativeFree();
}
