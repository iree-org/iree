/*
 * Copyright 2020 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package com.google.iree;

import android.util.Log;

/** A function reference. */
final class Function {
  private static final String TAG = Function.class.getCanonicalName();

  public static class Signature {
    String callingConvention;

    private Signature(String callingConvention) {
      this.callingConvention = callingConvention;
    }
  }

  public Function() {
    nativeAddress = nativeNew();
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
    String debugString = String.format("Function debug string\n"
            + "-Name: %s\n"
            + "-Calling convention: %s\n",
        name, signature.callingConvention);
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

  private native String nativeGetName();

  private native Signature nativeGetSignature();

  private native void nativeFree();
}
