/*
 * Copyright 2020 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package com.google.iree;

/**
 * Shared runtime instance responsible for routing Context events, enumerating and creating hardware
 * device interfaces, and managing device resource pools.
 */
final class Instance {
  /**
   * Loads the native IREE shared library. This must be called first before doing anything with the
   * IREE Java API.
   */
  public static void loadNativeLibrary() {
    System.loadLibrary("iree");
    loaded = true;
  }

  public Instance() throws Exception {
    if (!loaded) {
      throw new IllegalStateException("Native library is not loaded");
    }

    nativeAddress = nativeNew();
    Status status = Status.fromCode(nativeCreate());

    if (!status.isOk()) {
      throw status.toException("Could not create Instance");
    }
  }

  public long getNativeAddress() {
    return nativeAddress;
  }

  public void free() {
    nativeFree();
  }

  private static boolean loaded = false;

  private final long nativeAddress;

  private native long nativeNew();

  private native int nativeCreate();

  private native void nativeFree();
}
