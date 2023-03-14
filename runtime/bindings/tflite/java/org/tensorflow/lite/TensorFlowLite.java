/*
 * Copyright 2021 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package org.tensorflow.lite;

/** Static utility methods loading the TensorFlowLite runtime. */
public final class TensorFlowLite {
  private static final Throwable LOAD_LIBRARY_EXCEPTION;
  private static volatile boolean isInitialized = false;

  static {
    Throwable loadLibraryException = null;
    try {
      System.loadLibrary("iree_bindings_tflite");
    } catch (UnsatisfiedLinkError e) {
      loadLibraryException = e;
    }
    LOAD_LIBRARY_EXCEPTION = loadLibraryException;
  }

  private TensorFlowLite() {}

  /** Returns the version of the underlying TensorFlowLite runtime. */
  public static String runtimeVersion() {
    init();
    return nativeRuntimeVersion();
  }

  /**
   * Ensure the TensorFlowLite native library has been loaded.
   *
   * @throws UnsatisfiedLinkError with the appropriate error message.
   */
  public static void init() {
    if (isInitialized) {
      return;
    }

    try {
      // Try to invoke a native method (the method itself doesn't really matter) to ensure that
      // native libs are available.
      nativeRuntimeVersion();
      isInitialized = true;
    } catch (UnsatisfiedLinkError e) {
      // Prefer logging the original library loading exception if native methods are unavailable.
      Throwable exceptionToLog = LOAD_LIBRARY_EXCEPTION != null ? LOAD_LIBRARY_EXCEPTION : e;
      throw new UnsatisfiedLinkError("Failed to load native TensorFlow Lite methods. Check "
          + "that the correct native libraries are present, and, if using "
          + "a custom native library, have been properly loaded via System.loadLibrary():\n  "
          + exceptionToLog);
    }
  }

  private static native String nativeRuntimeVersion();
}
