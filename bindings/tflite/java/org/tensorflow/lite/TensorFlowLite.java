/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
