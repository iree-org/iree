/*
 * Copyright 2020 Google LLC
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
