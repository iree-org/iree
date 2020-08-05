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

import android.util.Log;

/** A function reference. */
final class Function {
  private static final String TAG = Function.class.getCanonicalName();

  public static class Signature {
    int argumentCount;
    int resultCount;

    private Signature(int argumentCount, int resultCount) {
      this.argumentCount = argumentCount;
      this.resultCount = resultCount;
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
            + "-Argument count: %d\n"
            + "-Result count: %d",
        name, signature.argumentCount, signature.resultCount);
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
