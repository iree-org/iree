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
   * schemas for the flatbuffer are available at https://github.com/google/iree/tree/main/iree/schemas.
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
