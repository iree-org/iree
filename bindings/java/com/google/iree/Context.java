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

/** An isolated execution context. */
final class Context {
  public Context(Instance instance) throws Exception {
    nativeAddress = nativeNew();
    Status status = Status.fromCode(nativeCreate(instance.getNativeAddress()));

    if (!status.isOk()) {
      throw status.toException("Could not create Context");
    }
  }

  public int getId() {
    return nativeGetId();
  }

  public void free() {
    nativeFree();
  }

  private final long nativeAddress;

  private native long nativeNew();

  private native int nativeCreate(long instanceAddress);

  private native void nativeFree();

  private native int nativeGetId();
}
