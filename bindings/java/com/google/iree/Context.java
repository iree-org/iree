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

import java.util.List;

/** An isolated execution context. */
final class Context {
  public Context(Instance instance) throws Exception {
    isStatic = false;
    nativeAddress = nativeNew();
    Status status = Status.fromCode(nativeCreate(instance.getNativeAddress()));

    if (!status.isOk()) {
      throw status.toException("Could not create Context");
    }
  }

  // TODO(jennik): Consider using ImmutableList here.
  public Context(Instance instance, List<Module> modules) throws Exception {
    isStatic = true;
    nativeAddress = nativeNew();
    long[] moduleAdresses = getModuleAdresses(modules);
    Status status =
        Status.fromCode(nativeCreateWithModules(instance.getNativeAddress(), moduleAdresses));
    if (!status.isOk()) {
      throw status.toException("Could not create Context");
    }
  }

  public void registerModules(List<Module> modules) throws Exception {
    if (isStatic) {
      throw new IllegalStateException("Cannot register modules to a static context");
    }

    long[] moduleAdresses = getModuleAdresses(modules);
    Status status = Status.fromCode(nativeRegisterModules(moduleAdresses));
    if (!status.isOk()) {
      throw status.toException("Could not register Modules");
    }
  }

  public Function resolveFunction(String name) throws Exception {
    Function function = new Function();
    Status status = Status.fromCode(nativeResolveFunction(function.getNativeAddress(), name));
    if (!status.isOk()) {
      throw status.toException("Could not resolve function");
    }
    return function;
  }

  public int getId() {
    return nativeGetId();
  }

  public void free() {
    nativeFree();
  }

  private static long[] getModuleAdresses(List<Module> modules) {
    long[] moduleAddresses = new long[modules.size()];
    for (int i = 0; i < modules.size(); i++) {
      moduleAddresses[i] = modules.get(i).getNativeAddress();
    }
    return moduleAddresses;
  }

  private final long nativeAddress;

  private final boolean isStatic;

  private native long nativeNew();

  private native int nativeCreate(long instanceAddress);

  private native int nativeCreateWithModules(long instanceAddress, long[] moduleAddresses);

  private native int nativeRegisterModules(long[] moduleAddresses);

  private native int nativeResolveFunction(long functionAddress, String name);

  private native void nativeFree();

  private native int nativeGetId();
}
