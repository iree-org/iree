/*
 * Copyright 2020 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package com.google.iree;

import java.nio.FloatBuffer;
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

  public void invokeFunction(
      Function function, FloatBuffer[] inputs, int inputElementCount, FloatBuffer output)
      throws Exception {
    Status status =
        Status.fromCode(
            nativeInvokeFunction(function.getNativeAddress(), inputs, inputElementCount, output));
    if (!status.isOk()) {
      throw status.toException("Could not invoke function");
    }
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

  // TODO(jennik): 'output' should be a Floatbuffer[].
  private native int nativeInvokeFunction(
      long functionAddress, FloatBuffer[] inputs, int inputElementCount, FloatBuffer output);

  private native void nativeFree();

  private native int nativeGetId();
}
