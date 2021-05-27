/*
 * Copyright 2020 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package com.google.iree;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.fail;

import android.content.Context;
import android.content.res.Resources;
import android.util.Log;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class IntegrationTest {
  private static final String TAG = IntegrationTest.class.getCanonicalName();

  @Test
  public void throwsExceptionWithoutNativeLib() throws Exception {
    try {
      new Instance();
      fail();
    } catch (IllegalStateException expected) {
    }
  }

  @Test
  public void simpleMulWithStaticContext() throws Exception {
    Instance.loadNativeLibrary();
    Instance instance = new Instance();

    Context context = ApplicationProvider.getApplicationContext();
    Resources resources = context.getResources();
    InputStream moduleInputStream = resources.openRawResource(R.raw.simple_mul_bytecode_module);
    ByteBuffer moduleByteBuffer = convertInputStreamToByteBuffer(moduleInputStream);
    Module module = new Module(moduleByteBuffer);
    module.printDebugString();

    List<Module> modules = new ArrayList<>();
    modules.add(module);
    com.google.iree.Context ireeContext = new com.google.iree.Context(instance, modules);

    assertNotEquals(ireeContext.getId(), -1);

    String functionName = "module.simple_mul";
    Function function = ireeContext.resolveFunction(functionName);
    function.printDebugString();

    int elementCount = 4;
    FloatBuffer x = ByteBuffer.allocateDirect(elementCount * /*sizeof(float)=*/4)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                        .put(new float[] {4.0f, 4.0f, 4.0f, 4.0f});
    FloatBuffer y = ByteBuffer.allocateDirect(elementCount * /*sizeof(float)=*/4)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                        .put(new float[] {2.0f, 2.0f, 2.0f, 2.0f});
    FloatBuffer[] inputs = {x, y};

    // TODO(jennik): Allocate outputs in C++ rather than here.
    FloatBuffer outputBuffer = ByteBuffer.allocateDirect(elementCount * /*sizeof(float)=*/4)
                                   .order(ByteOrder.nativeOrder())
                                   .asFloatBuffer()
                                   .put(new float[] {1.0f, 2.0f, 3.0f, 4.0f});
    ireeContext.invokeFunction(function, inputs, elementCount, outputBuffer);

    float[] output = new float[elementCount];
    outputBuffer.position(0);
    outputBuffer.get(output);
    Log.d(TAG, "Output: " + Arrays.toString(output));
    assertArrayEquals(new float[] {8.0f, 8.0f, 8.0f, 8.0f}, output, 0.1f);

    function.free();
    module.free();
    ireeContext.free();
    instance.free();
  }

  @Test
  public void simpleMulWithDynamicContext() throws Exception {
    Instance.loadNativeLibrary();
    Instance instance = new Instance();
    com.google.iree.Context ireeContext = new com.google.iree.Context(instance);

    assertNotEquals(ireeContext.getId(), -1);

    Context context = ApplicationProvider.getApplicationContext();
    Resources resources = context.getResources();
    InputStream moduleInputStream = resources.openRawResource(R.raw.simple_mul_bytecode_module);
    ByteBuffer moduleByteBuffer = convertInputStreamToByteBuffer(moduleInputStream);
    Module module = new Module(moduleByteBuffer);
    module.printDebugString();

    List<Module> modules = new ArrayList<>();
    modules.add(module);
    ireeContext.registerModules(modules);

    String functionName = "module.simple_mul";
    Function function = ireeContext.resolveFunction(functionName);
    function.printDebugString();

    int elementCount = 4;
    FloatBuffer x = ByteBuffer.allocateDirect(elementCount * /*sizeof(float)=*/4)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                        .put(new float[] {4.0f, 4.0f, 4.0f, 4.0f});
    FloatBuffer y = ByteBuffer.allocateDirect(elementCount * /*sizeof(float)=*/4)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                        .put(new float[] {2.0f, 2.0f, 2.0f, 2.0f});
    FloatBuffer[] inputs = {x, y};

    FloatBuffer outputBuffer = ByteBuffer.allocateDirect(elementCount * /*sizeof(float)=*/4)
                                   .order(ByteOrder.nativeOrder())
                                   .asFloatBuffer()
                                   .put(new float[] {1.0f, 2.0f, 3.0f, 4.0f});
    ireeContext.invokeFunction(function, inputs, elementCount, outputBuffer);

    float[] output = new float[elementCount];
    outputBuffer.position(0);
    outputBuffer.get(output);
    Log.d(TAG, "Output: " + Arrays.toString(output));
    assertArrayEquals(new float[] {8.0f, 8.0f, 8.0f, 8.0f}, output, 0.1f);

    function.free();
    module.free();
    ireeContext.free();
    instance.free();
  }

  private static ByteBuffer convertInputStreamToByteBuffer(InputStream inputStream)
      throws IOException {
    byte[] bytes = IOUtils.toByteArray(inputStream);
    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bytes.length);
    byteBuffer.put(bytes, 0, bytes.length);
    return byteBuffer;
  }
}
