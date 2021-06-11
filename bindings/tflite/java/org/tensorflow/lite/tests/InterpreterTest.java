/*
 * Copyright 2021 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package org.tensorflow.lite;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import android.content.Context;
import android.content.res.Resources;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.Interpreter.Options;

@RunWith(AndroidJUnit4.class)
public final class InterpreterTest {
  private static final String TAG = InterpreterTest.class.getCanonicalName();

  private static final float EPSILON = 0.0001f;

  private static final int BYTES_IN_FLOAT = 4;

  @Test
  public void test() throws Exception {
    Context context = ApplicationProvider.getApplicationContext();
    Resources resources = context.getResources();
    InputStream moduleInputStream = resources.openRawResource(R.raw.simple_add_bytecode_module);
    ByteBuffer moduleByteBuffer = convertInputStreamToByteBuffer(moduleInputStream);

    Options options = new Options();
    options.setNumThreads(2);
    try (Interpreter interpreter = new Interpreter(moduleByteBuffer, options)) {
      assertEquals(1, interpreter.getInputTensorCount());
      assertEquals(1, interpreter.getOutputTensorCount());
      interpreter.allocateTensors();
      Tensor inputTensor = interpreter.getInputTensor(0);
      assertEquals(DataType.FLOAT32, inputTensor.dataType());
      assertEquals(1, inputTensor.numDimensions());
      assertEquals(8, inputTensor.numBytes());
      assertEquals(2, inputTensor.numElements());
      assertArrayEquals(new int[] {2}, inputTensor.shape());
      float[] input = {1, 3};
      float[] output = new float[2];
      FloatBuffer inputBuffer = allocateNativeFloatBuffer(input.length);
      inputBuffer.put(input);
      FloatBuffer outputBuffer = allocateNativeFloatBuffer(output.length);
      interpreter.run(inputBuffer, outputBuffer);
      Tensor outputTensor = interpreter.getOutputTensor(0);
      assertEquals(DataType.FLOAT32, outputTensor.dataType());
      assertEquals(1, outputTensor.numDimensions());
      assertEquals(8, outputTensor.numBytes());
      assertEquals(2, outputTensor.numElements());
      assertArrayEquals(new int[] {2}, outputTensor.shape());
      outputBuffer.get(output);
      float[] expectedOutput = {2, 6};
      assertArrayEquals(expectedOutput, output, EPSILON);
    }
  }

  private static FloatBuffer allocateNativeFloatBuffer(int floatLength) {
    return ByteBuffer.allocateDirect(BYTES_IN_FLOAT * floatLength)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer();
  }

  private static ByteBuffer convertInputStreamToByteBuffer(InputStream inputStream)
      throws IOException {
    byte[] byteArray = new byte[inputStream.available()];
    inputStream.read(byteArray);
    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(byteArray.length);
    byteBuffer.order(ByteOrder.nativeOrder());
    byteBuffer.put(byteArray, 0, byteArray.length);
    return byteBuffer;
  }
}
