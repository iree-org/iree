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

import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.fail;

import android.content.Context;
import android.content.res.Resources;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class IntegrationTest {
  @Test
  public void throwsExceptionWithoutNativeLib() throws Exception {
    try {
      new Instance();
      fail();
    } catch (IllegalStateException expected) {
    }
  }

  @Test
  public void simpleMul() throws Exception {
    Instance.loadNativeLibrary();
    Instance instance = new Instance();
    com.google.iree.Context ireeContext = new com.google.iree.Context(instance);

    assertNotEquals(ireeContext.getId(), -1);

    Context context = ApplicationProvider.getApplicationContext();
    Resources resources = context.getResources();
    InputStream moduleInputStream = resources.openRawResource(R.raw.simple_mul_bytecode_module);
    ByteBuffer moduleByteBuffer = convertInputStreamToByteBuffer(moduleInputStream);
    Module module = new Module(moduleByteBuffer);
    // TODO(jennik): Register modules with the context.

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
