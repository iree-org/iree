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

import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;

// TODO(jennik): This is not really a context test anymore, so rename it as an e2e test.
@RunWith(AndroidJUnit4.class)
public final class ContextTest {
  @Test
  public void create_throwsExceptionWithoutNativeLib() {
    try {
      new Instance();
      fail();
    } catch (IllegalStateException e) {
      // Expected exception.
    }
  }

  @Test
  public void create_createsContextWithId() {
    Instance.loadNativeLibrary();
    Instance instance = new Instance();
    Context context = new Context(instance);

    assertNotEquals(context.getId(), -1);

    context.free();
    instance.free();
  }
}
