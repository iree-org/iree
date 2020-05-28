// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <jni.h>

#include "bindings/java/com/google/iree/native/instance_wrapper.h"
#include "iree/base/logging.h"

#define JNI_FUNC extern "C" JNIEXPORT
#define JNI_PREFIX(METHOD) Java_com_google_iree_Instance_##METHOD

using iree::java::InstanceWrapper;

namespace {

// Returns a pointer to the native IREE instance stored by the InstanceWrapper
// object.
static InstanceWrapper* GetInstanceWrapper(JNIEnv* env, jobject obj) {
  jclass clazz = env->GetObjectClass(obj);
  CHECK(clazz);

  jfieldID field = env->GetFieldID(clazz, "nativeAddress", "J");
  CHECK(field);

  return reinterpret_cast<InstanceWrapper*>(env->GetLongField(obj, field));
}

}  // namespace

JNI_FUNC jlong JNI_PREFIX(nativeNew)(JNIEnv* env, jobject thiz) {
  return reinterpret_cast<jlong>(new InstanceWrapper());
}

JNI_FUNC void JNI_PREFIX(nativeFree)(JNIEnv* env, jobject thiz, jlong handle) {
  InstanceWrapper* instance = GetInstanceWrapper(env, thiz);
  CHECK_NE(instance, nullptr);
  delete instance;
}

JNI_FUNC void JNI_PREFIX(nativeCreate)(JNIEnv* env, jobject thiz) {
  InstanceWrapper* instance = GetInstanceWrapper(env, thiz);
  CHECK_NE(instance, nullptr);

  auto status = instance->Create();

  // TODO(jennik): Propogate this status through to java side.
  if (!status.ok()) {
    LOG(FATAL) << status.message();
  }
}
