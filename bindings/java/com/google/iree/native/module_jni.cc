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

#include "bindings/java/com/google/iree/native/module_wrapper.h"

#define JNI_FUNC extern "C" JNIEXPORT
#define JNI_PREFIX(METHOD) Java_com_google_iree_Module_##METHOD

using iree::java::ModuleWrapper;

namespace {

// Returns a pointer to the native IREE module stored by the ModuleWrapper
// object.
static ModuleWrapper* GetModuleWrapper(JNIEnv* env, jobject obj) {
  jclass clazz = env->GetObjectClass(obj);
  CHECK(clazz);

  jfieldID field = env->GetFieldID(clazz, "nativeAddress", "J");
  CHECK(field);

  return reinterpret_cast<ModuleWrapper*>(env->GetLongField(obj, field));
}

}  // namespace

JNI_FUNC jlong JNI_PREFIX(nativeNew)(JNIEnv* env, jobject thiz) {
  return reinterpret_cast<jlong>(new ModuleWrapper());
}

JNI_FUNC void JNI_PREFIX(nativeFree)(JNIEnv* env, jobject thiz) {
  ModuleWrapper* module = GetModuleWrapper(env, thiz);
  CHECK_NE(module, nullptr);
  delete module;
}

JNI_FUNC jint JNI_PREFIX(nativeCreate)(JNIEnv* env, jobject thiz, jobject buf) {
  ModuleWrapper* module = GetModuleWrapper(env, thiz);
  CHECK_NE(module, nullptr);

  const uint8_t* data = (uint8_t*)env->GetDirectBufferAddress(buf);
  int length = env->GetDirectBufferCapacity(buf);
  auto status = module->Create(data, length);
  return (jint)status.code();
}
