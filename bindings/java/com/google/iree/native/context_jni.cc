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

#include "bindings/java/com/google/iree/native/context_wrapper.h"
#include "bindings/java/com/google/iree/native/function_wrapper.h"
#include "bindings/java/com/google/iree/native/instance_wrapper.h"
#include "bindings/java/com/google/iree/native/module_wrapper.h"
#include "iree/base/logging.h"

#define JNI_FUNC extern "C" JNIEXPORT
#define JNI_PREFIX(METHOD) Java_com_google_iree_Context_##METHOD

using iree::java::ContextWrapper;
using iree::java::FunctionWrapper;
using iree::java::InstanceWrapper;
using iree::java::ModuleWrapper;

namespace {

// Returns a pointer to the native IREE context stored by the ContextWrapper
// object.
static ContextWrapper* GetContextWrapper(JNIEnv* env, jobject obj) {
  jclass clazz = env->GetObjectClass(obj);
  CHECK(clazz);

  jfieldID field = env->GetFieldID(clazz, "nativeAddress", "J");
  CHECK(field);

  return reinterpret_cast<ContextWrapper*>(env->GetLongField(obj, field));
}

std::vector<ModuleWrapper*> GetModuleWrappersFromAdresses(
    JNIEnv* env, jlongArray moduleAddresses) {
  // Get the addresses of the ModuleWrappers.
  jsize modules_size = env->GetArrayLength(moduleAddresses);
  std::vector<int64_t> module_addresses(modules_size);
  env->GetLongArrayRegion(moduleAddresses, 0, modules_size,
                          reinterpret_cast<jlong*>(module_addresses.data()));

  // Convert the addresses to ModuleWrappers.
  std::vector<ModuleWrapper*> modules(modules_size);
  for (int i = 0; i < modules_size; i++) {
    modules[i] = (ModuleWrapper*)module_addresses[i];
  }
  return modules;
}

}  // namespace

JNI_FUNC jlong JNI_PREFIX(nativeNew)(JNIEnv* env, jobject thiz) {
  return reinterpret_cast<jlong>(new ContextWrapper());
}

JNI_FUNC void JNI_PREFIX(nativeFree)(JNIEnv* env, jobject thiz, jlong handle) {
  ContextWrapper* context = GetContextWrapper(env, thiz);
  CHECK_NE(context, nullptr);
  delete context;
}

JNI_FUNC jint JNI_PREFIX(nativeCreate)(JNIEnv* env, jobject thiz,
                                       jlong instanceAddress) {
  ContextWrapper* context = GetContextWrapper(env, thiz);
  CHECK_NE(context, nullptr);

  auto instance = (InstanceWrapper*)instanceAddress;
  auto status = context->Create(*instance);
  return (jint)status.code();
}

JNI_FUNC jint JNI_PREFIX(nativeCreateWithModules)(JNIEnv* env, jobject thiz,
                                                  jlong instanceAddress,
                                                  jlongArray moduleAddresses) {
  ContextWrapper* context = GetContextWrapper(env, thiz);
  CHECK_NE(context, nullptr);

  auto instance = (InstanceWrapper*)instanceAddress;
  auto modules = GetModuleWrappersFromAdresses(env, moduleAddresses);

  auto status = context->CreateWithModules(*instance, modules);
  return (jint)status.code();
}

JNI_FUNC jint JNI_PREFIX(nativeRegisterModules)(JNIEnv* env, jobject thiz,
                                                jlongArray moduleAddresses) {
  ContextWrapper* context = GetContextWrapper(env, thiz);
  CHECK_NE(context, nullptr);

  auto modules = GetModuleWrappersFromAdresses(env, moduleAddresses);
  auto status = context->RegisterModules(modules);
  return (jint)status.code();
}

JNI_FUNC jint JNI_PREFIX(nativeResolveFunction)(JNIEnv* env, jobject thiz,
                                                jlong functionAddress,
                                                jstring name) {
  ContextWrapper* context = GetContextWrapper(env, thiz);
  CHECK_NE(context, nullptr);

  auto function = (FunctionWrapper*)functionAddress;
  const char* native_name = env->GetStringUTFChars(name, /*isCopy=*/nullptr);

  auto status = context->ResolveFunction(
      *function, iree_string_view_t{native_name, strlen(native_name)});
  env->ReleaseStringUTFChars(name, native_name);
  return (jint)status.code();
}

JNI_FUNC jint JNI_PREFIX(nativeGetId)(JNIEnv* env, jobject thiz) {
  ContextWrapper* context = GetContextWrapper(env, thiz);
  CHECK_NE(context, nullptr);

  int context_id = context->id();
  return (jint)context_id;
}
