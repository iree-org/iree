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

#include "bindings/java/com/google/iree/native/function_wrapper.h"
#include "iree/base/logging.h"

#define JNI_FUNC extern "C" JNIEXPORT
#define JNI_PREFIX(METHOD) Java_com_google_iree_Function_##METHOD

using iree::java::FunctionWrapper;

namespace {

// Returns a pointer to the native IREE function stored by the FunctionWrapper
// object.
static FunctionWrapper* GetFunctionWrapper(JNIEnv* env, jobject obj) {
  jclass clazz = env->GetObjectClass(obj);
  CHECK(clazz);

  jfieldID field = env->GetFieldID(clazz, "nativeAddress", "J");
  CHECK(field);

  return reinterpret_cast<FunctionWrapper*>(env->GetLongField(obj, field));
}

}  // namespace

JNI_FUNC jlong JNI_PREFIX(nativeNew)(JNIEnv* env, jobject thiz) {
  return reinterpret_cast<jlong>(new FunctionWrapper());
}

JNI_FUNC void JNI_PREFIX(nativeFree)(JNIEnv* env, jobject thiz) {
  FunctionWrapper* function = GetFunctionWrapper(env, thiz);
  CHECK_NE(function, nullptr);
  delete function;
}

JNI_FUNC jstring JNI_PREFIX(nativeGetName)(JNIEnv* env, jobject thiz) {
  FunctionWrapper* function = GetFunctionWrapper(env, thiz);
  CHECK_NE(function, nullptr);

  iree_string_view_t function_name = function->name();
  return env->NewStringUTF(function_name.data);
}

JNI_FUNC jobject JNI_PREFIX(nativeGetSignature)(JNIEnv* env, jobject thiz) {
  FunctionWrapper* function = GetFunctionWrapper(env, thiz);
  CHECK_NE(function, nullptr);

  // TODO(jennik): Look into caching the results of these lookups.
  iree_vm_function_signature_t function_signature = function->signature();
  jclass cls = env->FindClass("com/google/iree/Function$Signature");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(II)V");
  return env->NewObject(cls, constructor, function_signature.argument_count,
                        function_signature.result_count);
}
