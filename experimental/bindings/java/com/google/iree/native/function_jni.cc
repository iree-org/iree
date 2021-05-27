// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <jni.h>

#include "experimental/bindings/java/com/google/iree/native/function_wrapper.h"
#include "iree/base/logging.h"

#define JNI_FUNC extern "C" JNIEXPORT
#define JNI_PREFIX(METHOD) Java_com_google_iree_Function_##METHOD

using iree::java::FunctionWrapper;

namespace {

// Returns a pointer to the native IREE function stored by the FunctionWrapper
// object.
static FunctionWrapper* GetFunctionWrapper(JNIEnv* env, jobject obj) {
  jclass clazz = env->GetObjectClass(obj);
  IREE_CHECK(clazz);

  jfieldID field = env->GetFieldID(clazz, "nativeAddress", "J");
  IREE_CHECK(field);

  return reinterpret_cast<FunctionWrapper*>(env->GetLongField(obj, field));
}

}  // namespace

JNI_FUNC jlong JNI_PREFIX(nativeNew)(JNIEnv* env, jobject thiz) {
  return reinterpret_cast<jlong>(new FunctionWrapper());
}

JNI_FUNC void JNI_PREFIX(nativeFree)(JNIEnv* env, jobject thiz) {
  FunctionWrapper* function = GetFunctionWrapper(env, thiz);
  IREE_CHECK_NE(function, nullptr);
  delete function;
}

JNI_FUNC jstring JNI_PREFIX(nativeGetName)(JNIEnv* env, jobject thiz) {
  FunctionWrapper* function = GetFunctionWrapper(env, thiz);
  IREE_CHECK_NE(function, nullptr);

  iree_string_view_t function_name = function->name();
  return env->NewStringUTF(function_name.data);
}

JNI_FUNC jobject JNI_PREFIX(nativeGetSignature)(JNIEnv* env, jobject thiz) {
  FunctionWrapper* function = GetFunctionWrapper(env, thiz);
  IREE_CHECK_NE(function, nullptr);

  // TODO(jennik): Look into caching the results of these lookups.
  iree_vm_function_signature_t function_signature = function->signature();
  jclass cls = env->FindClass("com/google/iree/Function$Signature");
  jmethodID constructor =
      env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
  return env->NewObject(
      cls, constructor,
      env->NewStringUTF(function_signature.calling_convention.data));
}
