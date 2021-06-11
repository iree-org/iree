// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <jni.h>

#include "experimental/bindings/java/com/google/iree/native/instance_wrapper.h"
#include "iree/base/logging.h"

#define JNI_FUNC extern "C" JNIEXPORT
#define JNI_PREFIX(METHOD) Java_com_google_iree_Instance_##METHOD

using iree::java::InstanceWrapper;

namespace {

// Returns a pointer to the native IREE instance stored by the InstanceWrapper
// object.
static InstanceWrapper* GetInstanceWrapper(JNIEnv* env, jobject obj) {
  jclass clazz = env->GetObjectClass(obj);
  IREE_CHECK(clazz);

  jfieldID field = env->GetFieldID(clazz, "nativeAddress", "J");
  IREE_CHECK(field);

  return reinterpret_cast<InstanceWrapper*>(env->GetLongField(obj, field));
}

}  // namespace

JNI_FUNC jlong JNI_PREFIX(nativeNew)(JNIEnv* env, jobject thiz) {
  return reinterpret_cast<jlong>(new InstanceWrapper());
}

JNI_FUNC void JNI_PREFIX(nativeFree)(JNIEnv* env, jobject thiz, jlong handle) {
  InstanceWrapper* instance = GetInstanceWrapper(env, thiz);
  IREE_CHECK_NE(instance, nullptr);
  delete instance;
}

JNI_FUNC jint JNI_PREFIX(nativeCreate)(JNIEnv* env, jobject thiz) {
  InstanceWrapper* instance = GetInstanceWrapper(env, thiz);
  IREE_CHECK_NE(instance, nullptr);

  auto status = instance->Create();
  return (jint)status.code();
}
