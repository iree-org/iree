// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <jni.h>

#include "iree/base/api.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api.h"

#define JNI_FUNC extern "C" JNIEXPORT
#define JNI_PREFIX(METHOD) Java_org_tensorflow_lite_Interpreter_##METHOD

namespace {

// Returns a pointer to the native IREE module stored by the GetInterpreter
// object.
static TfLiteInterpreter* GetInterpreter(JNIEnv* env, jobject obj) {
  jclass clazz = env->GetObjectClass(obj);
  IREE_ASSERT(clazz);

  jfieldID field = env->GetFieldID(clazz, "nativeAddress", "J");
  IREE_ASSERT(field);

  if (env->ExceptionCheck()) {
    return nullptr;  // Failed to get field, returning null.
  }

  return reinterpret_cast<TfLiteInterpreter*>(env->GetLongField(obj, field));
}

}  // namespace

JNI_FUNC jlong JNI_PREFIX(nativeNew)(JNIEnv* env, jobject thiz,
                                     jobject model_byte_buffer,
                                     jint num_threads) {
  // Create model
  const char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(model_byte_buffer));
  jlong capacity = env->GetDirectBufferCapacity(model_byte_buffer);
  TfLiteModel* model = TfLiteModelCreate(buf, capacity);

  // Create options
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, num_threads);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  return reinterpret_cast<jlong>(interpreter);
}

JNI_FUNC void JNI_PREFIX(nativeFree)(JNIEnv* env, jobject thiz) {
  TfLiteInterpreter* interpreter = GetInterpreter(env, thiz);
  IREE_ASSERT_NE(interpreter, nullptr);
  TfLiteInterpreterDelete(interpreter);
}

JNI_FUNC jint JNI_PREFIX(nativeInputTensorCount)(JNIEnv* env, jobject thiz) {
  TfLiteInterpreter* interpreter = GetInterpreter(env, thiz);
  IREE_ASSERT_NE(interpreter, nullptr);
  return (jint)TfLiteInterpreterGetInputTensorCount(interpreter);
}

JNI_FUNC jint JNI_PREFIX(nativeOutputTensorCount)(JNIEnv* env, jobject thiz) {
  TfLiteInterpreter* interpreter = GetInterpreter(env, thiz);
  IREE_ASSERT_NE(interpreter, nullptr);

  return (jint)TfLiteInterpreterGetOutputTensorCount(interpreter);
}

JNI_FUNC jint JNI_PREFIX(nativeAllocateTensors)(JNIEnv* env, jobject thiz) {
  TfLiteInterpreter* interpreter = GetInterpreter(env, thiz);
  if (!interpreter) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }

  return (jint)TfLiteInterpreterAllocateTensors(interpreter);
}

JNI_FUNC jint JNI_PREFIX(nativeResizeInputTensor)(JNIEnv* env, jobject thiz,
                                                  jint input_index,
                                                  jintArray dims) {
  TfLiteInterpreter* interpreter = GetInterpreter(env, thiz);
  if (!interpreter) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }

  const int dims_size = env->GetArrayLength(dims);
  jint* dims_data = env->GetIntArrayElements(dims, 0);
  jint status = TfLiteInterpreterResizeInputTensor(interpreter, input_index,
                                                   dims_data, dims_size);

  env->ReleaseIntArrayElements(dims, dims_data, 0);

  return status;
}

JNI_FUNC jint JNI_PREFIX(nativeInvoke)(JNIEnv* env, jobject thiz) {
  TfLiteInterpreter* interpreter = GetInterpreter(env, thiz);
  if (!interpreter) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }

  return (jint)TfLiteInterpreterInvoke(interpreter);
}
