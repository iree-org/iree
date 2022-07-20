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
#define JNI_PREFIX(METHOD) Java_org_tensorflow_lite_Tensor_##METHOD

namespace {

// Returns a pointer to the native IREE module stored by the GetTensor
// object.
static TfLiteTensor* GetTensor(JNIEnv* env, jobject obj) {
  jclass clazz = env->GetObjectClass(obj);
  IREE_ASSERT(clazz);

  jfieldID field = env->GetFieldID(clazz, "nativeAddress", "J");
  IREE_ASSERT(field);

  if (env->ExceptionCheck()) {
    return nullptr;  // Failed to get field, returning null.
  }

  return reinterpret_cast<TfLiteTensor*>(env->GetLongField(obj, field));
}

}  // namespace

JNI_FUNC jlong JNI_PREFIX(nativeCreateInput)(JNIEnv* env, jclass clazz,
                                             jlong interpreter_handle,
                                             jint input_index) {
  TfLiteInterpreter* interpreter = (TfLiteInterpreter*)interpreter_handle;
  if (!interpreter) {
    return kTfLiteError;  // Null handle input. Returning to error in Java.
  }

  TfLiteTensor* input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, input_index);
  return reinterpret_cast<jlong>(input_tensor);
}

JNI_FUNC jlong JNI_PREFIX(nativeCreateOutput)(JNIEnv* env, jclass clazz,
                                              jlong interpreter_handle,
                                              jint output_index) {
  TfLiteInterpreter* interpreter = (TfLiteInterpreter*)interpreter_handle;
  if (!interpreter) {
    return kTfLiteError;  // Null handle input. Returning to error in Java.
  }

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, output_index);
  return reinterpret_cast<jlong>(output_tensor);
}

JNI_FUNC jint JNI_PREFIX(nativeType)(JNIEnv* env, jobject thiz) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }
  return (jint)TfLiteTensorType(tensor);
}

JNI_FUNC jint JNI_PREFIX(nativeNumDims)(JNIEnv* env, jobject thiz) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }
  return (jint)TfLiteTensorNumDims(tensor);
}

JNI_FUNC jint JNI_PREFIX(nativeDim)(JNIEnv* env, jobject thiz, jint dim_index) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }
  return (jint)TfLiteTensorDim(tensor, dim_index);
}

JNI_FUNC jint JNI_PREFIX(nativeBytesSize)(JNIEnv* env, jobject thiz) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }                       // size_t is an unsigned int. This may roll over;
  return (jint)TfLiteTensorByteSize(tensor);
}

JNI_FUNC jstring JNI_PREFIX(nativeName)(JNIEnv* env, jobject thiz) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return nullptr;  // Failed get handle. Returning to error in Java.
  }
  auto msg = TfLiteTensorName(tensor);
  return env->NewStringUTF(msg);
}

JNI_FUNC jfloat JNI_PREFIX(nativeQuantizationScale)(JNIEnv* env, jobject thiz) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }
  auto params = TfLiteTensorQuantizationParams(tensor);
  return (jfloat)params.scale;
}

JNI_FUNC jint JNI_PREFIX(nativeQuantizationZeroPoint)(JNIEnv* env,
                                                      jobject thiz) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }
  auto params = TfLiteTensorQuantizationParams(tensor);
  return (jfloat)params.zero_point;
}

JNI_FUNC jint JNI_PREFIX(nativeCopyFromDirectBuffer)(
    JNIEnv* env, jobject thiz, jobject input_byte_buffer) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }
  const char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(input_byte_buffer));

  // Note: We're using the tensor size rather than the buffer size since non
  // ByteBuffer direct buffers missreport capacity based on the data type. This
  // relies on proper capacity checks in Java.
  return (jint)TfLiteTensorCopyFromBuffer(tensor, buf,
                                          TfLiteTensorByteSize(tensor));
}

JNI_FUNC jint JNI_PREFIX(nativeCopyToDirectBuffer)(JNIEnv* env, jobject thiz,
                                                   jobject output_byte_buffer) {
  const TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return kTfLiteError;  // Failed get handle. Returning to error in Java.
  }
  char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(output_byte_buffer));

  // Note: We're using the tensor size rather than the buffer size since non
  // ByteBuffer direct buffers missreport capacity based on the data type. This
  // relies on proper capacity checks in Java.
  return (jint)TfLiteTensorCopyToBuffer(tensor, buf,
                                        TfLiteTensorByteSize(tensor));
}

JNI_FUNC jobject JNI_PREFIX(nativeGetByteBuffer)(JNIEnv* env, jobject thiz) {
  TfLiteTensor* tensor = GetTensor(env, thiz);
  if (!tensor) {
    return nullptr;  // Failed get handle. Returning to error in Java.
  }
  return env->NewDirectByteBuffer(
      static_cast<void*>(TfLiteTensorData(tensor)),
      static_cast<jlong>(TfLiteTensorByteSize(tensor)));
}
