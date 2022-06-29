// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <jni.h>

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "runtime/bindings/tflite/include/tensorflow/lite/c/c_api.h"

#define JNI_FUNC extern "C" JNIEXPORT
#define JNI_PREFIX(METHOD) Java_org_tensorflow_lite_TensorFlowLite_##METHOD

JNI_FUNC jstring JNI_PREFIX(nativeRuntimeVersion)(JNIEnv* env,
                                                  jclass /*clazz*/) {
  return env->NewStringUTF(TfLiteVersion());
}
