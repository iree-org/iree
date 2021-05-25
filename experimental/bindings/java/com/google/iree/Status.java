/*
 * Copyright 2020 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

package com.google.iree;

import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeoutException;

/** Well-known status codes matching iree_status_code_t values. */
public enum Status {
  OK,
  CANCELLED,
  UNKNOWN,
  INVALID_ARGUMENT,
  DEADLINE_EXCEEDED,
  NOT_FOUND,
  ALREADY_EXISTS,
  PERMISSION_DENIED,
  UNAUTHENTICATED,
  RESOURCE_EXHAUSTED,
  FAILED_PRECONDITION,
  ABORTED,
  OUT_OF_RANGE,
  UNIMPLEMENTED,
  INTERNAL,
  UNAVAILABLE,
  DATA_LOSS;

  public boolean isOk() {
    return this == Status.OK;
  }

  public Exception toException(String message) {
    String messageWithStatus = this + ": " + message;
    switch (this) {
      case CANCELLED:
        return new CancellationException(messageWithStatus);
      case UNKNOWN:
        return new RuntimeException(messageWithStatus);
      case INVALID_ARGUMENT:
        return new IllegalArgumentException(messageWithStatus);
      case DEADLINE_EXCEEDED:
        return new TimeoutException(messageWithStatus);
      case NOT_FOUND:
        return new RuntimeException(messageWithStatus);
      case ALREADY_EXISTS:
        return new IllegalStateException(messageWithStatus);
      case PERMISSION_DENIED:
        return new IllegalAccessException(messageWithStatus);
      case RESOURCE_EXHAUSTED:
        return new RuntimeException(messageWithStatus);
      case FAILED_PRECONDITION:
        return new IllegalStateException(messageWithStatus);
      case ABORTED:
        return new InterruptedException(messageWithStatus);
      case OUT_OF_RANGE:
        return new IndexOutOfBoundsException(messageWithStatus);
      case UNIMPLEMENTED:
        return new UnsupportedOperationException(messageWithStatus);
      case INTERNAL:
        return new RuntimeException(messageWithStatus);
      case UNAVAILABLE:
        return new IllegalStateException(messageWithStatus);
      case DATA_LOSS:
        return new RuntimeException(messageWithStatus);
      case UNAUTHENTICATED:
        return new IllegalStateException(messageWithStatus);
      default:
        return new RuntimeException(messageWithStatus);
    }
  }

  public static Status fromCode(int code) {
    return Status.values()[code];
  }
}
