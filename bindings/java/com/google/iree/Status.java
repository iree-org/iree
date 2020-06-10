/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
