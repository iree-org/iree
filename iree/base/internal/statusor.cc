// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/statusor.h"

namespace iree {

namespace internal_statusor {

void Helper::HandleInvalidStatusCtorArg(Status* status) {
  const char* kMessage =
      "An OK status is not a valid constructor argument to StatusOr<T>";
  IREE_LOG(ERROR) << kMessage;
  *status = Status(StatusCode::kInternal, kMessage);
  abort();
}

void Helper::Crash(const Status& status) {
  IREE_LOG(FATAL) << "Attempting to fetch value instead of handling error "
                  << status;
  abort();
}

}  // namespace internal_statusor

}  // namespace iree
