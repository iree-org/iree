// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_PROJECTS_IREE_DIALECTS_C_DIALECTS_H
#define IREE_LLVM_PROJECTS_IREE_DIALECTS_C_DIALECTS_H

#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(IREEPublic, iree_public);

#ifdef __cplusplus
}
#endif

#endif  // IREE_LLVM_PROJECTS_IREE_DIALECTS_C_DIALECTS_H
