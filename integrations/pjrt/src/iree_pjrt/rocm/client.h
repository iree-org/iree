// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_ROCM_CLIENT_H_
#define IREE_PJRT_PLUGIN_PJRT_ROCM_CLIENT_H_

#include "experimental/rocm/api.h"
#include "iree_pjrt/common/api_impl.h"

namespace iree::pjrt::rocm {

class ROCMClientInstance final : public ClientInstance {
 public:
  ROCMClientInstance(std::unique_ptr<Platform> platform);
  ~ROCMClientInstance();
  iree_status_t CreateDriver(iree_hal_driver_t** out_driver) override;
  bool SetDefaultCompilerFlags(CompilerJob* compiler_job) override;

 private:
};

}  // namespace iree::pjrt::rocm

#endif  // IREE_PJRT_PLUGIN_PJRT_ROCM_CLIENT_H_
