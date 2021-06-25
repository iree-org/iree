// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/utils/yaml_util.h"

#include <yaml.h>

// TODO(benvanik): yaml parsing/printing to vm types.

void yaml_util_dummy() {
  // Just here to make linking issues visible.
  yaml_get_version_string();
}
