# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers import *

llama2_7b_f16qi4_source = fetch_source("foobar")

@pytest.mark.plat_rdna_vulkan
@pytest.mark.presubmit
@pytest.mark.unstable_linalg
def test_step_rdna_vulkan(llama2_7b_f16qi4_source):
    print("GOOD")
    print(llama2_7b_f16qi4_source)
