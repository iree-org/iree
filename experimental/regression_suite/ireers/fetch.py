# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def fetch_source(url: str):
    @pytest.fixture
    def fetcher(tmp_path_factory, worker_id):
        return f"Hi: {tmp_path_factory}, {worker_id}"

    return fetcher
