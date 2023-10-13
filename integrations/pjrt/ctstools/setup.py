# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import setup

setup(
    name="openxla_pjrt_ctstools",
    packages=["openxla.cts"],
    entry_points={
        "pytest11": ["openxla_pjrt_artifacts = openxla.cts.pytest_artifact_saver"]
    },
    classifiers=["Framework :: Pytest"],
)
