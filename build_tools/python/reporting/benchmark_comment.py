#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

GIST_LINK_PLACEHORDER = "<<gist-link-placeholder>>"


@dataclass(frozen=True)
class CommentData(object):
  type_id: str
  abbr_md: str
  full_md: str
