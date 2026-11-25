# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ExternalTensorTrait:
    """Represents a 'trait' that can be applied to a Tensor to signal that
    it is to be loaded by name from an external archive at AOT execution time.
    """

    external_scope: str
    external_name: str

    @staticmethod
    def get(from_tensor: torch.Tensor) -> Optional["ExternalTensorTrait"]:
        existing = getattr(from_tensor, "_iree_external_tensor_trait", None)
        if existing is None:
            return None
        assert isinstance(existing, ExternalTensorTrait)
        return existing

    def set(self, to_tensor: torch.Tensor):
        to_tensor._iree_external_tensor_trait = self  # type: ignore
