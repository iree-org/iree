# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The procedural package has circular dependencies due to its
# nature. In an effort to modularize the code, we do allow circular
# imports and when used, they must be coherent with the load
# order here and must perform the import at the end of the module.
