# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utility function for loading external HAL drivers.
"""

def get_attr_values(external_hal_drivers, attr_name):
    """Return the list of values of a specific attribute of the enabled external HAL drivers.

    Args:
        external_hal_drivers: List of external HAL drivers as returned by
            get_external_hal_drivers() from @iree_external_hal_drivers.
        attr_name (str): Name of the attribute to return.

    Returns:
        Chain of selections of list of strings and list of strings.
    """
    values = []
    for e_d in external_hal_drivers:
        if not hasattr(e_d, "enabled_by"):
            # Use += instead of append(): values may already be a select().
            values += [getattr(e_d, attr_name)]
            continue
        values += select({
            e_d.enabled_by: [getattr(e_d, attr_name)],
            "//conditions:default": [],
        })
    return values

def get_targets(external_hal_drivers):
    """Return list of enabled external driver targets."""
    return get_attr_values(external_hal_drivers, "target")

def get_register_fns(external_hal_drivers):
    """Return list of enabled external driver registration functions."""
    return get_attr_values(external_hal_drivers, "register_fn")
