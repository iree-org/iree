# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Process-global driver instantiation and discovery."""

from typing import Collection, Dict, Optional, Sequence, Union

import logging
import os
from threading import RLock

from ._binding import get_cached_hal_driver, HalDevice, HalDriver

_GLOBAL_DEVICES_BY_URI = {}  # type: Dict[str, Union[HalDevice, Exception]]

_LOCK = RLock()

DEFAULT_DRIVER_NAMES = "local-task,cuda,vulkan"


def query_available_drivers() -> Collection[str]:
    """Returns a collection of driver names that are available."""
    return HalDriver.query()


def get_driver(device_uri: str) -> HalDriver:
    """Returns a HAL driver by device_uri (or driver name).

    Args:
      device_uri: The URI of the device, either just a driver name for the
        default or a fully qualified "driver://path?params".
    """
    return get_cached_hal_driver(device_uri)


def get_device(device_uri: str, cache: bool = True) -> HalDevice:
    """Gets a cached device by URI.

    Args:
      device_uri: The URI of the device, either just a driver name for the
        default or a fully qualified "driver://path?params".
      cache: Whether to cache the device (default True).
    Returns:
      A HalDevice.
    """
    with _LOCK:
        if cache:
            existing = _GLOBAL_DEVICES_BY_URI.get(device_uri)
            if existing is not None:
                return existing

        driver = get_driver(device_uri)
        device = driver.create_device_by_uri(device_uri)

        if cache:
            _GLOBAL_DEVICES_BY_URI[device_uri] = device
        return device


def get_first_device(
    device_uris: Optional[Sequence[str]] = None, cache: bool = True
) -> HalDevice:
    """Gets the first valid (cached) device for a prioritized list of names.

    If no driver_names are given, and an environment variable of
    IREE_DEFAULT_DEVICE is available, then it is treated as a comma delimitted
    list of driver names to try.

    This is meant to be used for default/automagic startup and is not suitable
    for any kind of multi-device setup.

    Args:
      device_uris: Explicit list of device URIs to try.
      cache: Whether to cache the device (default True).
    Returns:
      A HalDevice instance.
    """
    # Parse from environment or defaults if not explicitly provided.
    if device_uris is None:
        device_uris = os.environ.get("IREE_DEFAULT_DEVICE")
        if device_uris is None:
            device_uris = DEFAULT_DRIVER_NAMES
        device_uris = [s.strip() for s in device_uris.split(",")]

    last_exception = None
    for device_uri in device_uris:
        try:
            return get_device(device_uri, cache=cache)
        except ValueError:
            # Driver not known.
            continue
        except Exception as ex:
            # Failure to create driver.
            logging.info(f"Failed to create device {device_uri}: {ex}")
            last_exception = ex
            continue

    if last_exception:
        raise RuntimeError(
            "Could not create device. " "Exception for last tried follows."
        ) from last_exception
    else:
        raise ValueError(f"No device found from list {device_uris}")
