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

from ._binding import _create_hal_driver, HalDevice, HalDriver

_GLOBAL_DRIVERS = {}  # type: Dict[str, Union[HalDriver, Exception]]
_GLOBAL_DEVICES_BY_NAME = {}  # type: Dict[str, Union[HalDevice, Exception]]

_LOCK = RLock()

DEFAULT_DRIVER_NAMES = "dylib,cuda,vulkan,vmvx"


def query_available_drivers() -> Collection[str]:
  """Returns a collection of driver names that are available."""
  return HalDriver.query()


def get_driver(driver_name: str) -> HalDriver:
  """Gets or creates a driver by name."""
  with _LOCK:
    existing = _GLOBAL_DRIVERS.get(driver_name)
    if existing is not None:
      if isinstance(existing, Exception):
        raise existing
      else:
        return existing

    available_names = query_available_drivers()
    if driver_name not in available_names:
      raise ValueError(
          f"Driver '{driver_name}' is not compiled into this binary")

    try:
      driver = _create_hal_driver(driver_name)
    except Exception as ex:
      _GLOBAL_DRIVERS[driver_name] = ex
      raise
    _GLOBAL_DRIVERS[driver_name] = driver
    return driver


def get_device_by_name(name_spec: str) -> HalDevice:
  """Gets a cached device by name.

  Args:
    name_spec: The name of a driver or "{driver_name}:{index}" to create
      a specific device. If the indexed form is not used, the driver
      will be asked to create its default device.
  Returns:
    A HalDevice.
  """
  with _LOCK:
    existing = _GLOBAL_DEVICES_BY_NAME.get(name_spec)
    if existing is None:
      # Not existing.
      # Split into driver_name[:device_index]
      try:
        colon_pos = name_spec.index(":")
      except ValueError:
        driver_name = name_spec
        device_index = None
      else:
        driver_name = name_spec[0:colon_pos]
        device_index = name_spec[colon_pos + 1:]
        try:
          device_index = int(device_index)
        except ValueError:
          raise ValueError(f"Could not parse device name {name_spec}")

      driver = get_driver(driver_name)
      device_id = None
      if device_index is not None:
        device_infos = driver.query_available_devices()
        if device_index >= len(device_infos):
          raise ValueError(f"Device index {device_index} is out of range. "
                           f"Found devices {device_infos}")
        device_id, device_name = device_infos[0]

      try:
        if device_id is None:
          logging.info("Creating default device for driver %s", driver_name)
          device = driver.create_default_device()
        else:
          logging.info("Creating device %d (%s) for driver %s", device_id,
                       device_name, driver_name)
          device = driver.create_device(device_id)
      except Exception as ex:
        _GLOBAL_DEVICES_BY_NAME[name_spec] = ex
        raise
      else:
        _GLOBAL_DEVICES_BY_NAME[name_spec] = device
        return device

    # Existing.
    if isinstance(existing, Exception):
      raise existing
    return existing


def get_first_device_by_name(
    name_specs: Optional[Sequence[str]] = None) -> HalDevice:
  """Gets the first valid (cached) device for a prioritized list of names.

  If no driver_names are given, and an environment variable of
  IREE_DEFAULT_DEVICE is available, then it is treated as a comma delimitted
  list of driver names to try.

  This is meant to be used for default/automagic startup and is not suitable
  for any kind of multi-device setup.

  Args:
    name_specs: Search list of device names to probe.
  Returns:
    A HalDevice instance.
  """
  # Parse from environment or defaults if not explicitly provided.
  if name_specs is None:
    name_specs = os.environ.get("IREE_DEFAULT_DEVICE")
    if name_specs is None:
      name_specs = DEFAULT_DRIVER_NAMES
    name_specs = [s.strip() for s in name_specs.split(",")]

  last_exception = None
  for name_spec in name_specs:
    try:
      return get_device_by_name(name_spec)
    except ValueError:
      # Driver not known.
      continue
    except Exception as ex:
      # Failure to create driver.
      logging.info(f"Failed to create device {name_spec}: {ex}")
      last_exception = ex
      continue

  if last_exception:
    raise RuntimeError("Could not create device. "
                       "Exception for last tried follows.") from last_exception
  else:
    raise ValueError(f"No device found from list {name_specs}")
