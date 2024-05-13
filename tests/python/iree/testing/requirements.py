# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import multiprocessing
from typing import Dict, List
import sys
import logging

logger = logging.getLogger("iree.testing")


def has_compiler_module() -> bool:
    spec = importlib.util.find_spec("iree.compiler")
    return spec is not None


def has_runtime_module() -> bool:
    spec = importlib.util.find_spec("iree.compiler")
    return spec is not None


if has_compiler_module():
    import iree.compiler

if has_runtime_module():
    import iree.runtime


def get_device_count_inplace(driver: str, queue: multiprocessing.Queue) -> None:
    device_count = get_device_count(driver)
    queue.put(device_count)


def get_device_count(driver: str, use_subprocess: bool = False) -> int:
    """
    Parameters
    ----------
    use_subprocess:
    Run in a subprocess to avoid initializing the driver context in this
    process as it may interfere when other subprocesses need it.
    """
    if use_subprocess:
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=get_device_count_inplace, args=[driver, queue]
        )
        process.start()
        process.join()
        assert process.exitcode == 0
        return queue.get()

    available_driver_names = iree.runtime.query_available_drivers()
    if driver not in available_driver_names:
        return 0
    try:
        hal_driver = iree.runtime.get_driver(driver)
    except iree.runtime.ErrorUnavailable:
        # If the driver is unavailable we do not consider it a hard error.
        # It means there are no devices associated with that driver.
        return 0
    except:
        raise

    device_infos = hal_driver.query_available_devices()
    return len(device_infos)


def has_compiler_requirement() -> bool:
    res = has_compiler_module()
    if not res:
        logger.info("Python module iree.compiler is missing.")
    return res


def has_compiler_target_backends_requirement(target_backends: List[str] = []) -> bool:
    res = True
    available_backends = iree.compiler.query_available_targets()
    for required_backend in target_backends:
        has_backend = required_backend in available_backends
        if not has_backend:
            logger.info(
                f"Compiler has missing required target backend {required_backend}."
            )
        res &= has_backend
    return res


def has_runtime_requirement() -> bool:
    res = has_compiler_module()
    if not res:
        logger.info("Python module iree.runtime is missing.")
    return res


def has_device_count_requirement(driver: str, count: int) -> bool:
    device_count = get_device_count(driver=driver, use_subprocess=True)
    res = device_count >= count
    if not res:
        logger.info(
            f"Found only {device_count} devices for driver {driver}, but {count} were required."
        )
    return res


def has_requirements(
    compiler: bool = False,
    runtime: bool = False,
    compiler_target_backends: List[str] = [],
    device_count: Dict[str, int] = {},
) -> bool:
    """
    Check if has the IREE requirements.

    Parameters
    ----------
    device_count: Map of (driver name -> device count).
    """
    if len(device_count) > 0:
        runtime = True
    if len(compiler_target_backends) > 0:
        compiler = True
    res = True
    if compiler:
        res &= has_compiler_requirement()
    if runtime:
        res &= has_runtime_requirement()
    if has_runtime_module():
        for driver, count in device_count.items():
            res &= has_device_count_requirement(driver=driver, count=count)
    if has_compiler_module():
        res &= has_compiler_target_backends_requirement(compiler_target_backends)
    return res
