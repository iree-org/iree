# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""
A number of optional arguments to the compiler can be useful for debugging:

* `extended_diagnostics=True` - Outputs verbose attached operations to \
  diagnostics. Can output a large volume of information.
* `crash_reproducer_path=... some .mlir file path...` - On a crash or error,\
  a reproducer will be output at the listed path.
* `extra_args=[...]` - Passes extra arguments to the compiler. Useful for \
  various standard features of MLIR based compilers like
  `-mlir-print-ir-after-all`.


In addition, the core compiler and frontend compiler APIs have a unified
mechanism for saving their temporary files, which are often useful for post
mortem debugging. Since the need for this is often as part of a larger system,
it is exposed both via an environment variable and an API.

In order to save all temporaries and reproducers, set the `IREE_SAVE_TEMPS`
environment variable to a directory in which to dump artifacts. For complex
programs that invoke the compiler many times, it will typically be necessary
to further qualify the path, and there are a few placeholders that will be
expanded:

* `{id}` - A per-process monotonically increasing number for each compiler
  invocation. Can be overridden by the API if a better symbolic name is
  available (i.e. test case, etc).
* `{pid}` - Process ID of the current process.
* `{main}` - Basename of `sys.argv[0]`, which is typically the name of the
  Python main file.

For interactive use, the following (on a Unix-like system) should provide
value:

.. code-block:: bash

  export IREE_SAVE_TEMPS="/tmp/ireedumps/{main}/{id}"

For the context manager based API, refer to the
`iree.compiler.tools.debugging.TempFileSaver` class.
"""

from typing import Optional

import logging
import os
import shutil
import sys
import threading

_thread_locals = threading.local()
_invocation_id = 0


def _get_temp_file_saver_stack():
    try:
        return _thread_locals.temp_file_saver_stack
    except AttributeError:
        stack = []
        _thread_locals.temp_file_saver_stack = stack
        return stack


def _interpolate_path_pattern(path_pattern: str, *, invocation_id: str):
    # We do not use str.format() because we do not know the providence of
    # path_pattern. Instead, handle a fixed set of replacements.
    path_pattern = path_pattern.replace("{id}", str(invocation_id))
    path_pattern = path_pattern.replace("{pid}", str(os.getpid()))
    path_pattern = path_pattern.replace("{main}", os.path.basename(sys.argv[0]))
    return path_pattern


class TempFileSaver:
    """Manages the saving of temp files resulting from tool invocations.

    The TempFileSaver is a thread-local context bound object. An attempt to
    create a new one will return the most recent instance created and entered
    as a context manager. This allows up-stack callers to establish the
    policy for saving temporaries and deep implementations will inherit it.

    Proper usage from users wishing to establish a saver context:

    .. code-block:: python

      with TempFileSaver():
        # Do things with temp files.

    Proper usage for implementors wishing to use an established saver context
    or set up a new one:

    .. code-block:: python

      with TempFileSaver.implicit() as tfs:
        # Do things with temp files.

    The outer-most creator can customize it with explicit arguments to __init__
    but these will be ignored if an instance is already thread bound.
    """

    TEMP_PATH_ENV_KEY = "IREE_SAVE_TEMPS"

    @staticmethod
    def implicit():
        stack = _get_temp_file_saver_stack()
        if stack:
            return stack[-1]
        return TempFileSaver()

    def __init__(
        self,
        temp_path_pattern: Optional[str] = None,
        *,
        invocation_id: Optional[str] = None,
    ):
        self.retained = False
        self._refcount = 0
        if temp_path_pattern is None:
            temp_path_pattern = os.environ.get(TempFileSaver.TEMP_PATH_ENV_KEY)
        if temp_path_pattern is None:
            return

        global _invocation_id
        if invocation_id is not None:
            self.invocation_id = invocation_id
        else:
            self.invocation_id = _invocation_id
            _invocation_id += 1

        self.retained_path = _interpolate_path_pattern(
            temp_path_pattern, invocation_id=self.invocation_id
        )
        self.retained = True
        self._retained_file_names = set()
        self._copy_on_finalize = list()  # Of (source_path, target_path)

    def __enter__(self):
        _get_temp_file_saver_stack().append(self)
        self._refcount += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del _get_temp_file_saver_stack()[-1]
        self._refcount -= 1
        if self._refcount == 0:
            self._finalize()

    @staticmethod
    def current():
        try:
            return _get_temp_file_saver_stack()[-1]
        except KeyError:
            raise RuntimeError("No current TempFileSaver")

    def alloc_optional(
        self, file_name: str, *, export_as: Optional[str] = None
    ) -> Optional[str]:
        """Allocates an optional temporary file.


        When in non-retained mode, the return value is 'export_as', meaning that the
        file is just some user specified output file.

        When in retained mode, the output file will be an index-mangled variant
        of 'file_name' under the temp_path. In addition, a mapping will be added
        so that upon finalization, the file is also exported to 'export_as' if
        specified.

        Returns None if neither a user-specified 'export_as' is specified nor in
        retained mode.

        The distinction between retained temporaries and exports is to help in
        cases for when the caller has requested that an artifact be written to
        a specific place (i.e. an output file) but for debuggability, we also
        want to save it as a temporary. In this case, we save it to the temporary
        location and then conclude by moving artifacts to their final location
        once the saver goes out of scope.
        """
        if not self.retained:
            return export_as
        alloced_path = self._alloc_retained_path(file_name)
        if export_as:
            self._copy_on_finalize.append((alloced_path, export_as))
        return alloced_path

    def _alloc_retained_path(self, file_name: str) -> str:
        assert self.retained
        index = 0
        original_file_name = file_name
        while True:
            if file_name not in self._retained_file_names:
                # First use of this name.
                self._retained_file_names.add(file_name)
                os.makedirs(self.retained_path, exist_ok=True)
                return os.path.join(self.retained_path, file_name)
            index += 1
            stem, ext = os.path.splitext(original_file_name)
            file_name = f"{stem}_{index}{ext}"

    def _finalize(self):
        if not self.retained:
            return
        # See which files were materialized.
        was_materialized = []
        for file_name in self._retained_file_names:
            file_path = os.path.join(self.retained_path, file_name)
            if os.path.exists(file_path):
                was_materialized.append((file_name, file_path))
        if was_materialized:
            logging.info(
                "**** IREE Compiler retained temporary files (%s)***:\n%s",
                self.invocation_id,
                "\n".join(
                    [
                        f"  * {file_name} : {file_path}"
                        for file_name, file_path in was_materialized
                    ]
                ),
            )
        for source_path, target_path in self._copy_on_finalize:
            if os.path.exists(source_path):
                logging.info(
                    "Copy retained file to output: %s -> %s", source_path, target_path
                )
                shutil.copyfile(source_path, target_path)
