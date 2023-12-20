# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dynamic library binding to libIREECompiler.so, using ctypes."""

from ctypes import *
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

import ctypes
import logging
import platform
import weakref

__all__ = [
    "Invocation",
    "Output",
    "Session",
    "Source",
]

_dylib = None

_GET_FLAG_CALLBACK = CFUNCTYPE(None, c_void_p, c_size_t, c_void_p)


def _setsig(f, restype, argtypes):
    f.restype = restype
    f.argtypes = argtypes


def _init_dylib():
    global _dylib
    if _dylib:
        return
    dylib_path = _probe_iree_compiler_dylib()
    if dylib_path is None:
        # TODO: Look for a bundled dylib.
        raise RuntimeError("Could not find libIREECompiler.so")
    _dylib = cdll.LoadLibrary(dylib_path)

    _setsig(
        _dylib.ireeCompilerSetupGlobalCL,
        None,
        [c_int, POINTER(c_char_p), c_char_p, c_bool],
    )
    _setsig(_dylib.ireeCompilerGlobalInitialize, None, [])
    _setsig(_dylib.ireeCompilerGlobalShutdown, None, [])

    # Setup signatures.
    # Error
    _setsig(_dylib.ireeCompilerErrorDestroy, None, [c_void_p])
    _setsig(_dylib.ireeCompilerErrorGetMessage, c_char_p, [c_void_p])

    # Invocation
    _setsig(_dylib.ireeCompilerInvocationCreate, c_void_p, [c_void_p])
    _setsig(_dylib.ireeCompilerInvocationDestroy, None, [c_void_p])
    _setsig(_dylib.ireeCompilerInvocationEnableConsoleDiagnostics, None, [c_void_p])
    _setsig(_dylib.ireeCompilerInvocationParseSource, c_bool, [c_void_p, c_void_p])
    _setsig(_dylib.ireeCompilerInvocationPipeline, c_bool, [c_void_p, c_int])
    _setsig(_dylib.ireeCompilerInvocationRunPassPipeline, c_bool, [c_void_p, c_char_p])
    _setsig(_dylib.ireeCompilerInvocationOutputIR, c_void_p, [c_void_p, c_void_p])
    _setsig(
        _dylib.ireeCompilerInvocationOutputIRBytecode,
        c_void_p,
        [c_void_p, c_void_p, c_int],
    )
    _setsig(
        _dylib.ireeCompilerInvocationOutputVMBytecode, c_void_p, [c_void_p, c_void_p]
    )

    # Output
    _setsig(_dylib.ireeCompilerOutputDestroy, None, [c_void_p])
    _setsig(_dylib.ireeCompilerOutputOpenFile, c_void_p, [c_char_p, c_void_p])
    _setsig(_dylib.ireeCompilerOutputOpenMembuffer, c_void_p, [c_void_p])
    _setsig(_dylib.ireeCompilerOutputKeep, None, [c_void_p])
    _setsig(
        _dylib.ireeCompilerOutputWrite, c_void_p, [c_void_p, POINTER(c_char), c_size_t]
    )
    _setsig(
        _dylib.ireeCompilerOutputMapMemory,
        c_void_p,
        [c_void_p, c_void_p, POINTER(c_uint64)],
    )

    # Session
    _setsig(_dylib.ireeCompilerSessionCreate, c_void_p, [])
    _setsig(_dylib.ireeCompilerSessionDestroy, None, [c_void_p])
    _setsig(
        _dylib.ireeCompilerSessionGetFlags,
        None,
        [c_void_p, c_bool, c_void_p, c_void_p],
    )
    _setsig(
        _dylib.ireeCompilerSessionSetFlags,
        c_void_p,
        [c_void_p, c_int, c_void_p],
    )
    # From MLIRInterop.h.
    _setsig(
        _dylib.ireeCompilerSessionStealContext,
        c_void_p,
        [c_void_p],
    )
    _setsig(
        _dylib.ireeCompilerInvocationImportBorrowModule,
        c_bool,
        [c_void_p, c_void_p],
    )
    _setsig(
        _dylib.ireeCompilerInvocationExportStealModule,
        c_void_p,
        [c_void_p],
    )
    # Source
    _setsig(_dylib.ireeCompilerSourceDestroy, None, [c_void_p])
    _setsig(
        _dylib.ireeCompilerSourceOpenFile,
        c_void_p,
        [
            c_void_p,  # session
            c_char_p,  # filePath
            c_void_p,  # out_source
        ],
    )
    _setsig(
        _dylib.ireeCompilerSourceWrapBuffer,
        c_void_p,
        [
            c_void_p,  # session
            c_char_p,  # bufferName
            POINTER(c_char),  # buffer
            c_size_t,  # length
            c_bool,  # isNullTerminated
            c_void_p,  # out_source
        ],
    )


# Capsule interop
PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p

PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
PyCapsule_GetPointer.restype = ctypes.c_void_p

# mlir-c/Bindings/Python/Interop.h defines how we pass capsules around for
# interop with the MLIR Python API. This is parameterized on the package
# prefix that we built the bindings for (in this case "iree.compiler" but
# we could auto-detect it from the current module in the future if needed).
# Each type has a capsule type string associated with it that we mirror here.
# Since they are used as byte strings, we encode them all.
#
# Python types that can be created from a capsule expose a static `_CAPICreate`
# function (see MLIR_PYTHON_CAPI_FACTORY_ATTR in Interop.h). A capsule can be
# obtained by calling `_CAPIPtr` on an instance (see MLIR_PYTHON_CAPI_PTR_ATTR).
MLIR_CAPSULE_PREFIX = "iree.compiler."
MLIR_PYTHON_CAPSULE_CONTEXT = (MLIR_CAPSULE_PREFIX + "ir.Context._CAPIPtr").encode()
MLIR_PYTHON_CAPSULE_OPERATION = (MLIR_CAPSULE_PREFIX + "ir.Operation._CAPIPtr").encode()


def _handle_error(err_p, exc_type=ValueError):
    if err_p is None:
        return
    message = _dylib.ireeCompilerErrorGetMessage(err_p).decode("UTF-8")
    _dylib.ireeCompilerErrorDestroy(err_p)
    raise exc_type(message)


def _initializeGlobalCL(*cl_args: str):
    arg_buffers = [create_string_buffer(s.encode()) for s in cl_args]
    arg_pointers = (c_char_p * len(cl_args))(*map(addressof, arg_buffers))
    _dylib.ireeCompilerSetupGlobalCL(len(cl_args), arg_pointers, b"ctypes", False)


class Session:
    def __init__(self):
        self._global_init = _global_init
        self._session_p = _dylib.ireeCompilerSessionCreate()
        # If a context has been requested, then the session has released
        # its ownership of it, so we must cache the new Python-level MLIRContext
        # so its lifetime extends at least to our own.
        self._owned_context = None

    def __del__(self):
        _dylib.ireeCompilerSessionDestroy(self._session_p)

    @property
    def context(self):
        if self._owned_context is None:
            from .. import ir

            context_void = _dylib.ireeCompilerSessionStealContext(self._session_p)
            if not context_void:
                raise RuntimeError(
                    "Session context could not be initialized. This either indicated"
                    "an error setting it up or an attempt to steal the context "
                    "multiple times (which could indicate a memory leak)."
                )
            context_cp = PyCapsule_New(context_void, MLIR_PYTHON_CAPSULE_CONTEXT, None)
            self._owned_context = ir.Context._CAPICreate(context_cp)
        return self._owned_context

    def invocation(self) -> "Invocation":
        return Invocation(self)

    def get_flags(self, non_default_only: bool = False) -> Sequence[str]:
        results = []

        @_GET_FLAG_CALLBACK
        def callback(flag_pointer, length, user_data):
            flag_bytes = string_at(flag_pointer, length)
            flag_value = flag_bytes.decode("UTF-8")
            results.append(flag_value)

        _dylib.ireeCompilerSessionGetFlags(
            self._session_p, non_default_only, callback, c_void_p(0)
        )
        return results

    def set_flags(self, *flags: str):
        argv_type = c_char_p * len(flags)
        argv = argv_type(*[flag.encode("UTF-8") for flag in flags])
        _handle_error(
            _dylib.ireeCompilerSessionSetFlags(self._session_p, len(argv), argv)
        )


class Output:
    """Wraps an iree_compiler_output_t."""

    def __init__(self, output_p: c_void_p):
        self._output_p = output_p
        self._local_dylib = _dylib

    def __del__(self):
        self.close()

    def close(self):
        if self._output_p:
            self._local_dylib.ireeCompilerOutputDestroy(self._output_p)
            self._output_p = None

    @staticmethod
    def open_file(file_path: str) -> "Output":
        output_p = c_void_p()
        _handle_error(
            _dylib.ireeCompilerOutputOpenFile(file_path.encode(), byref(output_p))
        )
        return Output(output_p)

    @staticmethod
    def open_membuffer() -> "Output":
        output_p = c_void_p()
        _handle_error(_dylib.ireeCompilerOutputOpenMembuffer(byref(output_p)))
        return Output(output_p)

    def keep(self) -> "Output":
        _dylib.ireeCompilerOutputKeep(self._output_p)
        return self

    def write(self, buffer):
        _handle_error(
            _dylib.ireeCompilerOutputWrite(self._output_p, buffer, len(buffer))
        )

    def map_memory(self) -> memoryview:
        contents = c_void_p()
        size = c_uint64()
        _handle_error(
            _dylib.ireeCompilerOutputMapMemory(
                self._output_p, byref(contents), byref(size)
            )
        )
        size = size.value
        pointer = (c_char * size).from_address(contents.value)
        # When the pointer is free'd, the no-op callback is invoked with
        # the argument `self`. This implicitly keeps `self` alive until
        # the callback is invoked, which keeps the compiler Output alive.
        # The typical use of this pointer is to read it via the buffer
        # protocol, and that will keep the pointer alive. Therefore, the
        # chain is secure.
        weakref.finalize(pointer, lambda x: ..., self)
        return pointer


class Source:
    """Wraps an iree_compiler_source_t."""

    def __init__(self, session: c_void_p, source_p: c_void_p, backing_ref):
        self._session: c_void_p = session  # Keeps ref alive.
        self._source_p: c_void_p = source_p
        self._backing_ref = backing_ref
        self._local_dylib = _dylib

    def __del__(self):
        self.close()

    def close(self):
        if self._source_p:
            s = self._source_p
            self._source_p = c_void_p()
            self._local_dylib.ireeCompilerSourceDestroy(s)
            self._backing_ref = None
            self._session = c_void_p()

    def __repr__(self):
        return f"<Source {self._source_p}>"

    @staticmethod
    def open_file(session: Session, file_path: str) -> "Source":
        source_p = c_void_p()
        _handle_error(
            _dylib.ireeCompilerSourceOpenFile(
                session._session_p, file_path.encode(), byref(source_p)
            )
        )
        return Source(session, source_p, None)

    @staticmethod
    def wrap_buffer(
        session: Session, buffer, *, buffer_name: str = "source.mlir"
    ) -> "Source":
        view = memoryview(buffer)
        if not view.c_contiguous:
            raise ValueError("Buffer must be c_contiguous")
        source_p = c_void_p()
        buffer_len = len(buffer)
        _handle_error(
            _dylib.ireeCompilerSourceWrapBuffer(
                session._session_p,
                buffer_name.encode(),
                buffer,
                buffer_len,
                # Detect if nul terminated.
                True if buffer_len > 0 and view[-1] == 0 else False,
                byref(source_p),
            )
        )
        return Source(session, source_p, buffer)


class PipelineType(IntEnum):
    IREE_COMPILER_PIPELINE_STD = 0
    IREE_COMPILER_PIPELINE_HAL_EXECUTABLE = 1
    IREE_COMPILER_PIPELINE_PRECOMPILE = 2


class Invocation:
    def __init__(self, session: Session):
        self._session = session
        self._inv_p = _dylib.ireeCompilerInvocationCreate(self._session._session_p)
        self._sources: list[Source] = []
        self._local_dylib = _dylib
        # If we are importing from a module, then the MLIR/Python Operation
        # will own the module, so we need to make sure that it outlives the
        # Invocation.
        self._retained_module_op = None

    @property
    def session(self) -> Session:
        return self._session

    def __del__(self):
        self.close()

    def close(self):
        if self._inv_p:
            self._local_dylib.ireeCompilerInvocationDestroy(self._inv_p)
            self._inv_p = c_void_p()
            for s in self._sources:
                s.close()
            self._sources.clear()

    def enable_console_diagnostics(self):
        _dylib.ireeCompilerInvocationEnableConsoleDiagnostics(self._inv_p)

    def export_module(self):
        """Exports the module."""
        if self.session._owned_context is None:
            raise RuntimeError(
                "In order to export a module, the context must first be exported from "
                "the session (i.e. `session.context`)."
            )

        from .. import ir

        if self._retained_module_op:
            return self._retained_module_op
        module_ptr = _dylib.ireeCompilerInvocationExportStealModule(self._inv_p)
        if not module_ptr:
            raise RuntimeError("Module is not available to export")
        capsule = PyCapsule_New(module_ptr, MLIR_PYTHON_CAPSULE_OPERATION, None)
        operation = ir.Operation._CAPICreate(capsule)
        self._retained_module_op = operation
        return operation

    def import_module(self, module_op) -> bool:
        self._retained_module_op = module_op
        # Import module.
        if module_op is not None:
            module_capsule = module_op._CAPIPtr
            module_ptr = PyCapsule_GetPointer(
                module_capsule, MLIR_PYTHON_CAPSULE_OPERATION
            )
            if not _dylib.ireeCompilerInvocationImportBorrowModule(
                self._inv_p, module_ptr
            ):
                # TODO: Capture diagnostics.
                raise RuntimeError("Failed to import module into Invocation")

    def parse_source(self, source: Source) -> bool:
        self._sources.append(source)
        return _dylib.ireeCompilerInvocationParseSource(self._inv_p, source._source_p)

    def execute(
        self, pipeline: PipelineType = PipelineType.IREE_COMPILER_PIPELINE_STD
    ) -> bool:
        return _dylib.ireeCompilerInvocationPipeline(self._inv_p, pipeline)

    def execute_text_pass_pipeline(self, text_pipeline_spec: str) -> bool:
        return _dylib.ireeCompilerInvocationRunPassPipeline(
            self._inv_p, text_pipeline_spec.encode()
        )

    def output_ir(self, output: Output):
        _handle_error(
            _dylib.ireeCompilerInvocationOutputIR(self._inv_p, output._output_p)
        )

    def output_ir_bytecode(self, output: Output, bytecode_version: int = -1):
        _handle_error(
            _dylib.ireeCompilerInvocationOutputIRBytecode(
                self._inv_p, output._output_p, bytecode_version
            )
        )

    def output_vm_bytecode(self, output: Output):
        _handle_error(
            _dylib.ireeCompilerInvocationOutputVMBytecode(self._inv_p, output._output_p)
        )


def _probe_iree_compiler_dylib() -> str:
    """Probes an installed iree.compiler for the compiler dylib."""
    # TODO: Make this an API on iree.compiler itself. Burn this with fire.
    from .. import _mlir_libs

    try:
        from .. import version

        version_dict = version.VERSION
        dev_mode = False
    except ImportError:
        # Development setups often lack this.
        version_dict = {}
        dev_mode = True

    # Try to find development mode library, falling back to normal
    # locations.
    paths = None
    if dev_mode and len(_mlir_libs.__path__) == 1:
        # Traverse up and find CMakeCache.txt
        build_dir = Path(_mlir_libs.__path__[0]).parent
        while True:
            anchor_files = [
                build_dir / "tools" / f"iree-compile",
                build_dir / "tools" / f"iree-compile.exe",
            ]
            if any([f.exists() for f in anchor_files]):
                # Most OS's keep their libs in lib. Windows keeps them
                # in bin (tools in the dev tree). Just check them all.
                paths = [build_dir / "lib", build_dir / "tools", build_dir / "bin"]
                break
            new_dir = build_dir.parent
            if new_dir == build_dir:
                break
            build_dir = new_dir
    if not paths:
        paths = _mlir_libs.__path__

    logging.debug("Found installed iree-compiler package %r", version_dict)
    dylib_basename = "libIREECompiler.so"
    system = platform.system()
    if system == "Darwin":
        dylib_basename = "libIREECompiler.dylib"
    elif system == "Windows":
        dylib_basename = "IREECompiler.dll"

    for p in paths:
        dylib_path = Path(p) / dylib_basename
        if dylib_path.exists():
            logging.debug("Found IREE compiler dylib=%s", dylib_path)
            return str(dylib_path)
    raise ValueError(f"Could not find {dylib_basename} in {paths}")


class _GlobalInit:
    def __init__(self):
        self.local_dylib = None
        _init_dylib()
        _dylib.ireeCompilerGlobalInitialize()
        # Cache locally so as to not have it go out of scope first
        # during shutdown.
        self.local_dylib = _dylib

    def __del__(self):
        if self.local_dylib:
            self.local_dylib.ireeCompilerGlobalShutdown()


# Keep one reference for the life of the module.
_global_init = _GlobalInit()
