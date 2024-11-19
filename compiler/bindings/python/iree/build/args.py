# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Generator, TypeVar

import argparse
import contextlib
import functools
import inspect
import threading

from typing import Callable

_locals = threading.local()
_ALL_ARG_REGISTRARS: list[Callable[[argparse.ArgumentParser], None]] = []
_ALL_ARG_HANDLERS: list[Callable[[argparse.Namespace], None]] = []


def register_arg_parser_callback(registrar: Callable[[argparse.ArgumentParser], None]):
    """Decorator that adds a global argument registration callback.

    This callback will be invoked when a new ArgumentParser is constructed.
    """
    _ALL_ARG_REGISTRARS.append(registrar)
    return registrar


def register_arg_handler_callback(handler: Callable[[argparse.Namespace], None]):
    """Decorator that registers a handler to be run on global arguments at startup."""
    _ALL_ARG_HANDLERS.append(handler)
    return handler


def configure_arg_parser(p: argparse.ArgumentParser):
    """Invokes all callbacks from `register_arg_parser_callback` on the parser."""
    for callback in _ALL_ARG_REGISTRARS:
        callback(p)


def run_global_arg_handlers(ns: argparse.Namespace):
    """Invokes all global argument handlers."""
    for h in _ALL_ARG_HANDLERS:
        h(ns)


@contextlib.contextmanager
def argument_namespace_context(ns: argparse.Namespace):
    """Establish that given namespace as the current namespace for this thread.

    Note that as a thread local, this does not propagate to child threads or
    sub-processes. This means that all argument management must be done during
    action setup and action invocations will not typically have access to args.
    """
    if not hasattr(_locals, "arg_ns_stack"):
        _locals.arg_ns_stack = []
    _locals.arg_ns_stack.append(ns)
    try:
        yield ns
    finally:
        _locals.arg_ns_stack.pop()


def current_args_namespace() -> argparse.Namespace:
    try:
        return _locals.arg_ns_stack[-1]
    except (AttributeError, IndexError):
        raise AssertionError(
            "No current argument namespace: Is it possible you are trying to resolve "
            "arguments from another thread or process"
        )


_Decorated = TypeVar("_Decorated", bound=Callable)


def expand_cl_arg_defaults(wrapped: _Decorated) -> _Decorated:
    sig = inspect.signature(wrapped)

    def wrapper(*args, **kwargs):
        args_ns = current_args_namespace()
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        def filter(arg):
            if isinstance(arg, ClArgRef):
                return arg.resolve(args_ns)
            return arg

        new_args = [filter(arg) for arg in bound.args]
        new_kwargs = {k: filter(v) for k, v in bound.kwargs.items()}
        return wrapped(*new_args, **new_kwargs)

    functools.update_wrapper(wrapper, wrapped)
    return wrapper


class ClArgRef:
    """Used in default values of function arguments to indicate that the default should
    be derived from an argument reference.

    Actually defining the argument must be done elsewhere.

    See `cl_arg_ref()` for canonical use.
    """

    def __init__(self, dest: str):
        self.dest = dest

    def resolve(self, arg_namespace: argparse.Namespace):
        try:
            return getattr(arg_namespace, self.dest)
        except AttributeError as e:
            raise RuntimeError(
                f"Unable to resolve command line argument '{self.dest}' in namespace"
            ) from e


def cl_arg_ref(dest: str):
    """Used as a default value for functions wrapped in @expand_cl_defaults to indicate
    that an argument must come from the command line environment.

    Note that this does not have a typing annotation, allowing the argument to be
    annotated with a type, assuming that resolution will happen dynamically in some
    fashion.
    """
    return ClArgRef(dest)


class ClArg(ClArgRef):
    """Used in default values of function arguments to indicate that an argument needs
    to be defined and referenced.

    This is used in user-defined entry points, and the executor has special logic to
    collect all needed arguments automatically.

    See `cl_arg()` for canonical use.
    """

    def __init__(self, name, dest: str, **add_argument_kw):
        super().__init__(dest)
        self.name = name
        self.add_argument_kw = add_argument_kw

    def define_arg(self, parser: argparse.ArgumentParser):
        parser.add_argument(f"--{self.name}", dest=self.dest, **self.add_argument_kw)


def cl_arg(name: str, *, action=None, default=None, type=None, help=None):
    """Used to define or reference a command-line argument from within actions
    and entry-points.

    Keywords have the same interpretation as `ArgumentParser.add_argument()`.

    Any ClArg set as a default value for an argument to an `entrypoint` will be
    added to the global argument parser. Any particular argument name can only be
    registered once and must not conflict with a built-in command line option.
    The implication of this is that for single-use arguments, the `=cl_arg(...)`
    can just be added as a default argument. Otherwise, for shared arguments,
    it should be created at the module level and referenced.

    When called, any entrypoint arguments that do not have an explicit keyword
    set will get their value from the command line environment.

    Note that this does not have a typing annotation, allowing the argument to be
    annotated with a type, assuming that resolution will happen dynamically in some
    fashion.
    """
    if name.startswith("-"):
        raise ValueError("cl_arg name must not be prefixed with dashes")
    dest = name.replace("-", "_")
    return ClArg(name, action=action, default=default, type=type, dest=dest, help=help)


def extract_cl_arg_defs(callable: Callable) -> Generator[ClArg, None, None]:
    """Extracts all `ClArg` default values from a callable.

    This is used in order to eagerly register argument definitions for some set
    of functions.
    """
    sig = inspect.signature(callable)
    for p in sig.parameters.values():
        def_value = p.default
        if isinstance(def_value, ClArg):
            yield def_value
