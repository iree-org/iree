# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The catalog of IREE's Torchax registrations.

One process-local catalog holds an entry per torch operation: a composite
handler that runs on Torchax, or an ordinary decomposition for the IREE
exporter. The family modules' built-ins and plugin registrations share it.
Registering changes no Torchax state: each export installs the catalog on its
own environment, and `install(env)` installs it on an environment the caller
chooses. A composite goes in through a managed adapter, recognised by its
identity, so installation tells IREE's current or superseded handlers from a
user's override.
"""

import dataclasses
import functools
import importlib
from typing import Callable

import torch

COMPOSITE = "composite"
DECOMPOSITION = "decomposition"

_BUILTINS = {
    "iree.attention": "iree_torchax.torchax_attention",
}


@functools.cache
def _builtin_keys():
    """The torch keys of the built-ins, by provider. The family modules load
    Torchax and JAX, so registration checks against these instead."""
    aten = torch.ops.aten
    attention = [
        torch.nn.functional.scaled_dot_product_attention,
        aten.scaled_dot_product_attention.default,
    ]
    keys = dict.fromkeys(attention, "iree.attention")
    return keys


@dataclasses.dataclass(frozen=True)
class Entry:
    """A torch operation and what IREE does with it.

    A composite handler takes the operation's arguments and the calling Torchax
    environment as `env=`; a decomposition takes the arguments only.
    """

    provider: str
    key: object
    kind: str
    handler: Callable
    # A torch function such as torch.nn.functional.*, rather than an ATen or
    # PT2E operator overload.
    is_function: bool = False
    export_handler: Callable | None = None


class _Adapter:
    """The handler IREE publishes for one entry. Each registration makes a new
    one, so an older adapter marks a superseded registration."""

    __slots__ = ("entry",)

    def __init__(self, entry):
        self.entry = entry

    def __call__(self, *args, env, **kwargs):
        return self.entry.handler(*args, env=env, **kwargs)


class _ExportAdapter(_Adapter):
    """An entry's handler for one export, its export handler when it has one."""

    __slots__ = ()

    def __call__(self, *args, env, **kwargs):
        handler = self.entry.export_handler or self.entry.handler
        return handler(*args, env=env, **kwargs)


_catalog: dict[object, Entry] = {}
_adapters: dict[object, _Adapter] = {}
_builtins_loaded = False


def catalog() -> tuple[Entry, ...]:
    """The current entries, built-ins first."""
    _load_builtins()
    return tuple(_catalog.values())


def snapshot() -> tuple[Entry, ...]:
    """The entries an export captures; later registrations leave it alone."""
    return catalog()


def lookup(ops, key):
    """The handler `ops` holds for `key`, by overload and then by its packet,
    as Torchax resolves it."""
    op = ops.get(key)
    if op is None and isinstance(key, torch._ops.OpOverload):
        op = ops.get(key.overloadpacket)
    return op


def _adapter(entry):
    adapter = _adapters.get(entry.key)
    if adapter is None or adapter.entry is not entry:
        adapter = _adapters[entry.key] = _Adapter(entry)
    return adapter


def _owned_adapter(op):
    func = getattr(op.func, "func", op.func)
    return func if isinstance(func, _Adapter) else None


def _is_foreign(op):
    """Whether `op` is a user's Torchax handler, neither a Torchax default nor
    an IREE adapter."""
    return op is not None and op.is_user_defined and _owned_adapter(op) is None


def _same(lhs, rhs):
    """Whether two entries register the same callables for the same key."""
    return (
        lhs.key is rhs.key
        and lhs.provider == rhs.provider
        and lhs.kind == rhs.kind
        and lhs.handler is rhs.handler
        and lhs.is_function == rhs.is_function
        and lhs.export_handler is rhs.export_handler
    )


def _validate(op, handler, provider, kind, is_function, export_handler, replace):
    """Returns the entry to record, or None for a repeated registration."""
    if not isinstance(provider, str):
        raise TypeError(f"provider must be a string, got {provider!r}")
    if not provider:
        raise ValueError("provider must be nonempty")
    if not callable(handler):
        raise TypeError(f"the {kind} of {op} must be callable, got {handler!r}")
    if export_handler is not None and not callable(export_handler):
        raise TypeError(f"the export_handler of {op} must be callable")
    if not isinstance(is_function, bool) or not isinstance(replace, bool):
        raise TypeError("is_function and replace must be bool")
    is_overload = isinstance(op, torch._ops.OpOverload)
    if is_function:
        if is_overload or isinstance(op, torch._ops.OpOverloadPacket):
            raise TypeError(f"{op} is an operator, not a torch function")
        if not callable(op):
            raise TypeError(f"torch function {op!r} must be callable")
    elif not is_overload:
        raise TypeError(f"{op!r} must be an individual torch OpOverload")

    entry = Entry(provider, op, kind, handler, is_function, export_handler)
    existing = _catalog.get(op)
    builtin = None if _builtins_loaded else _builtin_keys().get(op)
    if existing is None and builtin is not None:
        if kind != COMPOSITE:
            raise ValueError(
                f"{op} is registered as a composite by {builtin}; {provider} "
                f"cannot register it as a {kind}"
            )
        if not replace:
            raise ValueError(
                f"{op} is registered by {builtin}; the registration by "
                f"{provider} conflicts with it; pass replace=True to replace it"
            )
    if existing is not None and _same(existing, entry):
        return None
    if existing is not None:
        if existing.kind != kind:
            raise ValueError(
                f"{op} is registered as a {existing.kind} by "
                f"{existing.provider}; {provider} cannot register it as a {kind}"
            )
        if not replace:
            raise ValueError(
                f"{op} is registered by {existing.provider}; the registration "
                f"by {provider} conflicts with it; pass replace=True to replace it"
            )
    return entry


def _register(op, handler, provider, kind, is_function, export_handler, replace):
    entry = _validate(op, handler, provider, kind, is_function, export_handler, replace)
    if entry is not None:
        _catalog[op] = entry


def register_composite(
    op, handler, *, provider, is_function=False, export_handler=None, replace=False
):
    """Registers `handler(*args, env=...)` for the torch overload `op`, or for
    the torch function `op` with `is_function=True`. Export calls
    `export_handler` instead when one is given."""
    _register(op, handler, provider, COMPOSITE, is_function, export_handler, replace)


def register_decomposition(op, decomposition, *, provider, replace=False):
    """Registers a Torch callable that expands the overload `op` before export."""
    _register(op, decomposition, provider, DECOMPOSITION, False, None, replace)


def _install_entry(env, entry, replace):
    op = lookup(env._ops, entry.key)
    adapter = _adapter(entry)
    if op is not None and _owned_adapter(op) is adapter:
        return
    if _is_foreign(op) and not replace:
        return
    env.override_op_definition(entry.key, functools.partial(adapter, env=env))


def install(env, *, replace=False, provider=None):
    """Installs the catalog's composites, or those of `provider`, on `env`.

    Missing entries, Torchax defaults and superseded IREE handlers are
    replaced. A user's override stays unless `replace` is set.
    """
    for entry in catalog():
        if entry.kind == COMPOSITE and provider in (None, entry.provider):
            _install_entry(env, entry, replace)


def install_snapshot(env, entries):
    """Installs the export handlers of `entries` on an export environment; a
    user's handler it inherited from Torchax stays."""
    for entry in entries:
        if entry.kind != COMPOSITE or _is_foreign(lookup(env._ops, entry.key)):
            continue
        env.override_op_definition(
            entry.key, functools.partial(_ExportAdapter(entry), env=env)
        )


def _load_builtins():
    """Records the family modules' entries without publishing them."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    builtins = {}
    for provider, module in _BUILTINS.items():
        for key, handler, is_function, export_handler in importlib.import_module(
            module
        ).entries():
            builtins[key] = Entry(
                provider, key, COMPOSITE, handler, is_function, export_handler
            )
    if {key: entry.provider for key, entry in builtins.items()} != _builtin_keys():
        raise RuntimeError("the built-in keys differ from the family modules'")
    # A plugin's replacement of a built-in stays.
    catalog = {key: entry for key, entry in builtins.items() if key not in _catalog}
    catalog.update(_catalog)
    _catalog.clear()
    _catalog.update(catalog)
    _builtins_loaded = True
