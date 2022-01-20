# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import ModuleType
from typing import Any, Mapping, Optional, Sequence, Union

import contextlib
import inspect

from ... import builtin as builtin_d
from ... import iree_pydm as d
from .... import ir
# TODO: Upstream emit_error and use that instead.
from ...._mlir_libs._ireeDialects import emit_error as _emit_error, lookup_nearest_symbol_from as _lookup_nearest_symbol_from


class EmittedError(Exception):
  """Exception subclass that indicates an error diagnostic has been emitted.
  By throwing, this lets us abort and handle at a higher level so as not
  to duplicate diagnostics.
  """

  def __init__(self, loc: ir.Location, message: str):
    super().__init__(loc, message)

  @property
  def loc(self) -> ir.Location:
    return self.args[0]

  @property
  def message(self) -> str:
    return self.args[1]


class UserReportableError(Exception):
  """Used to raise an error with a message that should be reported to the user.
  Raising this error indicates that the error message is well formed and
  makes sense without a traceback.
  """

  def __init__(self, message: str):
    super().__init__(message)

  @property
  def message(self) -> str:
    return self.args[0]


class ImportContext:
  """Context for importing Python structures into IR."""

  def __init__(self,
               *,
               context: Optional[ir.Context] = None,
               module: Optional[builtin_d.ModuleOp] = None):
    self.context = context if context else create_context()
    self.loc = ir.Location.unknown(context=self.context)
    self._root_module: Optional[ir.Module] = None
    if module:
      self.module = module
    else:
      self._root_module = ir.Module.create(self.loc)
      self.module = self._root_module.operation
    # TODO: Add a "body" attribute to builtin.module.
    self.module_body = self.module.regions[0].blocks[0]
    self._ip_stack = []

  def __str__(self):
    if self._root_module:
      return str(self._root_module)
    else:
      return str(self.module)

  def set_file_line_col(self, file: str, line: int, col: int):
    self.loc = ir.Location.file(file, line, col, context=self.context)

  @contextlib.contextmanager
  def scoped_ip(self, scoped_ip: ir.InsertionPoint):
    self.push_ip(scoped_ip)
    try:
      yield scoped_ip
    finally:
      self.pop_ip()

  def push_ip(self, scoped_ip: ir.InsertionPoint):
    self._ip_stack.append(scoped_ip)

  def pop_ip(self):
    assert self._ip_stack, "Mismatched push_ip/pop_ip: stack is empty on pop"
    del self._ip_stack[-1]

  @property
  def ip(self) -> ir.InsertionPoint:
    assert self._ip_stack, "InsertionPoint requested but stack is empty"
    return self._ip_stack[-1]

  def reset_ip(self, ip: ir.InsertionPoint):
    """Resets the TOS insertion point.

    This is needed if splitting exection across blocks.
    """
    assert self._ip_stack, "InsertionPoint requested but stack is empty"
    self._ip_stack[-1] = ip

  def abort(self, message: str):
    """Emits an error diagnostic and raises an exception to abort."""
    loc = self.loc
    _emit_error(loc, message)
    raise EmittedError(loc, message)

  def lookup_symbol(self, symbol_attr):
    return _lookup_nearest_symbol_from(self.module, symbol_attr)

  def box(self, value: ir.Value, to_typed: Optional[bool] = True) -> ir.Value:
    """Boxes a value if necessary."""
    with self.ip, self.loc:
      t = value.type
      if d.ObjectType.isinstance(t):
        # Already boxed.
        return value
      boxed_type = d.ObjectType.get_typed(t) if to_typed else d.ObjectType.get()
      return d.BoxOp(boxed_type, value).result

  def unbox(self, to_type: ir.Type, value: ir.Value) -> ir.Value:
    with self.ip, self.loc:
      exc_result, unboxed = d.UnboxOp(d.ExceptionResultType.get(), to_type,
                                      value).results
      d.RaiseOnFailureOp(exc_result)
      return unboxed

  def emit_constant(self, value: Any) -> ir.Value:
    """Emits a constant for a supported Python value."""
    # Handle the various primitives directly.
    with self.loc, self.ip:
      if value is None:
        return d.NoneOp(d.NoneType.get()).result
      elif value is True or value is False:
        return d.ConstantOp(
            d.BoolType.get(),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(1),
                               1 if value else 0)).result
      elif isinstance(value, int):
        return d.ConstantOp(
            d.IntegerType.get(),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)).result
      elif isinstance(value, float):
        return d.ConstantOp(d.RealType.get(),
                            ir.FloatAttr.get(ir.F64Type.get(), value)).result
      elif isinstance(value, str):
        return d.ConstantOp(d.StrType.get(), ir.StringAttr.get(value)).result
      elif isinstance(value, bytes):
        return d.ConstantOp(d.BytesType.get(), ir.StringAttr.get(value)).result
    self.abort(
        f"unsupported Python constant value '{value}' (an {value.__class__}))")

  def create_functional_if_op(self, results, condition: ir.Value,
                              with_else_region: bool):
    """Sugar to create a `functional_if`.
    Returns:
      (if_op, then_ip, else_ip) if with_else_region, otherwise (if_op, then_ip)
    """
    # TODO: ODS for these style of ops with variable regions needs work.
    op = ir.Operation.create("iree_pydm.functional_if",
                             results=results,
                             operands=[condition],
                             regions=2 if with_else_region else 1,
                             loc=self.loc,
                             ip=self.ip)
    then_region = op.regions[0]
    then_block = then_region.blocks.append()
    if with_else_region:
      else_region = op.regions[1]
      else_block = else_region.blocks.append()
      return op, ir.InsertionPoint(then_block), ir.InsertionPoint(else_block)
    else:
      return op, ir.InsertionPoint(then_block)


class ImportStage:
  """Base class for activities representing isolated import activity.

  This is used, for example, to isolate activities, targeting the same
  module but different functions.
  """
  __slots__ = [
      "ic",
      "hooks",
  ]

  def __init__(self, ic: ImportContext, hooks: "ImportHooks"):
    self.ic = ic
    self.hooks = hooks


class _UnboundValue:

  def __repr__(self):
    return "<UnboundValue>"


def _get_module_type():
  import abc  # Not special - just standard.
  return type(abc)


_ModuleType = _get_module_type()


class Intrinsic:
  """An object that controls its own interaction with the AST and IR.

  Intrinsics are typically returned as a result of evaluating globals in the
  hosting Python process. They have methods on them for controlling how
  evaluation and IR emission should proceed. They can also implenent
  __call__, __getattr__, etc in order to support dual use, either in the host
  process or the compiled process.
  """
  UNBOUND_VALUE = _UnboundValue()

  def resolve_static_getattr(self, stage: ImportStage,
                             attr_name: str) -> "ResolveOutcome":
    return Intrinsic.UNBOUND_VALUE

  def emit_call(self, stage: ImportStage, args: Sequence[ir.Value],
                keywords: Sequence[Any]) -> ir.Value:
    stage.ic.abort(f"the compiler intrinsic {self} does not support calls")

  def emit_immediate(self, stage: ImportStage) -> ir.Value:
    """Emits this object as an immediate value.

    On failure, abort with error.
    """
    stage.ic.abort(
        f"the compiler intrinsic {self} can not be serialized as a value")

  @staticmethod
  def make_singleton(cls) -> "Intrinsic":
    """Class decorator to instantiate a singleton intrinsic class."""
    assert issubclass(cls, Intrinsic)
    return cls()


ResolveOutcome = Union[_UnboundValue, Intrinsic, ir.Value]


class FuncProvidingIntrinsic(Intrinsic):
  """An intrinsic which provides an IR function in some way.

  This provides an additional entry point for retrieving the provided
  function symbol.
  """

  def get_or_create_provided_func_symbol(self, stage: ImportStage) -> str:
    raise NotImplementedError()


class ImportHooks:
  """Hooks for customizing the import process."""

  def resolve_annotation_to_type(self, stage: ImportStage, annot) -> ir.Type:
    """Resolves a live, function annotation to a type.

    TODO: This currently has some dependency on whether crossing a
    ref-providence boundary. May need to untangle.
    """
    return d.ObjectType.get(context=stage.ic.context)

  def resolve_global(
      self,
      stage: ImportStage,
      name: str,
      *,
      host_closure_vars: Optional[inspect.ClosureVars] = None
  ) -> ResolveOutcome:
    """Resolves a global name.

    By default, this returns NO_VALUE, indicating that the global cannot
    be found.

    Typical implementations will consult the provided 'globals' dict and
    make a decision on a type of Intrinsic to return, bridging the host
    runtime namespace to what the compiler should consider. There are many
    strategies for doing this, each providing unique features and user
    experiences.
    """
    return Intrinsic.UNBOUND_VALUE


class DefaultImportHooks(ImportHooks):
  """Hooks that provide some default behavior.

  This has not been fully thought through yet with respect to layering
  for real users. This may just become for testing.
  """

  def resolve_annotation_to_type(self, stage: ImportStage, annot) -> ir.Type:
    """Resolves a live, function annotation to a type.

    TODO: This currently has some dependency on whether crossing a
    ref-providence boundary. May need to untangle.
    """
    ic = stage.ic
    # Handle built-in primitive mappings.
    with ic.context:
      # Value types.
      if annot is bool:
        return d.BoolType.get()
      if annot is int:
        return d.IntegerType.get()
      if annot is float:
        return d.RealType.get()
      if annot is None:
        return d.NoneType.get()
      if annot is inspect.Signature.empty:
        # Special value for return annotations to signal no annotation.
        return d.ObjectType.get()

      # Reference types. We always box these across function boundaries
      # to preserve providence.
      # TODO: Better heuristic?
      # TODO: Support typing annotations, not just raw types.
      # TODO: Box/unbox differently on argument/return boundaries.
      if annot is str:
        return d.ObjectType.get_typed(d.StrType.get())
      if annot is list:
        return d.ObjectType.get_typed(d.ListType.get())
      if annot is tuple:
        # TODO: Tuple is used exclusively at the moment for multi results in the
        # RTL so we let it through unboxed. This should be a special type hint
        # enabling this behavior.
        return d.TupleType.get()
      if annot is type:
        return d.ObjectType.get_typed(d.TypeType.get())

    # Fall-back.
    return super().resolve_annotation_to_type(stage, annot)

  def resolve_global(
      self,
      stage: ImportStage,
      name: str,
      *,
      host_closure_vars: Optional[inspect.ClosureVars] = None) -> Any:
    ic = stage.ic
    root = PassthroughModuleIntrinsic(
        host_closure_vars.globals if host_closure_vars else dict())
    found = root.resolve_static_getattr(stage, name)
    if found is not Intrinsic.UNBOUND_VALUE:
      return found
    # Resolve against builtins.
    if (not name in host_closure_vars.builtins):  # pytype: disable=attribute-error
      return Intrinsic.UNBOUND_VALUE

    from . import builtins_intrinsics
    if hasattr(builtins_intrinsics, name):
      return getattr(builtins_intrinsics, name)
    ic.abort(f"builtin {name} is defined for the host Python but not yet"
             f"implemented for this compiler")


def create_context(*, debug: bool = False) -> ir.Context:
  context = ir.Context()
  if debug:
    context.enable_multithreading(False)
  d.register_dialect(context)
  return context


class PassthroughModuleIntrinsic(Intrinsic):
  """Represents a host Python module, returning intrinsics for sub modules."""

  def __init__(self, m: Union[Mapping[str, Any], ModuleType]):
    self.m = m

  def resolve_static_getattr(self, stage: ImportStage,
                             attr_name: str) -> ResolveOutcome:
    ic = stage.ic
    m = self.m
    try:
      if isinstance(m, dict):
        child = m[attr_name]
      else:
        child = getattr(m, attr_name)
    except KeyError:
      return Intrinsic.UNBOUND_VALUE
    except AttributeError:
      return Intrinsic.UNBOUND_VALUE

    # We only return values that are modules or intrinsics and primitive,
    # immutable python values (which then get inlined).
    if isinstance(child, _ModuleType):
      return PassthroughModuleIntrinsic(child)
    elif isinstance(child, Intrinsic):
      return child
    elif child is None or _isinstance_multi(child, bool, int, float):
      return stage.ic.emit_constant(child)
    else:
      ic.abort(f"when resolving '{attr_name}' against module {m}, "
               f"encountered an unsupported type ({child.__class__})")


def _isinstance_multi(value, *types):
  for t in types:
    if isinstance(value, t):
      return True
  return False
