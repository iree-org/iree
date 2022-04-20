# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Optional, Sequence, Set, Tuple, List, Union

import ast
import inspect
import logging
import sys
import textwrap

from .util import DefaultImportHooks, ImportContext, ImportHooks, ImportStage, Intrinsic

from ... import iree_pydm as d
from ... import cf as cf_d
from .... import ir


class Importer:
  """Imports a Python construct into IR."""
  __slots__ = [
      "ic",
      "hooks",
  ]

  def __init__(self, ic: ImportContext, hooks: Optional[ImportHooks] = None):
    self.ic = ic
    self.hooks = hooks or DefaultImportHooks()

  def import_global_function(self,
                             f,
                             *,
                             symbol: Optional[str] = None,
                             visibility: Optional[str] = None) -> d.FuncOp:
    """Imports a live Python global function.

    This is just a placeholder of the simplest possible thing until a proper,
    general mechanism is created.
    """
    ic = self.ic
    filename, root_node = _get_function_ast(f)
    fd_node = root_node.body[0]  # pytype: disable=attribute-error
    self.ic.set_file_line_col(filename, fd_node.lineno, fd_node.col_offset)

    if not symbol:
      symbol = fd_node.name

    # Define the function.
    # TODO: Much more needs to be done here (arg/result mapping, etc)
    logging.debug(":::::::")
    logging.debug("::: Importing global function %s:\n%s", symbol,
                  ast.dump(fd_node, include_attributes=False))

    # Main import uses a FunctionContext but we aren't ready to create it yet.
    dummy_stage = ImportStage(ic, self.hooks)

    # Things we are short-cutting by inspecting the live function:
    #   - freevars
    #   - cellvars
    #   - globals
    #   - arg definitions
    #   - annotation parsing
    # Also, since this is just a toy right now, sticking to pos params.
    code_object = f.__code__
    with ic.scoped_ip(ir.InsertionPoint(ic.module_body)) as ip, ip, ic.loc:
      f_signature = inspect.signature(f)
      f_params = f_signature.parameters
      arg_names = list(f_params.keys())
      var_names = list(code_object.co_varnames)
      f_input_types = [
          self.hooks.resolve_annotation_to_type(dummy_stage, p.annotation)
          for p in f_params.values()
      ]
      f_arg_names = ir.ArrayAttr.get(
          [ir.StringAttr.get(name) for name in arg_names])
      f_var_names = ir.ArrayAttr.get(
          [ir.StringAttr.get(name) for name in var_names])
      f_return_type = self.hooks.resolve_annotation_to_type(
          dummy_stage, f_signature.return_annotation)
      ir_f_type = ir.FunctionType.get(
          f_input_types, [d.ExceptionResultType.get(), f_return_type],
          context=ic.context)
      f_op = d.FuncOp(
          ir.StringAttr.get(symbol),
          function_type=ir.TypeAttr.get(ir_f_type),
          arg_names=f_arg_names,
          free_vars=f_var_names,
          cell_vars=ir.ArrayAttr.get([]),
          sym_visibility=ir.StringAttr.get(visibility) if visibility else None)
      entry_block = f_op.add_entry_block()

    fctx = FunctionContext(self.ic,
                           self.hooks,
                           f_op,
                           filename=filename,
                           host_closure_vars=inspect.getclosurevars(f))
    body_importer = FunctionDefBodyImporter(fctx)
    with ic.scoped_ip(ir.InsertionPoint(entry_block)):
      body_importer.declare_variables()
      body_importer.import_body(fd_node)
    return f_op


class FunctionContext(ImportStage):
  """Represents a function import in progress.

  Note that construction of the outer FuncOp is performed externally. This
  allows for multiple modes of operation:
    - Bootstrapping a func from a live Python callable (via inspection)
    - Parsing a function declaration purely from AST
  """
  __slots__ = [
      "f_op",
      "filename",
      "arg_names",
      "free_vars",
      "cell_vars",
      "host_closure_vars",
      "variable_value_map",
  ]

  def __init__(self,
               ic: ImportContext,
               hooks: ImportHooks,
               f_op: d.FuncOp,
               *,
               host_closure_vars: Optional[inspect.ClosureVars] = None,
               filename: str = "<anonymous>"):
    super().__init__(ic, hooks)
    self.f_op = f_op
    self.host_closure_vars = host_closure_vars
    self.filename = filename

    # Keep sets of free and cell var names so that we know what kinds of
    # loads to issue.
    self.arg_names: Set[str] = set(
        [ir.StringAttr(attr).value for attr in self.f_op.arg_names])
    self.free_vars: Set[str] = set(
        [ir.StringAttr(attr).value for attr in self.f_op.free_vars])
    self.cell_vars: Set[str] = set(
        [ir.StringAttr(attr).value for attr in self.f_op.cell_vars])

    self.variable_value_map: Dict[str, ir.Value] = {}

  def declare_free_var(self, name: str):
    """Declares a free-variable SSA value for the given name."""
    ic = self.ic
    if name in self.variable_value_map:
      ic.abort(f"attempt to duplicate declare variable {name}")
    with ic.ip, ic.loc:
      t = d.FreeVarRefType.get()
      self.variable_value_map[name] = (d.AllocFreeVarOp(t,
                                                        ir.StringAttr.get(name),
                                                        index=None).result)

  def find_variable(self, name: str) -> ir.Value:
    try:
      return self.variable_value_map[name]
    except KeyError:
      self.ic.abort(f"attempt to reference variable not declared: {name}")

  def update_loc(self, ast_node):
    self.ic.set_file_line_col(self.filename, ast_node.lineno,
                              ast_node.col_offset)

  def cast_to_return_type(self, value: ir.Value) -> ir.Value:
    """Casts an arbitrary value to the declared function return type."""
    ic = self.ic
    input_type = value.type
    return_type = self.f_op.py_return_type
    if input_type == return_type:
      return value
    if d.ObjectType.isinstance(return_type):
      # Function returns a boxed value.
      if d.ObjectType.isinstance(input_type):
        # Already an object type but annotated differently. Something has
        # gone wrong.
        ic.abort(f"function declared return type {return_type} "
                 f"is incompatible with actual return type {input_type}")
      return ic.box(value)
    else:
      # Function returns a primitive value.
      return ic.unbox(return_type, value)


class BaseNodeVisitor(ast.NodeVisitor):
  """Base class of a node visitor that aborts on unhandled nodes."""
  IMPORTER_TYPE = "<unknown>"
  __slots__ = [
      "fctx",
  ]

  def __init__(self, fctx: FunctionContext):
    super().__init__()
    self.fctx = fctx

  def visit(self, node):
    # Some psuedo-nodes (like old 'Index' types do not have location info).
    if hasattr(node, "lineno"):
      self.fctx.update_loc(node)
    return super().visit(node)

  def generic_visit(self, ast_node: ast.AST):
    logging.debug("UNHANDLED NODE: %s", ast.dump(ast_node))
    self.fctx.ic.abort(f"unhandled python {self.IMPORTER_TYPE} "
                       f"AST node {ast_node.__class__.__name__}: {ast_node}")


class FunctionDefBodyImporter(BaseNodeVisitor):
  """AST visitor for importing a function's statements.
  Handles nodes that are direct children of a FunctionDef.
  """
  IMPORTER_TYPE = "statement"
  __slots__ = [
      "break_block",
      "continue_block",
      "successor_block",
      "terminated",
  ]

  def __init__(self,
               fctx: FunctionContext,
               *,
               successor_block: Optional[ir.Block] = None,
               break_block: Optional[ir.Block] = None,
               continue_block: Optional[ir.Block] = None):
    super().__init__(fctx)
    self.successor_block = successor_block
    self.break_block = break_block
    self.continue_block = continue_block
    self.terminated = False

  def declare_variables(self):
    fctx = self.fctx
    for name in fctx.free_vars:
      fctx.declare_free_var(name)

  def import_body(self, ast_fd: ast.FunctionDef):
    ic = self.fctx.ic
    # Function prologue: Initialize arguments.
    for arg_index, arg_name in enumerate(
        [ir.StringAttr(attr).value for attr in self.fctx.f_op.arg_names]):
      self.initialize_argument(arg_index, arg_name)
    # Import statements.
    self.import_block(ast_fd.body)

  def import_block(self, stmts: Sequence[ast.AST]):
    ic = self.fctx.ic
    for ast_stmt in stmts:
      self.terminated = False
      logging.debug("STMT: %s", ast.dump(ast_stmt, include_attributes=True))
      self.visit(ast_stmt)
    if not self.terminated:
      with ic.ip, ic.loc:
        # Add a default terminator.
        if self.successor_block:
          # Branch to the successor.
          cf_d.BranchOp([], dest=self.successor_block)
        else:
          # Return from function.
          none_value = d.NoneOp(d.NoneType.get()).result
          d.ReturnOp(none_value)

  def initialize_argument(self, index, name):
    fctx = self.fctx
    ic = fctx.ic
    entry_block = fctx.f_op.entry_block
    arg_value = entry_block.arguments[index]
    arg_value = ic.box(arg_value)
    with ic.loc, ic.ip:
      d.StoreVarOp(fctx.find_variable(name), arg_value)

  def visit_Assign(self, node: ast.Assign):
    if self.terminated:
      return
    fctx = self.fctx
    ic = fctx.ic
    expr = ExpressionImporter(fctx)
    expr.visit(node.value)
    for target in node.targets:
      fctx.update_loc(target)
      # All assignment nodes (Attribute, Subscript, Starred, Name, List, Tuple)
      # have a `ctx`.
      target_ctx = target.ctx  # pytype: disable=attribute-error
      if not isinstance(target_ctx, ast.Store):
        # TODO: Del, AugStore, etc
        ic.abort(
            f"unsupported assignment context type {target_ctx.__class__.__name__}"
        )

      if isinstance(target, ast.Name):
        boxed = ic.box(expr.get_immediate())
        with ic.loc, ic.ip:
          target_id = target.id
          d.StoreVarOp(fctx.find_variable(target_id), boxed)
      elif isinstance(target, ast.Subscript):
        subscript_target_expr = ExpressionImporter(fctx)
        subscript_target_expr.visit(target.value)
        subscript_slice_expr = ExpressionImporter(fctx)
        subscript_slice_expr.visit(target.slice)
        fctx.update_loc(node)
        with ic.loc, ic.ip:
          exc_result = d.AssignSubscriptOp(
              d.ExceptionResultType.get(),
              subscript_target_expr.get_immediate(),
              subscript_slice_expr.get_immediate(), expr.get_immediate()).result
          d.RaiseOnFailureOp(exc_result)
      else:
        ic.abort(f"unsupported assignment target: {target.__class__.__name__}")

  def visit_Break(self, node: ast.Break):
    if self.terminated:
      return
    fctx = self.fctx
    ic = fctx.ic
    if not self.break_block:
      ic.abort(f"cannot 'break' outside of a loop")
    with ic.ip, ic.loc:
      cf_d.BranchOp([], self.break_block)
    self.terminated = True

  def visit_Continue(self, node: ast.Continue):
    if self.terminated:
      return
    fctx = self.fctx
    ic = fctx.ic
    if not self.continue_block:
      ic.abort(f"cannot 'continue' outside of a loop")
    with ic.ip, ic.loc:
      cf_d.BranchOp([], self.continue_block)
    self.terminated = True

  def visit_Expr(self, node: ast.Expr):
    if self.terminated:
      return
    fctx = self.fctx
    ic = fctx.ic

    expr = ExpressionImporter(fctx)
    expr.visit(node.value)
    with ic.loc, ic.ip:
      d.ExprStatementDiscardOp(expr.get_immediate())

  def visit_If(self, node: ast.If):
    if self.terminated:
      return
    fctx = self.fctx
    ic = fctx.ic
    # Emit the test.
    test_expr = ExpressionImporter(fctx)
    test_expr.visit(node.test)

    # We create a successor block that a non terminating block will branch to.
    predecessor_block = ic.ip.block
    successor_block = predecessor_block.create_after()

    with ic.ip, ic.loc:
      test_bool = d.AsBoolOp(d.BoolType.get(), test_expr.get_immediate()).result
      test_pred = d.BoolToPredOp(ir.IntegerType.get_signless(1),
                                 test_bool).result

    # Emit the false block
    if not node.orelse:
      # Else just jumps to the successor.
      false_block = successor_block
    else:
      # Emit the false body.
      false_block = predecessor_block.create_after()
      with ic.scoped_ip(ir.InsertionPoint(false_block)):
        else_importer = FunctionDefBodyImporter(
            fctx,
            successor_block=successor_block,
            break_block=self.break_block,
            continue_block=self.continue_block)
        else_importer.import_block(node.orelse)
    # Emit the true body.
    true_block = predecessor_block.create_after()
    with ic.scoped_ip(ir.InsertionPoint(true_block)):
      body_importer = FunctionDefBodyImporter(
          fctx,
          successor_block=successor_block,
          break_block=self.break_block,
          continue_block=self.continue_block)
      body_importer.import_block(node.body)

    # Now that we have true/false blocks, emit the cond_br in the original
    # block.
    fctx.update_loc(node)
    with ic.ip, ic.loc:
      cf_d.CondBranchOp(condition=test_pred,
                        trueDestOperands=[],
                        falseDestOperands=[],
                        trueDest=true_block,
                        falseDest=false_block)

    # And emission continues here.
    ic.reset_ip(ir.InsertionPoint(successor_block))

  def visit_Pass(self, ast_node):
    if self.terminated:
      return
    pass

  def visit_Return(self, ast_node):
    if self.terminated:
      return
    ic = self.fctx.ic
    with ic.loc, ic.ip:
      expr = ExpressionImporter(self.fctx)
      expr.visit(ast_node.value)
      d.ReturnOp(self.fctx.cast_to_return_type(expr.get_immediate()))
    self.terminated = True

  def visit_While(self, node: ast.While):
    if self.terminated:
      return
    fctx = self.fctx
    ic = fctx.ic

    # Blocks:
    #   entry: branch to condition
    #   condition: evalute the test expression and branch to body or orelse
    #   body: main body of the loop
    #   orelse: optional block to execute when condition is no longer true
    #   successor: statements following the loop
    predecessor_block = ic.ip.block
    condition_block = predecessor_block.create_after()
    body_block = condition_block.create_after()
    if node.orelse:
      orelse_block = body_block.create_after()
      successor_block = orelse_block.create_after()
    else:
      successor_block = body_block.create_after()

    # Unconditional branch to the condition block.
    with ic.ip, ic.loc:
      cf_d.BranchOp([], condition_block)

    # Emit test.
    with ic.scoped_ip(ir.InsertionPoint(condition_block)):
      test_expr = ExpressionImporter(fctx)
      test_expr.visit(node.test)
      with ic.ip, ic.loc:
        test_bool = d.AsBoolOp(d.BoolType.get(),
                               test_expr.get_immediate()).result
        test_pred = d.BoolToPredOp(ir.IntegerType.get_signless(1),
                                   test_bool).result
        cf_d.CondBranchOp(
            condition=test_pred,
            trueDestOperands=[],
            falseDestOperands=[],
            trueDest=body_block,
            falseDest=orelse_block if node.orelse else successor_block)

    # Emit body.
    with ic.scoped_ip(ir.InsertionPoint(body_block)):
      body_importer = FunctionDefBodyImporter(fctx,
                                              successor_block=condition_block,
                                              break_block=successor_block,
                                              continue_block=condition_block)
      body_importer.import_block(node.body)

    # Emit orelse.
    if node.orelse:
      with ic.scoped_ip(ir.InsertionPoint(orelse_block)):
        orelse_importer = FunctionDefBodyImporter(
            fctx, successor_block=successor_block)
        orelse_importer.import_block(node.orelse)

    # Done.
    ic.reset_ip(ir.InsertionPoint(successor_block))


ExpressionResult = Union[Intrinsic, ir.Value]


class ExpressionImporter(BaseNodeVisitor):
  """Imports expression nodes.
  Visitor methods must either raise an exception or call _set_result.
  """
  IMPORTER_TYPE = "expression"
  __slots__ = [
      "_result",
  ]

  def __init__(self, fctx: FunctionContext):
    super().__init__(fctx)
    self._result: Optional[ExpressionResult] = None

  def visit(self, node):
    super().visit(node)
    assert self._result is not None, (
        f"ExpressionImporter did not assign a value ({ast.dump(node)})")

  def get_immediate(self) -> ir.Value:
    """Gets the expression result by emitting it as an immediate value."""
    if isinstance(self._result, ir.Value):
      return self._result
    else:
      # Intrinsic.
      return self._result.emit_immediate(self.fctx.ic)

  def get_call_result(self, args: Sequence[ir.Value]) -> ir.Value:
    """Perfoms a call against the expression result, returning the value."""
    if isinstance(self._result, ir.Value):
      return self.fctx.ic.abort(
          f"TODO: User defined function call not supported")
    else:
      # Intrinsic.
      return self._result.emit_call(self.fctx, args=args, keywords=[])

  def get_static_attribute(self, attr_name: str) -> ExpressionResult:
    fctx = self.fctx
    ic = fctx.ic
    if isinstance(self._result, ir.Value):
      # Immediate.
      ic.abort(f"TODO: Runtime attribute resolution NYI")
    else:
      # Intrinsic.
      resolved = self._result.resolve_static_getattr(ic, attr_name)
      if resolved is Intrinsic.UNBOUND_VALUE:
        ic.abort(f"attribute {attr_name} not found for compile time intrinsic "
                 f"{self._result}")
      return resolved

  def _set_result(self, result: ExpressionResult):
    assert not self._result
    assert isinstance(result, (Intrinsic, ir.Value)), (
        f"Not an ExpressionResult: is a {result.__class__} ({result})")
    self._result = result

  def visit_Name(self, node: ast.Name):
    fctx = self.fctx
    ic = fctx.ic
    if not isinstance(node.ctx, ast.Load):
      # Note that the other context types (Store, Del, Star) cannot appear
      # in expressions.
      fctx.abort(f"Unsupported expression name context type %s")

    # Handle free variables (also includes args).
    with ic.loc:
      if node.id in self.fctx.free_vars:
        self._set_result(
            d.LoadVarOp(d.ObjectType.get(),
                        fctx.find_variable(node.id),
                        ip=ic.ip).result)
        return

    # Fall-back to global resolution.
    resolved = fctx.hooks.resolve_global(
        fctx, node.id, host_closure_vars=fctx.host_closure_vars)
    if resolved == Intrinsic.UNBOUND_VALUE:
      ic.abort(f"could not resolve global {node.id}")
    self._set_result(resolved)

  def visit_Attribute(self, node: ast.Attribute):
    sub_eval = ExpressionImporter(self.fctx)
    sub_eval.visit(node.value)
    self._set_result(sub_eval.get_static_attribute(node.attr))

  def visit_BinOp(self, node: ast.BinOp):
    fctx = self.fctx
    ic = fctx.ic
    op = node.op
    for op_type, dunder_name in _AST_BINOP_TYPE_TO_DUNDER:
      if isinstance(op, op_type):
        break
    else:
      ic.abort(f"unsupported binary operation {op}")

    left = ExpressionImporter(fctx)
    left.visit(node.left)
    right = ExpressionImporter(fctx)
    right.visit(node.right)
    fctx.update_loc(node)

    with ic.loc, ic.ip:
      object_type = d.ObjectType.get()
      # TODO: There are some exceptions to blanket binary promotion:
      #   - Truediv has its own promotion rules
      #   - Shl, Shr are different
      left_prime, right_prime = d.DynamicBinaryPromoteOp(
          object_type, object_type, left.get_immediate(),
          right.get_immediate()).results
      result = d.ApplyBinaryOp(object_type, ir.StringAttr.get(dunder_name),
                               left_prime, right_prime).result
      self._set_result(result)

  def visit_BoolOp(self, node):
    fctx = self.fctx
    ic = fctx.ic
    if isinstance(node.op, ast.And):
      return_first_true = False
    elif isinstance(node.op, ast.Or):
      return_first_true = True
    else:
      ic.abort(f"unknown bool op {ast.dump(node.op)}")

    def emit_next(next_nodes):
      next_node = next_nodes[0]
      next_nodes = next_nodes[1:]

      # Evaluate sub-expression.
      sub_expression = ExpressionImporter(fctx)
      sub_expression.visit(next_node)
      next_value = sub_expression.get_immediate()
      if not next_nodes:
        return next_value

      condition_value = d.AsBoolOp(d.BoolType.get(), next_value,
                                   ip=ic.ip).result
      # TODO: See if we can re-organize this to not force boxing through the
      # if.
      if_op, then_ip, else_ip = ic.create_functional_if_op([d.ObjectType.get()],
                                                           condition_value,
                                                           True)
      # Short-circuit return case.
      with ic.scoped_ip(then_ip if return_first_true else else_ip):
        next_value_casted = ic.box(next_value)
        d.YieldOp([next_value_casted], loc=ic.loc, ip=ic.ip)

      # Nested evaluate next case.
      with ic.scoped_ip(else_ip if return_first_true else then_ip):
        nested_value = emit_next(next_nodes)
        nested_value_casted = next_value_casted = ic.box(nested_value)
        d.YieldOp([nested_value_casted], loc=ic.loc, ip=ic.ip)

      return if_op.result

    with ic.loc:
      self._set_result(emit_next(node.values))

  def visit_Compare(self, node: ast.Compare):
    # Short-circuit comparison (degenerates to binary comparison when just
    # two operands).
    fctx = self.fctx
    ic = fctx.ic
    false_value = ic.emit_constant(False)

    def emit_next(left_value, comparisons):
      operation, right_node = comparisons[0]
      comparisons = comparisons[1:]

      # Determine operation type.
      for (ast_type, op_name, reflective_op_name,
           needs_promotion) in _AST_COMPAREOP_TYPE_TO_INFO:
        if isinstance(operation, ast_type):
          break
      else:
        ic.abort(f"unsupported comparison op: {operation}")

      # Lazy evaluate the right.
      right_expr = ExpressionImporter(fctx)
      right_expr.visit(right_node)
      right_value = right_expr.get_immediate()
      with ic.ip, ic.loc:
        object_type = d.ObjectType.get()
        # Promote if needed.
        if needs_promotion:
          left_prime, right_prime = d.DynamicBinaryPromoteOp(
              object_type, object_type, left_value, right_value).results
        else:
          left_prime = left_value
          right_prime = right_expr.get_immediate()

        # Apply comparison.
        compare_result = d.ApplyCompareOp(d.BoolType.get(),
                                          ir.StringAttr.get(op_name),
                                          left_prime, right_prime).result
      # Terminate by yielding the final compare result.
      if not comparisons:
        return compare_result

      # Emit if for short circuit eval.
      # Since this is an 'and', all else clauses yield a false value.
      with ic.ip, ic.loc:
        if_op, then_ip, else_ip = ic.create_functional_if_op([d.BoolType.get()],
                                                             compare_result,
                                                             True)
      # Build the else clause.
      with ic.scoped_ip(else_ip):
        d.YieldOp([false_value], loc=ic.loc, ip=ic.ip)

      # Build the then clause.
      with ic.scoped_ip(then_ip):
        nested_result = emit_next(right_value, comparisons)
        d.YieldOp([nested_result], loc=ic.loc, ip=ic.ip)

      return if_op.result

    # Compute left and recurse for lazy evaluation.
    left_expr = ExpressionImporter(fctx)
    left_expr.visit(node.left)
    self._set_result(
        emit_next(left_expr.get_immediate(),
                  list(zip(node.ops, node.comparators))))

  def visit_Call(self, node: ast.Call):
    fctx = self.fctx
    ic = fctx.ic
    func_expr = ExpressionImporter(fctx)
    func_expr.visit(node.func)

    args = []
    for ast_arg in node.args:
      arg_expr = ExpressionImporter(fctx)
      arg_expr.visit(ast_arg)
      args.append(arg_expr.get_immediate())

    if node.keywords:
      ic.abort(f"TODO: keyword calls are not yet supported")
    fctx.update_loc(node)
    self._set_result(func_expr.get_call_result(args=args))

  def visit_List(self, node: ast.List):
    fctx = self.fctx
    ic = fctx.ic

    element_values: List[ir.Value] = []
    for elt in node.elts:
      sub_expression = ExpressionImporter(fctx)
      sub_expression.visit(elt)
      element_values.append(ic.box(sub_expression.get_immediate()))
    fctx.update_loc(node)

    with ic.ip, ic.loc:
      self._set_result(d.MakeListOp(d.ListType.get(), element_values).result)

  def visit_Subscript(self, node: ast.Subscript):
    fctx = self.fctx
    ic = fctx.ic
    value = ExpressionImporter(fctx)
    value.visit(node.value)
    slice = ExpressionImporter(fctx)
    slice.visit(node.slice)

    fctx.update_loc(node)
    with ic.ip, ic.loc:
      exc_result, result = d.SubscriptOp(d.ExceptionResultType.get(),
                                         d.ObjectType.get(),
                                         value.get_immediate(),
                                         slice.get_immediate()).results
      d.RaiseOnFailureOp(exc_result)
    self._set_result(result)

  def visit_Tuple(self, node: ast.Tuple):
    fctx = self.fctx
    ic = fctx.ic

    element_values: List[ir.Value] = []
    for elt in node.elts:
      sub_expression = ExpressionImporter(fctx)
      sub_expression.visit(elt)
      element_values.append(sub_expression.get_immediate())
    fctx.update_loc(node)

    with ic.ip, ic.loc:
      self._set_result(d.MakeTupleOp(d.TupleType.get(), element_values).result)

  def visit_UnaryOp(self, node: ast.UnaryOp):
    fctx = self.fctx
    ic = fctx.ic
    with ic.ip, ic.loc:
      op = node.op

      # Evaluate sub-expression.
      sub_expression = ExpressionImporter(fctx)
      sub_expression.visit(node.operand)
      fctx.update_loc(node)
      operand_value = sub_expression.get_immediate()

      if isinstance(op, ast.Not):
        # Special handling for logical-not.
        bool_value = d.AsBoolOp(d.BoolType.get(), operand_value).result
        true_value = ic.emit_constant(True)
        false_value = ic.emit_constant(False)
        self._set_result(
            d.SelectOp(d.BoolType.get(), bool_value, false_value,
                       true_value).result)
      elif isinstance(op, ast.USub):
        self._set_result(d.NegOp(operand_value.type, operand_value).result)
      else:
        ic.abort(f"Unknown unary op {ast.dump(op)}")

  def visit_IfExp(self, node: ast.IfExp):
    fctx = self.fctx
    ic = fctx.ic

    # Evaluate test sub-expression.
    sub_expression = ExpressionImporter(fctx)
    sub_expression.visit(node.test)
    fctx.update_loc(node)
    test_value = sub_expression.get_immediate()

    # Interpret as bool.
    test_bool = d.AsBoolOp(d.BoolType.get(), test_value, ip=ic.ip,
                           loc=ic.loc).result

    # TODO: There is a hazard here if then and else refine to different
    # boxed types. Needs a derefine cast. Also we are boxing to type erased
    # types to satisfy scf.if verifier. Do something better.
    if_op, then_ip, else_ip = ic.create_functional_if_op(
        [d.ObjectType.get(ic.context)], test_bool, True)
    # Build the then clause
    with ic.scoped_ip(then_ip):
      # Evaluate the true clause within the if body.
      sub_expression = ExpressionImporter(fctx)
      sub_expression.visit(node.body)
      then_result = sub_expression.get_immediate()
      d.YieldOp([ic.box(then_result, to_typed=False)], loc=ic.loc, ip=ic.ip)

    # Build the then clause.
    with ic.scoped_ip(else_ip):
      sub_expression = ExpressionImporter(fctx)
      sub_expression.visit(node.orelse)
      orelse_result = sub_expression.get_immediate()
      d.YieldOp([ic.box(orelse_result, to_typed=False)], loc=ic.loc, ip=ic.ip)

    self._set_result(if_op.result)

  if sys.version_info < (3, 8, 0):
    # <3.8 breaks these out into separate AST classes.
    def visit_Num(self, ast_node):
      self._set_result(self.fctx.ic.emit_constant(ast_node.n))

    def visit_Str(self, ast_node):
      self._set_result(self.fctx.ic.emit_constant(ast_node.s))

    def visit_Bytes(self, ast_node):
      self._set_result(self.fctx.ic.emit_constant(ast_node.s))

    def visit_NameConstant(self, ast_node):
      self._set_result(self.fctx.ic.emit_constant(ast_node.value))

    def visit_Ellipsis(self, ast_node):
      self._set_result(self.fctx.ic.emit_constant(...))
  else:
    # >= 3.8
    def visit_Constant(self, ast_node):
      self._set_result(self.fctx.ic.emit_constant(ast_node.value))

  if sys.version_info < (3, 9, 0):
    # Starting in 3.9, Index nodes are no longer generated (they used to be
    # a layer of indirection in subscripts). They aren't "real" nodes and
    # we just pass them through.
    def visit_Index(self, ast_node):
      self.visit(ast_node.value)


def _get_function_ast(f) -> Tuple[str, ast.AST]:
  filename = inspect.getsourcefile(f)
  source_lines, start_lineno = inspect.getsourcelines(f)
  source = "".join(source_lines)
  source = textwrap.dedent(source)
  ast_root = ast.parse(source, filename=filename)
  ast.increment_lineno(ast_root, start_lineno - 1)
  return filename, ast_root


# Maps an AST type (from BinOp.op) to the dunder name in the Python data
# model.
_AST_BINOP_TYPE_TO_DUNDER = (
    (ast.Add, "add"),
    (ast.Sub, "sub"),
    (ast.Mult, "mul"),
    (ast.MatMult, "matmul"),
    (ast.Div, "truediv"),
    (ast.FloorDiv, "floordiv"),
    (ast.Mod, "mod"),
    (ast.Pow, "pow"),
    (ast.LShift, "lshift"),
    (ast.RShift, "rshift"),
    (ast.BitAnd, "and"),
    (ast.BitOr, "or"),
    (ast.BitXor, "xor"),
)

# Maps AST Compare op type. Fields are:
#   [0] = ast type
#   [1] = op name (root of the dunder name for rich compare ops)
#   [2] = reflective op name
#   [3] = whether numeric promotion should take place
_AST_COMPAREOP_TYPE_TO_INFO = (
    (ast.Lt, "lt", "gte", True),
    (ast.LtE, "le", "gt", True),
    (ast.Eq, "eq", "eq", True),
    (ast.NotEq, "ne", "ne", True),
    (ast.Gt, "gt", "le", True),
    (ast.GtE, "ge", "lt", True),
    (ast.Is, "is", "is", False),
    (ast.IsNot, "isnot", "isnot", False),
    (ast.In, "in", "in", False),
    (ast.NotIn, "notin", "notin", False),
)
