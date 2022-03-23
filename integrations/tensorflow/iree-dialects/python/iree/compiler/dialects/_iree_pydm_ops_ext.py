# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pytype: disable=attribute-error

from .. import ir


class FuncOp:
  """Specialization for the func op class."""

  @property
  def body(self):
    return self.regions[0]

  @property
  def type(self):
    return ir.FunctionType(ir.TypeAttr(self.attributes["function_type"]).value)

  @property
  def py_return_type(self) -> ir.Type:
    return self.type.results[1]

  @property
  def entry_block(self):
    return self.regions[0].blocks[0]

  # TODO: Why aren't these getters being auto-generated?
  @property
  def arg_names(self) -> ir.ArrayAttr:
    return ir.ArrayAttr(self.attributes["arg_names"])

  @property
  def free_vars(self) -> ir.ArrayAttr:
    return ir.ArrayAttr(self.attributes["free_vars"])

  @property
  def cell_vars(self) -> ir.ArrayAttr:
    return ir.ArrayAttr(self.attributes["cell_vars"])

  def add_entry_block(self):
    """Add an entry block to the function body using the function signature to
    infer block arguments. Returns the newly created block.
    """
    self.body.blocks.append(*self.type.inputs)
    return self.body.blocks[0]
