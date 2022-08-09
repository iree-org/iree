# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import os
import iree.compiler as compiler
import iree.runtime as rt

TOKEN_TABLE = [
    b"hi",
    b"thanks",
    b"bye",
    b"for",
    b"all",
    b"so",
    b"the",
    b"fish",
    b"the",
    b"end",
    b"long",
    b"and",
    b".",
    b"now",
    b"there",
]


def create_tokenizer_module():
  """Creates a module which defines some custom methods for decoding."""

  class Detokenizer:

    def __init__(self, iface):
      # Any class state here is maintained per-context.
      self.start_of_text = True
      self.start_of_sentence = True

    def reset(self):
      self.start_of_text = True
      self.start_of_sentence = True

    def accumtokens(self, ids_tensor_ref, token_list_ref):
      # TODO: This little dance to turn BufferView refs into real arrays... is not good.
      ids_bv = ids_tensor_ref.deref(rt.HalBufferView)
      ids_array = ids_bv.map().asarray(
          ids_bv.shape, rt.HalElementType.map_to_dtype(ids_bv.element_type))
      token_list = token_list_ref.deref(rt.VmVariantList)
      for index in range(ids_array.shape[0]):
        token_id = ids_array[index]
        token = TOKEN_TABLE[token_id]

        # And this dance to make a buffer... is also not good.
        # A real implementation would just map the constant memory, etc.
        buffer = rt.VmBuffer(len(token))
        buffer_view = memoryview(buffer)
        buffer_view[:] = token
        token_list.push_ref(buffer)
      return ids_array.shape[0]

    def jointokens(self, token_list_ref):
      # The world's dumbest detokenizer. Ideally, the state tracking
      # would be in a module private type that got retained and passed
      # back through.
      token_list = token_list_ref.deref(rt.VmVariantList)
      text = bytearray()
      for i in range(len(token_list)):
        item = bytes(token_list.get_as_object(i, rt.VmBuffer))
        if item == b".":
          text.extend(b".")
          self.start_of_sentence = True
        else:
          if not self.start_of_text:
            text.extend(b" ")
          else:
            self.start_of_text = False
          if self.start_of_sentence:
            text.extend(item[0:1].decode("utf-8").upper().encode("utf-8"))
            text.extend(item[1:])
            self.start_of_sentence = False
          else:
            text.extend(item)

      # TODO: This dance to make a buffer is still bad.
      results = rt.VmBuffer(len(text))
      memoryview(results)[:] = text
      return results.ref

  iface = rt.PyModuleInterface("detokenizer", Detokenizer)
  iface.export("accumtokens", "0rr_i", Detokenizer.accumtokens)
  iface.export("jointokens", "0r_r", Detokenizer.jointokens)
  iface.export("reset", "0v_v", Detokenizer.reset)
  return iface.create()


def compile():
  return compiler.tools.compile_file(os.path.join(os.path.dirname(__file__),
                                                  "main.mlir"),
                                     target_backends=["vmvx"])


def main():
  print("Compiling...")
  vmfb_contents = compile()
  print("Decoding secret message...")
  config = rt.Config("local-sync")
  main_module = rt.VmModule.from_flatbuffer(config.vm_instance, vmfb_contents)
  modules = config.default_vm_modules + (
      create_tokenizer_module(),
      main_module,
  )
  context = rt.SystemContext(vm_modules=modules, config=config)

  # First message.
  count = context.modules.main.add_tokens(
      np.asarray([5, 10, 11, 1, 3, 4, 5, 7, 12], dtype=np.int32))
  print(f"ADDED {count} tokens")

  # Second message.
  count = context.modules.main.add_tokens(np.asarray([2, 13], dtype=np.int32))
  print(f"ADDED {count} tokens")

  text = bytes(context.modules.main.get_results().deref(rt.VmBuffer))
  print(f"RESULTS: {text}")

  assert text == b"So long and thanks for all so fish. Bye now"

  # Reset and decode some more.
  context.modules.main.reset()
  count = context.modules.main.add_tokens(
      np.asarray([0, 14, 12], dtype=np.int32))
  print(f"ADDED {count} tokens")
  text = bytes(context.modules.main.get_results().deref(rt.VmBuffer))
  print(f"RESULTS: {text}")
  assert text == b"Hi there."


if __name__ == "__main__":
  main()
