# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import iree.runtime as rt


class VmTypesTest(unittest.TestCase):

  def testRefProtocol(self):
    lst1 = rt.VmVariantList(0)
    ref = lst1.__iree_vm_ref__
    ref2 = lst1.ref
    print(ref)
    print(ref2)
    self.assertEqual(ref, ref2)
    self.assertNotEqual(ref, False)
    lst2 = rt.VmVariantList.__iree_vm_cast__(ref)
    print(lst2)
    lst3 = ref.deref(rt.VmVariantList)
    print(lst3)
    self.assertEqual(lst1, lst2)
    self.assertEqual(lst2, lst3)
    self.assertNotEqual(lst1, False)
    self.assertTrue(ref.isinstance(rt.VmVariantList))


if __name__ == "__main__":
  unittest.main()
