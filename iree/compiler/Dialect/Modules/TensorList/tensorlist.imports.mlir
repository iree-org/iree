// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

vm.module @tensorlist {

// Maps to IREE::TensorList::Reserve.
vm.import @reserve(
  %element_shape : !vm.ref<!hal.buffer_view>,
  %num_elements : !vm.ref<!hal.buffer_view>,
  %element_type : i32
) -> !vm.ref<!tensorlist.list>
attributes {nosideeffects}

// Maps to IREE::TensorList::GetItem.
vm.import @get_item(
  %list : !vm.ref<!tensorlist.list>,
  %index : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Maps to IREE:TensorList::SetItem
vm.import @set_item(
  %list : !vm.ref<!tensorlist.list>,
  %index : !vm.ref<!hal.buffer_view>,
  %item : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!tensorlist.list>
attributes {nosideeffects}

// Maps to IREE:TensorList::FromTensor
vm.import @from_tensor(
  %tensor : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!tensorlist.list>
attributes {nosideeffects}

// Maps to IREE:TensorList::Concat
vm.import @concat(
  %allocator : !vm.ref<!hal.allocator>,
  %list : !vm.ref<!tensorlist.list>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Maps to IREE:TensorList::Stack
vm.import @stack(
  %allocator : !vm.ref<!hal.allocator>,
  %list : !vm.ref<!tensorlist.list>,
  %num_elements : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

}  // vm.module
