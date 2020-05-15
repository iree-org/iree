// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

vm.module @tensorlist {

// Maps to IREE::TensorList::Reserve.
vm.import @reserve(
  %element_shape : !vm.ref<!hal.buffer_view>,
  %num_elements : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!tensorlist.list>
attributes {nosideeffects}

// Maps to IREE::TensorList::GetItem.
vm.import @get_item(
  %list : !vm.ref<!tensorlist.list>,
  %index : !vm.ref<!hal.buffer_view>,
  %element_shape: !vm.ref<!hal.buffer_view>
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
  %tensor : !vm.ref<!hal.buffer_view>,
  %element_shape : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!tensorlist.list>
attributes {nosideeffects}

// Maps to IREE:TensorList::Concat
vm.import @concat(
  %list : !vm.ref<!tensorlist.list>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Maps to IREE:TensorList::Stack
vm.import @stack(
  %list : !vm.ref<!tensorlist.list>,
  %element_shape : !vm.ref<!hal.buffer_view>,
  %num_elements : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

}  // vm.module
