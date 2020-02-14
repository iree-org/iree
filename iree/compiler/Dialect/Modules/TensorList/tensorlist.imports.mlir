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
  %element_shape : !iree.ref<!hal.buffer_view>,
  %num_elements : !iree.ref<!hal.buffer_view>
) -> !iree.ref<!tensorlist.list>
attributes {nosideeffects}

// Maps to IREE::TensorList::GetItem.
vm.import @get_item(
  %list : !iree.ref<!tensorlist.list>,
  %index : !iree.ref<!hal.buffer_view>,
  %element_shape: !iree.ref<!hal.buffer_view>
) -> !iree.ref<!hal.buffer_view>
attributes {nosideeffects}

// Maps to IREE:TensorList::SetItem
vm.import @set_item(
  %list : !iree.ref<!tensorlist.list>,
  %index : !iree.ref<!hal.buffer_view>,
  %item : !iree.ref<!hal.buffer_view>
) -> !iree.ref<!tensorlist.list>
attributes {nosideeffects}

}  // vm.module
