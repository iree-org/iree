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

#ifndef IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_OPS
#define IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/IREE/IR/IREEBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CUSTOM_Dialect : Dialect {
  let name = "custom";
  let cppNamespace = "IREE::Custom";

  let summary = [{
    A custom dialect demonstrating custom ops and implementations.
  }];
  let description = [{
    The ops in this dialect are lowered to vm.imports as described in the
    `custom_modules/dialect/custom.imports.mlir` file and the implementations
    are provided by the `custom_modules/native_module.cc` file.
  }];
}

def CUSTOM_Message : DialectType<
    CUSTOM_Dialect,
    CPred<"$_self.isa<IREE::Custom::MessageType>()">,
    "message"> {
  let typeDescription = [{
    A string message that can be printed. These types will be reference counted
    at runtime and must derive from iree_vm_ref_t but otherwise can be whatever
    the dialect wants.
  }];
}

def CUSTOM_TensorToMessageOp : Op<CUSTOM_Dialect, "tensor_to_message"> {
  let summary = [{formats tensor contents as a message}];
  let description = [{
    Formats the tensor using the IREE buffer printer to have a shape/type and
    the contents as a string.

    This demonstrates handling tensor->buffer conversion through the HAL layer.
    This op (tensor_to_message) is only used prior to translation and instances
    of it will be converted to the lower-level HAL-based buffer_to_message op.
  }];

  let arguments = (ins
    AnyTensor:$operand
  );
  let results = (outs
    CUSTOM_Message:$result
  );
}

def CUSTOM_BufferToMessageOp : Op<CUSTOM_Dialect, "buffer_to_message"> {
  let summary = [{formats buffer contents as a message}];
  let description = [{
    Formats the tensor using the IREE buffer printer to have a shape/type and
    the contents as a string.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    CUSTOM_Message:$result
  );
}

def CUSTOM_MessageToTensorOp : Op<CUSTOM_Dialect, "message_to_tensor"> {
  let summary = [{parses message contents as a tensor}];
  let description = [{
    Parses the message containing a IREE buffer parser-formatted tensor.
  }];

  let arguments = (ins
    CUSTOM_Message:$message
  );
  let results = (outs
    AnyTensor:$result
  );
}

def CUSTOM_MessageToBufferOp : Op<CUSTOM_Dialect, "message_to_buffer"> {
  let summary = [{parses message contents as a buffer}];
  let description = [{
    Parses the message containing a IREE buffer parser-formatted tensor.
  }];

  let arguments = (ins
    CUSTOM_Message:$message
  );
  let results = (outs
    HAL_BufferView:$result
  );
}

def CUSTOM_PrintOp : Op<CUSTOM_Dialect, "print"> {
  let summary = [{prints a message zero or more times}];
  let description = [{
    Prints the input message zero or more times with a newline inbetween.

    This demonstrates passing arguments from the application (the original
    message).
  }];

  let arguments = (ins
    CUSTOM_Message:$message,
    I32:$count
  );
}

def CUSTOM_ReverseOp : Op<CUSTOM_Dialect, "reverse", [NoSideEffect]> {
  let summary = [{reverses the characters in the given the message}];
  let description = [{
    Reverses the message characters and returns the new value.

    This demonstrates allocating new values from within the op implementation
    (the reversed result). Note that this op (as well as the vm.import) are
    marked with NoSideEffect/nosideeffect to enable additional compiler
    optimization.
  }];

  let arguments = (ins
    CUSTOM_Message:$operand
  );
  let results = (outs
    CUSTOM_Message:$result
  );
}

def CUSTOM_GetUniqueMessageOp : Op<CUSTOM_Dialect, "get_unique_message", [NoSideEffect]> {
  let summary = [{returns a per-context unique message}];
  let description = [{
    Returns a unique message allocated per context using the module.

    This demonstrates context-specific global storage. Mutation methods could
    exist that update the global storage as desired.
  }];

  let results = (outs
    CUSTOM_Message:$result
  );
}

#endif  // IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_OPS
