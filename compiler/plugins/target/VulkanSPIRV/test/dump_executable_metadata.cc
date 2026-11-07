// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prints serialized metadata for lit tests without relying on FlatBuffer
// offsets.
#include <cstdlib>
#include <cstring>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/vulkan_executable_def_json_printer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

static mlir::LogicalResult
printMetadata(mlir::iree_compiler::IREE::HAL::ExecutableBinaryOp binary) {
  auto data = mlir::dyn_cast<mlir::DenseIntElementsAttr>(binary.getData());
  if (!data || data.isSplat()) {
    return binary.emitError("expected dense executable bytes");
  }
  auto bytes = data.getRawData();
  iree_const_byte_span_t flatbuffer = {};
  iree_status_t status = iree_hal_read_executable_flatbuffer_header_versioned(
      iree_make_const_byte_span(bytes.data(), bytes.size()), false, "VKE1", 1,
      &flatbuffer);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return binary.emitError("invalid Vulkan executable header");
  }
  iree_flatbuffer_file_header_t header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  llvm::outs() << "binary @" << binary.getSymName() << " version "
               << header.version << "\n";
  flatcc_json_printer_t printer;
  flatcc_json_printer_init_dynamic_buffer(&printer, 0);
  flatcc_json_printer_set_indent(&printer, 2);
  flatcc_json_printer_set_force_default(&printer, true);
  if (vulkan_executable_def_print_json(
          &printer, reinterpret_cast<const char *>(flatbuffer.data),
          flatbuffer.data_length) == -1) {
    flatcc_json_printer_clear(&printer);
    return binary.emitError("cannot print Vulkan executable metadata");
  }
  size_t length = 0;
  char *json = static_cast<char *>(
      flatcc_json_printer_finalize_dynamic_buffer(&printer, &length));
  llvm::outs().write(json, length);
  free(json);
  llvm::outs() << "\n";
  return mlir::success();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: dump_executable_metadata <serialized.mlir>\n";
    return 1;
  }
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::iree_compiler::IREE::HAL::HALDialect>();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module) {
    return 1;
  }
  auto result = module->walk(
      [](mlir::iree_compiler::IREE::HAL::ExecutableBinaryOp binary) {
        return mlir::failed(printMetadata(binary))
                   ? mlir::WalkResult::interrupt()
                   : mlir::WalkResult::advance();
      });
  return result.wasInterrupted() ? 1 : 0;
}
