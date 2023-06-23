// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from mlir-tblgen.cpp. Simply delegates through to MlirTblgenMain.

#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"

using namespace llvm;
using namespace mlir;

// Generator that prints records.
GenRegistration printRecords("print-records", "Print all records to stdout",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               os << records;
                               return false;
                             });

int main(int argc, char **argv) { return MlirTblgenMain(argc, argv); }
