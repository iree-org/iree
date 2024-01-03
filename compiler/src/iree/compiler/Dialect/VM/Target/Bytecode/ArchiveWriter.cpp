// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/Bytecode/ArchiveWriter.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/schemas/bytecode_module_def_json_printer.h"
#include "llvm/Support/CRC.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::IREE::VM {

// Alignment applied to each segment of the archive.
// All embedded file contents (FlatBuffers, rodata, etc) are aligned to this
// boundary.
static constexpr unsigned kArchiveSegmentAlignment = 64;

//====---------------------------------------------------------------------===//
// JSONArchiveWriter
//====---------------------------------------------------------------------===//

JSONArchiveWriter::JSONArchiveWriter(Location loc, llvm::raw_ostream &os)
    : loc(loc), os(os) {}

JSONArchiveWriter::~JSONArchiveWriter() { os.flush(); }

ArchiveWriter::File JSONArchiveWriter::declareFile(
    std::string fileName, uint64_t fileAlignment, uint64_t fileLength,
    std::function<LogicalResult(llvm::raw_ostream &os)> write) {
  File file;
  file.fileName = std::move(fileName);
  file.relativeOffset = 0;
  file.fileLength = fileLength;
  return file;
}

LogicalResult JSONArchiveWriter::flush(FlatbufferBuilder &fbb) {
  // Write the FlatBuffer contents out.
  if (failed(fbb.printJsonToStream(/*pretty=*/true,
                                   /*includeDefaults=*/false,
                                   bytecode_module_def_print_json, os))) {
    return mlir::emitError(loc)
           << "failed to print FlatBuffer emitter contents to output "
              "stream - possibly out of memory, possibly unprintable "
              "structure";
  }
  os.flush();
  return success();
}

//====---------------------------------------------------------------------===//
// FlatArchiveWriter
//====---------------------------------------------------------------------===//

FlatArchiveWriter::FlatArchiveWriter(Location loc, llvm::raw_ostream &os)
    : loc(loc), os(os) {}

FlatArchiveWriter::~FlatArchiveWriter() { os.flush(); }

ArchiveWriter::File FlatArchiveWriter::declareFile(
    std::string fileName, uint64_t fileAlignment, uint64_t fileLength,
    std::function<LogicalResult(llvm::raw_ostream &os)> write) {
  File file;
  file.fileName = std::move(fileName);
  file.relativeOffset = IREE::Util::align(tailFileOffset, fileAlignment);
  tailFileOffset = file.relativeOffset + fileLength;
  file.fileLength = fileLength;
  file.write = std::move(write);
  files.push_back(file);
  return file;
}

LogicalResult FlatArchiveWriter::flush(FlatbufferBuilder &fbb) {
  // Write the FlatBuffer contents out.
  if (failed(fbb.copyToStream(os))) {
    return mlir::emitError(loc)
           << "failed to copy FlatBuffer emitter contents to the output stream "
              "- possibly out of memory or storage";
  }

  // Pad out to the start of the external rodata segment.
  // This ensures we begin writing at an aligned offset; all relative offsets
  // in the embedded files assume this.
  uint64_t baseOffset = os.tell();
  uint64_t basePadding =
      IREE::Util::align(baseOffset, kArchiveSegmentAlignment) - baseOffset;
  os.write_zeros(basePadding);
  baseOffset = os.tell();

  // Flush all files.
  for (auto &file : files) {
    // Pad out with zeros to the start of the file.
    // Compute padding bytes required to align the file contents.
    unsigned filePadding = static_cast<unsigned>(
        baseOffset + file.relativeOffset + file.prefixLength - os.tell());
    os.write_zeros(filePadding);

    // Issue the callback to write the file to the stream at the current offset.
    if (failed(file.write(os))) {
      return mlir::emitError(loc)
             << "failed to write embedded file to the output stream - possibly "
                "out of memory or storage (file size: "
             << file.fileLength << ")";
    }
  }

  os.flush();
  return success();
}

//====---------------------------------------------------------------------===//
// ZIP data structures
//====---------------------------------------------------------------------===//
// These come from the ZIP APPNOTE.TXT:
// https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
// ZIP is not a good file format.
//
// We always use the 64-bit extended ZIP format (aka zip64 aka v4.5) for
// simplicity. It's basically the same as basic ZIP but with even more sharp
// edges: some fields may optionally be sentinel values (UINT32_MAX) to indicate
// that their actual values are stored in a separate data structure, while in
// other cases entirely new structures are used. The weirdest one is that the
// 32-bit zip header must exist with bogus values _as well as_ the 64-bit one.

namespace {
using llvm::support::ulittle16_t;
using llvm::support::ulittle32_t;
using llvm::support::ulittle64_t;
} // namespace

LLVM_PACKED_START

struct ZIPLocalFileHeader {
  ulittle32_t signature; // 0x04034B50
  ulittle16_t versionToExtract;
  ulittle16_t generalPurposeFlag;
  ulittle16_t compressionMethod;
  ulittle16_t lastModifiedTime;
  ulittle16_t lastModifiedDate;
  ulittle32_t crc32;
  ulittle32_t compressedSize;
  ulittle32_t uncompressedSize;
  ulittle16_t fileNameLength;
  ulittle16_t extraFieldLength;
  // file name (variable size)
  // extra field (variable size)
};
static_assert(sizeof(ZIPLocalFileHeader) == 30, "bad packing");

struct ZIP64DataDescriptor {
  ulittle32_t signature; // 0x08074B50
  ulittle32_t crc32;
  ulittle64_t compressedSize;
  ulittle64_t uncompressedSize;
};
static_assert(sizeof(ZIP64DataDescriptor) == 24, "bad packing");

struct ZIPExtraFieldHeader {
  ulittle16_t id;
  ulittle16_t size;
};
static_assert(sizeof(ZIPExtraFieldHeader) == 4, "bad packing");

struct ZIP64LocalExtraField {
  ZIPExtraFieldHeader header;
  ulittle64_t uncompressedSize;
  ulittle64_t compressedSize;
};
static_assert(sizeof(ZIP64LocalExtraField) == 20, "bad packing");

struct ZIPCentralDirectoryRecord {
  ulittle32_t signature; // 0x02014B50
  ulittle16_t versionMadeBy;
  ulittle16_t versionToExtract;
  ulittle16_t generalPurposeFlags;
  ulittle16_t compressionMethod;
  ulittle16_t lastModifiedTime;
  ulittle16_t lastModifiedDate;
  ulittle32_t crc32;
  ulittle32_t compressedSize;
  ulittle32_t uncompressedSize;
  ulittle16_t fileNameLength;
  ulittle16_t extraFieldLength;
  ulittle16_t fileCommentLength;
  ulittle16_t diskStartNumber;
  ulittle16_t internalFileAttributes;
  ulittle32_t externalFileAttributes;
  ulittle32_t localHeaderOffset;
  // file name (variable size)
  // extra field (variable size)
  // file comment (variable size)
};
static_assert(sizeof(ZIPCentralDirectoryRecord) == 46, "bad packing");

struct ZIP64CentralExtraField {
  ZIPExtraFieldHeader header;
  ulittle64_t uncompressedSize;
  ulittle64_t compressedSize;
  ulittle64_t localHeaderOffset;
};
static_assert(sizeof(ZIP64CentralExtraField) == 28, "bad packing");

struct ZIPEndOfCentralDirectoryRecord {
  ulittle32_t signature; // 0x06054B50
  ulittle16_t diskNumber;
  ulittle16_t startDiskNumber;
  ulittle16_t entriesOnDisk;
  ulittle16_t entryCount;
  ulittle32_t directorySize;
  ulittle32_t directoryOffset;
  ulittle16_t commentLength;
  // comment (variable size)
};
static_assert(sizeof(ZIPEndOfCentralDirectoryRecord) == 22, "bad packing");

struct ZIPEndOfCentralDirectoryRecord64 {
  ulittle32_t signature; // 0x06064B50
  ulittle64_t sizeOfEOCD64Minus12;
  ulittle16_t versionMadeBy;
  ulittle16_t versionRequired;
  ulittle32_t diskNumber;
  ulittle32_t startDiskNumber;
  ulittle64_t entriesOnDisk;
  ulittle64_t entryCount;
  ulittle64_t directorySize;
  ulittle64_t directoryOffset;
  // comment (variable size up to EOCD64)
};
static_assert(sizeof(ZIPEndOfCentralDirectoryRecord64) == 56, "bad packing");

struct ZIPEndOfCentralDirectoryLocator64 {
  ulittle32_t signature; // 0x07064B50
  ulittle32_t recordDiskNumber;
  ulittle64_t recordOffset;
  ulittle32_t diskCount;
};
static_assert(sizeof(ZIPEndOfCentralDirectoryLocator64) == 20, "bad packing");

LLVM_PACKED_END

// A ZIP file reference into the output stream.
// This records where the contents are and enough information to build the
// central directory.
struct ZIPFileRef {
  // Name of the file used within the ZIP archive.
  std::string fileName;
  // Offset of the local file header in the stream relative to the stream start.
  uint64_t headerOffset;
  // Total size, in bytes, of the uncompressed file.
  uint64_t totalLength;
  // CRC32 of the file.
  uint32_t crc32;
};

// Computes the minimum length of the ZIP header we write preceeding the file.
// This can have any alignment. The result value is only a minimum as up to 64KB
// of padding can be added following it.
static uint64_t computeMinHeaderLength(StringRef fileName) {
  return sizeof(ZIPLocalFileHeader) + fileName.size() +
         sizeof(ZIP64LocalExtraField) + sizeof(ZIPExtraFieldHeader);
}

// Appends a ZIP local file header at the current location.
// The header is a prefix to the actual file contents. ZIP requires that the
// payload start immediately after the header with no padding.
static ZIPFileRef appendZIPLocalFileHeader(std::string fileName,
                                           uint64_t filePadding,
                                           uint64_t fileLength, uint32_t crc32,
                                           llvm::raw_ostream &os) {
  // Capture the header offset that will be recorded in the central directory to
  // locate this file.
  uint64_t headerOffset = os.tell();

  // The amount of padding we need to add between the header and the file.
  // This ensures that once the header is written it'll end immediately on the
  // alignment boundary required by the file.
  uint64_t interiorPadding = filePadding - computeMinHeaderLength(fileName);

  // Append local file header.
  ZIPLocalFileHeader fileHeader;
  fileHeader.signature = 0x04034B50u;
  fileHeader.versionToExtract = 0x2Du; // 4.5 (for zip64)
  fileHeader.generalPurposeFlag = 0;
  fileHeader.compressionMethod = 0; // COMP_STORED
  // https://docs.microsoft.com/en-us/windows/win32/api/oleauto/nf-oleauto-dosdatetimetovarianttime
  fileHeader.lastModifiedTime = 0u;
  fileHeader.lastModifiedDate = 0x21; // 1980-01-01
  fileHeader.crc32 = crc32;
  fileHeader.compressedSize = 0xFFFFFFFFu;   // in extra field
  fileHeader.uncompressedSize = 0xFFFFFFFFu; // in extra field
  fileHeader.fileNameLength = static_cast<uint16_t>(fileName.size());
  fileHeader.extraFieldLength = sizeof(ZIP64LocalExtraField) +
                                sizeof(ZIPExtraFieldHeader) + interiorPadding;
  os.write(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));

  // File name immediately follows the header with no NUL terminator.
  os.write(fileName.data(), fileName.size());

  // Interior padding field.
  // This shouldn't be required if we just pad out the extraFieldLength but some
  // ZIP tools ignore that field and try to parse each extra field. We do this
  // before we do the 64-bit size extra field because some ZIP tools are so
  // poorly written that they only ever look at the last field present for
  // getting the size. Have I mentioned how terrible of a format ZIP is?
  ZIPExtraFieldHeader paddingExtra;
  paddingExtra.id = 0xFECAu; // 'CAFE'; in the user prefix range
  paddingExtra.size = static_cast<uint16_t>(interiorPadding);
  os.write(reinterpret_cast<char *>(&paddingExtra), sizeof(paddingExtra));
  os.write_zeros(interiorPadding);

  // Zip64 extension for 64-bit offsets/lengths.
  // The -1 values above tell the extractor to use the values in this field
  // instead. For simplicity we always use these regardless of whether we
  // need to or not - we aren't optimizing for size when in this mode.
  ZIP64LocalExtraField sizesExtra;
  sizesExtra.header.id = 0x0001u;
  sizesExtra.header.size =
      static_cast<uint16_t>(sizeof(sizesExtra) - sizeof(ZIPExtraFieldHeader));
  sizesExtra.compressedSize = fileLength;
  sizesExtra.uncompressedSize = fileLength;
  os.write(reinterpret_cast<char *>(&sizesExtra), sizeof(sizesExtra));

  ZIPFileRef fileRef;
  fileRef.fileName = std::move(fileName);
  fileRef.headerOffset = headerOffset;
  fileRef.totalLength = fileLength;
  fileRef.crc32 = crc32;
  return fileRef;
}

// Computes an Adler32 CRC and sends the data into the void.
class null_crc32_ostream : public llvm::raw_ostream {
public:
  explicit null_crc32_ostream(uint32_t &crc32) : crc32(crc32) {
    SetUnbuffered();
  }

private:
  void write_impl(const char *Ptr, size_t Size) override {
    crc32 = llvm::crc32(
        crc32, ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Ptr), Size));
    pos += Size;
  }
  uint64_t current_pos() const override { return pos; }
  uint32_t &crc32;
  uint64_t pos = 0;
};

// appendZIPFile implementation used when |os| is a stream without random
// access (like stdout). This requires us to serialize the file twice in order
// to compute the total length and CRC32.
static std::optional<ZIPFileRef>
appendZIPFileToStream(std::string fileName, uint64_t filePadding,
                      uint64_t fileLength,
                      std::function<LogicalResult(llvm::raw_ostream &os)> write,
                      llvm::raw_ostream &os) {
  // Compute the Adler32 CRC as required in the local file header (and later the
  // central directory). Since we only have an unseekable raw_ostream we can't
  // go patch the header after we stream out the file and instead have to stream
  // it twice - first here to compute the CRC, we write the header, and second
  // for real following the header.
  //
  // I've tried streaming zips (general purpose flag bit 3 set and
  // ZIP64DataDescriptor suffixes on files) but several tools don't handle
  // simultaneous use of zip64 and streaming and the trailing descriptor makes
  // laying out files more complex. Ideally our write functions are fairly
  // efficient and polyglot files are debug mode so the double serialization
  // isn't too bad. Probably. Piping out multi-GB files is pretty silly, anyway.
  uint32_t crc32 = 0;
  null_crc32_ostream crcStream(crc32);
  if (failed(write(crcStream))) {
    return std::nullopt;
  }

  // Write the ZIP header and padding up to the start of the file.
  auto fileRef = appendZIPLocalFileHeader(std::move(fileName), filePadding,
                                          fileLength, crc32, os);

  // Stream out the file contents to the output stream.
  uint64_t start = os.tell();
  if (failed(write(os))) {
    return std::nullopt;
  }
  fileRef.totalLength = os.tell() - start;
  assert(fileRef.totalLength == fileLength && "declared length mismatch");

  return fileRef;
}

// Computes an Adler32 CRC and passes the data along to an underlying ostream.
class crc32_ostream : public llvm::raw_ostream {
public:
  explicit crc32_ostream(llvm::raw_ostream &impl, uint32_t &crc32)
      : impl(impl), crc32(crc32) {
    SetUnbuffered();
  }

private:
  void write_impl(const char *Ptr, size_t Size) override {
    crc32 = llvm::crc32(
        crc32, ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Ptr), Size));
    impl.write(Ptr, Size);
  }
  uint64_t current_pos() const override { return impl.tell(); }
  llvm::raw_ostream &impl;
  uint32_t &crc32;
};

// appendZIPFile implementation used when |os| is a file with random access.
// This allows us to write the header and backpatch the CRC computed while while
// serializing the file contents.
static std::optional<ZIPFileRef>
appendZIPFileToFD(std::string fileName, uint64_t filePadding,
                  uint64_t fileLength,
                  std::function<LogicalResult(llvm::raw_ostream &os)> write,
                  llvm::raw_fd_ostream &os) {
  // Write the ZIP header and padding up to the start of the file.
  // We write a dummy CRC we'll patch up after we compute it while serializing
  // the file contents.
  auto fileRef = appendZIPLocalFileHeader(std::move(fileName), filePadding,
                                          fileLength, /*crc32=*/0, os);

  // Stream out the file contents to the output stream.
  uint64_t start = os.tell();
  {
    crc32_ostream crcStream(os, fileRef.crc32);
    if (failed(write(crcStream))) {
      return std::nullopt;
    }
    crcStream.flush();
  }
  fileRef.totalLength = os.tell() - start;
  assert(fileRef.totalLength == fileLength && "declared length mismatch");

  // Patch the CRC back into the header.
  uint64_t end = os.tell();
  os.seek(fileRef.headerOffset + offsetof(ZIPLocalFileHeader, crc32));
  os.write(reinterpret_cast<char *>(&fileRef.crc32), sizeof(fileRef.crc32));
  os.seek(end);

  return fileRef;
}

// Appends a file wrapped in a ZIP header and data descriptor.
// |write| is used to stream the file contents to |os| while also capturing its
// CRC as required for the central directory.
static std::optional<ZIPFileRef>
appendZIPFile(std::string fileName, uint64_t filePadding, uint64_t fileLength,
              std::function<LogicalResult(llvm::raw_ostream &os)> write,
              llvm::raw_ostream &os) {
  if (os.get_kind() == llvm::raw_ostream::OStreamKind::OK_FDStream) {
    auto &osFD = static_cast<llvm::raw_fd_ostream &>(os);
    if (osFD.supportsSeeking()) {
      // Output stream is backed by a file descriptor and supports
      // random-access; this allows us to write out the file contents much more
      // efficiently.
      return appendZIPFileToFD(std::move(fileName), filePadding, fileLength,
                               std::move(write), osFD);
    }
  }
  // Output stream does not support seeking and needs to perform extra work to
  // get the CRC required for the ZIP header.
  return appendZIPFileToStream(std::move(fileName), filePadding, fileLength,
                               std::move(write), os);
}

// Appends a ZIP central directory to |os| with the references to all of
// |zipFileRefs|. Must follow all of the local file headers.
static void appendZIPCentralDirectory(ArrayRef<ZIPFileRef> fileRefs,
                                      llvm::raw_ostream &os) {
  // Append the central directory, which contains the local file headers with
  // some extra junk and references back to where the local headers are in the
  // file.
  uint64_t centralDirectoryStartOffset = os.tell();
  for (auto fileRef : fileRefs) {
    // Fixed-size header.
    ZIPCentralDirectoryRecord cdr;
    cdr.signature = 0x02014B50u;
    cdr.versionMadeBy = 0x031E;
    cdr.versionToExtract = 0x2Du; // 4.5 (for zip64)
    cdr.generalPurposeFlags = 0;
    cdr.compressionMethod = 0; // COMP_STORED
    // https://docs.microsoft.com/en-us/windows/win32/api/oleauto/nf-oleauto-dosdatetimetovarianttime
    cdr.lastModifiedTime = 0u;
    cdr.lastModifiedDate = 0x21; // 1980-01-01
    cdr.crc32 = fileRef.crc32;
    cdr.compressedSize = 0xFFFFFFFFu;   // in extra field
    cdr.uncompressedSize = 0xFFFFFFFFu; // in extra field
    cdr.fileNameLength = static_cast<uint16_t>(fileRef.fileName.size());
    cdr.extraFieldLength =
        static_cast<uint16_t>(sizeof(ZIP64CentralExtraField));
    cdr.fileCommentLength = 0;
    cdr.diskStartNumber = 0;
    cdr.internalFileAttributes = 0;
    cdr.externalFileAttributes = 0;
    cdr.localHeaderOffset = 0xFFFFFFFFu;
    os.write(reinterpret_cast<const char *>(&cdr), sizeof(cdr));
    os.write(fileRef.fileName.data(), fileRef.fileName.size());

    // Zip64 extension for 64-bit offsets/lengths.
    // The -1 values above tell the extractor to use the values in this field
    // instead. For simplicity we always use these regardless of whether we
    // need to or not - we aren't optimizing for size when in this mode.
    ZIP64CentralExtraField zip64Extra;
    zip64Extra.header.id = 0x0001u;
    zip64Extra.header.size =
        static_cast<uint16_t>(sizeof(zip64Extra) - sizeof(ZIPExtraFieldHeader));
    zip64Extra.localHeaderOffset = fileRef.headerOffset;
    zip64Extra.compressedSize = fileRef.totalLength;
    zip64Extra.uncompressedSize = fileRef.totalLength;
    os.write(reinterpret_cast<const char *>(&zip64Extra), sizeof(zip64Extra));
  }
  uint64_t centralDirectoryEndOffset = os.tell();

  // Append the central directory record.
  ZIPEndOfCentralDirectoryRecord64 endOfCDR64;
  endOfCDR64.signature = 0x06064B50u;
  endOfCDR64.sizeOfEOCD64Minus12 = sizeof(endOfCDR64) - 12;
  endOfCDR64.versionMadeBy = 0x002Du;
  endOfCDR64.versionRequired = 0x002Du; // 4.5 (for zip64)
  endOfCDR64.diskNumber = 0;
  endOfCDR64.startDiskNumber = 0;
  endOfCDR64.entriesOnDisk = static_cast<uint64_t>(fileRefs.size());
  endOfCDR64.entryCount = static_cast<uint64_t>(fileRefs.size());
  endOfCDR64.directorySize = static_cast<uint64_t>(centralDirectoryEndOffset -
                                                   centralDirectoryStartOffset);
  endOfCDR64.directoryOffset =
      static_cast<uint64_t>(centralDirectoryStartOffset);
  os.write(reinterpret_cast<const char *>(&endOfCDR64), sizeof(endOfCDR64));

  // End of central directory locator; must be at the end of the file.
  ZIPEndOfCentralDirectoryLocator64 locator;
  locator.signature = 0x07064B50u;
  locator.recordDiskNumber = 0;
  locator.recordOffset = centralDirectoryEndOffset;
  locator.diskCount = 1;
  os.write(reinterpret_cast<const char *>(&locator), sizeof(locator));

  // Append the final ZIP file footer.
  // NOTE: this must come at the very end of the file. Even though we have the
  // EOCD64 record above this is still required for extractors to recognize the
  // file as a zip file. The offset of -1 will cause incompatible extractors
  // (like on MS-DOS I guess?) to fail and compatible ones to look for the
  // locator.
  ZIPEndOfCentralDirectoryRecord endOfCDR;
  endOfCDR.signature = 0x06054B50u;
  endOfCDR.diskNumber = 0;
  endOfCDR.startDiskNumber = 0;
  endOfCDR.entriesOnDisk = static_cast<uint16_t>(endOfCDR64.entriesOnDisk);
  endOfCDR.entryCount = static_cast<uint16_t>(endOfCDR64.entryCount);
  endOfCDR.directorySize = static_cast<uint32_t>(endOfCDR64.directorySize);
  endOfCDR.directoryOffset = 0xFFFFFFFF;
  endOfCDR.commentLength = 0;
  os.write(reinterpret_cast<const char *>(&endOfCDR), sizeof(endOfCDR));
}

ZIPArchiveWriter::ZIPArchiveWriter(Location loc, llvm::raw_ostream &os)
    : loc(loc), os(os) {}

ZIPArchiveWriter::~ZIPArchiveWriter() { os.flush(); }

// Files are serialized with a ZIP local file header followed by the file bytes.
// The critical alignment applies only to the file bytes and the header
// alignment doesn't matter (zip only requires byte alignment). The file
// contents must immediately follow the header.
//
//   [padding] [header] [file] [data descriptor]
//                      ^-- aligned
//
// Note that the offset we record is for the header.
ArchiveWriter::File ZIPArchiveWriter::declareFile(
    std::string fileName, uint64_t fileAlignment, uint64_t fileLength,
    std::function<LogicalResult(llvm::raw_ostream &os)> write) {
  // Align the file offset; the header will be prepended.
  uint64_t headerOffset = tailFileOffset;
  uint64_t headerLength = computeMinHeaderLength(fileName);
  uint64_t fileOffset =
      IREE::Util::align(headerOffset + headerLength, fileAlignment);
  tailFileOffset = fileOffset + fileLength;

  File file;
  file.fileName = std::move(fileName);
  file.relativeOffset = headerOffset;
  file.prefixLength = fileOffset - headerOffset;
  file.fileLength = fileLength;
  file.write = std::move(write);
  files.push_back(file);
  return file;
}

LogicalResult ZIPArchiveWriter::flush(FlatbufferBuilder &fbb) {
  SmallVector<ZIPFileRef> fileRefs;
  fileRefs.reserve(files.size() + 1);

  // Compute padding of the header to ensure the module FlatBuffer ends up with
  // the proper alignment.
  auto moduleName = "module.fb";
  uint64_t startOffset = os.tell();
  uint64_t moduleHeaderLength = computeMinHeaderLength(moduleName);
  uint64_t modulePadding = IREE::Util::align(startOffset + moduleHeaderLength,
                                             kArchiveSegmentAlignment);

  // Serialize the module FlatBuffer to a binary blob in memory.
  // Ideally we'd stream out using fbb.copyToStream but we have no way of
  // computing the size without serializing and we need that for the ZIP header.
  std::string moduleData;
  {
    llvm::raw_string_ostream moduleStream(moduleData);
    if (failed(fbb.copyToStream(moduleStream))) {
      return mlir::emitError(loc)
             << "failed to serialize FlatBuffer emitter "
                "contents to memory - possibly out of memory";
    }
    moduleStream.flush();
  }

  // Pad out the module data so we can easily compute the relative offsets.
  auto paddedModuleLength = static_cast<flatbuffers_uoffset_t>(
      IREE::Util::align(sizeof(flatbuffers_uoffset_t) + moduleData.size(),
                        kArchiveSegmentAlignment) -
      sizeof(flatbuffers_uoffset_t));

  // Stream out the FlatBuffer contents.
  auto zipFile = appendZIPFile(
      moduleName, modulePadding, paddedModuleLength,
      [&](llvm::raw_ostream &os) -> LogicalResult {
        os.write(reinterpret_cast<char *>(&paddedModuleLength),
                 sizeof(flatbuffers_uoffset_t));
        os.write(moduleData.data() + sizeof(flatbuffers_uoffset_t),
                 moduleData.size() - sizeof(flatbuffers_uoffset_t));
        os.write_zeros(paddedModuleLength - moduleData.size());
        return success();
      },
      os);
  if (!zipFile.has_value()) {
    return mlir::emitError(loc) << "failed to serialize flatbuffer module";
  }
  fileRefs.push_back(*zipFile);

  // Pad out to the start of the external rodata segment.
  // This ensures we begin writing at an aligned offset; all relative offsets
  // in the embedded files assume this.
  uint64_t baseOffset = os.tell();
  uint64_t basePadding =
      IREE::Util::align(baseOffset, kArchiveSegmentAlignment) - baseOffset;
  os.write_zeros(basePadding);
  baseOffset = os.tell();

  // Flush all declared files.
  for (auto &file : files) {
    // Compute padding bytes required to align the file contents.
    unsigned filePadding = static_cast<unsigned>(
        baseOffset + file.relativeOffset + file.prefixLength - os.tell());

    // Write file header and payload.
    auto zipFile = appendZIPFile(
        file.fileName, filePadding, file.fileLength,
        [this, file](llvm::raw_ostream &os) -> LogicalResult {
          if (failed(file.write(os))) {
            return mlir::emitError(loc)
                   << "failed to write embedded file to the output stream - "
                      "possibly out of memory or storage (file size: "
                   << file.fileLength << ")";
          }
          return success();
        },
        os);
    if (!zipFile.has_value())
      return failure();
    fileRefs.push_back(*zipFile);
  }

  // Append the central directory containing an index of all the files.
  appendZIPCentralDirectory(fileRefs, os);

  os.flush();
  return success();
}

} // namespace mlir::iree_compiler::IREE::VM
