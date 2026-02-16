/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Gemmini.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace buddy {
namespace gemmini {

class GemminiDialect : public ::mlir::Dialect {
  explicit GemminiDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~GemminiDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("gemmini");
  }
};
} // namespace gemmini
} // namespace buddy
MLIR_DECLARE_EXPLICIT_TYPE_ID(::buddy::gemmini::GemminiDialect)
