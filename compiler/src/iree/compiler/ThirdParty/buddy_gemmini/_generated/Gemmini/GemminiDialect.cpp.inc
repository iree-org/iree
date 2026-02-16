/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Gemmini.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::GemminiDialect)
namespace buddy {
namespace gemmini {

GemminiDialect::GemminiDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<GemminiDialect>())
    
     {
  
  initialize();
}

GemminiDialect::~GemminiDialect() = default;

} // namespace gemmini
} // namespace buddy
