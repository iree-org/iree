#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"

namespace mlir::iree_compiler {
namespace {

struct RISCVIMESession
    : public PluginSession<RISCVIMESession, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<riscv_ime::RISCVIMEDialect>();
  }
};

} // namespace
} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_hal_target_riscv_ime(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::RISCVIMESession>(
      "hal_target_riscv_ime");
  return true;
}
