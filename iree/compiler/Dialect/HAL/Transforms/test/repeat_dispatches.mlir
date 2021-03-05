// RUN: iree-opt -split-input-file -iree-hal-repeat-dispatches -iree-hal-repeat-dispatch-num=32 -canonicalize %s | IreeFileCheck %s

hal.variable @_executable_0 : !hal.executable
func @multiple_reads_no_writes() {
  %0 = hal.variable.load @_executable_0 : !hal.executable
  %1 = hal.variable.load @_executable_0 : !hal.executable
  %2 = hal.variable.load @_executable_0 : !hal.executable

  %c1 = constant 1 : index
  %dev = hal.ex.shared_device : !hal.device
  %cmd = hal.command_buffer.create %dev, "OneShot", "Transfer|Dispatch" : !hal.command_buffer
  hal.command_buffer.begin %cmd
  hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.dispatch %cmd, %2, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
  hal.command_buffer.end %cmd
  return
}
// CHECK-DAG: %[[C0:.+]] = constant 0 : index
// CHECK-DAG: %[[C1:.+]] = constant 1 : index
// CHECK-DAG: %[[C32:.+]] = constant 32 : index
// CHECK:     scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:       hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
// CHECK:     scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:       hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
// CHECK:     scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:       hal.command_buffer.dispatch %cmd, %2, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
