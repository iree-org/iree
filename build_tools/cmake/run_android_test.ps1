# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

param(
  [Parameter(Position=0, Mandatory, ValueFromRemainingArguments=$true)]
  [ValidateNotNullOrEmpty()]
  [string[]]
    $adb_cmd
)

# NOTE: to debug first run `$DebugPreference = 'Continue'` in your shell.
# $DebugPreference = "Continue"

Write-Host -ForegroundColor Yellow "Requested adb command: $adb_cmd"

$adb_path = $(Get-Command adb -Type Application).Path
Write-Host -ForegroundColor Yellow "Using adb executable: $adb_path"

# Compose arguments to `adb shell`. It should be of the form:
#   "cd /data/local/tmp && <requested-adb-command>"
$adb_cmd_prefix = "cd", "/data/local/tmp", "&&"
$adb_params = $adb_cmd_prefix + $adb_cmd
$adb_params = $adb_params -join " "
$adb_params = ('"', $adb_params, '"') -join ""

$adb_params = "shell " + $adb_params
Write-Host -ForegroundColor Yellow "Full adb arguments: $adb_params"

$process = Start-Process -FilePath $adb_path -ArgumentList $adb_params -NoNewWindow -PassThru
# HACK: Start-Process is broken... wow.
# https://stackoverflow.com/questions/10262231/obtaining-exitcode-using-start-process-and-waitforexit-instead-of-wait
$handle = $process.Handle
$exitcode = 1
$timeout_millis = 120 * 1000
if ($process.WaitForExit($timeout_millis) -eq $false) {
  Write-Error "Test timed out after $timeout_millis millis"
} else {
  $exitcode = $process.ExitCode
  Write-Debug "Test returned in time with exit code $($process.ExitCode)"
}

Write-Debug "Test exited with $exitcode"
exit $exitcode
