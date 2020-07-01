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

# Wrapper script to push build artifacts and run tests on an Android device.
#
# This script expects the following arguments:
#   <test-binary> [<test-args>]..
# Where <test-binary> should be a path relative to /data/local/tmp/ on device.
#
# This script reads the following environment variables:
# - TEST_ANDROID_ABS_DIR: the absolute path on Android device for the build
#   artifact.
# - TEST_ARTIFACT: the build artifact to push to the Android device.
# - TEST_ARTIFACT_IS_EXECUTABLE: whether the build artifact should be marked
#   as executable on the Android device.
# - TEST_ARTIFACT_NAME: the build artifact's filename.
# - TEST_TMPDIR: optional; temporary directory on the Android device for
#   running tests.
#
# This script pushes $env:TEST_ARTIFACT onto the device as
# $env:TEST_ANDROID_ABS_DIR/$env:TEST_ARTIFACT_NAME before running
# <test-binary> with all <test-args> under /data/local/tmp.

param(
  [Parameter(Position=0, Mandatory)]
  [ValidateNotNullOrEmpty()]
  [string]
    $test_binary,
  [Parameter(Position=1, ValueFromRemainingArguments=$true)]
  [string[]]
    $test_args = @()
)

# NOTE: to debug first run `$DebugPreference = 'Continue'` in your shell.
# $DebugPreference = "Continue"

# Composes arguments for adb to push `$data` to `$dest` on device.
function Compose-AdbPushCommand {
  param([string]$data, [string]$dest)

  $adb_params = "push", $data, $dest
  $adb_params = $adb_params -join " "

  return $adb_params
}

# Composes arguments for adb to run the given `$adb_cmd` and returns a string:
#   shell "cd /data/local/tmp && $adb_cmd"
function Compose-AdbShellCommand {
  param([string[]]$adb_cmd)

  $adb_cmd_prefix = "cd", "/data/local/tmp", "&&"
  $adb_params = $adb_cmd_prefix + $adb_cmd
  $adb_params = $adb_params -join " "
  $adb_params = ('"', $adb_params, '"') -join ""

  $adb_params = "shell " + $adb_params
  Write-Host -ForegroundColor Yellow "adb parameters: $adb_params"
  return $adb_params
}

# Invokes adb at `$adb_path` with the given parameters `$adb_params`.
function Invoke-Adb {
  param([string]$adb_path, [string]$adb_params)

  $process = Start-Process -FilePath $adb_path -ArgumentList $adb_params -NoNewWindow -PassThru
  # HACK: Start-Process is broken... wow.
  # https://stackoverflow.com/questions/10262231/obtaining-exitcode-using-start-process-and-waitforexit-instead-of-wait
  $handle = $process.Handle
  $exitcode = 1
  $timeout_millis = 120 * 1000
  if ($process.WaitForExit($timeout_millis) -eq $false) {
    Write-Error "adb timed out after $timeout_millis millis"
  } else {
    $exitcode = $process.ExitCode
    Write-Debug "adb returned in time with exit code $($exitcode)"
  }

  if ($process.ExitCode -ne 0) {
    Write-Debug "adb exited with $exitcode"
    exit $exitcode
  }
}

Write-Host -ForegroundColor Yellow "Requested adb command: $test_binary $test_args"

$adb_path = $(Get-Command adb -Type Application).Path
Write-Host -ForegroundColor Yellow "Using adb executable: $adb_path"

# Push the artifact needed for testing to Android device.
$adb_push_params = Compose-AdbPushCommand $env:TEST_ARTIFACT $env:TEST_ANDROID_ABS_DIR/$env:TEST_ARTIFACT_NAME
Invoke-Adb $adb_path $adb_push_params

# Optionally mark it as executable.
if ($env:TEST_ARTIFACT_IS_EXECUTABLE) {
  $adb_mark_executable_params = Compose-AdbShellCommand "chmod","+x",$env:TEST_ANDROID_ABS_DIR/$env:TEST_ARTIFACT_NAME
  Invoke-Adb $adb_path $adb_mark_executable_params
}

if ($env:TEST_TMPDIR -ne $null) {
  $adb_mkdir_params = Compose-AdbShellCommand "mkdir","-p",$env:TEST_TMPDIR
  Invoke-Adb $adb_path $adb_mkdir_params
  $tmpdir = "TEST_TMPDIR=" + $tmpdir
} else {
  $tmpdir = ""
}

$adb_shell_params = Compose-AdbShellCommand $tmpdir,$test_binary,$test_args
Invoke-Adb $adb_path $adb_shell_params

if ($env:TEST_TMPDIR -ne $null) {
  $adb_rm_params = Compose-AdbShellCommand "rm","-rf",$env:TEST_TMPDIR
  Invoke-Adb $adb_path $adb_rm_params
}

exit 0
