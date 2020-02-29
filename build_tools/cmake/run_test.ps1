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

# Create/cleanup the test output directory (where gtest files are written, etc).
$test_tmpdir = $env:TEST_TMPDIR
if ($null -eq $test_tmpdir) {
  Write-Error "TEST_TMPDIR environment variable not set" -Category InvalidArgument
  Get-ChildItem env:
  exit 1
}
Write-Debug "Preparing test output path $test_tmpdir..."
if (Test-Path $test_tmpdir) {
  Write-Debug "Removing existing folder at $test_tmpdir"
  Remove-Item $test_tmpdir -Recurse -Force
}
New-Item -Path $test_tmpdir -ItemType Directory -Force | Out-Null
Write-Debug "Created new folder at $test_tmpdir"
trap {
  if (Test-Path $test_tmpdir) {
    Write-Debug "Cleaning up $test_tmpdir on error..."
    Remove-Item $test_tmpdir -Recurse -Force
  }
  Write-Error $_
  exit 1
}

# Run the test executable with all arguments we were passed.
Write-Host -ForegroundColor Blue "Running test:"
Write-Host -ForegroundColor Yellow "$test_binary $test_args"
$process = $null
if ($test_args.Count -gt 0) {
  $process = Start-Process -FilePath $test_binary -ArgumentList $test_args -NoNewWindow -PassThru
} else {
  $process = Start-Process -FilePath $test_binary -NoNewWindow -PassThru
}
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

# Cleanup test tempdir.
Write-Debug "Cleaning up $test_tmpdir..."
Remove-Item $test_tmpdir -Recurse -Force
Write-Debug "Test exited with $exitcode"
exit $exitcode
