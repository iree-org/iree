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
    $test_file,
  [Parameter(Position=1, ValueFromRemainingArguments=$true)]
  [string[]]
    $test_data = @()
)

# NOTE: to debug first run `$DebugPreference = 'Continue'` in your shell.
# $DebugPreference = "Continue"

trap {
  Write-Error $_
  exit 1
}

# Get all of the directories we'll want to put on our path for the test.
$test_dirs = [System.Collections.ArrayList]@()
foreach ($test_path in $test_data) {
  $test_dir = Split-Path -Path $test_path -Parent
  $test_dirs.Add($test_dir) | Out-Null
}
Write-Debug "Test data directories: $test_dirs"
$test_dirs.AddRange($env:Path.Split(";"))
$env:Path = $test_dirs -join ";"
Write-Debug "Test PATH:"
Write-Debug "$env:PATH"

$test_lines = Get-Content -Path $test_file
foreach ($test_line in $test_lines) {
  if (!$test_line.StartsWith("// RUN:")) {
    continue
  }
  $test_line = $test_line.Substring("// RUN: ".Length)
  $test_line = $test_line -replace "%s", $test_file
  Write-Host -ForegroundColor Blue "Running test command:"
  Write-Host -ForegroundColor Yellow "$test_line"
  & "bash.exe" -c $test_line | Out-Default
  if ($LASTEXITCODE -gt 0) {
    Write-Host -ForegroundColor Red "Test failed with $LASTEXITCODE, command:"
    Write-Host -ForegroundColor Yellow "$test_line"
    exit $LASTEXITCODE
  }
}

Write-Debug "All run commands completed successfully"
exit 0
