// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Colors taken and slightly modified from the Just the Docs theme to match
// the rest of the site.
let SUCCESS_COLOR = '#7253ed66';
let FAILURE_COLOR = '#f5f6fa';

window.onload = () => {
  let successes = document.body.getElementsByClassName('success-table-element');
  Array.prototype.forEach.call(successes, element => {
    element.parentElement.style.backgroundColor = SUCCESS_COLOR;
  })
  let failures = document.body.getElementsByClassName('failure-table-element');
  Array.prototype.forEach.call(failures, element => {
    element.parentElement.style.backgroundColor = FAILURE_COLOR;
  })
}
