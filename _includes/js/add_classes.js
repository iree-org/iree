/**
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https:www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

document.addEventListener('DOMContentLoaded', () => {
  const successes =
    document.body.getElementsByClassName('success-table-element');
  Array.prototype.forEach.call(successes, element => {
    element.parentElement.classList.add('success-table-cell');
  });
  const failures =
    document.body.getElementsByClassName('failure-table-element');
  Array.prototype.forEach.call(failures, element => {
    element.parentElement.classList.add('failure-table-cell');
  });
});
