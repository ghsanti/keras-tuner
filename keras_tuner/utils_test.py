# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from keras_tuner import utils


def test_to_list_with_many_return_list():
    many = [(1, 2, 3), "abc", 1, 1.0, "a", {"a": "b"}]
    for item in many:
        result = utils.to_list(item)
        assert isinstance(result, list)
        # test some cases.
        if isinstance(item, tuple):
            assert result == [1, 2, 3]
        elif isinstance(item, list):
            assert result == ["a", "b", "c"]
        elif isinstance(item, int):
            assert result == [1]
        elif isinstance(item, float):
            assert result == [1.0]


def test_try_clear_without_ipython():
    is_notebook = utils.IS_NOTEBOOK
    utils.IS_NOTEBOOK = False
    utils.try_clear()
    utils.IS_NOTEBOOK = is_notebook


def test_create_directory_and_remove_existing(tmp_path):
    utils.create_directory(tmp_path, remove_existing=True)
