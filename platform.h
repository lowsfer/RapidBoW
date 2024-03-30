/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

//
// Created by yao on 9/3/19.
//

#pragma once

//for IDE parser
#if defined(Q_CREATOR_RUN) || defined(__CLION_IDE__) || defined (__INTELLISENSE__) || defined(IN_KDEVELOP_PARSER) || defined(__JETBRAINS_IDE__)
#define IS_IN_IDE_PARSER 1
#else
#define IS_IN_IDE_PARSER 0
#endif
