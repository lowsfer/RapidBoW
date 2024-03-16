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
