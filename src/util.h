#include <cstdio>
#include <vector>
#include <iostream>

#define PEEK_VECTOR(v, n) {\
  printf("Peeking vector [%s]: [", #v);\
  for (unsigned int i = 0; i < v.size(); ++i) {\
    printf("%f ", v[i]);\
    if (n > 0 && n < v.size() && i >= n-1) {\
      printf(" ... (%d more)", (int)v.size()-i-1);\
      break;\
    }\
  }\
  printf("]\n");\
};

#define LOG_STATS(key, value) {\
  std::cout << "STATS " << "rank -1 " << key << " " << value << std::endl;\
};

#define LOG_STATS_TAGGED(tag, key, value) {\
  std::cout << "STATS " << "rank " << tag << " " << key << " " << value << std::endl;\
};

#define USE_FBVECTOR 0

#if USE_FBVECTOR
#include <folly/FBVector.h>
template <typename T>
using Vector = folly::fbvector<T>;
#else
template <typename T>
using Vector = std::vector<T>;
#endif
