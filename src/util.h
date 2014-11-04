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
  std::cout << "STATS " << "-1 " << key << " " << value << std::endl;\
};

#define LOG_STATS_TAGGED(tag, key, value) {\
  std::cout << "STATS " << tag << " " << key << " " << value << std::endl;\
};
