#include <cstdio>
#include <vector>

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
