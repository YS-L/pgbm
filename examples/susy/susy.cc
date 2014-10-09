#include "data.h"
#include "boosting.h"

#include <glog/logging.h>

int main(int argc, char** argv) {

  LOG(INFO) << "Susy started";

  DataMatrix data;
  data.Load("../../Scripts/susy/susy.svmlight.50k");
  LOG(INFO) << "Size: " << data.Size()
            << " x "
            << data.Dimension();

  Booster booster(50, 0.05);
  booster.Train(data);

  return 0;
};
