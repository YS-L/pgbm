#include "histogram.h"
#include "mpi_util.h"

#include <folly/FBVector.h>
#include <glog/logging.h>

int main() {

  /*
  folly::fbvector<int> numbers({0, 1, 2, 3});
  numbers.reserve(10);
  for (int i = 4; i < 10; i++) {
    numbers.push_back(i * 2);
  }
  assert(numbers[6] == 12);
  folly::fbvector<int>& msg = numbers;
  */

  unsigned int N = 10;
  Histogram hist(N);
  for (unsigned int i = 1; i <= 2 * N; ++i) {
    hist.Update(i, 1);
  }
  Histogram& msg = hist;

  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;

  if (world.rank() == 0) {
    world.send(1, 0, msg);
    world.recv(1, 1, msg);
    LOG(INFO) << "Rank 0 done";
  } else {
    world.recv(0, 0, msg);
    world.send(0, 1, msg);
    LOG(INFO) << "Rank 1 done";
  }
}
