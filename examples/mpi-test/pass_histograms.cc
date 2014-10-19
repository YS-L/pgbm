#include "histogram.h"
#include "mpi-util.h"

#include <glog/logging.h>

int main() {
  unsigned int N = 10;
  Histogram hist(N);
  for (unsigned int i = 1; i <= 2 * N; ++i) {
    hist.Update(i, 1);
  }
  mpi::environment& env = MPIHandle::Get().env;
  mpi::communicator& world = MPIHandle::Get().world;

  if (world.rank() == 0) {
    world.send(1, 0, hist);
    Histogram msg;
    world.recv(1, 1, msg);
    LOG(INFO) << "Rank 0 done";
  } else {
    Histogram msg;
    world.recv(0, 0, msg);
    std::cout << msg << ", ";
    std::cout.flush();
    world.send(0, 1, hist);
    LOG(INFO) << "Rank 1 done";
  }
}
