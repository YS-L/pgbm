#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

class MPIHandle
{
  public:
    static MPIHandle& Get() {
      static MPIHandle instance; // Guaranteed to be destroyed.
                                 // Instantiated on first use.
      return instance;
    }

    mpi::environment env;
    mpi::communicator world;

  private:
    MPIHandle() {};
    // Dont forget to declare these two. You want to make sure they
    // are unaccessable otherwise you may accidently get copies of
    // your singleton appearing.
    MPIHandle(MPIHandle const&); // Don't Implement
    void operator=(MPIHandle const&); // Don't implement
};
