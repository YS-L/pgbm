#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

class MPIHandle
{
  public:
    static MPIHandle& Get() {
      static MPIHandle instance; // Guaranteed to be destroyed.
                                 // Instantiated on first use.
      return instance;
    }
    boost::mpi::environment env;
    boost::mpi::communicator world;

  private:
    MPIHandle() {};
    // Dont forget to declare these two. You want to make sure they
    // are unaccessable otherwise you may accidently get copies of
    // your singleton appearing.
    MPIHandle(MPIHandle const&); // Don't Implement
    void operator=(MPIHandle const&); // Don't implement
};
