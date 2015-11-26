Introduction
------------

An implementation of the Parallel Gradient Boosted Regression Trees algorithm
[1] for a course project. Parallelization is based on MPI.

Building
--------
Dependencies:

* g++ (a version supporting c++11)
* cmake
* glogs
* gflags
* gtest
* boost (mpi, serialization)

To build, execute the following:

```bash
mkdir build
cd build
cmake ../
make
```

Current status
--------------
Not usable yet (duh...). More to come.


References
----------
[1] Stephen Tyree, Kilian Q. Weinberger, Kunal Agrawal, and Jennifer Paykin.
Parallel Boosted Regression Trees for Web Search Ranking. Proceedings of the
20th international conference on World Wide Web (WWW), pages 387-396, ACM,
2011.
