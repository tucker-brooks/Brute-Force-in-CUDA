# Brute-Force-in-CUDA

Demonstrated the profound impact of growing GPU computational power by significantly reducing time complexity in password brute forcing compared to the recursive CPU-based approach, achieving a 3.71x speedup with outermost loop parallelization and an extraordinary 18,570x speedup with one-to-one mapping.

BruteForceCPU is a recursive CPU approach

BruteForceGPU is parallelizing the outermost for loop on the GPU

oneToOneMap is a full one to one mapping taking full advantage of the GPU arcitecture
