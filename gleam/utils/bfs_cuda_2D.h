#ifndef BFS_CUDA_H
#define BFS_CUDA_H

void BFS_kernel_launcher_2D(
    const float* occupancy_maps,
    const int* starts,
    const int* goals,
    float* path_lengths,
    int num_env,
    int H,
    int W
);

#endif