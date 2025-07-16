#include <torch/extension.h>
#include "bfs_cuda_2D.h"

void BFS_CUDA_2D(
    torch::Tensor occupancy_maps,
    torch::Tensor starts,
    torch::Tensor goals,
    torch::Tensor path_lengths)
{
    int num_env = occupancy_maps.size(0);
    int H = occupancy_maps.size(1);
    int W = occupancy_maps.size(2);

    BFS_kernel_launcher_2D(
        // occupancy_maps.data_ptr<int>(),
        occupancy_maps.data_ptr<float>(),
        starts.data_ptr<int>(),
        goals.data_ptr<int>(),
        path_lengths.data_ptr<float>(),
        num_env,
        H,
        W
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("BFS_CUDA_2D", &BFS_CUDA_2D, "BFS Pathfinding on CUDA");
}