#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "bfs_cuda_2D.h"

#define MAX_PATH_LENGTH 16384   // 128 * 128

__global__ void BFS_kernel_2D(
    const float* __restrict__ occupancy_maps,   // Grid map (1 = walkable, 0 = obstacle)
    const int* __restrict__ starts,
    const int* __restrict__ goals,
    float* __restrict__ path_lengths,
    int num_env,
    int H,
    int W)
{
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= num_env) return;

    // Compute offsets for this environment
    const int occupancy_offset = env_id * H * W;

    // Extract start and goal positions
    int start_x = starts[env_id * 2];
    int start_y = starts[env_id * 2 + 1];
    int goal_x = goals[env_id * 2];
    int goal_y = goals[env_id * 2 + 1];

    // Use global memory for visited and queue
    bool* visited = new bool[H * W];
    int* distances = new int[H * W];
    int* queue = new int[MAX_PATH_LENGTH];

    // Initialize arrays
    for (int i = 0; i < H * W; i++) {
        visited[i] = false;
        distances[i] = 0;
    }

    // Initialize BFS
    int front = 0;
    int back = 1;
    int start_pos = start_x * W + start_y;
    queue[0] = start_pos;
    visited[start_pos] = true;
    distances[start_pos] = 1;
    bool found = false;
    float path_length = -1.0f;

    // BFS loop - stop if path length exceeds MAX_PATH_LENGTH
    while (front < back && back < MAX_PATH_LENGTH && !found) {
        int current_pos = queue[front];
        int current_x = current_pos / W;
        int current_y = current_pos % W;

        // If current distance is already MAX_PATH_LENGTH, break
        if (distances[current_pos] >= MAX_PATH_LENGTH) {
            break;
        }

        if (current_x == goal_x && current_y == goal_y) {
            found = true;
            path_length = (float)distances[current_pos];
            break;
        }

        // // Check 8-connected neighbors (diagonal movement allowed)
        // const int dx[8] = {-1, 0, 1, 0, 1, 1, -1, -1};
        // const int dy[8] = {0, 1, 0, -1, 1, -1, 1, -1};

        // Check 4-connected neighbors (no diagonal movement)
        const int dx[4] = {-1, 0, 1, 0};
        const int dy[4] = {0, 1, 0, -1};

        for (int i = 0; i < 4; i++) {
            int nx = current_x + dx[i];
            int ny = current_y + dy[i];
            
            if (nx >= 0 && nx < H && ny >= 0 && ny < W) {
                int neighbor_pos = nx * W + ny;
                if (occupancy_maps[occupancy_offset + neighbor_pos] == 1 && !visited[neighbor_pos]) {
                    queue[back] = neighbor_pos;
                    visited[neighbor_pos] = true;
                    distances[neighbor_pos] = distances[current_pos] + 1;
                    back++;
                    if (back >= MAX_PATH_LENGTH) break;
                }
            }
        }
        front++;
    }

    path_lengths[env_id] = path_length;

    // Clean up
    delete[] queue;
    delete[] visited;
    delete[] distances;
}

void BFS_kernel_launcher_2D(
    const float* occupancy_maps,
    const int* starts,
    const int* goals,
    float* path_lengths,
    int num_env,
    int H,
    int W)
{
    int threads = 256;
    int blocks = (num_env + threads - 1) / threads;
    BFS_kernel_2D<<<blocks, threads>>>(
        occupancy_maps,
        starts,
        goals,
        path_lengths,
        num_env,
        H,
        W
    );
}