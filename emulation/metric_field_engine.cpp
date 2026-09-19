#include "spatiotemporal_memory.hpp"
#include <cmath>
#include <iostream>

namespace elysia::emulation {

#ifndef __CUDACC__

// ============================================================================
// CPU Fallback implementations of CUDA kernels and methods
// ============================================================================

static MetricTensor3x3 invert_3x3_cpu(const MetricTensor3x3& m) {
    MetricTensor3x3 inv;
    float det = m.g[0][0] * (m.g[1][1] * m.g[2][2] - m.g[1][2] * m.g[2][1])
              - m.g[0][1] * (m.g[1][0] * m.g[2][2] - m.g[1][2] * m.g[2][0])
              + m.g[0][2] * (m.g[1][0] * m.g[2][1] - m.g[1][1] * m.g[2][0]);

    float inv_det = 1.0f / (std::max(std::abs(det), 1e-7f) * (det < 0.0f ? -1.0f : 1.0f));

    inv.g[0][0] =  (m.g[1][1] * m.g[2][2] - m.g[1][2] * m.g[2][1]) * inv_det;
    inv.g[0][1] = -(m.g[0][1] * m.g[2][2] - m.g[0][2] * m.g[2][1]) * inv_det;
    inv.g[0][2] =  (m.g[0][1] * m.g[1][2] - m.g[0][2] * m.g[1][1]) * inv_det;

    inv.g[1][0] = -(m.g[1][0] * m.g[2][2] - m.g[1][2] * m.g[2][0]) * inv_det;
    inv.g[1][1] =  (m.g[0][0] * m.g[2][2] - m.g[0][2] * m.g[2][0]) * inv_det;
    inv.g[1][2] = -(m.g[0][0] * m.g[1][2] - m.g[0][2] * m.g[1][0]) * inv_det;

    inv.g[2][0] =  (m.g[1][0] * m.g[2][1] - m.g[1][1] * m.g[2][0]) * inv_det;
    inv.g[2][1] = -(m.g[0][0] * m.g[2][1] - m.g[0][1] * m.g[2][0]) * inv_det;
    inv.g[2][2] =  (m.g[0][0] * m.g[1][1] - m.g[0][1] * m.g[1][0]) * inv_det;

    return inv;
}

MetricFieldEngine::MetricFieldEngine(size_t grid_dim_x, size_t grid_dim_y, size_t grid_dim_z)
    : grid_x_(grid_dim_x), grid_y_(grid_dim_y), grid_z_(grid_dim_z),
      total_cells_(grid_dim_x * grid_dim_y * grid_dim_z) {

    size_t bytes = total_cells_ * sizeof(MetricTensor3x3);
    cudaMalloc(reinterpret_cast<void**>(&d_g_mem_field_), bytes);

    std::vector<MetricTensor3x3> host_init(total_cells_);
    for (size_t k = 0; k < total_cells_; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                host_init[k].g[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    cudaMemcpy(d_g_mem_field_, host_init.data(), bytes, cudaMemcpyHostToDevice);
}

MetricFieldEngine::~MetricFieldEngine() {
    if (d_g_mem_field_) {
        cudaFree(d_g_mem_field_);
    }
}

void MetricFieldEngine::launch_riemannian_gradient_kernel(
    const float* d_pos_x, float* d_grad_out, size_t num_particles, cudaStream_t stream) {
    (void)stream;
    for (size_t idx = 0; idx < num_particles; ++idx) {
        float px = d_pos_x[idx * 3 + 0];
        float py = d_pos_x[idx * 3 + 1];
        float pz = d_pos_x[idx * 3 + 2];

        int gx = std::min(std::max(static_cast<int>(px * grid_x_), 0), static_cast<int>(grid_x_) - 1);
        int gy = std::min(std::max(static_cast<int>(py * grid_y_), 0), static_cast<int>(grid_y_) - 1);
        int gz = std::min(std::max(static_cast<int>(pz * grid_z_), 0), static_cast<int>(grid_z_) - 1);
        size_t cell_idx = gx + gy * grid_x_ + gz * grid_x_ * grid_y_;

        MetricTensor3x3 g_inv = invert_3x3_cpu(d_g_mem_field_[cell_idx]);
        float grad_euc[3] = { px - 0.5f, py - 0.5f, pz - 0.5f };

        d_grad_out[idx * 3 + 0] = g_inv.g[0][0] * grad_euc[0] + g_inv.g[0][1] * grad_euc[1] + g_inv.g[0][2] * grad_euc[2];
        d_grad_out[idx * 3 + 1] = g_inv.g[1][0] * grad_euc[0] + g_inv.g[1][1] * grad_euc[1] + g_inv.g[1][2] * grad_euc[2];
        d_grad_out[idx * 3 + 2] = g_inv.g[2][0] * grad_euc[0] + g_inv.g[2][1] * grad_euc[1] + g_inv.g[2][2] * grad_euc[2];
    }
}

void MetricFieldEngine::launch_plasticity_update_kernel(
    const float* d_pos_x, const float* d_velocity, const Rotor* d_torques, float alpha, float beta,
    size_t num_particles, cudaStream_t stream) {
    (void)stream;
    for (size_t idx = 0; idx < num_particles; ++idx) {
        float px = d_pos_x[idx * 3 + 0];
        float py = d_pos_x[idx * 3 + 1];
        float pz = d_pos_x[idx * 3 + 2];

        int gx = std::min(std::max(static_cast<int>(px * grid_x_), 0), static_cast<int>(grid_x_) - 1);
        int gy = std::min(std::max(static_cast<int>(py * grid_y_), 0), static_cast<int>(grid_y_) - 1);
        int gz = std::min(std::max(static_cast<int>(pz * grid_z_), 0), static_cast<int>(grid_z_) - 1);
        size_t cell_idx = gx + gy * grid_x_ + gz * grid_x_ * grid_y_;

        float vx = d_velocity[idx * 3 + 0];
        float vy = d_velocity[idx * 3 + 1];
        float vz = d_velocity[idx * 3 + 2];

        float v_outer[3][3] = {
            { vx * vx, vx * vy, vx * vz },
            { vy * vx, vy * vy, vy * vz },
            { vz * vx, vz * vy, vz * vz }
        };

        Rotor r = d_torques[idx];
        float omega_skew[3][3] = {
            { 0.0f,          -r.bivector_xy,  r.bivector_zx },
            {  r.bivector_xy, 0.0f,          -r.bivector_yz },
            { -r.bivector_zx, r.bivector_yz,  0.0f          }
        };

        MetricTensor3x3* target_cell = &d_g_mem_field_[cell_idx];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float delta = alpha * v_outer[i][j] + beta * omega_skew[i][j];
                target_cell->g[i][j] += delta;
            }
        }
    }
}

void MetricFieldEngine::launch_entropy_decay_kernel(
    float gamma, float diffusion_D, float delta_t, cudaStream_t stream) {
    (void)stream;
    std::vector<MetricTensor3x3> g_out(total_cells_);

    auto fetch_cell = [&](int x, int y, int z) {
        x = std::min(std::max(x, 0), static_cast<int>(grid_x_) - 1);
        y = std::min(std::max(y, 0), static_cast<int>(grid_y_) - 1);
        z = std::min(std::max(z, 0), static_cast<int>(grid_z_) - 1);
        return d_g_mem_field_[x + y * grid_x_ + z * grid_x_ * grid_y_];
    };

    for (int gz = 0; gz < static_cast<int>(grid_z_); ++gz) {
        for (int gy = 0; gy < static_cast<int>(grid_y_); ++gy) {
            for (int gx = 0; gx < static_cast<int>(grid_x_); ++gx) {
                size_t center_idx = gx + gy * grid_x_ + gz * grid_x_ * grid_y_;
                MetricTensor3x3 center = d_g_mem_field_[center_idx];

                MetricTensor3x3 px = fetch_cell(gx + 1, gy, gz);
                MetricTensor3x3 nx = fetch_cell(gx - 1, gy, gz);
                MetricTensor3x3 py = fetch_cell(gx, gy + 1, gz);
                MetricTensor3x3 ny = fetch_cell(gx, gy - 1, gz);
                MetricTensor3x3 pz = fetch_cell(gx, gy, gz + 1);
                MetricTensor3x3 nz = fetch_cell(gx, gy, gz - 1);

                MetricTensor3x3 updated;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        float laplacian = (px.g[i][j] + nx.g[i][j] +
                                           py.g[i][j] + ny.g[i][j] +
                                           pz.g[i][j] + nz.g[i][j] - 6.0f * center.g[i][j]);

                        float identity = (i == j) ? 1.0f : 0.0f;
                        float decay = -gamma * (center.g[i][j] - identity);

                        float dg_dt = decay + diffusion_D * laplacian;
                        updated.g[i][j] = center.g[i][j] + delta_t * dg_dt;
                    }
                }
                g_out[center_idx] = updated;
            }
        }
    }

    std::memcpy(d_g_mem_field_, g_out.data(), total_cells_ * sizeof(MetricTensor3x3));
}

#endif // __CUDACC__

} // namespace elysia::emulation
