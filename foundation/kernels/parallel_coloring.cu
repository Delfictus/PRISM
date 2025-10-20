// GPU-Accelerated Parallel Graph Coloring
// Each thread attempts one complete greedy coloring with different random seed
// Enables 10,000+ parallel coloring attempts for massive solution space exploration

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" {

// Parallel greedy coloring kernel
// Each thread colors one graph independently with phase guidance
__global__ void parallel_greedy_coloring_kernel(
    const bool* adjacency,        // n×n adjacency matrix (row-major)
    const double* phases,         // n phase values
    const int* vertex_order,      // n vertex ordering (by Kuramoto phase)
    const double* coherence,      // n×n coherence matrix
    int* colorings,               // Output: n_attempts × n (all colorings)
    int* chromatic_numbers,       // Output: n_attempts (chromatic number for each)
    int* conflicts,               // Output: n_attempts (conflicts for each)
    int n_vertices,
    int n_attempts,
    int max_colors,
    unsigned long long base_seed
) {
    int attempt_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (attempt_id >= n_attempts) return;

    // Initialize RNG for this attempt
    curandState_t rng_state;
    curand_init(base_seed, attempt_id, 0, &rng_state);

    // Each thread works on its own coloring
    int* my_coloring = colorings + attempt_id * n_vertices;

    // Initialize all vertices as uncolored
    for (int i = 0; i < n_vertices; i++) {
        my_coloring[i] = -1;  // -1 = uncolored
    }

    // Tiny deterministic variation per attempt (not random!)
    // This explores solution space without destroying phase coherence signal
    double variation_scale = (double)(attempt_id % 100) / 100000.0;  // 0.00000 to 0.00099

    // Color vertices in Kuramoto phase order
    for (int idx = 0; idx < n_vertices; idx++) {
        int v = vertex_order[idx];

        // Find forbidden colors (used by neighbors)
        bool forbidden[256] = {false};
        int forbidden_count = 0;

        for (int u = 0; u < n_vertices; u++) {
            if (adjacency[v * n_vertices + u] && my_coloring[u] >= 0) {
                if (my_coloring[u] < 256) {
                    forbidden[my_coloring[u]] = true;
                    forbidden_count++;
                }
            }
        }

        // Find best color using PURE phase coherence (like CPU algorithm)
        double best_score = -1e9;
        int best_color = 0;

        for (int c = 0; c < max_colors; c++) {
            if (forbidden[c]) continue;

            // Compute phase coherence score for this color (EXACT same as CPU)
            double score = 0.0;
            int count = 0;

            // Average coherence with vertices already using this color
            for (int u = 0; u < n_vertices; u++) {
                if (my_coloring[u] == c) {
                    score += coherence[v * n_vertices + u];
                    count++;
                }
            }

            if (count > 0) {
                score /= count;
            } else {
                score = 1.0;  // New color - neutral score
            }

            // Add TINY deterministic tie-breaker (preserves phase signal!)
            // Different attempts explore slightly different tie-breaking
            score += variation_scale * (double)c;  // Deterministic, tiny (0.00001 scale)

            if (score > best_score) {
                best_score = score;
                best_color = c;
            }
        }

        my_coloring[v] = best_color;
    }

    // Compute chromatic number (max color + 1)
    int max_color_used = -1;
    for (int i = 0; i < n_vertices; i++) {
        if (my_coloring[i] > max_color_used) {
            max_color_used = my_coloring[i];
        }
    }
    chromatic_numbers[attempt_id] = max_color_used + 1;

    // Count conflicts
    int conflict_count = 0;
    for (int i = 0; i < n_vertices; i++) {
        for (int j = i + 1; j < n_vertices; j++) {
            if (adjacency[i * n_vertices + j] &&
                my_coloring[i] >= 0 &&
                my_coloring[j] >= 0 &&
                my_coloring[i] == my_coloring[j]) {
                conflict_count++;
            }
        }
    }
    conflicts[attempt_id] = conflict_count;
}

// Simulated annealing kernel - multiple chains in parallel
__global__ void parallel_sa_kernel(
    const bool* adjacency,
    int* colorings,               // Input/Output: n_chains × n
    int* chromatic_numbers,       // Output: n_chains
    int n_vertices,
    int n_chains,
    int max_colors,
    int iterations_per_chain,
    double initial_temperature,
    unsigned long long base_seed
) {
    int chain_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chain_id >= n_chains) return;

    curandState_t rng_state;
    curand_init(base_seed, chain_id, 0, &rng_state);

    int* my_coloring = colorings + chain_id * n_vertices;
    int current_chromatic = chromatic_numbers[chain_id];
    int best_chromatic = current_chromatic;

    double temperature = initial_temperature;
    double cooling_rate = 0.9995;

    // Local copy for best coloring
    int best_coloring[4096];  // Max 4096 vertices
    for (int i = 0; i < n_vertices; i++) {
        best_coloring[i] = my_coloring[i];
    }

    for (int iter = 0; iter < iterations_per_chain; iter++) {
        // Random move: recolor a vertex
        int v = (int)(curand_uniform_double(&rng_state) * n_vertices);
        int old_color = my_coloring[v];
        int new_color = (int)(curand_uniform_double(&rng_state) * (max_colors + 1));

        // Try the move
        my_coloring[v] = new_color;

        // Count new conflicts
        int new_conflicts = 0;
        for (int u = 0; u < n_vertices; u++) {
            if (adjacency[v * n_vertices + u] && my_coloring[u] == new_color && u != v) {
                new_conflicts++;
            }
        }

        // Compute new chromatic number
        int new_max = -1;
        for (int i = 0; i < n_vertices; i++) {
            if (my_coloring[i] > new_max) {
                new_max = my_coloring[i];
            }
        }
        int new_chromatic = new_max + 1;

        // Accept/reject with Metropolis criterion
        int delta = (new_chromatic - current_chromatic) * 100 + new_conflicts * 10;

        if (delta < 0 || curand_uniform_double(&rng_state) < exp(-delta / temperature)) {
            // Accept
            current_chromatic = new_chromatic;

            // Update best if valid and better
            if (new_conflicts == 0 && new_chromatic < best_chromatic) {
                best_chromatic = new_chromatic;
                for (int i = 0; i < n_vertices; i++) {
                    best_coloring[i] = my_coloring[i];
                }
            }
        } else {
            // Reject - revert
            my_coloring[v] = old_color;
        }

        temperature *= cooling_rate;
    }

    // Write back best solution
    for (int i = 0; i < n_vertices; i++) {
        my_coloring[i] = best_coloring[i];
    }
    chromatic_numbers[chain_id] = best_chromatic;
}

} // extern "C"
