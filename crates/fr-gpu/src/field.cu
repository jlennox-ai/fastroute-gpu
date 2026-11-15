// The guide field on the GPU: the raster of fr_engine::autoroute::guide::raster and the
// cost-to-go relaxation whose fixpoint is the Dijkstra field of the CPU solver.
//
// Layouts (all per cell arrays are layer-major: [layer][y][x]):
// * polygon metadata: POLYGON_WORDS u32 per polygon (first half-plane, half-plane count,
//   layer, kind, item, cost bits, min x bits, min y bits, max x bits, max y bits);
// * half-planes: 3 floats (nx, ny, c), inside if nx * x + ny * y <= c;
// * soft keys: (cost bits << 32) | item, the cheapest soft item per cell (atomicMin);
// * move masks: bit m set if move m (MOVE_DX/MOVE_DY) is allowed from the cell.
//
// The raster tests use __fmul_rn/__fadd_rn/__fdiv_rn so that they are never contracted into
// fused multiply-adds and give the same answers as the CPU.

#define POLYGON_WORDS 10
#define KIND_HARD 0u
#define KIND_SOFT 1u
#define KIND_DESTINATION 2u
#define NO_ITEM 0xffffffffu
#define NO_SOFT 0xffffffffffffffffull
// move mask flag: no soft charge on any move from the cell
#define CHARGE_FREE (1u << 17)
#define TILE 32
#define HALO 2
#define SPAN (TILE + 2 * HALO)

__constant__ int MOVE_DX[16] = {1, -1, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 2, 2, -2, -2};
__constant__ int MOVE_DY[16] = {0, 0, 1, -1, 1, -1, 1, -1, 2, -2, 2, -2, 1, -1, 1, -1};
// the two cells a diagonal or knight move passes (raster::MOVE_PASSES)
__constant__ int PASS_AX[16] = {0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1};
__constant__ int PASS_AY[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0};
__constant__ int PASS_BX[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};
__constant__ int PASS_BY[16] = {0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1};

__device__ __forceinline__ float cell_centre(int index, float cell_size) {
    return __fmul_rn(__fadd_rn((float)index, 0.5f), cell_size);
}

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

extern "C" __global__ void fill_bytes(unsigned char* bytes, unsigned char value, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        bytes[i] = value;
    }
}

extern "C" __global__ void clear_grid(unsigned char* hard, unsigned char* destination, unsigned long long* soft_key, int cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cells) {
        hard[i] = 0;
        destination[i] = 0;
        soft_key[i] = NO_SOFT;
    }
}

// One block per polygon (grid-stride over polygons), the block's threads over the cells of the
// polygon's bounding box.
extern "C" __global__ void rasterize(const unsigned int* meta, const float* half_planes, int polygon_count, int width, int height,
                                     float cell_size, unsigned char* hard, unsigned char* destination, unsigned long long* soft_key) {
    for (int p = blockIdx.x; p < polygon_count; p += gridDim.x) {
        const unsigned int* m = meta + (size_t)p * POLYGON_WORDS;
        unsigned int first = m[0];
        unsigned int count = m[1];
        unsigned int layer = m[2];
        unsigned int kind = m[3];
        unsigned int item = m[4];
        unsigned int cost_bits = m[5];
        float min_x = __uint_as_float(m[6]);
        float min_y = __uint_as_float(m[7]);
        float max_x = __uint_as_float(m[8]);
        float max_y = __uint_as_float(m[9]);
        int x0 = clamp_int((int)floorf(__fdiv_rn(min_x, cell_size)) - 1, 0, width);
        int x1 = clamp_int((int)floorf(__fdiv_rn(max_x, cell_size)) + 2, 0, width);
        int y0 = clamp_int((int)floorf(__fdiv_rn(min_y, cell_size)) - 1, 0, height);
        int y1 = clamp_int((int)floorf(__fdiv_rn(max_y, cell_size)) + 2, 0, height);
        int span_x = x1 - x0;
        int total = span_x * (y1 - y0);
        size_t plane = (size_t)layer * width * height;
        for (int c = threadIdx.x; c < total; c += blockDim.x) {
            int x = x0 + c % span_x;
            int y = y0 + c / span_x;
            float cx = cell_centre(x, cell_size);
            float cy = cell_centre(y, cell_size);
            bool inside = true;
            for (unsigned int h = 0; h < count && inside; h++) {
                const float* hp = half_planes + 3 * (size_t)(first + h);
                inside = __fadd_rn(__fmul_rn(hp[0], cx), __fmul_rn(hp[1], cy)) <= hp[2];
            }
            if (!inside) {
                continue;
            }
            size_t i = plane + (size_t)y * width + x;
            if (kind == KIND_HARD) {
                hard[i] = 1;
            } else if (kind == KIND_SOFT) {
                atomicMin(soft_key + i, ((unsigned long long)cost_bits << 32) | item);
            } else {
                destination[i] = 1;
            }
        }
        if (kind == KIND_DESTINATION && threadIdx.x == 0) {
            // the anchor: the cell of the bounding box centre
            float ax = floorf(__fdiv_rn(__fmul_rn(__fadd_rn(min_x, max_x), 0.5f), cell_size));
            float ay = floorf(__fdiv_rn(__fmul_rn(__fadd_rn(min_y, max_y), 0.5f), cell_size));
            if (ax >= 0.0f && ay >= 0.0f && ax < (float)width && ay < (float)height) {
                destination[plane + (size_t)ay * width + (size_t)ax] = 1;
            }
        }
    }
}

// Blocked flags, soft items and costs, and the start values (0 in the destination cells).
extern "C" __global__ void finalize(const unsigned char* hard, const unsigned char* destination, const unsigned long long* soft_key, int cells,
                                    unsigned char* blocked, unsigned int* soft_item, float* soft_cost, float* dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) {
        return;
    }
    bool is_destination = destination[i] != 0;
    blocked[i] = (hard[i] != 0 && !is_destination) ? 1 : 0;
    unsigned long long key = soft_key[i];
    if (key == NO_SOFT) {
        soft_item[i] = NO_ITEM;
        soft_cost[i] = 0.0f;
    } else {
        soft_item[i] = (unsigned int)(key & 0xffffffffull);
        soft_cost[i] = __uint_as_float((unsigned int)(key >> 32));
    }
    dist[i] = is_destination ? 0.0f : __int_as_float(0x7f800000);
}

// Boundary seeds: pairs (cell index, value bits); applied in free cells.
extern "C" __global__ void seed_boundary(const unsigned int* seeds, int seed_count, const unsigned char* blocked, float* dist) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seed_count) {
        return;
    }
    unsigned int i = seeds[2 * s];
    float value = __uint_as_float(seeds[2 * s + 1]);
    if (!blocked[i]) {
        dist[i] = fminf(dist[i], value);
    }
}

// The charge of entering cell i from a cell holding soft item from_item.
__device__ __forceinline__ bool is_free(const unsigned char* blocked_plane, int width, int height, int x, int y) {
    return x >= 0 && y >= 0 && x < width && y < height && !blocked_plane[(size_t)y * width + x];
}

// The allowed moves of every cell (fr_engine::autoroute::guide::raster::Raster::move_cost) in
// bits 0-15, and CHARGE_FREE if no move from the cell can be charged: every cell within two
// cells holds no soft item or the cell's own.
extern "C" __global__ void move_masks(const unsigned char* blocked, const unsigned int* soft_item, int width, int height, int layers,
                                      unsigned int* masks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int plane_cells = width * height;
    if (i >= plane_cells * layers) {
        return;
    }
    int layer = i / plane_cells;
    int r = i % plane_cells;
    int x = r % width;
    int y = r / width;
    const unsigned char* b = blocked + (size_t)layer * plane_cells;
    unsigned int mask = 0;
    if (is_free(b, width, height, x, y)) {
        for (int m = 0; m < 16; m++) {
            int dx = MOVE_DX[m];
            int dy = MOVE_DY[m];
            if (!is_free(b, width, height, x + dx, y + dy)) {
                continue;
            }
            bool passable = true;
            if (m >= 4) {
                bool free_a = is_free(b, width, height, x + PASS_AX[m], y + PASS_AY[m]);
                bool free_b = is_free(b, width, height, x + PASS_BX[m], y + PASS_BY[m]);
                passable = m < 8 ? (free_a || free_b) : (free_a && free_b);
            }
            if (passable) {
                mask |= 1u << m;
            }
        }
        const unsigned int* items = soft_item + (size_t)layer * plane_cells;
        unsigned int own = items[(size_t)y * width + x];
        bool charge_free = true;
        for (int dy = -2; dy <= 2 && charge_free; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                    continue;
                }
                unsigned int item = items[(size_t)ny * width + nx];
                if (item != NO_ITEM && item != own) {
                    charge_free = false;
                    break;
                }
            }
        }
        if (charge_free) {
            mask |= CHARGE_FREE;
        }
    }
    masks[i] = mask;
}

// Shared memory of relax_tiles. The halo arrays (cost-to-go, soft item, soft cost) have rows of
// PITCH words and the tile array (move mask with CHARGE_FREE, BLOCKED_BIT) rows of TILE_PITCH words:
// odd pitches, so that the lanes of a left/right sweep (a lane per row) hit different banks.
#define PITCH (SPAN + 1)
#define TILE_PITCH (TILE + 1)
#define BLOCKED_BIT (1u << 16)
#define SHARED_PER_LAYER (4 * (3 * SPAN * PITCH + TILE * TILE_PITCH))

// The moves pulled by each sweep, from the cells already swept (left to right: moves with
// dx < 0, right to left: dx > 0, bottom to top: dy < 0, top to bottom: dy > 0): the orthogonal
// move, the two diagonal moves, the four knight moves.
__constant__ int SWEEP_MOVES[4][7] = {
    {1, 6, 7, 10, 11, 14, 15},
    {0, 4, 5, 8, 9, 12, 13},
    {3, 5, 7, 9, 11, 13, 15},
    {2, 4, 6, 8, 10, 12, 14},
};

// One block per tile of TILE x TILE cells, a warp per layer and sweep direction (blockDim
// (32, 4, layers)). An active tile loads its cells and a halo of HALO cells into shared memory
// and relaxes its cells with directional sweeps (a lane per row or column, vias in every step;
// the four directions run at the same time) until a round changes nothing (or max_inner
// rounds), then writes them back. It wakes (marks active for the next launch) the neighbours
// whose halo it changed, and itself if it did not settle. Every move is in some sweep, so the
// fixpoint is the one of all moves. Values only decrease, so concurrent updates (of the other
// sweeps and layers of the tile, of neighbouring tiles) are safe: a tile that read a stale halo
// is woken again by the neighbour that changed it.
// active_in and active_out are swapped between launches; a tile clears its active_in flag, so
// both are empty again when the field is done.
// rounds_done (profile): [0] rounds of the tiles, [1] tiles that did not settle.
extern "C" __global__ void relax_tiles(float* dist, const unsigned int* masks, const unsigned char* blocked, const unsigned int* soft_item,
                                       const float* soft_cost, const float* lengths, int width, int height, int layers, float via_cost,
                                       unsigned char* active_in, unsigned char* active_out, int* changed, int max_inner,
                                       unsigned long long* rounds_done) {
    extern __shared__ unsigned int shared_words[];
    __shared__ int s_changed;
    __shared__ int s_tile_changed;
    // bit (dy + 1) * 3 + (dx + 1): a cell in the halo of neighbour (dx, dy) changed; bit 4: the
    // tile did not settle
    __shared__ unsigned int s_wake;
    __shared__ float s_lengths[16 * 8];
    int tile = blockIdx.y * gridDim.x + blockIdx.x;
    // One thread reads and clears the flag (this buffer is the next launch's active_out, which
    // has to start empty); the block decides on the copy, so that no warp sees it cleared.
    __shared__ int s_active;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        s_active = active_in[tile];
        active_in[tile] = 0;
    }
    __syncthreads();
    if (!s_active) {
        return;
    }
    float* s_dist = (float*)shared_words;
    unsigned int* s_item = shared_words + layers * SPAN * PITCH;
    float* s_cost = (float*)(s_item + layers * SPAN * PITCH);
    unsigned int* s_cell = (unsigned int*)(s_cost + layers * SPAN * PITCH);
    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y * blockDim.z;
    int tx0 = blockIdx.x * TILE;
    int ty0 = blockIdx.y * TILE;
    size_t plane_cells = (size_t)width * height;
    for (int i = tid; i < layers * 16; i += threads) {
        s_lengths[i] = lengths[i];
    }
    for (int i = tid; i < layers * SPAN * SPAN; i += threads) {
        int l = i / (SPAN * SPAN);
        int r = i % (SPAN * SPAN);
        int hx = r % SPAN;
        int hy = r / SPAN;
        int gx = tx0 + hx - HALO;
        int gy = ty0 + hy - HALO;
        int o = l * SPAN * PITCH + hy * PITCH + hx;
        if (gx >= 0 && gy >= 0 && gx < width && gy < height) {
            size_t g = l * plane_cells + (size_t)gy * width + gx;
            s_dist[o] = dist[g];
            s_item[o] = soft_item[g];
            s_cost[o] = soft_cost[g];
        } else {
            s_dist[o] = __int_as_float(0x7f800000);
            s_item[o] = NO_ITEM;
            s_cost[o] = 0.0f;
        }
    }
    for (int i = tid; i < layers * TILE * TILE; i += threads) {
        int l = i / (TILE * TILE);
        int r = i % (TILE * TILE);
        int cx = r % TILE;
        int cy = r / TILE;
        int gx = tx0 + cx;
        int gy = ty0 + cy;
        int o = l * TILE * TILE_PITCH + cy * TILE_PITCH + cx;
        if (gx < width && gy < height) {
            size_t g = l * plane_cells + (size_t)gy * width + gx;
            s_cell[o] = masks[g] | (blocked[g] ? BLOCKED_BIT : 0u);
        } else {
            s_cell[o] = BLOCKED_BIT;
        }
    }
    if (tid == 0) {
        s_tile_changed = 0;
        s_wake = 0;
    }
    __syncthreads();
    int lane = threadIdx.x;
    int sweep = threadIdx.y;
    int layer = threadIdx.z;
    int base = layer * SPAN * PITCH;
    // the sweep's moves (one orthogonal (k = 0), two diagonal, four knight moves): offsets in the
    // halo arrays and lengths
    int move_offset[7];
    int pass_a_offset[7];
    int pass_b_offset[7];
    float move_length[7];
#pragma unroll
    for (int k = 0; k < 7; k++) {
        int m = SWEEP_MOVES[sweep][k];
        move_offset[k] = MOVE_DY[m] * PITCH + MOVE_DX[m];
        pass_a_offset[k] = PASS_AY[m] * PITCH + PASS_AX[m];
        pass_b_offset[k] = PASS_BY[m] * PITCH + PASS_BX[m];
        move_length[k] = s_lengths[layer * 16 + m];
    }
    unsigned int wake = 0;
    bool settled = false;
    int rounds = 0;
    for (int round = 0; round < max_inner; round++) {
        rounds++;
        if (tid == 0) {
            s_changed = 0;
        }
        __syncthreads();
        bool local_changed = false;
        for (int step = 0; step < TILE; step++) {
            int cx = sweep == 0 ? step : (sweep == 1 ? TILE - 1 - step : lane);
            int cy = sweep == 2 ? step : (sweep == 3 ? TILE - 1 - step : lane);
            unsigned int cell = s_cell[layer * TILE * TILE_PITCH + cy * TILE_PITCH + cx];
            if (!(cell & BLOCKED_BIT)) {
                int s = base + (cy + HALO) * PITCH + cx + HALO;
                float best = s_dist[s];
                if (cell & CHARGE_FREE) {
#pragma unroll
                    for (int k = 0; k < 7; k++) {
                        if ((cell >> SWEEP_MOVES[sweep][k]) & 1u) {
                            best = fminf(best, s_dist[s + move_offset[k]] + move_length[k]);
                        }
                    }
                } else {
                    unsigned int item_p = s_item[s];
#pragma unroll
                    for (int k = 0; k < 7; k++) {
                        if (!((cell >> SWEEP_MOVES[sweep][k]) & 1u)) {
                            continue;
                        }
                        int q = s + move_offset[k];
                        unsigned int item_q = s_item[q];
                        float enter = (item_q != NO_ITEM && item_q != item_p) ? s_cost[q] : 0.0f;
                        if (k > 0) {
                            int a = s + pass_a_offset[k];
                            int b = s + pass_b_offset[k];
                            float pass_a = (s_item[a] != NO_ITEM && s_item[a] != item_p) ? s_cost[a] : 0.0f;
                            float pass_b = (s_item[b] != NO_ITEM && s_item[b] != item_p) ? s_cost[b] : 0.0f;
                            enter = fmaxf(enter, k < 3 ? fminf(pass_a, pass_b) : fmaxf(pass_a, pass_b));
                        }
                        best = fminf(best, s_dist[q] + move_length[k] + enter);
                    }
                }
                for (int l2 = 0; l2 < layers; l2++) {
                    if (l2 == layer || (s_cell[l2 * TILE * TILE_PITCH + cy * TILE_PITCH + cx] & BLOCKED_BIT)) {
                        continue;
                    }
                    int other = s + (l2 - layer) * SPAN * PITCH;
                    float via_charge = s_item[other] != NO_ITEM ? s_cost[other] : 0.0f;
                    best = fminf(best, s_dist[other] + via_cost + via_charge);
                }
                if (best < s_dist[s]) {
                    s_dist[s] = best;
                    local_changed = true;
                    int xs = cx < HALO ? -1 : (cx >= TILE - HALO ? 1 : 0);
                    int ys = cy < HALO ? -1 : (cy >= TILE - HALO ? 1 : 0);
                    if (xs != 0) {
                        wake |= 1u << (4 + xs);
                    }
                    if (ys != 0) {
                        wake |= 1u << (4 + 3 * ys);
                    }
                    if (xs != 0 && ys != 0) {
                        wake |= 1u << (4 + 3 * ys + xs);
                    }
                }
            }
            __syncwarp();
        }
        if (local_changed) {
            s_changed = 1;
        }
        __syncthreads();
        if (!s_changed) {
            settled = true;
            break;
        }
        if (tid == 0) {
            s_tile_changed = 1;
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(rounds_done, (unsigned long long)rounds);
        atomicAdd(rounds_done + 1, settled ? 0ull : 1ull);
    }
    if (wake != 0) {
        atomicOr(&s_wake, wake);
    }
    if (!settled && tid == 0) {
        atomicOr(&s_wake, 1u << 4);
    }
    __syncthreads();
    if (!s_tile_changed) {
        return;
    }
    for (int i = tid; i < layers * TILE * TILE; i += threads) {
        int l = i / (TILE * TILE);
        int r = i % (TILE * TILE);
        int cx = r % TILE;
        int cy = r / TILE;
        int gx = tx0 + cx;
        int gy = ty0 + cy;
        if (gx < width && gy < height) {
            dist[l * plane_cells + (size_t)gy * width + gx] = s_dist[l * SPAN * PITCH + (cy + HALO) * PITCH + cx + HALO];
        }
    }
    if (tid < 9 && (s_wake >> tid) & 1u) {
        int nx = (int)blockIdx.x + tid % 3 - 1;
        int ny = (int)blockIdx.y + tid / 3 - 1;
        if (nx >= 0 && ny >= 0 && nx < (int)gridDim.x && ny < (int)gridDim.y) {
            active_out[ny * gridDim.x + nx] = 1;
            *changed = 1;
        }
    }
}
