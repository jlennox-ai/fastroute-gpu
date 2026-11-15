# GPU acceleration (`--gpu`)

`--gpu` computes a cost-to-go field for each connection on an NVIDIA GPU (CUDA) and uses it as
the estimate of the maze search. The search itself, insertion, shoving, pull-tight and DRC stay
on the CPU and unchanged; the field only changes the order in which the search expands rooms.

```sh
cargo build --release -p fastroute --features gpu     # needs no CUDA toolkit to build
fastroute -de board.dsn -do board.ses --gpu ...
```

At run time `--gpu` needs the NVIDIA driver and NVRTC (`nvrtc64_*.dll` / `libnvrtc.so` from a
CUDA 12 toolkit, on the library path). The kernels are compiled for the device found at start;
both libraries are loaded at run time, so a binary built with the feature runs without them
until `--gpu` is given. Without the feature, `--gpu` stops with a message.

## Is there a GPU fastroute already?

No (October 2026). Neither fastroute, its forks, nor Freerouting has GPU code. Related work:

* [OrthoRoute](https://github.com/bbenchoff/OrthoRoute): a GPU (CUDA/Metal) PathFinder router
  on a Manhattan lattice for very large, regular backplanes; its author recommends other
  routers for conventional boards.
* [pcbroute](https://github.com/techcantv/pcbroute): a grid-based negotiated-congestion router
  with an optional CUDA back end for long searches (new, one commit).
* GPU global routers for chips (GAMER, FastGR, ...), unrelated to this fastroute despite the
  name of the chip router FastRoute.
* The Bradley-Cooley fork of fastroute tried global routing with corridors (negotiated
  congestion) and a PathFinder history on the CPU; neither helped (its `docs/IMPROVEMENTS.md`,
  "Not adopted either"). That ruled out the obvious GPU project: a fast global router would
  only speed up a stage that does not improve the routing.

## What makes sense on a GPU

Profile of a single-threaded run on the test board (samply, passes 1 to 3):

| | share of the run |
|---|---|
| maze search (`occupy_next_element`) | 87 % |
| of which: room completion (`complete_shape`, search tree queries) | 43 % |
| of which: via checks (`check_layer_with_any_matching_via`) | 18 % |
| of which: shove checks (`shove_trace_room`) | 9 % |
| insertion of the found connection | 7 % |
| pull-tight of the changed area | 3 % |

The hot path is a gridless A* over expansion rooms: a priority queue in Java order, rooms built
on demand from the shape search tree, each step a few microseconds of pointer-heavy geometry
that depends on the previous one. There is nothing to batch, and a round trip to the GPU costs
more than a step. Porting it would not be faster.

What the GPU can do is make the search smaller. Per-connection statistics
(`FASTROUTE_MAZE_STATS=file`, see below) showed why the search is large: the estimate is the
weighted distance to the bounding boxes of the destination items, and the connections found
cost 1.29 (pass 1), 1.47 (pass 2) and 1.62 (pass 3) times the estimate at their start. A* expands
every room whose cost plus estimate is below the cost of the connection, so even a 7 % gap on a
100 mm connection sweeps a band of tens of thousands of small rooms. An estimate that knows the
copper in between is what shrinks the search, and computing one per connection is a raster
shortest-path problem over a million cells: work a GPU does well and a CPU does not (432 ms for
one field with Dijkstra on the CPU, 7 to 12 ms on an RTX 3080).

Inflating the estimate instead (weighted A*) was tried first: ×1.1 to ×1.5 make pass 1 1.4 to 4
times faster but leave 8 to 30 % more connections unrouted after it, because the paths are worse.
The field keeps the paths and still shrinks the search.

Not done, and why:

* Global routing / corridors on the GPU: no gain on the CPU (see above), so none on the GPU.
* The gridless search, room completion, shove and insertion: sequential, see above.
* DRC (clearance checks of new items): a few percent of the run.

## The guide field

For a connection whose start and destination items span at least `--guide-min-span-mm` (10 mm),
the window is their bounding box enlarged by a quarter of the span (at least 3 mm), rastered
at `--guide-cell-um` (100 µm; larger windows get larger cells, at most 2^20 cells per layer).
On each routing layer, from the shapes of the autoroute search tree:

* items of other nets that cannot be ripped (pins, fixed wiring, keepouts, the outline) block
  the cells whose centres they cover;
* items that can be ripped charge a share (`--guide-soft-cost`, 0.5) of their rip-up cost when a
  move enters them;
* the destination items' cells are 0, the window border starts at the plain estimate (so that
  paths leaving the window are not overestimated).

Moves are the 8 neighbours and the 8 knight moves (a straight distance is overestimated by at
most 2.7 %), weighted by the layer's horizontal and vertical trace costs; a via costs the
cheapest via cost. The field is the cheapest cost from each cell to the destination. In the
search, the estimate of a point is the larger of the plain estimate and the field's value at
the point (lowered by half a cell's cost); where the field has no value the plain estimate is
used. Code: `crates/fr-engine/src/autoroute/guide/` (problem, raster rules, CPU reference
solver), `crates/fr-gpu/` (CUDA solver).

### The CUDA solver

1. Rasterize: one block per polygon tests the cell centres of its bounding box; soft items
   keep the cheapest per cell (`atomicMin` on cost and item).
2. Start values, boundary seeds, and a 16-bit mask of the allowed moves of each cell.
3. Relax: one block per 32 × 32 tile, a warp per layer and sweep direction. A tile loads its
   cells and a 2-cell halo into shared memory and sweeps left to right, right to left, bottom
   to top and top to bottom at the same time (each step pulls the 7 moves from the cells
   already swept, plus vias), at most 2 rounds per launch, then writes back and wakes the
   neighbours whose halo it changed. Launches repeat until nothing changes.

The relaxation only lowers values, so the order of updates does not matter; its fixpoint is the
Dijkstra field. `crates/fr-gpu/tests/matches_cpu.rs` compares the two on random problems (hard
and soft obstacles, 1 to 3 layers, boundary seeds, destinations smaller than a cell): equal up
to float rounding (relative 6e-6). The raster tests use `__fmul_rn`/`__fadd_rn` so that both
sides decide every cell the same way. Results are deterministic: the field depends only on the
board the connection is routed on.

Kernel tuning on a board-sized field (64 × 64 mm, 0.1 mm, 2 layers, 1800 obstacles):

| step | ms per field |
|---|---|
| tile relaxation, all moves per round, global memory | 72 |
| tile data in shared memory | 46 |
| directional sweeps (one warp per layer) | 35 |
| four sweep directions concurrently | 15 |
| wake only the neighbours whose halo changed, odd shared-memory pitch, unrolled moves | 12 |
| at most 2 rounds per tile and launch | 7 |

A field needs about 40 launches: a change crosses one tile per launch.

## Results

The 6-layer GameTank board (3 signal layers, 1053 connections to route, 3 router threads, no
fanout or optimizer; the board was held out of the PGO training), on an Intel Core i9-10900K
with an NVIDIA GeForce RTX 3080, one run each:

| | without `--gpu` | `--gpu` |
|---|---|---|
| first pass | 180 s (405 left) | 62 s (427 left) |
| down to 1 unrouted | 866 s | 301 s |
| whole run (70 passes, stopped by the stagnation rule) | 901 s | 509 s |
| unrouted | 1 | 1 |
| vias | 2001 | 2006 |
| track length | 33.40 m | 33.49 m |

## Statistics

`FASTROUTE_MAZE_STATS=FILE` writes one line per connection attempt: start and destination
items, polled and expanded maze elements, rooms, the estimate at the start and the cost found,
whether a guide field was used, its build, solve and download times, relaxation launches,
cells and obstacles, and the time of the search, location, insertion and pull-tight.
