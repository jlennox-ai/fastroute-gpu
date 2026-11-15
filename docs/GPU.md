# GPU acceleration (`--gpu`)

With `--gpu`, fastroute uses an NVIDIA GPU to help the router decide where to search. The routing
itself still runs on the CPU and follows the same rules; the GPU makes each search look at far
less of the board. Setup: see [What you need for `--gpu`](../README.md#what-you-need-for---gpu).

## Where the time goes

A single-threaded routing run on a 6-layer board:

| | share of the run |
|---|---|
| finding a path for a connection (maze search) | 87 % |
| - of which: building the free areas the path moves through | 43 % |
| - of which: checking where vias fit | 18 % |
| - of which: checking whether traces can be pushed aside | 9 % |
| inserting the found path | 7 % |
| tidying the changed traces | 3 % |

The maze search is an A* search: it grows outward from the start, always continuing from the
most promising spot, where "promising" is the cost so far plus an estimate of the cost still to
go. Each step is a few microseconds of geometry that depends on the step before it. That cannot
be split into thousands of independent pieces, which is what a GPU needs, so moving the search
itself to the GPU would make it slower.

## What the GPU does instead

The better the estimate, the less the search explores. The router's estimate is the straight distance to
the destination, which ignores everything in the way. On real boards the paths found cost 1.3 to
1.6 times that estimate, so the search sweeps a wide band of the board around every connection.

With `--gpu`, fastroute computes a better estimate for each connection before searching: a map
of the cheapest cost from every point near the connection to its destination, going around the
pins, keepouts and traces in the way. Computing that map is the same simple calculation repeated
over a million grid cells, which is exactly what a GPU is good at: 7 to 12 ms per connection on an
RTX 3080, against about 430 ms on the CPU.

The map only steers the search; the search still checks every step against the real board, so
the result obeys the same design rules. Simply making the plain estimate larger also shrinks the
search, but then the router settles for worse paths and leaves 8 to 30 % more connections
unrouted after the first pass. The map shrinks the search without that cost.

## How the map is built

Only connections spanning at least 10 mm get a map (`--guide-min-span-mm`); short ones are cheap
anyway. The map covers the box around the connection's start and destination plus a margin of a
quarter of its size, at a resolution of 0.1 mm (`--guide-cell-um`; coarser for large boxes), on
every routing layer:

* Things the router may never move (pins of other nets, fixed traces, keepouts, the board
  outline) block the cells they cover.
* Traces the router may rip up and reroute make a cell more expensive
  (`--guide-soft-cost`, half their rip-up cost).
* The destination costs 0. At the edge of the box the map falls back to the plain estimate, so
  paths that leave the box are not judged too expensive.

From each cell a path can step to its 8 neighbours or by a knight's move (which keeps diagonal
distances within 3 % of the true length), or change layer through a via. The cost of each step
follows the router's own trace and via costs.

In the search, the estimate for a point is the larger of the plain estimate and the map's value
there. The code is in `crates/fr-engine/src/autoroute/guide/` (the map's rules and a CPU version,
used by `--guide=cpu` and the tests) and `crates/fr-gpu/` (the GPU version).

## On the GPU

The board is split into tiles of 32 × 32 cells. Each tile sweeps its cells in all four directions
at once, lowering every cell's cost wherever a neighbour offers a cheaper way to the destination,
and wakes the neighbouring tiles it changed. This repeats until nothing changes, about 40 rounds
for a board-sized map. Since costs only ever go down, the order of updates does not matter and
the result is exact: the tests compare it with the CPU version on random boards and it matches to
within float rounding. The map depends only on the board, so runs are repeatable.

## Results

The 6-layer board above (1053 connections to route, 3 router threads), Intel Core i9-10900K with
an NVIDIA GeForce RTX 3080, one run each:

| | without `--gpu` | `--gpu` |
|---|---|---|
| first pass | 180 s (405 left) | 62 s (427 left) |
| down to 1 unrouted | 866 s | 301 s |
| whole run | 901 s | 509 s |
| unrouted | 1 | 1 |
| vias | 2001 | 2006 |
| track length | 33.40 m | 33.49 m |

## Statistics

`FASTROUTE_MAZE_STATS=FILE` writes one line per connection the router tries: how much the search
explored, the estimate at the start against the cost found, whether a map was used and how long
it took to build, and how long the search, insertion and tidying took.
