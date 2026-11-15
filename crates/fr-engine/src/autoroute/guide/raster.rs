//! The raster semantics shared by all field solvers: which cells an obstacle or a destination
//! covers, which moves between cells are allowed and what they cost. A GPU solver has to give
//! the field of [`super::cpu::CpuFieldSolver`], which is built on these functions (up to float
//! rounding of the path sums).
//!
//! * A polygon covers a cell if the cell centre satisfies all its half-planes. The test is made
//!   in `f32` (`nx * x + ny * y <= c`, two products and a sum, no fused multiply-add), so that it
//!   gives the same answer on every device.
//! * A cell is blocked on a layer if a hard obstacle covers it and no destination does. Each
//!   destination polygon also covers the cell of its bounding box centre (its anchor), so that
//!   small destinations are not lost.
//! * Of the soft obstacles covering a cell, the cell keeps the cheapest (ties: the lower item).
//! * Moves: the 8 neighbours and the 8 knight moves (straight distances are overestimated by at
//!   most 2.7 %). A diagonal or knight move passes two cells ([`MOVE_PASSES`]): a diagonal move
//!   needs one of them free, a knight move both. Cost: the weighted length
//!   (`hypot(ch * dx, cv * dy) * cell_size`) plus the charge of the cells entered, where a cell's
//!   charge is its soft cost if it holds a soft item other than the source cell's: the target's
//!   charge, or more if the passed cells charge more (diagonal: the cheaper of the two, as the
//!   move may slip through either; knight: the dearer one), so that no move jumps a thin soft
//!   item for free.
//! * A via changes the layer in the same cell: the via cost plus the soft cost of the target
//!   cell. Both cells have to be free.

use super::{FieldPolygon, FieldProblem, ObstacleKind};

/// No soft item in the cell.
pub const NO_ITEM: u32 = u32::MAX;

/// The moves from a cell: 8 neighbours, then 8 knight moves.
pub const MOVES: [(i32, i32); 16] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
];

/// The two cells a diagonal or knight move passes, relative to the source (unused for the
/// first four, orthogonal moves).
pub const MOVE_PASSES: [[(i32, i32); 2]; 16] = [
    [(0, 0), (0, 0)],
    [(0, 0), (0, 0)],
    [(0, 0), (0, 0)],
    [(0, 0), (0, 0)],
    [(1, 0), (0, 1)],
    [(1, 0), (0, -1)],
    [(-1, 0), (0, 1)],
    [(-1, 0), (0, -1)],
    [(0, 1), (1, 1)],
    [(0, -1), (1, -1)],
    [(0, 1), (-1, 1)],
    [(0, -1), (-1, -1)],
    [(1, 0), (1, 1)],
    [(1, 0), (1, -1)],
    [(-1, 0), (-1, 1)],
    [(-1, 0), (-1, -1)],
];

/// True for the diagonal moves, false for the orthogonal and knight moves.
fn is_diagonal(mv: usize) -> bool {
    (4..8).contains(&mv)
}

/// A polygon's half-planes and bounding box in `f32`, as every solver tests them.
#[derive(Clone, Debug)]
pub struct RasterPolygon {
    pub layer: usize,
    pub half_planes: Vec<[f32; 3]>,
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl RasterPolygon {
    pub fn new(polygon: &FieldPolygon) -> Self {
        RasterPolygon {
            layer: polygon.layer,
            half_planes: polygon.half_planes.iter().map(|&[nx, ny, c]| [nx as f32, ny as f32, c as f32]).collect(),
            min: [polygon.min[0] as f32, polygon.min[1] as f32],
            max: [polygon.max[0] as f32, polygon.max[1] as f32],
        }
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        self.half_planes.iter().all(|&[nx, ny, c]| nx * x + ny * y <= c)
    }
}

/// The centre of cell `index` along one axis.
pub fn cell_centre(index: usize, cell_size: f32) -> f32 {
    (index as f32 + 0.5) * cell_size
}

/// The range of cell indices whose centres may lie in `[min, max]` (one cell extra on each
/// side; the exact test decides).
pub fn candidate_range(min: f32, max: f32, cell_size: f32, count: usize) -> std::ops::Range<usize> {
    let first = ((min / cell_size).floor() as i64 - 1).clamp(0, count as i64) as usize;
    let last = ((max / cell_size).floor() as i64 + 2).clamp(0, count as i64) as usize;
    first..last
}

/// Calls `visit(x, y)` for every cell whose centre the polygon covers.
pub fn covered_cells(polygon: &RasterPolygon, cell_size: f32, width: usize, height: usize, mut visit: impl FnMut(usize, usize)) {
    for y in candidate_range(polygon.min[1], polygon.max[1], cell_size, height) {
        let cy = cell_centre(y, cell_size);
        for x in candidate_range(polygon.min[0], polygon.max[0], cell_size, width) {
            if polygon.contains(cell_centre(x, cell_size), cy) {
                visit(x, y);
            }
        }
    }
}

/// The cell of the bounding box centre of a destination polygon, if it is in the window.
pub fn anchor_cell(polygon: &RasterPolygon, cell_size: f32, width: usize, height: usize) -> Option<(usize, usize)> {
    let x = ((polygon.min[0] + polygon.max[0]) * 0.5 / cell_size).floor();
    let y = ((polygon.min[1] + polygon.max[1]) * 0.5 / cell_size).floor();
    if x < 0.0 || y < 0.0 || x >= width as f32 || y >= height as f32 {
        return None;
    }
    Some((x as usize, y as usize))
}

/// The cells on the border of a `width` x `height` raster, each once: the bottom and top rows,
/// then the left and right columns without their ends.
pub fn border_cells(width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut cells = Vec::with_capacity(2 * (width + height));
    for x in 0..width {
        cells.push((x, 0));
        if height > 1 {
            cells.push((x, height - 1));
        }
    }
    for y in 1..height.saturating_sub(1) {
        cells.push((0, y));
        if width > 1 {
            cells.push((width - 1, y));
        }
    }
    cells
}

/// The weighted length of each move on each layer (`[layer][move]`).
pub fn move_lengths(problem: &FieldProblem) -> Vec<[f64; 16]> {
    problem
        .layer_costs
        .iter()
        .map(|costs| {
            let mut lengths = [0.0; 16];
            for (i, &(dx, dy)) in MOVES.iter().enumerate() {
                lengths[i] = (costs[0] * dx as f64).hypot(costs[1] * dy as f64) * problem.cell_size;
            }
            lengths
        })
        .collect()
}

/// The rasterized problem: per layer and cell whether it is blocked or a destination, and its
/// soft item.
#[derive(Clone, Debug)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
    pub layers: usize,
    pub blocked: Vec<bool>,
    pub destination: Vec<bool>,
    pub soft_cost: Vec<f32>,
    pub soft_item: Vec<u32>,
}

impl Raster {
    pub fn new(problem: &FieldProblem) -> Self {
        let (width, height, layers) = (problem.width, problem.height, problem.layers);
        let cells = problem.cell_count();
        let cell_size = problem.cell_size as f32;
        let mut hard = vec![false; cells];
        let mut destination = vec![false; cells];
        let mut soft_cost = vec![0.0f32; cells];
        let mut soft_item = vec![NO_ITEM; cells];
        for obstacle in &problem.obstacles {
            let polygon = RasterPolygon::new(&obstacle.polygon);
            let plane = polygon.layer * width * height;
            covered_cells(&polygon, cell_size, width, height, |x, y| {
                let i = plane + y * width + x;
                match obstacle.kind {
                    ObstacleKind::Hard => hard[i] = true,
                    ObstacleKind::Soft { item, cost } => {
                        if soft_item[i] == NO_ITEM || (cost.to_bits(), item) < (soft_cost[i].to_bits(), soft_item[i]) {
                            soft_cost[i] = cost;
                            soft_item[i] = item;
                        }
                    }
                }
            });
        }
        for polygon in &problem.destinations {
            let polygon = RasterPolygon::new(polygon);
            let plane = polygon.layer * width * height;
            covered_cells(&polygon, cell_size, width, height, |x, y| destination[plane + y * width + x] = true);
            if let Some((x, y)) = anchor_cell(&polygon, cell_size, width, height) {
                destination[plane + y * width + x] = true;
            }
        }
        let blocked = hard.iter().zip(&destination).map(|(&h, &d)| h && !d).collect();
        Raster { width, height, layers, blocked, destination, soft_cost, soft_item }
    }

    pub fn index(&self, layer: usize, x: usize, y: usize) -> usize {
        (layer * self.height + y) * self.width + x
    }

    /// True if the cell is inside the raster and not blocked.
    pub fn free(&self, layer: usize, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height && !self.blocked[self.index(layer, x as usize, y as usize)]
    }

    /// The charge of entering cell `to` from a cell holding soft item `from_item`.
    fn charge(&self, from_item: u32, to: usize) -> f32 {
        let item = self.soft_item[to];
        if item != NO_ITEM && item != from_item {
            self.soft_cost[to]
        } else {
            0.0
        }
    }

    /// True if move `mv` from the free cell `(x, y)` on `layer` is allowed.
    pub fn move_allowed(&self, layer: usize, x: usize, y: usize, mv: usize) -> bool {
        let (x, y) = (x as i64, y as i64);
        let (dx, dy) = MOVES[mv];
        if !self.free(layer, x, y) || !self.free(layer, x + dx as i64, y + dy as i64) {
            return false;
        }
        if mv < 4 {
            return true;
        }
        let [(ax, ay), (bx, by)] = MOVE_PASSES[mv];
        let free_a = self.free(layer, x + ax as i64, y + ay as i64);
        let free_b = self.free(layer, x + bx as i64, y + by as i64);
        if is_diagonal(mv) {
            free_a || free_b
        } else {
            free_a && free_b
        }
    }

    /// The cost of move `mv` from cell `(x, y)` on `layer`, or `None` if the move is not allowed.
    pub fn move_cost(&self, lengths: &[[f64; 16]], layer: usize, x: usize, y: usize, mv: usize) -> Option<f64> {
        if !self.move_allowed(layer, x, y, mv) {
            return None;
        }
        let (dx, dy) = MOVES[mv];
        let from_item = self.soft_item[self.index(layer, x, y)];
        let at = |ox: i32, oy: i32| self.index(layer, (x as i64 + ox as i64) as usize, (y as i64 + oy as i64) as usize);
        let mut charge = self.charge(from_item, at(dx, dy));
        if mv >= 4 {
            let [(ax, ay), (bx, by)] = MOVE_PASSES[mv];
            let (pass_a, pass_b) = (self.charge(from_item, at(ax, ay)), self.charge(from_item, at(bx, by)));
            charge = charge.max(if is_diagonal(mv) { pass_a.min(pass_b) } else { pass_a.max(pass_b) });
        }
        Some(lengths[layer][mv] + charge as f64)
    }

    /// The cost of a via from `from_layer` to `to_layer` in cell `(x, y)`, or `None` if one of the
    /// two cells is blocked.
    pub fn via_cost(&self, via_cost: f64, from_layer: usize, to_layer: usize, x: usize, y: usize) -> Option<f64> {
        if !self.free(from_layer, x as i64, y as i64) || !self.free(to_layer, x as i64, y as i64) {
            return None;
        }
        let to = self.index(to_layer, x, y);
        let charge = if self.soft_item[to] != NO_ITEM { self.soft_cost[to] as f64 } else { 0.0 };
        Some(via_cost + charge)
    }

    /// The start values: 0 in the destination cells, the boundary seeds in free cells, infinity
    /// elsewhere.
    pub fn initial_values(&self, problem: &FieldProblem) -> Vec<f64> {
        let mut values: Vec<f64> = self.destination.iter().map(|&d| if d { 0.0 } else { f64::INFINITY }).collect();
        for seed in &problem.boundary {
            let i = self.index(seed.layer, seed.x, seed.y);
            if !self.blocked[i] {
                values[i] = values[i].min(seed.value as f64);
            }
        }
        values
    }
}
