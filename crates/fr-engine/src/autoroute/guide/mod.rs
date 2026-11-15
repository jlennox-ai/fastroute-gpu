//! fastroute: a cost-to-go field that guides the maze search (`--guide`, `--gpu`).
//!
//! The maze search is an A* over expansion rooms. Its estimate of the remaining cost
//! ([`DestinationDistance`]) is the weighted distance to the bounding boxes of the destination
//! items and knows nothing of the copper in between. On a crowded board the connections found
//! cost 1.3 to 1.6 times the estimate at their start, and the search expands every room whose
//! cost plus estimate stays below that: on long connections a band of thousands of rooms.
//!
//! The field is a better estimate: the cost of reaching the destination on a raster of the
//! board around the connection, with
//! * the obstacles of other nets that cannot be ripped (pins, fixed wiring, keepouts, the
//!   outline) as walls,
//! * a charge for entering an item that can be ripped (a share of its rip-up cost),
//! * the trace costs of each layer and the via costs.
//!
//! It only orders the search: the estimate of an expansion is the larger of the two, and where
//! the field has no value (outside its window, inactive layers) the old estimate is used.
//!
//! The raster semantics shared by all solvers are in [`raster`]; [`cpu::CpuFieldSolver`] is the
//! reference (Dijkstra), the GPU solver lives in crate `fr-gpu`.

pub mod cpu;
pub mod raster;

use std::fmt;
use std::sync::Arc;

use fr_geom::{FloatPoint, IntBox, RegularTileShape, TileShape};

use crate::board::{ItemSet, RoutingBoard};
use crate::ids::LayerNo;
use crate::structure::Unit;

use super::control::AutorouteControl;
use super::destination::DestinationDistance;

/// Computes the field of a [`FieldProblem`].
pub trait FieldSolver: Send + Sync {
    /// A short name for logs ("cpu", "cuda").
    fn name(&self) -> &str;

    /// The cost to the destination of every cell, layer-major (`[layer][y][x]`);
    /// `f32::INFINITY` where the destination cannot be reached. An error is a failure of the
    /// device: the routing stops (see [`build_field`]).
    fn solve(&self, problem: &FieldProblem) -> Result<Vec<f32>, String>;
}

/// How the guide field is built (`--guide-*` options).
#[derive(Clone)]
pub struct GuideSettings {
    pub solver: Arc<dyn FieldSolver>,
    /// Edge length of a raster cell.
    pub cell_size_um: f64,
    /// Cells per layer at most: larger windows use larger cells.
    pub max_cells_per_layer: usize,
    /// The window is the bounding box of the start and destination items, enlarged by the
    /// larger of these two margins.
    pub margin_mm: f64,
    pub margin_fraction: f64,
    /// Connections whose start and destination items fit into a square of this size use the
    /// old estimate only (their search is cheap).
    pub min_span_mm: f64,
    /// The charge for entering a rippable item, as a share of its rip-up cost.
    pub soft_cost_factor: f64,
    /// The field value is scaled by this before it is compared with the old estimate.
    pub estimate_factor: f64,
}

impl fmt::Debug for GuideSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuideSettings")
            .field("solver", &self.solver.name())
            .field("cell_size_um", &self.cell_size_um)
            .field("max_cells_per_layer", &self.max_cells_per_layer)
            .field("margin_mm", &self.margin_mm)
            .field("margin_fraction", &self.margin_fraction)
            .field("min_span_mm", &self.min_span_mm)
            .field("soft_cost_factor", &self.soft_cost_factor)
            .field("estimate_factor", &self.estimate_factor)
            .finish()
    }
}

impl GuideSettings {
    /// The defaults with the given solver.
    pub fn new(solver: Arc<dyn FieldSolver>) -> Self {
        GuideSettings {
            solver,
            cell_size_um: 100.0,
            max_cells_per_layer: 1 << 20,
            margin_mm: 3.0,
            margin_fraction: 0.25,
            min_span_mm: 10.0,
            soft_cost_factor: 0.5,
            estimate_factor: 1.0,
        }
    }
}

/// A convex polygon of a [`FieldProblem`] as half-planes `nx * x + ny * y <= c` in window
/// coordinates (board units relative to the window's lower left corner).
#[derive(Clone, Debug, PartialEq)]
pub struct FieldPolygon {
    pub layer: usize,
    pub half_planes: Vec<[f64; 3]>,
    /// Lower left and upper right corner of the bounding box.
    pub min: [f64; 2],
    pub max: [f64; 2],
}

impl FieldPolygon {
    /// The polygon of `shape` (board coordinates) shifted by `-origin`; `None` if the shape is
    /// not two-dimensional.
    pub fn from_tile_shape(shape: &TileShape, layer: usize, origin: [f64; 2]) -> Option<Self> {
        if shape.dimension() < 2 {
            return None;
        }
        let corners: Vec<[f64; 2]> = (0..shape.border_line_count())
            .map(|i| {
                let corner = shape.corner_approx(i);
                [corner.x - origin[0], corner.y - origin[1]]
            })
            .collect();
        Self::from_corners(&corners, layer)
    }

    /// The convex polygon with these corners (in order, either orientation).
    pub fn from_corners(corners: &[[f64; 2]], layer: usize) -> Option<Self> {
        let n = corners.len();
        if n < 3 {
            return None;
        }
        let mut twice_area = 0.0;
        for i in 0..n {
            let start = corners[i];
            let end = corners[(i + 1) % n];
            twice_area += start[0] * end[1] - end[0] * start[1];
        }
        if twice_area == 0.0 {
            return None;
        }
        let orientation = twice_area.signum();
        let mut half_planes = Vec::with_capacity(n);
        let mut min = [f64::MAX, f64::MAX];
        let mut max = [f64::MIN, f64::MIN];
        for i in 0..n {
            let start = corners[i];
            let end = corners[(i + 1) % n];
            min = [min[0].min(start[0]), min[1].min(start[1])];
            max = [max[0].max(start[0]), max[1].max(start[1])];
            let dx = end[0] - start[0];
            let dy = end[1] - start[1];
            let length = (dx * dx + dy * dy).sqrt();
            if length == 0.0 {
                continue;
            }
            // outward normal: right of the edge for a counterclockwise polygon
            let nx = orientation * dy / length;
            let ny = -orientation * dx / length;
            half_planes.push([nx, ny, nx * start[0] + ny * start[1]]);
        }
        Some(FieldPolygon { layer, half_planes, min, max })
    }

    /// True if the point is inside or on the border.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        self.half_planes.iter().all(|&[nx, ny, c]| nx * x + ny * y <= c)
    }
}

/// What an obstacle of a [`FieldProblem`] is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObstacleKind {
    /// Cannot be passed (cannot be ripped).
    Hard,
    /// Can be ripped: entering it costs `cost` once. `item` tells items apart, so that moving on
    /// inside the same item is free.
    Soft { item: u32, cost: f32 },
}

/// An obstacle of a [`FieldProblem`].
#[derive(Clone, Debug, PartialEq)]
pub struct FieldObstacle {
    pub polygon: FieldPolygon,
    pub kind: ObstacleKind,
}

/// A start value on the border of the window: the old estimate there, so that paths leaving
/// the window are not overestimated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundarySeed {
    pub layer: usize,
    pub x: usize,
    pub y: usize,
    pub value: f32,
}

/// The input of a [`FieldSolver`]: a raster window with obstacles, destinations and costs.
/// Cell `(x, y)` covers `[x * cell_size, (x + 1) * cell_size)` (likewise for y) in window
/// coordinates; the raster tests are made at the cell centres (see [`raster`]).
#[derive(Clone, Debug, PartialEq)]
pub struct FieldProblem {
    pub width: usize,
    pub height: usize,
    pub layers: usize,
    /// Edge length of a cell (board units).
    pub cell_size: f64,
    /// Horizontal and vertical trace cost of each layer.
    pub layer_costs: Vec<[f64; 2]>,
    /// The cost of a layer change.
    pub via_cost: f64,
    pub obstacles: Vec<FieldObstacle>,
    pub destinations: Vec<FieldPolygon>,
    pub boundary: Vec<BoundarySeed>,
}

impl FieldProblem {
    pub fn cell_count(&self) -> usize {
        self.width * self.height * self.layers
    }
}

/// The solved field of a connection, looked up by the maze search.
#[derive(Clone, Debug)]
pub struct GuideField {
    origin: [f64; 2],
    cell_size: f64,
    width: usize,
    height: usize,
    /// The field layer of each board layer.
    field_layer: Vec<Option<usize>>,
    values: Vec<f32>,
    /// The cost of a point's distance to its cell centre is at most this times the distance.
    max_weight: f64,
    estimate_factor: f64,
}

impl GuideField {
    /// The field's estimate of the cost from `point` on `layer` to the destination, if it has
    /// one there. Lowered by the largest cost from the point to the cell centre, so that it
    /// stays below the cost of the cell's path.
    pub fn estimate(&self, point: &FloatPoint, layer: LayerNo) -> Option<f64> {
        let field_layer = (*self.field_layer.get(layer as usize)?)?;
        let fx = (point.x - self.origin[0]) / self.cell_size;
        let fy = (point.y - self.origin[1]) / self.cell_size;
        if !(fx >= 0.0 && fy >= 0.0 && fx < self.width as f64 && fy < self.height as f64) {
            return None;
        }
        let (cx, cy) = (fx as usize, fy as usize);
        let plane = &self.values[field_layer * self.width * self.height..(field_layer + 1) * self.width * self.height];
        let own = plane[cy * self.width + cx] as f64;
        let half_diagonal = std::f64::consts::FRAC_1_SQRT_2 * self.cell_size * self.max_weight;
        let value = if own.is_finite() {
            own - half_diagonal
        } else {
            // the point is in a cell whose centre is covered: the best neighbour, lowered by the
            // distance to its centre
            let mut best = f64::INFINITY;
            for ny in cy.saturating_sub(1)..(cy + 2).min(self.height) {
                for nx in cx.saturating_sub(1)..(cx + 2).min(self.width) {
                    best = best.min(plane[ny * self.width + nx] as f64);
                }
            }
            if !best.is_finite() {
                return None;
            }
            best - 3.0 * half_diagonal
        };
        Some(value.max(0.0) * self.estimate_factor)
    }
}

/// The guide field of a connection from `start_items` to `destination_items` (`base`: the
/// destination distance with the destination items joined), or `None` if the connection is
/// short or the window has no destination.
pub fn build_field(
    board: &RoutingBoard,
    tree: usize,
    ctrl: &AutorouteControl,
    start_items: &ItemSet,
    destination_items: &ItemSet,
    base: &DestinationDistance,
    settings: &GuideSettings,
) -> Option<GuideField> {
    let (problem, origin, field_layer) = build_problem(board, tree, ctrl, start_items, destination_items, base, settings)?;
    // A failing solver (a lost GPU) is not a routing failure: the search code catches panics as
    // Java exceptions and would carry on without the guide, so stop here, loudly. The session
    // file already holds the best board so far.
    let solve_start = std::time::Instant::now();
    let values = settings.solver.solve(&problem).unwrap_or_else(|e| {
        log::error!("guide field: the {} solver failed: {e}", settings.solver.name());
        eprintln!("fastroute: guide field: the {} solver failed: {e}", settings.solver.name());
        std::process::abort();
    });
    assert_eq!(values.len(), problem.cell_count(), "guide solver '{}' returned a field of the wrong size", settings.solver.name());
    super::stats::update(|connection| {
        connection.field_solve_time += solve_start.elapsed();
        connection.field_cells += problem.cell_count();
        connection.field_obstacles += problem.obstacles.len();
    });
    let max_weight = problem.layer_costs.iter().flat_map(|costs| costs.iter().copied()).fold(0.0, f64::max);
    Some(GuideField {
        origin,
        cell_size: problem.cell_size,
        width: problem.width,
        height: problem.height,
        field_layer,
        values,
        max_weight,
        estimate_factor: settings.estimate_factor,
    })
}

/// The [`FieldProblem`] of a connection, its window origin (board coordinates) and the field
/// layer of each board layer.
pub fn build_problem(
    board: &RoutingBoard,
    tree: usize,
    ctrl: &AutorouteControl,
    start_items: &ItemSet,
    destination_items: &ItemSet,
    base: &DestinationDistance,
    settings: &GuideSettings,
) -> Option<(FieldProblem, [f64; 2], Vec<Option<usize>>)> {
    let units_per_mm = board.communication.get_resolution(Unit::Mm);
    let mut field_layer = vec![None; ctrl.layer_count as usize];
    let mut board_layers = Vec::new();
    for layer in 0..ctrl.layer_count {
        if ctrl.layer_active[layer as usize] {
            field_layer[layer as usize] = Some(board_layers.len());
            board_layers.push(layer);
        }
    }
    if board_layers.is_empty() {
        return None;
    }
    let mut items_box: Option<IntBox> = None;
    for key in start_items.iter().chain(destination_items.iter()) {
        for i in 0..board.tree_shape_count(tree, key) {
            if let Some(shape) = board.tree_shape(tree, key, i) {
                let shape_box = shape.bounding_box();
                items_box = Some(items_box.map_or(shape_box.clone(), |b| b.union_int_box(&shape_box)));
            }
        }
    }
    let items_box = items_box?;
    let span = ((items_box.ur.x - items_box.ll.x).max(items_box.ur.y - items_box.ll.y)) as f64;
    if span < settings.min_span_mm * units_per_mm {
        return None;
    }
    let margin = (settings.margin_mm * units_per_mm).max(settings.margin_fraction * span);
    let window = items_box.offset(margin).intersection_int_box(&board.bounding_box);
    if window.is_empty() {
        return None;
    }
    let window_width = (window.ur.x - window.ll.x) as f64;
    let window_height = (window.ur.y - window.ll.y) as f64;
    let cell_size = (settings.cell_size_um * units_per_mm / 1000.0).max((window_width * window_height / settings.max_cells_per_layer as f64).sqrt());
    let width = ((window_width / cell_size).ceil() as usize).max(1);
    let height = ((window_height / cell_size).ceil() as usize).max(1);
    let origin = [window.ll.x as f64, window.ll.y as f64];

    let window_shape = RegularTileShape::IntBox(window.clone());
    let net = ctrl.net_number;
    let mut obstacles = Vec::new();
    for (index, &layer) in board_layers.iter().enumerate() {
        let soft_half_width = ctrl.trace_half_width[layer as usize] as f64;
        for leaf in board.tree_overlaps_on_layer(tree, &window_shape, layer) {
            let (object, shape_index) = board.search_tree(tree).min_area_tree().leaf_payload(leaf);
            let Some(key) = object.item() else { continue };
            if board.object_shape_layer(tree, object, shape_index) != layer {
                continue;
            }
            let item = board.item(key);
            if !board.is_trace_obstacle(item, net) {
                continue;
            }
            let Some(shape) = board.tree_shape(tree, key, shape_index) else { continue };
            let Some(polygon) = FieldPolygon::from_tile_shape(&shape, index, origin) else { continue };
            let kind = if ctrl.ripup_allowed && item.is_routable() {
                let half_width = item.as_trace().map_or(soft_half_width, |t| t.half_width() as f64);
                let cost = settings.soft_cost_factor * ctrl.ripup_costs as f64 * half_width;
                ObstacleKind::Soft { item: key.index, cost: cost as f32 }
            } else {
                ObstacleKind::Hard
            };
            obstacles.push(FieldObstacle { polygon, kind });
        }
    }
    let mut destinations = Vec::new();
    for key in destination_items.iter() {
        for i in 0..board.tree_shape_count(tree, key) {
            let Some(index) = field_layer[board.shape_layer(key, i) as usize] else { continue };
            let Some(shape) = board.tree_shape(tree, key, i) else { continue };
            if let Some(polygon) = FieldPolygon::from_tile_shape(&shape, index, origin) {
                destinations.push(polygon);
            }
        }
    }
    if destinations.is_empty() {
        return None;
    }
    let mut boundary = Vec::new();
    for (index, &layer) in board_layers.iter().enumerate() {
        for (x, y) in raster::border_cells(width, height) {
            let centre = FloatPoint {
                x: origin[0] + (x as f64 + 0.5) * cell_size,
                y: origin[1] + (y as f64 + 0.5) * cell_size,
            };
            let value = base.calculate_point(&centre, layer);
            if value < i32::MAX as f64 {
                boundary.push(BoundarySeed { layer: index, x, y, value: value as f32 });
            }
        }
    }
    let layer_costs = board_layers
        .iter()
        .map(|&l| [ctrl.trace_costs[l as usize].horizontal, ctrl.trace_costs[l as usize].vertical])
        .collect();
    let problem = FieldProblem {
        width,
        height,
        layers: board_layers.len(),
        cell_size,
        layer_costs,
        via_cost: ctrl.min_normal_via_cost,
        obstacles,
        destinations,
        boundary,
    };
    Some((problem, origin, field_layer))
}
