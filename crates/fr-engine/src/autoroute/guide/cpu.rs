//! The reference field solver: Dijkstra from the destination cells over the raster of
//! [`super::raster`].

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::raster::{move_lengths, Raster, MOVES};
use super::{FieldProblem, FieldSolver};

/// Solves the field on the CPU (one thread per field).
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuFieldSolver;

#[derive(Clone, Copy, PartialEq)]
struct Cost(f64);

impl Eq for Cost {}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl FieldSolver for CpuFieldSolver {
    fn name(&self) -> &str {
        "cpu"
    }

    fn solve(&self, problem: &FieldProblem) -> Result<Vec<f32>, String> {
        let raster = Raster::new(problem);
        let lengths = move_lengths(problem);
        let mut values = raster.initial_values(problem);
        let mut heap: BinaryHeap<(Reverse<Cost>, usize)> =
            values.iter().enumerate().filter(|(_, value)| value.is_finite()).map(|(i, &value)| (Reverse(Cost(value)), i)).collect();
        let plane = raster.width * raster.height;
        while let Some((Reverse(Cost(value)), index)) = heap.pop() {
            if value > values[index] {
                continue;
            }
            let layer = index / plane;
            let x = index % raster.width;
            let y = (index % plane) / raster.width;
            // the cells that reach this one in one move
            for (mv, &(dx, dy)) in MOVES.iter().enumerate() {
                let (px, py) = (x as i64 - dx as i64, y as i64 - dy as i64);
                if !raster.free(layer, px, py) {
                    continue;
                }
                let (px, py) = (px as usize, py as usize);
                if let Some(cost) = raster.move_cost(&lengths, layer, px, py, mv) {
                    let predecessor = raster.index(layer, px, py);
                    let candidate = value + cost;
                    if candidate < values[predecessor] {
                        values[predecessor] = candidate;
                        heap.push((Reverse(Cost(candidate)), predecessor));
                    }
                }
            }
            for from_layer in 0..raster.layers {
                if from_layer == layer {
                    continue;
                }
                if let Some(cost) = raster.via_cost(problem.via_cost, from_layer, layer, x, y) {
                    let below = raster.index(from_layer, x, y);
                    let candidate = value + cost;
                    if candidate < values[below] {
                        values[below] = candidate;
                        heap.push((Reverse(Cost(candidate)), below));
                    }
                }
            }
        }
        Ok(values.into_iter().map(|value| value as f32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BoundarySeed, FieldObstacle, FieldPolygon, ObstacleKind};
    use super::*;

    fn square(layer: usize, x0: f64, y0: f64, x1: f64, y1: f64) -> FieldPolygon {
        FieldPolygon::from_corners(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], layer).unwrap()
    }

    fn problem(obstacles: Vec<FieldObstacle>) -> FieldProblem {
        FieldProblem {
            width: 20,
            height: 10,
            layers: 2,
            cell_size: 10.0,
            layer_costs: vec![[1.0, 1.0], [1.0, 2.0]],
            via_cost: 50.0,
            obstacles,
            destinations: vec![square(0, 0.0, 0.0, 10.0, 10.0)],
            boundary: Vec::new(),
        }
    }

    #[test]
    fn free_space_is_the_weighted_distance() {
        let values = CpuFieldSolver.solve(&problem(Vec::new())).unwrap();
        // cell (19, 0) on layer 0: 19 cells to the right of the destination cell
        assert!((values[19] - 190.0).abs() < 1e-3);
        // a knight move up and right: hypot(10, 20)
        assert!((values[2 * 20 + 1] - 500.0f32.sqrt()).abs() < 1e-3);
        // layer 1, same cell as the destination: one via
        assert!((values[200] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn a_wall_forces_a_detour_and_a_soft_wall_costs_its_charge() {
        let wall = square(0, 50.0, -1.0, 60.0, 81.0);
        let hard = problem(vec![FieldObstacle { polygon: wall.clone(), kind: ObstacleKind::Hard }]);
        let free = CpuFieldSolver.solve(&problem(Vec::new())).unwrap();
        let walled = CpuFieldSolver.solve(&hard).unwrap();
        let beyond = 9;
        assert!(walled[beyond] > free[beyond] + 10.0, "the wall makes the cells beyond it more expensive");
        let soft = problem(vec![FieldObstacle { polygon: wall, kind: ObstacleKind::Soft { item: 7, cost: 5.0 } }]);
        let charged = CpuFieldSolver.solve(&soft).unwrap();
        assert!((charged[beyond] - (free[beyond] + 5.0)).abs() < 1e-3, "crossing the soft wall costs its charge once");
    }

    #[test]
    fn boundary_seeds_cap_the_values() {
        let mut p = problem(Vec::new());
        p.boundary.push(BoundarySeed { layer: 0, x: 19, y: 9, value: 1.0 });
        let values = CpuFieldSolver.solve(&p).unwrap();
        assert!((values[9 * 20 + 19] - 1.0).abs() < 1e-6);
        assert!(values[9 * 20 + 18] <= 11.0 + 1e-3);
    }
}
