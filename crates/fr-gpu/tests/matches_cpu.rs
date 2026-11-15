//! The CUDA field must be the CPU reference field (up to float rounding of the path sums):
//! random problems with hard and soft obstacles, several layers, boundary seeds and
//! destinations smaller than a cell. Needs a CUDA device.

use fr_engine::autoroute::guide::cpu::CpuFieldSolver;
use fr_engine::autoroute::guide::{BoundarySeed, FieldObstacle, FieldPolygon, FieldProblem, FieldSolver, ObstacleKind};
use fr_gpu::CudaFieldSolver;

/// xorshift64*: deterministic test data without a dependency.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A random convex polygon: a rotated rectangle or an octagon around a random centre.
fn polygon(random: &mut Random, layer: usize, extent: [f64; 2], size: f64) -> FieldPolygon {
    let centre = [random.uniform(0.0, extent[0]), random.uniform(0.0, extent[1])];
    let corners: Vec<[f64; 2]> = if random.below(2) == 0 {
        let (a, b) = (random.uniform(0.2, 1.0) * size, random.uniform(0.05, 0.4) * size);
        let angle = random.uniform(0.0, std::f64::consts::PI);
        let (c, s) = (angle.cos(), angle.sin());
        [[-a, -b], [a, -b], [a, b], [-a, b]].iter().map(|p| [centre[0] + c * p[0] - s * p[1], centre[1] + s * p[0] + c * p[1]]).collect()
    } else {
        let r = random.uniform(0.1, 0.5) * size;
        (0..8)
            .map(|i| {
                let angle = std::f64::consts::PI / 8.0 + i as f64 * std::f64::consts::PI / 4.0;
                [centre[0] + r * angle.cos(), centre[1] + r * angle.sin()]
            })
            .collect()
    };
    FieldPolygon::from_corners(&corners, layer).expect("random polygon is two-dimensional")
}

fn random_problem(seed: u64) -> FieldProblem {
    let mut random = Random(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let width = 20 + random.below(300);
    let height = 20 + random.below(300);
    let layers = 1 + random.below(3);
    let cell_size = random.uniform(500.0, 1500.0);
    let extent = [width as f64 * cell_size, height as f64 * cell_size];
    let size = 10.0 * cell_size;
    let mut obstacles = Vec::new();
    for item in 0..random.below(400) {
        let layer = random.below(layers);
        let kind = if random.below(3) == 0 {
            ObstacleKind::Hard
        } else {
            ObstacleKind::Soft { item: item as u32, cost: random.uniform(0.0, 5.0 * cell_size) as f32 }
        };
        obstacles.push(FieldObstacle { polygon: polygon(&mut random, layer, extent, size), kind });
    }
    let destinations = (0..1 + random.below(4))
        .map(|_| {
            let layer = random.below(layers);
            // some destinations are smaller than a cell: only their anchor cell is set
            let destination_size = if random.below(2) == 0 { 0.3 * cell_size } else { size };
            polygon(&mut random, layer, extent, destination_size)
        })
        .collect();
    let mut boundary = Vec::new();
    if random.below(2) == 0 {
        for (x, y) in fr_engine::autoroute::guide::raster::border_cells(width, height) {
            if random.below(3) == 0 {
                boundary.push(BoundarySeed { layer: random.below(layers), x, y, value: random.uniform(0.0, 200.0 * cell_size) as f32 });
            }
        }
    }
    let layer_costs = (0..layers).map(|_| [random.uniform(1.0, 2.0), random.uniform(1.0, 2.0)]).collect();
    FieldProblem { width, height, layers, cell_size, layer_costs, via_cost: random.uniform(0.0, 50.0 * cell_size), obstacles, destinations, boundary }
}

fn assert_same_field(seed: u64, problem: &FieldProblem, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len());
    let mut finite = 0;
    for (i, (&c, &g)) in cpu.iter().zip(gpu).enumerate() {
        if c.is_infinite() || g.is_infinite() {
            assert_eq!(c, g, "seed {seed}: cell {i} reachable on one solver only (cpu {c}, gpu {g}) in a {}x{}x{} field", problem.width, problem.height, problem.layers);
            continue;
        }
        finite += 1;
        let tolerance = 1e-4 * c.abs().max(problem.cell_size as f32);
        assert!((c - g).abs() <= tolerance, "seed {seed}: cell {i}: cpu {c}, gpu {g}");
    }
    assert!(finite > 0, "seed {seed}: no reachable cell");
}

#[test]
fn random_fields_match_the_cpu_reference() {
    let gpu = CudaFieldSolver::new().expect("a CUDA device for the GPU tests");
    for seed in 0..60 {
        let problem = random_problem(seed);
        let cpu_field = CpuFieldSolver.solve(&problem).unwrap();
        let gpu_field = gpu.solve(&problem).unwrap();
        assert_same_field(seed, &problem, &cpu_field, &gpu_field);
    }
}

#[test]
fn a_large_open_field_converges() {
    // a long way across many tiles, around a wall with one gap
    let gpu = CudaFieldSolver::new().expect("a CUDA device for the GPU tests");
    let cell_size = 1000.0;
    let (width, height) = (900, 700);
    let wall = |x0: f64, y0: f64, x1: f64, y1: f64| FieldObstacle {
        polygon: FieldPolygon::from_corners(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0).unwrap(),
        kind: ObstacleKind::Hard,
    };
    let problem = FieldProblem {
        width,
        height,
        layers: 2,
        cell_size,
        layer_costs: vec![[1.0, 2.0], [2.0, 1.0]],
        via_cost: 50_000.0,
        obstacles: vec![wall(400_000.0, 0.0, 410_000.0, 650_000.0), wall(600_000.0, 50_000.0, 610_000.0, 700_000.0)],
        destinations: vec![FieldPolygon::from_corners(&[[10_000.0, 10_000.0], [12_000.0, 10_000.0], [12_000.0, 12_000.0], [10_000.0, 12_000.0]], 0).unwrap()],
        boundary: Vec::new(),
    };
    let cpu_field = CpuFieldSolver.solve(&problem).unwrap();
    let gpu_field = gpu.solve(&problem).unwrap();
    assert_same_field(u64::MAX, &problem, &cpu_field, &gpu_field);
}
