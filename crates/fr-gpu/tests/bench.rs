//! Timing of the CUDA field on a board-sized problem (`cargo test --release -p fr-gpu --test
//! bench -- --ignored --nocapture`): a 64 x 64 mm window at 0.1 mm on two layers, crossed by
//! traces (soft) and dotted with pads (hard), the destination in a corner.

use std::time::Instant;

use fr_engine::autoroute::guide::cpu::CpuFieldSolver;
use fr_engine::autoroute::guide::{FieldObstacle, FieldPolygon, FieldProblem, FieldSolver, ObstacleKind};
use fr_gpu::CudaFieldSolver;

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
}

fn rectangle(layer: usize, centre: [f64; 2], half_length: f64, half_width: f64, angle: f64) -> FieldPolygon {
    let (c, s) = (angle.cos(), angle.sin());
    let corners: Vec<[f64; 2]> = [[-half_length, -half_width], [half_length, -half_width], [half_length, half_width], [-half_length, half_width]]
        .iter()
        .map(|p| [centre[0] + c * p[0] - s * p[1], centre[1] + s * p[0] + c * p[1]])
        .collect();
    FieldPolygon::from_corners(&corners, layer).unwrap()
}

fn board_problem(seed: u64) -> FieldProblem {
    let mut random = Random(seed | 1);
    let cell_size = 1000.0;
    let (width, height) = (640, 640);
    let extent = width as f64 * cell_size;
    let mut obstacles = Vec::new();
    for item in 0..1200u32 {
        let layer = (item % 2) as usize;
        let centre = [random.uniform(0.0, extent), random.uniform(0.0, extent)];
        // traces prefer their layer's direction, at 0 / 45 / 90 degrees
        let angle = [0.0, std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_2][(random.next() % 3) as usize] + layer as f64 * std::f64::consts::FRAC_PI_2;
        let polygon = rectangle(layer, centre, random.uniform(5_000.0, 40_000.0), 1_900.0, angle);
        obstacles.push(FieldObstacle { polygon, kind: ObstacleKind::Soft { item, cost: 30_000.0 } });
    }
    for _ in 0..600 {
        let centre = [random.uniform(0.0, extent), random.uniform(0.0, extent)];
        for layer in 0..2 {
            obstacles.push(FieldObstacle { polygon: rectangle(layer, centre, 900.0, 900.0, 0.0), kind: ObstacleKind::Hard });
        }
    }
    FieldProblem {
        width,
        height,
        layers: 2,
        cell_size,
        layer_costs: vec![[1.0, 1.5], [1.5, 1.0]],
        via_cost: 50_000.0,
        obstacles,
        destinations: vec![rectangle(0, [30_000.0, 30_000.0], 800.0, 800.0, 0.0)],
        boundary: Vec::new(),
    }
}

#[test]
#[ignore]
fn fixed_cost_of_a_field() {
    let gpu = CudaFieldSolver::new().expect("a CUDA device");
    let mut problem = board_problem(7);
    problem.width = 32;
    problem.height = 32;
    problem.obstacles.clear();
    problem.destinations = vec![rectangle(0, [3_000.0, 3_000.0], 800.0, 800.0, 0.0)];
    gpu.solve(&problem).unwrap();
    let runs = 200;
    let start = Instant::now();
    for _ in 0..runs {
        gpu.solve(&problem).unwrap();
    }
    println!("one-tile field: {:.3} ms", start.elapsed().as_secs_f64() * 1000.0 / runs as f64);
}

#[test]
#[ignore]
fn board_sized_field() {
    let gpu = CudaFieldSolver::new().expect("a CUDA device");
    let (registers, blocks) = gpu.relax_occupancy(2).unwrap();
    println!("relaxation kernel: {registers} registers per thread, {blocks} blocks per multiprocessor (2 layers)");
    let problem = board_problem(7);
    let cpu_start = Instant::now();
    let cpu_field = CpuFieldSolver.solve(&problem).unwrap();
    println!("cpu: {:.1} ms", cpu_start.elapsed().as_secs_f64() * 1000.0);
    // warm up (allocations), then time
    let gpu_field = gpu.solve(&problem).unwrap();
    let worst = cpu_field.iter().zip(&gpu_field).filter(|(c, _)| c.is_finite()).map(|(c, g)| (c - g).abs() / c.max(1000.0)).fold(0.0f32, f32::max);
    println!("largest relative difference to the cpu field: {worst:e}");
    let runs = 20;
    let start = Instant::now();
    for _ in 0..runs {
        gpu.solve(&problem).unwrap();
    }
    println!("gpu: {:.2} ms per field", start.elapsed().as_secs_f64() * 1000.0 / runs as f64);
    gpu.set_profiling(true);
    for _ in 0..runs {
        gpu.solve(&problem).unwrap();
    }
    let p = gpu.profile().unwrap();
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / p.fields as f64;
    println!(
        "profiled, per field: upload {:.2} ms, raster {:.2} ms, relax {:.2} ms, download {:.2} ms; {:.1} launches, {:.1} active tiles per launch of {}",
        ms(p.upload),
        ms(p.raster),
        ms(p.relax),
        ms(p.download),
        p.launches as f64 / p.fields as f64,
        p.active_tiles as f64 / p.launches as f64,
        p.tiles / p.fields as u64
    );
    println!("relaxation kernels on the device: {:.2} ms per field", ms(p.relax_kernels));
    println!("rounds per active tile {:.2}, unsettled tiles per field {:.1}", p.tile_rounds as f64 / p.active_tiles as f64, p.unsettled_tiles as f64 / p.fields as f64);
}
