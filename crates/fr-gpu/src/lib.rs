//! The guide field of the maze search (`fr_engine::autoroute::guide`) on an NVIDIA GPU.
//!
//! The kernels (`field.cu`) are compiled with NVRTC for the device at start-up. The CUDA driver
//! and NVRTC are loaded at run time, so a fastroute built with this crate starts without them;
//! only `--gpu` needs them, and fails with a clear message where they are missing.
//!
//! A field is computed in five steps on one stream: the problem's polygons are rasterized (one
//! block per polygon), the blocked flags, soft items and start values derived, the boundary
//! seeds applied, the allowed moves of every cell computed, and the cost-to-go relaxed tile by
//! tile until no tile changes. The fixpoint of the relaxation is the Dijkstra field of
//! `fr_engine::autoroute::guide::cpu::CpuFieldSolver` (the tests compare the two).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg, ValidAsZeroBits};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use fr_engine::autoroute::guide::raster::{move_lengths, RasterPolygon};
use fr_engine::autoroute::guide::{FieldProblem, FieldSolver, ObstacleKind};
use fr_engine::autoroute::stats;

const SOURCE: &str = include_str!("field.cu");

/// Must match `field.cu`.
const POLYGON_WORDS: usize = 10;
const KIND_HARD: u32 = 0;
const KIND_SOFT: u32 = 1;
const KIND_DESTINATION: u32 = 2;
const TILE: usize = 32;
const SPAN: usize = TILE + 4;
/// `SHARED_PER_LAYER` of `field.cu`: three halo arrays with rows of `SPAN + 1` words and the
/// tile array with rows of `TILE + 1` words.
const SHARED_PER_LAYER: usize = 4 * (3 * SPAN * (SPAN + 1) + TILE * (TILE + 1));
/// The static shared memory of `relax_tiles` (and some room), on top of the dynamic part.
const STATIC_SHARED: usize = 1024;
/// `s_lengths` of `relax_tiles` holds the move lengths of this many layers.
const MAX_LAYERS: usize = 8;

/// Rounds of four sweeps of a tile per launch before it hands over to the next launch.
const MAX_INNER_ROUNDS: i32 = 2;
/// Launches between two reads of the changed flag (each read waits for the GPU).
const LAUNCHES_PER_CHECK: usize = 8;
/// A field that is still changing after this many launches is a bug.
const MAX_LAUNCHES: usize = 100_000;
const THREADS: u32 = 256;

/// Computes guide fields with CUDA. One field at a time: the routing threads share the device.
pub struct CudaFieldSolver {
    device_name: String,
    /// The most layers a field may have (the shared memory of a block holds a tile of each).
    max_layers: usize,
    gpu: Mutex<Gpu>,
}

struct Kernels {
    clear_grid: CudaFunction,
    rasterize: CudaFunction,
    finalize: CudaFunction,
    seed_boundary: CudaFunction,
    move_masks: CudaFunction,
    fill_bytes: CudaFunction,
    relax_tiles: CudaFunction,
}

struct Gpu {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    buffers: Buffers,
    /// Set while profiling (see [`CudaFieldSolver::set_profiling`]).
    profile: Option<Profile>,
}

/// Where the time of the fields went, while profiling: every phase then waits for the device,
/// which makes the solver slower.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub fields: u32,
    pub cells: u64,
    /// Flattening and copying the problem to the device.
    pub upload: Duration,
    /// Rasterizing, start values, boundary seeds and move masks.
    pub raster: Duration,
    pub relax: Duration,
    /// Of `relax`: the relaxation kernels on the device (events around each launch).
    pub relax_kernels: Duration,
    pub download: Duration,
    pub launches: u64,
    /// Active tiles summed over the launches.
    pub active_tiles: u64,
    pub tiles: u64,
    /// Rounds of the active tiles, and tiles that did not settle within their launch.
    pub tile_rounds: u64,
    pub unsettled_tiles: u64,
}

/// While profiling: waits for the device and adds the time since `since` to the phase.
fn mark(stream: &CudaStream, profile: &mut Option<Profile>, phase: fn(&mut Profile) -> &mut Duration, since: &mut Instant) -> Result<(), DriverError> {
    if let Some(totals) = profile {
        stream.synchronize()?;
        *phase(totals) += since.elapsed();
        *since = Instant::now();
    }
    Ok(())
}

/// A device buffer that grows to the largest size asked for and is reused.
struct Grow<T> {
    slice: Option<CudaSlice<T>>,
}

impl<T: DeviceRepr + ValidAsZeroBits> Grow<T> {
    fn new() -> Self {
        Grow { slice: None }
    }

    fn ensure(&mut self, stream: &Arc<CudaStream>, len: usize) -> Result<&mut CudaSlice<T>, DriverError> {
        let len = len.max(1);
        if self.slice.as_ref().is_none_or(|slice| slice.len() < len) {
            // grow by half again, so that slowly growing windows do not reallocate every time
            self.slice = Some(stream.alloc_zeros::<T>(len + len / 2)?);
        }
        Ok(self.slice.as_mut().expect("buffer allocated above"))
    }
}

struct Buffers {
    hard: Grow<u8>,
    destination: Grow<u8>,
    soft_key: Grow<u64>,
    blocked: Grow<u8>,
    soft_item: Grow<u32>,
    soft_cost: Grow<f32>,
    dist: Grow<f32>,
    masks: Grow<u32>,
    active_in: Grow<u8>,
    active_out: Grow<u8>,
    changed: Grow<i32>,
    /// Tile rounds done and tiles not settled (profile).
    rounds_done: Grow<u64>,
    meta: Grow<u32>,
    half_planes: Grow<f32>,
    seeds: Grow<u32>,
    lengths: Grow<f32>,
}

impl CudaFieldSolver {
    /// Opens the first CUDA device and compiles the kernels for it.
    pub fn new() -> Result<Self, String> {
        // cudarc panics on a missing library; both are loaded at run time
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("the NVIDIA driver's CUDA library (nvcuda.dll / libcuda.so) was not found; install the NVIDIA driver".into());
        }
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err("NVRTC (nvrtc64_120_0.dll / libnvrtc.so.12) was not found; install the CUDA Toolkit 12, or put its NVRTC \
                        libraries on the library path (Windows: PATH or next to fastroute.exe, Linux: LD_LIBRARY_PATH)"
                .into());
        }
        let ctx =CudaContext::new(0).map_err(|e| format!("no CUDA device: {e:?}"))?;
        let device_name = ctx.name().map_err(|e| format!("CUDA device name: {e:?}"))?;
        let (major, minor) = ctx.compute_capability().map_err(|e| format!("CUDA compute capability: {e:?}"))?;
        // one stream, buffers used by one field at a time: no event tracking needed
        unsafe { ctx.disable_event_tracking() };
        let ptx = compile(&format!("compute_{major}{minor}")).or_else(|e| {
            log::warn!(target: "fr_gpu", "NVRTC cannot compile for compute_{major}{minor} ({e}); compiling for compute_80 (the driver translates it)");
            compile("compute_80")
        })?;
        let module = ctx.load_module(ptx).map_err(|e| format!("loading the guide kernels: {e:?}"))?;
        let function = |name: &str| module.load_function(name).map_err(|e| format!("guide kernel {name}: {e:?}"));
        let shared_limit = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .map_err(|e| format!("CUDA shared memory limit: {e:?}"))? as usize;
        let max_layers = (shared_limit.saturating_sub(STATIC_SHARED) / SHARED_PER_LAYER).min(MAX_LAYERS);
        if max_layers == 0 {
            return Err(format!("{device_name}: {shared_limit} bytes of shared memory per block, the guide kernels need {}", SHARED_PER_LAYER + STATIC_SHARED));
        }
        let kernels = Kernels {
            clear_grid: function("clear_grid")?,
            rasterize: function("rasterize")?,
            finalize: function("finalize")?,
            seed_boundary: function("seed_boundary")?,
            move_masks: function("move_masks")?,
            fill_bytes: function("fill_bytes")?,
            relax_tiles: function("relax_tiles")?,
        };
        let buffers = Buffers {
            hard: Grow::new(),
            destination: Grow::new(),
            soft_key: Grow::new(),
            blocked: Grow::new(),
            soft_item: Grow::new(),
            soft_cost: Grow::new(),
            dist: Grow::new(),
            masks: Grow::new(),
            active_in: Grow::new(),
            active_out: Grow::new(),
            changed: Grow::new(),
            rounds_done: Grow::new(),
            meta: Grow::new(),
            half_planes: Grow::new(),
            seeds: Grow::new(),
            lengths: Grow::new(),
        };
        kernels
            .relax_tiles
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                (max_layers * SHARED_PER_LAYER) as i32,
            )
            .map_err(|e| format!("CUDA shared memory of the relaxation kernel: {e:?}"))?;
        let gpu = Gpu { stream: ctx.default_stream(), kernels, buffers, profile: None };
        Ok(CudaFieldSolver { device_name: format!("{device_name} (compute {major}.{minor})"), max_layers, gpu: Mutex::new(gpu) })
    }

    /// The name and compute capability of the device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Registers per thread of the relaxation kernel and its blocks per multiprocessor for a
    /// field of `layers` layers.
    pub fn relax_occupancy(&self, layers: usize) -> Result<(i32, u32), String> {
        let gpu = self.gpu.lock().expect("CUDA field solver poisoned");
        let relax = &gpu.kernels.relax_tiles;
        let registers = relax.num_regs().map_err(|e| format!("{e:?}"))?;
        let blocks = relax
            .occupancy_max_active_blocks_per_multiprocessor((TILE * 4 * layers) as u32, layers * SHARED_PER_LAYER, None)
            .map_err(|e| format!("{e:?}"))?;
        Ok((registers, blocks))
    }

    /// Starts (and resets) or stops the phase profile.
    pub fn set_profiling(&self, on: bool) {
        self.gpu.lock().expect("CUDA field solver poisoned").profile = on.then(Profile::default);
    }

    /// The phase profile since profiling started, if on.
    pub fn profile(&self) -> Option<Profile> {
        self.gpu.lock().expect("CUDA field solver poisoned").profile.clone()
    }
}

fn compile(architecture: &str) -> Result<cudarc::nvrtc::Ptx, String> {
    let options = CompileOptions {
        options: vec![format!("--gpu-architecture={architecture}")],
        name: Some("field.cu".to_string()),
        ..CompileOptions::default()
    };
    compile_ptx_with_opts(SOURCE, options).map_err(|e| match e {
        cudarc::nvrtc::CompileError::CompileError { log, .. } => format!("NVRTC: {}", log.to_string_lossy()),
        other => format!("NVRTC: {other:?}"),
    })
}

impl FieldSolver for CudaFieldSolver {
    fn name(&self) -> &str {
        "cuda"
    }

    fn solve(&self, problem: &FieldProblem) -> Result<Vec<f32>, String> {
        if problem.layers > self.max_layers {
            return Err(format!("a field of {} layers; {} supports up to {}", problem.layers, self.device_name, self.max_layers));
        }
        let mut gpu = self.gpu.lock().map_err(|_| "the CUDA field solver panicked earlier".to_string())?;
        gpu.solve(problem).map_err(|e| format!("CUDA: {e:?}"))
    }
}

/// The problem flattened as `field.cu` reads it.
struct Flat {
    meta: Vec<u32>,
    half_planes: Vec<f32>,
    seeds: Vec<u32>,
    lengths: Vec<f32>,
}

fn flatten(problem: &FieldProblem) -> Flat {
    let polygon_count = problem.obstacles.len() + problem.destinations.len();
    let mut meta = Vec::with_capacity(polygon_count * POLYGON_WORDS);
    let mut half_planes = Vec::new();
    let mut push = |polygon: &RasterPolygon, kind: u32, item: u32, cost: f32| {
        meta.extend_from_slice(&[
            (half_planes.len() / 3) as u32,
            polygon.half_planes.len() as u32,
            polygon.layer as u32,
            kind,
            item,
            cost.to_bits(),
            polygon.min[0].to_bits(),
            polygon.min[1].to_bits(),
            polygon.max[0].to_bits(),
            polygon.max[1].to_bits(),
        ]);
        for h in &polygon.half_planes {
            half_planes.extend_from_slice(h);
        }
    };
    for obstacle in &problem.obstacles {
        let polygon = RasterPolygon::new(&obstacle.polygon);
        match obstacle.kind {
            ObstacleKind::Hard => push(&polygon, KIND_HARD, 0, 0.0),
            ObstacleKind::Soft { item, cost } => push(&polygon, KIND_SOFT, item, cost),
        }
    }
    for destination in &problem.destinations {
        push(&RasterPolygon::new(destination), KIND_DESTINATION, 0, 0.0);
    }
    let mut seeds = Vec::with_capacity(problem.boundary.len() * 2);
    for seed in &problem.boundary {
        let index = (seed.layer * problem.height + seed.y) * problem.width + seed.x;
        seeds.push(u32::try_from(index).expect("guide field larger than 2^32 cells"));
        seeds.push(seed.value.to_bits());
    }
    let lengths = move_lengths(problem).iter().flat_map(|layer| layer.iter().map(|&length| length as f32)).collect();
    Flat { meta, half_planes, seeds, lengths }
}

fn blocks(count: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((count as u32).div_ceil(THREADS).max(1), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
}

impl Gpu {
    fn solve(&mut self, problem: &FieldProblem) -> Result<Vec<f32>, DriverError> {
        assert!(problem.layers >= 1, "guide field without layers");
        let cells = problem.cell_count();
        assert!(cells < i32::MAX as usize, "guide field of {cells} cells");
        let mut since = Instant::now();
        let flat = flatten(problem);
        let polygon_count = flat.meta.len() / POLYGON_WORDS;
        let stream = self.stream.clone();
        let kernels = &self.kernels;
        let buffers = &mut self.buffers;
        let profile = &mut self.profile;
        let (width, height, layers) = (problem.width as i32, problem.height as i32, problem.layers as i32);
        let cell_count = cells as i32;
        let cell_size = problem.cell_size as f32;
        let via_cost = problem.via_cost as f32;

        let meta = buffers.meta.ensure(&stream, flat.meta.len())?;
        stream.memcpy_htod(&flat.meta, &mut meta.slice_mut(0..flat.meta.len()))?;
        let half_planes = buffers.half_planes.ensure(&stream, flat.half_planes.len())?;
        stream.memcpy_htod(&flat.half_planes, &mut half_planes.slice_mut(0..flat.half_planes.len()))?;
        let seeds = buffers.seeds.ensure(&stream, flat.seeds.len())?;
        stream.memcpy_htod(&flat.seeds, &mut seeds.slice_mut(0..flat.seeds.len()))?;
        let lengths = buffers.lengths.ensure(&stream, flat.lengths.len())?;
        stream.memcpy_htod(&flat.lengths, &mut lengths.slice_mut(0..flat.lengths.len()))?;
        mark(&stream, profile, |profile| &mut profile.upload, &mut since)?;

        let hard = buffers.hard.ensure(&stream, cells)?;
        let destination = buffers.destination.ensure(&stream, cells)?;
        let soft_key = buffers.soft_key.ensure(&stream, cells)?;
        unsafe { stream.launch_builder(&kernels.clear_grid).arg(&mut *hard).arg(&mut *destination).arg(&mut *soft_key).arg(&cell_count).launch(blocks(cells))? };
        if polygon_count > 0 {
            let polygons = polygon_count as i32;
            let config = LaunchConfig { grid_dim: (polygon_count.min(65_535) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                stream
                    .launch_builder(&kernels.rasterize)
                    .arg(&*meta)
                    .arg(&*half_planes)
                    .arg(&polygons)
                    .arg(&width)
                    .arg(&height)
                    .arg(&cell_size)
                    .arg(&mut *hard)
                    .arg(&mut *destination)
                    .arg(&mut *soft_key)
                    .launch(config)?
            };
        }
        let blocked = buffers.blocked.ensure(&stream, cells)?;
        let soft_item = buffers.soft_item.ensure(&stream, cells)?;
        let soft_cost = buffers.soft_cost.ensure(&stream, cells)?;
        let dist = buffers.dist.ensure(&stream, cells)?;
        unsafe {
            stream
                .launch_builder(&kernels.finalize)
                .arg(&*hard)
                .arg(&*destination)
                .arg(&*soft_key)
                .arg(&cell_count)
                .arg(&mut *blocked)
                .arg(&mut *soft_item)
                .arg(&mut *soft_cost)
                .arg(&mut *dist)
                .launch(blocks(cells))?
        };
        let seed_count = (flat.seeds.len() / 2) as i32;
        if seed_count > 0 {
            unsafe { stream.launch_builder(&kernels.seed_boundary).arg(&*seeds).arg(&seed_count).arg(&*blocked).arg(&mut *dist).launch(blocks(seed_count as usize))? };
        }
        let masks = buffers.masks.ensure(&stream, cells)?;
        unsafe {
            stream
                .launch_builder(&kernels.move_masks)
                .arg(&*blocked)
                .arg(&*soft_item)
                .arg(&width)
                .arg(&height)
                .arg(&layers)
                .arg(&mut *masks)
                .launch(blocks(cells))?
        };
        mark(&stream, profile, |profile| &mut profile.raster, &mut since)?;

        let tiles_x = problem.width.div_ceil(TILE);
        let tiles_y = problem.height.div_ceil(TILE);
        let tiles = tiles_x * tiles_y;
        let tile_count = tiles as i32;
        let one: u8 = 1;
        stream.memset_zeros(buffers.active_out.ensure(&stream, tiles)?)?;
        let active_in = buffers.active_in.ensure(&stream, tiles)?;
        unsafe { stream.launch_builder(&kernels.fill_bytes).arg(&mut *active_in).arg(&one).arg(&tile_count).launch(blocks(tiles))? };
        buffers.changed.ensure(&stream, 1)?;
        let rounds_done = buffers.rounds_done.ensure(&stream, 2)?;
        stream.memset_zeros(&mut *rounds_done)?;
        let relax_config = LaunchConfig {
            grid_dim: (tiles_x as u32, tiles_y as u32, 1),
            block_dim: (TILE as u32, 4, problem.layers as u32),
            shared_mem_bytes: (problem.layers * SHARED_PER_LAYER) as u32,
        };
        let max_inner = MAX_INNER_ROUNDS;
        let mut launches = 0;
        loop {
            stream.memset_zeros(buffers.changed.slice.as_mut().expect("changed flag allocated"))?;
            for _ in 0..LAUNCHES_PER_CHECK {
                let active_out = buffers.active_out.slice.as_mut().expect("active flags allocated");
                if let Some(totals) = profile.as_mut() {
                    // the kernel clears the flags it reads: count them before
                    let active = stream.clone_dtoh(&buffers.active_in.slice.as_ref().expect("active flags allocated").slice(0..tiles))?;
                    totals.active_tiles += active.iter().filter(|&&flag| flag != 0).count() as u64;
                    totals.launches += 1;
                }
                let kernel_start = if profile.is_some() { Some(stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?) } else { None };
                let active_in = buffers.active_in.slice.as_mut().expect("active flags allocated");
                let changed = buffers.changed.slice.as_mut().expect("changed flag allocated");
                unsafe {
                    stream
                        .launch_builder(&kernels.relax_tiles)
                        .arg(&mut *dist)
                        .arg(&*masks)
                        .arg(&*blocked)
                        .arg(&*soft_item)
                        .arg(&*soft_cost)
                        .arg(&*lengths)
                        .arg(&width)
                        .arg(&height)
                        .arg(&layers)
                        .arg(&via_cost)
                        .arg(active_in)
                        .arg(active_out)
                        .arg(changed)
                        .arg(&max_inner)
                        .arg(&mut *rounds_done)
                        .launch(relax_config)?
                };
                if let Some(kernel_start) = kernel_start {
                    let kernel_end = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    let totals = profile.as_mut().expect("profiling");
                    totals.relax_kernels += Duration::from_secs_f64(kernel_start.elapsed_ms(&kernel_end)? as f64 / 1000.0);
                }
                std::mem::swap(&mut buffers.active_in, &mut buffers.active_out);
                launches += 1;
            }
            let changed = stream.clone_dtoh(&buffers.changed.slice.as_ref().expect("changed flag allocated").slice(0..1))?;
            stats::update(|connection| connection.field_launches += LAUNCHES_PER_CHECK as u32);
            if changed[0] == 0 {
                break;
            }
            assert!(launches < MAX_LAUNCHES, "guide field ({}x{}x{}) still changing after {launches} launches", problem.width, problem.height, problem.layers);
        }
        mark(&stream, profile, |profile| &mut profile.relax, &mut since)?;
        if let Some(totals) = profile.as_mut() {
            let counts = stream.clone_dtoh(&rounds_done.slice(0..2))?;
            totals.tile_rounds += counts[0];
            totals.unsettled_tiles += counts[1];
        }
        let download_start = Instant::now();
        let values = stream.clone_dtoh(&dist.slice(0..cells))?;
        stats::update(|connection| connection.field_download_time += download_start.elapsed());
        mark(&stream, profile, |profile| &mut profile.download, &mut since)?;
        if let Some(totals) = profile.as_mut() {
            totals.fields += 1;
            totals.cells += cells as u64;
            totals.tiles += tiles as u64;
        }
        Ok(values)
    }
}
