//! fastroute: per-connection statistics of the maze search (`FASTROUTE_MAZE_STATS=<file>`).
//!
//! One TSV line per connection attempt: the sizes of the start and destination sets, how much of
//! the board the search expanded, the heuristic at the start against the cost of the connection
//! found, and where the time went. Used to judge what a better maze heuristic would save; off
//! (and free) unless the variable is set.

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// The statistics of one connection attempt.
#[derive(Clone, Debug, Default)]
pub struct ConnectionStats {
    pub net: i32,
    pub pass_no: i32,
    pub start_items: usize,
    pub destination_items: usize,
    /// Area of the union of the destination boxes the heuristic measures against (mm²).
    pub destination_box_area_mm2: f64,
    /// Elements taken from the maze queue.
    pub polled: u64,
    /// Elements expanded (taken and not yet occupied).
    pub expanded: u64,
    /// Expansion rooms created by the search.
    pub rooms: usize,
    /// The smallest sorting value of the start elements (the heuristic at the start).
    pub start_estimate: f64,
    /// The expansion value of the destination element (the cost of the connection found), or -1.
    pub found_cost: f64,
    /// Time spent building the guide field (included in `maze_time`).
    pub field_time: Duration,
    /// Of `field_time`: solving the field (the rest is building the problem from the board).
    pub field_solve_time: Duration,
    /// Of `field_solve_time`: copying the field back from the device.
    pub field_download_time: Duration,
    /// Relaxation launches of the GPU solver.
    pub field_launches: u32,
    /// Cells of the field (all layers).
    pub field_cells: usize,
    /// Obstacles of the field problem.
    pub field_obstacles: usize,
    /// True if the search had a guide field.
    pub guided: bool,
    pub maze_time: Duration,
    pub locate_time: Duration,
    pub insert_time: Duration,
    pub optimize_time: Duration,
    pub state: String,
}

const HEADER: &str = "net\tpass\tstart_items\tdest_items\tdest_box_mm2\tpolled\texpanded\trooms\tstart_estimate\tfound_cost\tguided\tfield_ms\tsolve_ms\tdownload_ms\tlaunches\tcells\tobstacles\tmaze_ms\tlocate_ms\tinsert_ms\topt_ms\tstate";

fn writer() -> Option<&'static Mutex<BufWriter<File>>> {
    static WRITER: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
    WRITER
        .get_or_init(|| {
            let path = std::env::var_os("FASTROUTE_MAZE_STATS")?;
            let file = File::create(&path).unwrap_or_else(|e| panic!("FASTROUTE_MAZE_STATS: cannot create {path:?}: {e}"));
            let mut out = BufWriter::new(file);
            writeln!(out, "{HEADER}").expect("FASTROUTE_MAZE_STATS: write failed");
            Some(Mutex::new(out))
        })
        .as_ref()
}

/// True if the statistics are collected.
pub fn enabled() -> bool {
    writer().is_some()
}

thread_local! {
    static CURRENT: RefCell<ConnectionStats> = RefCell::new(ConnectionStats::default());
}

/// Changes the statistics of the connection being routed on this thread.
pub fn update(f: impl FnOnce(&mut ConnectionStats)) {
    if enabled() {
        CURRENT.with(|current| f(&mut current.borrow_mut()));
    }
}

/// Starts the statistics of a new connection on this thread.
pub fn begin(net: i32, pass_no: i32) {
    update(|stats| {
        *stats = ConnectionStats { net, pass_no, found_cost: -1.0, ..ConnectionStats::default() };
    });
}

/// Writes the statistics of the connection routed on this thread.
pub fn finish(state: &str) {
    let Some(w) = writer() else { return };
    let line = CURRENT.with(|c| {
        let s = c.borrow();
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        format!(
            "{}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{:.0}\t{:.0}\t{}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{}",
            s.net,
            s.pass_no,
            s.start_items,
            s.destination_items,
            s.destination_box_area_mm2,
            s.polled,
            s.expanded,
            s.rooms,
            s.start_estimate,
            s.found_cost,
            u8::from(s.guided),
            ms(s.field_time),
            ms(s.field_solve_time),
            ms(s.field_download_time),
            s.field_launches,
            s.field_cells,
            s.field_obstacles,
            ms(s.maze_time),
            ms(s.locate_time),
            ms(s.insert_time),
            ms(s.optimize_time),
            state
        )
    });
    let mut out = w.lock().expect("FASTROUTE_MAZE_STATS writer poisoned");
    writeln!(out, "{line}").expect("FASTROUTE_MAZE_STATS: write failed");
    out.flush().expect("FASTROUTE_MAZE_STATS: flush failed");
}
