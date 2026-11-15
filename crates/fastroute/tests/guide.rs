//! Routing with the guide field (`--guide=cpu`, the reference solver, so that no GPU is needed):
//! the field is used, results are deterministic, and the boards route as far as without it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn synthetic(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fr-io/testdata/dsn").join(name)
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fastroute-guide-{}-{name}", std::process::id()))
}

/// Routes `dsn` (autorouter only) and returns the session and the unrouted count of the report.
fn route(dsn: &Path, tag: &str, extra: &[&str], stats: Option<&Path>) -> (Vec<u8>, i64) {
    let ses = temp(&format!("{tag}.ses"));
    let report = temp(&format!("{tag}.json"));
    let mut command = Command::new(env!("CARGO_BIN_EXE_fastroute"));
    command
        .arg("-de")
        .arg(dsn)
        .arg("-do")
        .arg(&ses)
        .arg(format!("--report={}", report.display()))
        .args(["--router.optimizer.enabled=false", "--multi-start=1"])
        .args(extra)
        .stderr(std::process::Stdio::null())
        .stdout(std::process::Stdio::null());
    if let Some(stats) = stats {
        command.env("FASTROUTE_MAZE_STATS", stats);
    }
    assert!(command.status().expect("run fastroute").success(), "fastroute failed on {}", dsn.display());
    let session = std::fs::read(&ses).unwrap();
    let text = std::fs::read_to_string(&report).unwrap();
    let _ = std::fs::remove_file(&ses);
    let _ = std::fs::remove_file(&report);
    let at = text.find("\"unrouted\":").expect("unrouted count in the report") + "\"unrouted\":".len();
    let digits: String = text[at..].trim_start().chars().take_while(|c| c.is_ascii_digit()).collect();
    (session, digits.parse().unwrap())
}

/// Coarse cells keep the reference solver quick.
const GUIDE: &[&str] = &["--guide=cpu", "--guide-min-span-mm=0", "--guide-cell-um=400"];

#[test]
fn the_guide_field_is_used_and_deterministic() {
    let dsn = synthetic("synth_rules.dsn");
    let stats = temp("stats.tsv");
    let (first, _) = route(&dsn, "first", GUIDE, Some(&stats));
    let (second, _) = route(&dsn, "second", GUIDE, None);
    assert!(first == second, "two guided runs gave different sessions");
    let table = std::fs::read_to_string(&stats).unwrap();
    let _ = std::fs::remove_file(&stats);
    let header: Vec<&str> = table.lines().next().unwrap().split('\t').collect();
    let guided = header.iter().position(|&h| h == "guided").expect("guided column");
    let used = table.lines().skip(1).filter(|line| line.split('\t').nth(guided) == Some("1")).count();
    assert!(used > 0, "no connection was routed with a guide field");
}

#[test]
fn guided_routing_routes_as_far_as_plain_routing() {
    for name in ["synth_rules.dsn", "synth_order_placed.dsn"] {
        let dsn = synthetic(name);
        let (_, plain) = route(&dsn, &format!("plain-{name}"), &[], None);
        let (_, guided) = route(&dsn, &format!("guided-{name}"), GUIDE, None);
        assert!(guided <= plain, "{name}: {guided} unrouted with the guide, {plain} without");
    }
}

#[test]
fn guide_options_need_a_solver() {
    let status = Command::new(env!("CARGO_BIN_EXE_fastroute"))
        .arg("-de")
        .arg(synthetic("synth_rules.dsn"))
        .arg("--guide-cell-um=150")
        .stderr(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status()
        .expect("run fastroute");
    assert!(!status.success(), "--guide-* without --guide must be refused");
}
