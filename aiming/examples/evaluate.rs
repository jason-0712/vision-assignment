//! Evaluate the real controller with the evaluator, including draining pending shots.
use aiming_sim::{
    aim, physics,
    sim::{Config, Simulation},
};
use serde_json::json;
use std::{error::Error, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut reports = Vec::new();
    for name in ["stationary", "spinning", "moving"] {
        let config: Config = serde_json::from_str(&fs::read_to_string(
            directory.join(format!("scenarios/{name}.json")),
        )?)?;
        let mut simulation = Simulation::new(config)?;
        let end_s = config.duration_s + physics::LAUNCH_DELAY_S + physics::MAX_FLIGHT_S;
        let mut controller = |input: &aiming_sim::Input, debug: &mut aiming_sim::ui::Gizmos| {
            if input.now_s < config.duration_s {
                aim(input, debug)
            } else {
                None
            }
        };
        let final_frame = loop {
            let frame = simulation.step(&mut controller);
            if frame.time_s >= end_s {
                break frame;
            }
        };
        let stats = final_frame.stats;
        let resolved = stats.hits + stats.misses;
        if stats.shots != resolved || stats.invalid_commands != 0 {
            return Err(format!("{name}: unresolved shots or invalid commands").into());
        }
        reports.push(json!({
            "scenario": name,
            "config": config,
            "stats": stats,
            "hit_rate_resolved": if resolved == 0 { None } else { Some(stats.hits as f64 / resolved as f64) },
            "shots_per_simulation_second": stats.shots as f64 / config.duration_s,
            "resolved_shots": resolved,
            "pending_shots": stats.shots - resolved,
            "final_simulation_time_s": final_frame.time_s,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "controller": "observation age + launch delay + fixed-point flight time; fixed plate index",
            "physics": "unmodified assignment simulator",
            "scenarios": reports,
        }))?
    );
    Ok(())
}
