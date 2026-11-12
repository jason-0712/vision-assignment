//! Check the controller against the unchanged simulator's timing and collision checks.
use aiming_sim::{
    Input, Sample, Size, Stats, Target, aim, physics,
    sim::{Config, Frame, Simulation},
    ui::Gizmos,
};
use std::error::Error;

fn input(target: Target) -> Input {
    Input {
        now_s: 0.4,
        observation: Some(Sample {
            time_s: 0.35,
            target,
        }),
        size: Size::default(),
        stats: Stats::default(),
    }
}

fn run(config: Config, age_override: Option<f64>) -> Result<Frame, Box<dyn Error>> {
    let mut simulation = Simulation::new(config)?;
    let mut controller = |input: &Input, debug: &mut Gizmos| {
        if input.now_s >= config.duration_s {
            return None;
        }
        let mut delayed = *input;
        if let Some(age_s) = age_override {
            let capture_s = input.now_s - age_s;
            delayed.observation = (capture_s >= 0.0).then(|| Sample {
                time_s: capture_s,
                target: config.target.at(capture_s),
            });
        }
        aim(&delayed, debug)
    };
    let end_s = config.duration_s + physics::LAUNCH_DELAY_S + physics::MAX_FLIGHT_S;
    loop {
        let frame = simulation.step(&mut controller);
        if frame.time_s >= end_s {
            assert_eq!(frame.stats.invalid_commands, 0);
            assert_eq!(frame.stats.hits + frame.stats.misses, frame.stats.shots);
            return Ok(frame);
        }
    }
}

#[test]
fn missing_future_and_nonfinite_observations_do_not_fire() {
    let valid = input(Target::default());
    let mut missing = valid;
    missing.observation = None;
    assert!(aim(&missing, &mut Gizmos::default()).is_none());
    for (now, capture) in [
        (f64::NAN, 0.0),
        (0.4, 0.5),
        (0.4, -0.1),
        (0.4, f64::INFINITY),
    ] {
        let invalid = Input {
            now_s: now,
            observation: Some(Sample {
                time_s: capture,
                target: Target::default(),
            }),
            ..valid
        };
        assert!(aim(&invalid, &mut Gizmos::default()).is_none());
    }
    let invalid = input(Target {
        vx_m_s: f64::NAN,
        ..Target::default()
    });
    assert!(aim(&invalid, &mut Gizmos::default()).is_none());
    let invalid = Input {
        size: Size {
            radius_m: f64::NAN,
            ..valid.size
        },
        ..valid
    };
    assert!(aim(&invalid, &mut Gizmos::default()).is_none());
}

#[test]
fn unreachable_target_does_not_produce_a_command() {
    let unreachable = input(Target {
        x_m: 100.0,
        omega_rad_s: 0.0,
        ..Target::default()
    });
    assert!(aim(&unreachable, &mut Gizmos::default()).is_none());
}

#[test]
fn reachable_plate_facing_away_is_tracked_without_firing() -> Result<(), Box<dyn Error>> {
    let oblique = Input {
        size: Size {
            plate_center_z_m: 1.5,
            ..Size::default()
        },
        ..input(Target {
            x_m: 0.3,
            omega_rad_s: 0.0,
            ..Target::default()
        })
    };
    let command = aim(&oblique, &mut Gizmos::default()).ok_or("reachable elevated plate")?;
    assert!(
        !command.fire,
        "a reachable back face must not trigger a shot"
    );
    Ok(())
}

#[test]
fn angles_depend_on_observation_age_not_absolute_time() -> Result<(), Box<dyn Error>> {
    let original = input(Target {
        vx_m_s: 0.6,
        vy_m_s: -0.4,
        omega_rad_s: 4.0,
        ..Target::default()
    });
    let mut shifted = original;
    shifted.now_s += 100.0;
    if let Some(sample) = &mut shifted.observation {
        sample.time_s += 100.0;
    }
    let a = aim(&original, &mut Gizmos::default()).ok_or("reachable target")?;
    let b = aim(&shifted, &mut Gizmos::default()).ok_or("same target and age")?;
    assert!((a.yaw_rad - b.yaw_rad).abs() < 1e-10);
    assert!((a.pitch_rad - b.pitch_rad).abs() < 1e-10);
    assert_eq!(a.fire, b.fire);
    Ok(())
}

#[test]
fn supplied_scenarios_hit_and_settle_all_shots() -> Result<(), Box<dyn Error>> {
    for text in [
        include_str!("../scenarios/stationary.json"),
        include_str!("../scenarios/spinning.json"),
        include_str!("../scenarios/moving.json"),
    ] {
        let config: Config = serde_json::from_str(text)?;
        let frame = run(config, None)?;
        assert!(
            frame.stats.shots >= 60,
            "controller rarely fires: {:?}",
            frame.stats
        );
        assert_eq!(frame.stats.misses, 0, "scenario: {text}");
    }
    Ok(())
}

#[test]
fn extra_observation_age_is_compensated_without_double_counting() -> Result<(), Box<dyn Error>> {
    let config = Config {
        target: Target {
            x_m: 6.0,
            y_m: 0.5,
            vx_m_s: 0.4,
            vy_m_s: -0.5,
            omega_rad_s: -5.0,
            ..Target::default()
        },
        duration_s: 3.0,
        ..Config::default()
    };
    for age_s in [0.0, 0.05, 0.3, 0.75] {
        let frame = run(config, Some(age_s))?;
        assert!(frame.stats.hits >= 8, "age {age_s}: {:?}", frame.stats);
        assert_eq!(frame.stats.misses, 0, "age {age_s}");
    }
    Ok(())
}

#[test]
fn varied_distance_velocity_spin_and_height_use_the_same_controller() -> Result<(), Box<dyn Error>>
{
    // Different views, mount geometries and spin signs; none are supplied training clips.
    for (x, y, vx, vy, spin, height) in [
        (2.0, 1.0, 0.4, -0.2, -8.0, 0.4),
        (8.0, -3.0, -0.5, 0.4, 8.0, 0.6),
        (18.0, 2.0, -0.8, -0.5, -12.0, 1.2),
        (-6.0, 3.0, -0.2, -0.3, 4.0, 0.5),
        (12.0, -5.0, 1.2, 0.8, 20.0, 0.3),
        (5.0, 0.5, 0.0, 0.0, 0.0, 1.5),
    ] {
        let config = Config {
            target: Target {
                x_m: x,
                y_m: y,
                vx_m_s: vx,
                vy_m_s: vy,
                omega_rad_s: spin,
                ..Target::default()
            },
            size: Size {
                plate_center_z_m: height,
                ..Size::default()
            },
            duration_s: 3.0,
        };
        let frame = run(config, None)?;
        assert!(
            frame.stats.shots >= 6,
            "scenario {config:?}: {:?}",
            frame.stats
        );
        assert_eq!(frame.stats.misses, 0, "scenario {config:?}");
    }
    Ok(())
}
