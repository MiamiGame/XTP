//! Sonde des collisions d'identifiants signalées par 0xkn0wh3r3 (VIctory #2126).

use super::host::{Dispatch, Host, Root, View};
use super::tests::{compute, emit_for};
use crate::native::{ComputeIo, RootParam, RootType, StageIo};

fn probe(width: u32, height: u32, wave32: bool) -> crate::NativeProgram {
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000,                 // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F,                 // s_waitcnt lgkmcnt(0)
        0x1604_0200 | (128 + width), // v_mul_u32_u24 v2, largeur, v1
        0x4A04_0500,                 // v_add_nc_u32 v2, v0, v2
        0x3406_0482,                 // v_lshlrev_b32 v3, 2, v2
        0x4A04_0481,                 // v_add_nc_u32 v2, 1, v2
    ];
    if wave32 {
        words.extend([
            0xD765_0004,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v4, -1, 0
            0x3408_0890, // v_lshlrev_b32 v4, 16, v4
            0x4A04_0504, // v_add_nc_u32 v2, v4, v2
        ]);
    }
    words.extend([
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let mut p = compute(&words, false);
    p.layout.push(RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    });
    p.io = StageIo::Cs(ComputeIo {
        rsrc2: 1 << 1 | 1 << 11, // Table en s0, coordonnées locales en v0 et v1.
        threads: [width, height, 1],
    });
    p
}

fn check(host: &Host, wave32: bool) {
    let mut failures = Vec::new();
    for (width, height) in [(32, 2), (32, 8), (64, 4), (24, 8), (12, 12), (10, 10)] {
        let n = width * height;
        let cs = emit_for(&probe(width, height, wave32), host.target()).unwrap();
        assert_eq!(cs.wave_size(), wave32.then_some(32));
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; cs.user_data_dwords().max(1) as usize]),
                    )],
                    views: &[View::RawUav(n)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        // Chaque thread écrit une fois. Les n cases doivent donc toutes être
        // écrites, y compris la case zéro. Les bits hauts vérifient aussi la lane.
        let wrong = (0..n)
            .filter(|&i| {
                let lane = if wave32 { i % 32 } else { 0 };
                out[0][i as usize] != (lane << 16) + i + 1
            })
            .count();
        eprintln!("{width}×{height}, wave32={wave32} : {wrong}/{n} identifiants incorrects");
        if wrong != 0 {
            failures.push((width, height, wrong));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn flat_ids_on_warp() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = match Host::new(true) {
        Ok(host) => host,
        Err(e) => {
            assert!(
                std::env::var("XTP_REQUIRE_GPU").as_deref() != Ok("1"),
                "{e}"
            );
            eprintln!("WARP: {e}; skipped");
            return;
        }
    };
    check(&host, host.target().runs(32));
}

#[test]
#[ignore = "hardware: NVIDIA, wave32"]
fn flat_ids_and_lanes_on_nvidia() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = Host::on("NVIDIA").unwrap();
    assert!(host.target().runs(32));
    check(&host, true);
}
