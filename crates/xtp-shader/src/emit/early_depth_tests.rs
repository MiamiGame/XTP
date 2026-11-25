use crate::dxbc::{self, Container, FourCC};
use crate::emit::container::{SigElement, signature};
use crate::emit::{HostShader, translate_program};
use crate::native::{HwStage, NativeProgram, StageIo};

fn pixel(db_shader_control: u32) -> HostShader {
    let words = [
        0x7E00_02F2u32, // v_mov_b32 v0, 1.0
        0xF800_180F,
        0,           // exp mrt0, v0, v0, v0, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        float_mode: 0,
        vgprs: 1,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            rsrc2: 0,
            db_shader_control,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 15,
        half: false,
    };
    let bytes = dxbc::build(&[(FourCC::OSG1, &signature(&[target], false))]);
    translate_program(&Container::parse(&bytes).unwrap(), &native).unwrap()
}

#[test]
fn depth_before_shader_sets_only_the_dxil_entry_flag() {
    // Même programme, sans écriture UAV. Seul le bit 12 commande le test anticipé.
    let neighbors = (1 << 11) | (1 << 13);
    let plain = pixel(neighbors);
    let early = pixel(neighbors | (1 << 12));
    assert_eq!(plain.dxil, pixel(0).dxil);
    let plain_parts = Container::parse(&plain.dxil).unwrap();
    let early_parts = Container::parse(&early.dxil).unwrap();
    for part in &plain_parts.parts {
        let other = early_parts.part(part.fourcc).unwrap().data;
        if part.fourcc == FourCC::DXIL {
            assert_ne!(part.data, other);
        } else {
            assert_eq!(part.data, other);
        }
    }
    #[cfg(windows)]
    if let Some(ir) = crate::emit::smem_tests::disassembly(&early) {
        assert!(ir.contains("!{i32 0, i64 8}"), "{ir}");
    }
}
