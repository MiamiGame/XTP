//! What dxil.dll refuses in the containers the emitter writes (W6): the
//! order of the rows of `ISG1`/`OSG1`, which it compares byte for byte with
//! the parts it writes from the module, and the one interpolation mode it
//! allows per signature row. Small modules built with [`Shader`] (their
//! signatures as given), and one console pixel program translated whole.
//! Each container is checked by the system's dxil.dll where there is one
//! (a skip is said on standard error).

use super::bitcode::{Attr, Cast, Term};
use super::container::{self, SigElement};
use super::shader::{Shader, one_mode_per_row};
use super::validator::Validator;
use crate::dxbc::{Container, FourCC};

/// dxil.dll's verdict on `container`; `None` without a dxil.dll.
fn verdict(what: &str, container: &[u8]) -> Option<Result<(), String>> {
    match Validator::load(Validator::default_path().as_deref()) {
        Ok(mut v) => Some(v.validate(&mut container.to_vec())),
        Err(e) => {
            eprintln!("SKIPPED validation of {what}: {e}");
            None
        }
    }
}

/// Panics if dxil.dll refuses `container`.
fn validates(what: &str, container: &[u8]) {
    if let Some(Err(e)) = verdict(what, container) {
        panic!("{what}: validation: {e}");
    }
}

/// The validator the tests use knows the version the modules declare
/// (`dx.valver` 1.6, `Shader::finish`): an older one would refuse every
/// module, a newer one checks them by 1.6's rules. Its version and path
/// are printed, since the checks of every test depend on them.
#[test]
fn the_validator_knows_the_declared_version() {
    let path = Validator::default_path();
    let v = match Validator::load(path.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            return;
        }
    };
    let version = v.version().expect("IDxcVersionInfo");
    eprintln!("dxil.dll {path:?}: validator {}.{}", version.0, version.1);
    assert!(version >= (1, 6), "validator {version:?}");
}

#[allow(clippy::too_many_arguments)]
fn el(
    name: &str,
    index: u32,
    kind: u8,
    comp: u8,
    interp: u8,
    row: Option<u32>,
    col: u8,
    cols: u8,
) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp,
        interp,
        row,
        col,
        cols,
        usage: 0,
        half: false,
    }
}

/// `SV_Position` in row 0, as an output (noperspective, as DXC declares it).
fn position() -> SigElement {
    el("SV_Position", 0, 3, 9, 4, Some(0), 0, 4)
}

/// A 32-bit uint `TEXCOORD<index>` at `(1, col)`, one channel.
fn uint(index: u32, interp: u8, col: u8) -> SigElement {
    el("TEXCOORD", index, 0, 5, interp, Some(1), col, 1)
}

/// A float `TEXCOORD<index>` at `(1, col)`, one channel.
fn float(index: u32, interp: u8, col: u8) -> SigElement {
    el("TEXCOORD", index, 0, 9, interp, Some(1), col, 1)
}

fn front_face(interp: u8, col: u8) -> SigElement {
    el("SV_IsFrontFace", 0, 13, 5, interp, Some(1), col, 1)
}

/// A pixel shader reading the first channel of each element of `inputs`
/// named in `reads` into SV_Target0's channels in turn.
fn pixel(mut inputs: Vec<SigElement>, reads: &[usize]) -> Vec<u8> {
    let mut s = Shader::new(true);
    let b = s.entry;
    let mut vals = Vec::new();
    for &id in reads {
        let e = &mut inputs[id];
        e.usage |= 1;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (s.t.i32, ".i32")
        } else {
            (s.t.f32, ".f32")
        };
        let undef = s.m.undef(s.t.i32);
        let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(0), undef];
        let v = s.op(
            b,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        let f32t = s.t.f32;
        vals.push(if int {
            s.m.cast(b, Cast::Bitcast, v, f32t)
        } else {
            v
        });
    }
    s.inputs = inputs;
    for col in 0..4u8 {
        let v = match vals.get(usize::from(col)) {
            Some(&v) => v,
            None => s.m.f32(0),
        };
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), v];
        let void = s.t.void;
        s.op(b, "storeOutput.f32", Attr::NoUnwind, void, 5, &args);
    }
    let mut target = el("SV_Target", 0, 16, 9, 0, Some(0), 0, 4);
    target.usage = 0xf;
    s.outputs = vec![target];
    s.m.terminate(b, Term::Ret);
    s.finish()
}

/// A vertex shader writing every channel of `outputs` (zeros).
fn vertex(outputs: Vec<SigElement>) -> Vec<u8> {
    let mut s = Shader::new(false);
    let b = s.entry;
    for (id, e) in outputs.iter().enumerate() {
        for col in 0..e.cols {
            let (suffix, zero) = if e.comp == 9 {
                (".f32", s.m.f32(0))
            } else {
                (".i32", s.m.i32(0))
            };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), zero];
            let void = s.t.void;
            let name = format!("storeOutput{suffix}");
            s.op(b, &name, Attr::NoUnwind, void, 5, &args);
        }
    }
    s.outputs = outputs
        .into_iter()
        .map(|mut e| {
            e.usage = (1 << e.cols) - 1;
            e
        })
        .collect();
    s.m.terminate(b, Term::Ret);
    s.finish()
}

/// The elements of part `fourcc` of `container`, in the part's row order.
fn rows_of(container: &[u8], fourcc: FourCC) -> Vec<(String, u32, Option<u32>, u8)> {
    let c = Container::parse(container).expect("a container");
    let part = c.part(fourcc).expect("the part").data;
    super::signature(part)
        .expect("a signature")
        .into_iter()
        .map(|e| (e.name, e.index, e.row, e.col))
        .collect()
}

/// Two elements packed into one register, the one at column 3 first in
/// element order (as the console's signatures have them: rows are theirs).
/// dxil.dll writes the rows of a register in the order of their names,
/// which is element order: by column, ISG1 and OSG1 differed from its own
/// ("Container part 'Program Input/Output Signature' does not match
/// expected for module").
#[test]
fn the_rows_of_one_register_keep_element_order() {
    let elements = || {
        vec![
            position(),
            float(0, 2, 3),
            el("TEXCOORD", 1, 0, 9, 2, Some(1), 0, 3),
        ]
    };
    let expected = vec![
        ("SV_Position".to_owned(), 0, Some(0), 0),
        ("TEXCOORD".to_owned(), 0, Some(1), 3),
        ("TEXCOORD".to_owned(), 1, Some(1), 0),
    ];
    let vs = vertex(elements());
    assert_eq!(rows_of(&vs, FourCC::OSG1), expected);
    validates("vertex outputs", &vs);
    let ps = pixel(elements(), &[1, 2]);
    assert_eq!(rows_of(&ps, FourCC::ISG1), expected);
    validates("pixel inputs", &ps);
}

/// Rows still sort by register first, unallocated elements last; within a
/// register, by name (one per element, system values sharing theirs).
#[test]
fn rows_sort_by_register_then_name() {
    let depth = el("SV_Depth", 0, 17, 9, 0, None, 0, 1);
    let elements = [
        el("SV_Target", 1, 16, 9, 0, Some(1), 0, 4),
        depth,
        el("SV_Target", 0, 16, 9, 0, Some(0), 0, 4),
    ];
    let part = container::signature(&elements, false);
    let rows: Vec<(u32, Option<u32>)> = super::signature(&part)
        .unwrap()
        .into_iter()
        .map(|e| (e.index, e.row))
        .collect();
    assert_eq!(rows, [(0, Some(0)), (1, Some(1)), (0, None)]);
}

#[test]
fn an_unread_front_face_sharing_a_row_with_a_linear_read_is_omitted() {
    for (face_col, float_col) in [(0, 1), (1, 0)] {
        let face = front_face(1, face_col);
        let varying = float(0, 2, float_col);
        for (inputs, read) in [
            (vec![position(), face.clone(), varying.clone()], 2),
            (vec![position(), varying, face], 1),
        ] {
            let ps = pixel(inputs, &[read]);
            assert_eq!(
                rows_of(&ps, FourCC::ISG1),
                [
                    ("SV_Position".to_owned(), 0, Some(0), 0),
                    ("TEXCOORD".to_owned(), 0, Some(1), float_col)
                ]
            );
            validates("front face beside linear float", &ps);
        }
    }
}

/// The modes [`one_mode_per_row`] gives each row's elements.
fn modes(mut elements: Vec<SigElement>, read: &[usize], pixel_inputs: bool) -> Vec<u8> {
    for &i in read {
        elements[i].usage = 1;
    }
    one_mode_per_row(&mut elements, pixel_inputs);
    elements.iter().map(|e| e.interp).collect()
}

/// The console packs a flat channel beside interpolated ones in one
/// attribute row; DXIL allows one interpolation mode per row ("signature
/// element ... has interpolation mode that differs from another element
/// packed into the same row"). Elements whose mode changes nothing take
/// the row's: inputs no read chose a mode for, and outputs.
#[test]
fn a_row_gets_one_interpolation_mode() {
    // Pixel inputs nothing reads: constant where the row holds an integer
    // (DXIL wants integers constant or undefined), else the first's mode.
    assert_eq!(
        modes(vec![position(), uint(0, 1, 0), float(1, 2, 1)], &[], true),
        [4, 1, 1]
    );
    assert_eq!(
        modes(
            vec![position(), float(0, 2, 0), front_face(1, 1)],
            &[],
            true
        ),
        [4, 1, 1]
    );
    assert_eq!(
        modes(vec![position(), float(0, 3, 0), float(1, 2, 1)], &[], true),
        [4, 3, 3]
    );
    // A read input keeps the mode its reads chose; the others take it.
    assert_eq!(
        modes(vec![position(), float(0, 2, 0), float(1, 3, 1)], &[1], true),
        [4, 2, 2]
    );
    assert_eq!(
        modes(vec![position(), float(0, 4, 0), float(1, 2, 1)], &[1], true),
        [4, 4, 4]
    );
    assert_eq!(
        modes(vec![position(), float(0, 2, 0), uint(1, 1, 1)], &[2], true),
        [4, 1, 1]
    );
    // An integer beside a read float of another mode: undefined, which
    // dxil.dll takes before the row's other elements.
    assert_eq!(
        modes(vec![position(), uint(0, 1, 0), float(1, 2, 1)], &[2], true),
        [4, 0, 2]
    );
    // Outputs: their mode is the next stage's to say.
    assert_eq!(
        modes(
            vec![position(), uint(0, 1, 0), float(1, 2, 1)],
            &[1, 2],
            false
        ),
        [4, 1, 1]
    );
    assert_eq!(
        modes(
            vec![position(), float(0, 2, 0), float(1, 4, 1)],
            &[1, 2],
            false
        ),
        [4, 2, 2]
    );
    // Rows dxil.dll takes are left as they are: an undefined element
    // before a defined one, and one mode.
    assert_eq!(
        modes(vec![position(), uint(0, 0, 0), float(1, 2, 1)], &[], true),
        [4, 0, 2]
    );
    assert_eq!(
        modes(vec![position(), float(0, 2, 0), float(1, 2, 1)], &[], false),
        [4, 2, 2]
    );
    // Two modes read in one row: left to dxil.dll, which refuses them.
    assert_eq!(
        modes(
            vec![position(), float(0, 2, 0), float(1, 3, 1)],
            &[1, 2],
            true
        ),
        [4, 2, 3]
    );
}

/// The rows of [`a_row_gets_one_interpolation_mode`] in finished
/// containers: dxil.dll takes each.
#[test]
fn rows_of_one_mode_validate() {
    let ps = [
        (
            "uint and float unread",
            vec![position(), uint(0, 1, 0), float(1, 2, 1)],
            vec![],
        ),
        (
            "float unread beside the front face",
            vec![position(), float(0, 2, 0), front_face(1, 1)],
            vec![],
        ),
        (
            "linear read, centroid unread",
            vec![position(), float(0, 2, 0), float(1, 3, 1)],
            vec![1],
        ),
        (
            "noperspective read, linear unread",
            vec![position(), float(0, 4, 0), float(1, 2, 1)],
            vec![1],
        ),
        (
            "linear unread, uint read",
            vec![position(), float(0, 2, 0), uint(1, 1, 1)],
            vec![2],
        ),
        (
            "uint unread, then linear read",
            vec![position(), uint(0, 1, 0), float(1, 2, 1)],
            vec![2],
        ),
    ];
    for (what, inputs, reads) in ps {
        validates(&format!("pixel inputs: {what}"), &pixel(inputs, &reads));
    }
    let vs = [
        (
            "uint and float",
            vec![position(), uint(0, 1, 0), float(1, 2, 1)],
        ),
        (
            "float and uint",
            vec![position(), float(0, 2, 0), uint(1, 1, 1)],
        ),
        (
            "linear and noperspective",
            vec![position(), float(0, 2, 0), float(1, 4, 1)],
        ),
    ];
    for (what, outputs) in vs {
        validates(&format!("vertex outputs: {what}"), &vertex(outputs));
    }
}

/// What stays refused, and why: an integer after a read element of another
/// mode in its row can be neither that mode (integers are constant or
/// undefined) nor undefined (the row has a mode by then); its row needs
/// the read lowered otherwise (as a constant input and barycentric
/// weights). The test keeps dxil.dll's rules the row pass relies on in
/// sight: if a later dxil.dll takes this, the pass can do more.
#[test]
fn an_integer_after_a_read_float_of_another_mode_stays_refused() {
    let ps = pixel(vec![position(), float(0, 2, 0), uint(1, 1, 1)], &[1]);
    if let Some(r) = verdict("an integer after a linear float", &ps) {
        let e = r.expect_err("dxil.dll took an integer after a linear float in one row");
        assert!(e.contains("interpolation mode that differs"), "{e}");
    }
}

/// A console pixel program whose attribute rows hold a flat channel the
/// program reads (`v_interp_mov` P0) beside one it does not read, and an
/// uint beside a float, neither read: translated whole, each row has one
/// mode and dxil.dll takes the container.
#[test]
fn a_console_row_of_a_flat_channel_and_others_validates() {
    use crate::native::{NativeProgram, PsInput, StageIo, Usage};
    let words = [
        0xC80E_0002u32, // v_interp_mov_f32 v3, p0, attr0.x
        0xF800_180F,
        0x0303_0303, // exp mrt0, v3, v3, v3, v3 done vm
        0xBF81_0000,
    ];
    let input = |row: u8| PsInput {
        usage: Usage::Generic,
        row,
        mask: 0b0011,
        attr: row,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    };
    let native = NativeProgram {
        stage: crate::HwStage::Ps,
        wave64: false,
        float_mode: 0,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![input(0), input(1)],
            input_addr: 0,
            rsrc2: 0,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let at = |e: SigElement, row: u32| SigElement {
        row: Some(row),
        interp: 0,
        ..e
    };
    let isg1 = container::signature(
        &[
            at(float(0, 0, 0), 0),
            at(float(1, 0, 1), 0),
            at(uint(2, 0, 0), 1),
            at(float(3, 0, 1), 1),
        ],
        true,
    );
    let mut target = el("SV_Target", 0, 16, 9, 0, Some(0), 0, 4);
    target.usage = 0xf;
    let osg1 = container::signature(&[target], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let out = super::translate_program(&c, &native).unwrap_or_else(|(step, e)| {
        panic!("{}: {e}", step.name());
    });
    validates("the console pixel program", &out.dxil);
}

/// A vertex shader writing one dword to a raw UAV of the view heap, as the
/// emitter makes it for a console vertex program's buffer store, with
/// ShaderFlags `extra` besides those the shader collects.
fn vertex_writing_a_heap_uav(extra: u64) -> Vec<u8> {
    let mut s = Shader::new(false);
    let b = s.entry;
    let index = s.m.i32(0);
    let h = s.heap_handle(b, index, false, false, (11 | 0x1000, 0));
    let iu = s.m.undef(s.t.i32);
    let (zero, mask, align) = (s.m.i32(0), s.m.i8(1), s.m.i32(4));
    let args = [h, zero, iu, zero, iu, iu, iu, mask, align];
    let void = s.t.void;
    s.op(b, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
    s.flags |= extra;
    let mut pos = position();
    pos.usage = 0xf;
    for col in 0..4u8 {
        let zero = s.m.f32(0);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), zero];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, void, 5, &args);
    }
    s.outputs = vec![pos];
    s.m.terminate(b, Term::Ret);
    s.finish()
}

/// VIctory #1977 sets ShaderFlags bit 16 (UAVs at every stage, SFI0 0x4)
/// for vertex and geometry programs that write the UAVs its instrumentation
/// binds by register. The emitter binds no UAV by register: a vertex
/// program writes through the view heap, and for such a UAV dxil.dll counts
/// the heap and nothing else. It takes the module as the emitter declares
/// it, and refuses it with bit 16 ("Flags must match usage"): the change
/// does not apply here.
#[test]
fn a_vertex_shader_writing_a_heap_uav_declares_no_uavs_at_every_stage() {
    validates("a heap UAV store", &vertex_writing_a_heap_uav(0));
    if let Some(r) = verdict("with bit 16", &vertex_writing_a_heap_uav(1 << 16)) {
        let e = r.expect_err("dxil.dll took bit 16 for a heap UAV");
        assert!(e.contains("Flags must match usage"), "{e}");
    }
}
