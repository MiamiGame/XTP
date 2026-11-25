//! Control flow: each block's exit (branches, loops and their iteration
//! limit, the end of the program), the blocks that split edges, and the
//! phis, filled once every block is lowered.
//!
//! A lane stopped by the iteration limit leaves the program as killed,
//! through a conditional back edge as through an unconditional one: each
//! back edge goes through a latch of its own that sends a lane at the limit
//! to the abort block (Marvisak/VIctory#1483). Where a group barrier can
//! follow a loop, the lanes that end keep passing group barriers until
//! every thread of the group has ended ([`Drain`]).

use std::collections::{HashMap, HashSet};

use super::{Lower, Stop, Val, Why, unsupported};
use crate::emit::bitcode::{Attr, Bin, Rmw, Term, V, pred};
use crate::gfx10::{Encoding, Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// Lanes that end and keep passing group barriers.
///
/// A lane stopped by the iteration limit returns. A group barrier that the
/// other lanes reach later would wait for it: D3D12 does not define a
/// group sync once some threads have returned, and VItoriaBr `808a209`
/// reports it hanging. So in a program where a group barrier can follow a
/// loop ([`barrier_after_loop`]), every lane that ends, stopped by the limit
/// or at the program's end, counts itself ended and passes group barriers
/// until every thread of the group has ended; then all of them return after
/// the same barrier (item 94's form, VIctory #1433, for any flow).
///
/// The decision is exact. Every barrier is one that all threads pass, so
/// between two barriers all threads have passed the same number of them;
/// each lane carries that number (an SSA value, from each block's
/// `s_barrier`s). A lane that ends having passed `c` barriers adds 1 to the
/// count of parity `c % 2`; after each later barrier it reads only the
/// count of the other parity, which no thread writes before the next
/// barrier. The two counts it read on either side of a barrier sum to the
/// threads that ended before it, the same in every thread.
// shortcut: where a barrier is the wave's (AMD), a lane that ended waits,
// masked, while the rest of its wave goes on through barriers, so it counts
// fewer barriers than its wave passed, and its count can land in the parity
// others are reading. Threads may then return a barrier apart; there a wave
// that has ended leaves the barrier, so none is waited for, and every count
// still reaches the group's size.
#[derive(Default)]
pub(super) struct Drain {
    /// Whether the program's lanes drain (none: not decided yet; decided at
    /// the entry block's exit).
    on: Option<bool>,
    /// The barriers a lane has passed.
    passed: Carried,
    /// The two group-shared counts of the threads that ended, by the parity
    /// of the barriers they had passed.
    ended: Option<V>,
    /// The block an ending lane enters, and the phi of its barrier count.
    block: Option<(usize, V)>,
    /// That phi's incoming.
    incoming: Vec<(V, usize)>,
}

/// The back edges a lane has taken in all loops together, against
/// [`crate::emit::loop_budget`]: the loop limits count per loop and per
/// entry into it, so loops inside loops multiply theirs. A lane that takes
/// the budget's back edge leaves as a lane at a loop's limit does. Only in
/// programs where the loops together can pass the budget
/// ([`loops_can_pass`]).
#[derive(Default)]
pub(super) struct Budget {
    /// Whether the program's lanes count their back edges (none: not
    /// decided yet; decided at the entry block's exit).
    on: Option<bool>,
    /// The back edges a lane has taken.
    taken: Carried,
}

/// A count each lane carries from block to block ([`Drain`]'s barriers,
/// [`Budget`]'s back edges): a phi where each IR block starts (0 in the
/// entry block), its value where the block ends, and the value a back edge
/// carries instead.
#[derive(Default)]
struct Carried {
    /// Per IR block: the phi (filled by [`Lower::fill_phis`]).
    entry: Vec<(usize, V)>,
    /// Per IR block: the count where it ends.
    exit: HashMap<usize, V>,
    /// Per back edge (IR block, header), where it carries another count.
    back: HashMap<(usize, usize), V>,
}

/// Which [`Carried`] count.
#[derive(Clone, Copy)]
enum Count {
    Barriers,
    BackEdges,
}

/// Whether a lane can take more back edges than `budget` in the loops of
/// `p` together (`headers`, each with its own limit, `limit`): where one
/// loop is inside another (its passes multiply), or where the limits of
/// loops one after the other add up to it. Unreachable blocks are
/// `usize::MAX` in `rpo`.
pub(super) fn loops_can_pass(
    p: &Program,
    rpo: &[usize],
    headers: &[usize],
    limit: impl Fn(usize) -> u32,
    budget: u32,
) -> bool {
    let reachable = |q: usize| rpo[q] != usize::MAX;
    let nested = headers.iter().any(|&h| {
        // The loop's body: what reaches a latch without passing the header.
        let mut body = HashSet::from([h]);
        let mut work: Vec<usize> = p.blocks[h]
            .preds
            .iter()
            .copied()
            .filter(|&q| reachable(q) && rpo[h] <= rpo[q])
            .collect();
        while let Some(b) = work.pop() {
            if body.insert(b) {
                work.extend(p.blocks[b].preds.iter().filter(|&&q| reachable(q)));
            }
        }
        headers.iter().any(|&g| g != h && body.contains(&g))
    });
    // Each loop runs once otherwise: its limit's back edges at most.
    let total: u64 = headers
        .iter()
        .map(|&h| u64::from(limit(h).saturating_sub(1)))
        .sum();
    nested || total >= u64::from(budget)
}

#[cfg(test)]
thread_local! {
    /// Whether the program this thread lowered last drains (none: not
    /// decided). WARP does not wait at a group barrier for a thread that has
    /// returned, so no dispatch there shows what draining is for.
    pub(super) static DRAINS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether a group barrier (a block where `barrier` holds) is reachable from
/// one of the loop `headers`: a lane the limit stops in that loop leaves
/// before it, or before the next pass through it.
pub(super) fn barrier_after_loop(
    p: &Program,
    headers: impl IntoIterator<Item = usize>,
    barrier: impl Fn(usize) -> bool,
) -> bool {
    let mut seen = vec![false; p.blocks.len()];
    let mut work: Vec<usize> = headers.into_iter().collect();
    while let Some(b) = work.pop() {
        if std::mem::replace(&mut seen[b], true) {
            continue;
        }
        if barrier(b) {
            return true;
        }
        work.extend(p.blocks[b].exit.successors());
    }
    false
}

/// Headers of the loops a lane leaves once an induction variable passes a
/// value the loop does not change: a header phi that every iteration adds
/// the same constant to, compared with a constant or a loop-invariant
/// value by a test the lane passes every iteration (at the header, or at
/// the loop's only latch), and staying in the loop only while the variable
/// has not passed that value (`i < n` going up, `i > n` going down). Their
/// iteration count is the program's own, not data the host may not have
/// mirrored, so they keep the higher limit
/// ([`crate::emit::counted_loop_limit`]).
///
/// A loop that leaves only once the variable *equals* its bound (`i != n`,
/// even by steps of 1, the form compilers give `i < n`) is not one: where
/// the variable starts past the bound (a bound of 0 read from a buffer the
/// host has not written, or the console's own guard left out), it never
/// equals it before wrapping around, so the loop runs to its limit in every
/// lane (VIctory #71's pixel loop). It keeps the general
/// limit.
// shortcut: the bound itself may come from such data (a count in a buffer
// the host has not written), and then the loop runs to the higher limit.
pub(super) fn counted_loops(p: &Program, st: &Structured) -> HashSet<usize> {
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; p.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        rpo[b] = n;
    }
    let mut found = HashSet::new();
    for &h in &order {
        let latches: Vec<usize> = p.blocks[h]
            .preds
            .iter()
            .copied()
            .filter(|&q| rpo[q] != usize::MAX && rpo[h] <= rpo[q])
            .collect();
        if latches.is_empty() {
            continue;
        }
        let l = Counted::new(p, h, &latches, &rpo);
        let mut tests = vec![h];
        if let [latch] = latches[..]
            && latch != h
        {
            tests.push(latch);
        }
        if tests.into_iter().any(|b| l.exit_counts(st, b)) {
            found.insert(h);
        }
    }
    found
}

/// A compare's test of its first operand against its second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Test {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Test {
    fn parse(s: &str) -> Option<Test> {
        Some(match s {
            "lt" => Test::Lt,
            "le" => Test::Le,
            "gt" => Test::Gt,
            "ge" => Test::Ge,
            "eq" => Test::Eq,
            "ne" | "lg" => Test::Ne,
            _ => return None,
        })
    }

    /// The same test with the operands swapped.
    fn swapped(self) -> Test {
        match self {
            Test::Lt => Test::Gt,
            Test::Le => Test::Ge,
            Test::Gt => Test::Lt,
            Test::Ge => Test::Le,
            t => t,
        }
    }

    fn negated(self) -> Test {
        match self {
            Test::Lt => Test::Ge,
            Test::Le => Test::Gt,
            Test::Gt => Test::Le,
            Test::Ge => Test::Lt,
            Test::Eq => Test::Ne,
            Test::Ne => Test::Eq,
        }
    }
}

/// An integer operand: a value or a constant.
#[derive(Clone, Copy)]
enum Term32 {
    Value(Value),
    Const(i64),
}

fn term(inst: &Inst, op: &Operand) -> Option<Term32> {
    match *op {
        Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)).map(Term32::Value),
        Operand::Vector { reg, .. } => inst.read(Reg::V(reg)).map(Term32::Value),
        Operand::Const(c @ 128..=208) => {
            crate::resource::inline(c).map(|x| Term32::Const(i64::from(x as i32)))
        }
        Operand::Literal(x) => Some(Term32::Const(i64::from(x as i32))),
        _ => None,
    }
}

/// A loop for [`counted_loops`].
struct Counted<'a> {
    p: &'a Program,
    header: usize,
    body: HashSet<usize>,
    /// Block and instruction (none: a phi) of each value the loop writes.
    defs: HashMap<Value, (usize, Option<usize>)>,
}

/// How far [`Counted::resolve`] follows phis.
const PHI_DEPTH: u32 = 8;

impl<'a> Counted<'a> {
    fn new(p: &'a Program, header: usize, latches: &[usize], rpo: &[usize]) -> Self {
        let mut body = HashSet::from([header]);
        let mut work = latches.to_vec();
        while let Some(b) = work.pop() {
            if body.insert(b) {
                work.extend(p.blocks[b].preds.iter().filter(|&&q| rpo[q] != usize::MAX));
            }
        }
        let mut defs = HashMap::new();
        for &b in &body {
            for phi in &p.blocks[b].phis {
                defs.insert(phi.value, (b, None));
            }
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs.insert(v, (b, Some(i)));
                }
            }
        }
        Counted {
            p,
            header,
            body,
            defs,
        }
    }

    /// `v`, or the one value a phi of the loop merges with itself (a
    /// register the loop does not write still gets phis).
    fn resolve(&self, v: Value, depth: u32) -> Value {
        let Some(&(b, None)) = self.defs.get(&v) else {
            return v;
        };
        let Some(phi) = self.p.blocks[b].phis.iter().find(|q| q.value == v) else {
            return v;
        };
        if depth == 0 {
            return v;
        }
        let mut other = None;
        for &a in &phi.args {
            let a = self.resolve(a, depth - 1);
            if a == v {
                continue;
            }
            match other {
                None => other = Some(a),
                Some(o) if o == a => {}
                Some(_) => return v,
            }
        }
        other.unwrap_or(v)
    }

    fn invariant(&self, v: Value) -> bool {
        !self.defs.contains_key(&self.resolve(v, PHI_DEPTH))
    }

    fn def(&self, v: Value) -> Option<&'a Inst> {
        match *self.defs.get(&v)? {
            (b, Some(i)) => Some(&self.p.blocks[b].insts[i]),
            (_, None) => None,
        }
    }

    /// `q` as `x + step`: an addition or subtraction of a nonzero constant
    /// that writes `q` (its result, not a carry).
    fn added(&self, q: Value) -> Option<(Value, i64)> {
        let inst = self.def(q)?;
        let name = inst.name();
        let sum = |a: Term32, b: Term32, sign: i64| match (a, b) {
            (Term32::Value(x), Term32::Const(c)) => Some((self.resolve(x, PHI_DEPTH), sign * c)),
            (Term32::Const(c), Term32::Value(x)) if sign > 0 => {
                Some((self.resolve(x, PHI_DEPTH), c))
            }
            _ => None,
        };
        let (x, step) = match &inst.fields {
            Fields::Valu(v) => {
                let plain = v.neg == 0
                    && v.abs == 0
                    && v.sdwa.is_none()
                    && v.dpp.is_none()
                    && !v.clamp
                    && v.omod == 0;
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return None;
                };
                if !plain || inst.write(Reg::V(reg)) != Some(q) {
                    return None;
                }
                let (a, b) = (term(inst, v.src.first()?)?, term(inst, v.src.get(1)?)?);
                match name {
                    "v_add_nc_u32" | "v_add_co_u32" | "v_add_nc_i32" | "v_add_u32" => sum(a, b, 1)?,
                    "v_sub_nc_u32" | "v_sub_co_u32" | "v_sub_nc_i32" => sum(a, b, -1)?,
                    "v_subrev_nc_u32" | "v_subrev_co_u32" => sum(b, a, -1)?,
                    _ => return None,
                }
            }
            Fields::Sop { dst, src, imm } => {
                let Some(Operand::Scalar { code, .. }) = *dst else {
                    return None;
                };
                if inst.write(Reg::S(code)) != Some(q) {
                    return None;
                }
                match (name, imm) {
                    ("s_addk_i32", Some(k)) => {
                        let old = inst.read(Reg::S(code))?;
                        (self.resolve(old, PHI_DEPTH), i64::from(*k as i16))
                    }
                    ("s_add_u32" | "s_add_i32", None) => {
                        sum(term(inst, src.first()?)?, term(inst, src.get(1)?)?, 1)?
                    }
                    ("s_sub_u32" | "s_sub_i32", None) => {
                        sum(term(inst, src.first()?)?, term(inst, src.get(1)?)?, -1)?
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        (step != 0).then_some((x, step))
    }

    /// The constant every iteration adds to `x`, if `x` is a phi of the
    /// header whose back edges all bring it plus that constant, or such a
    /// phi plus it.
    fn induction(&self, x: Value) -> Option<i64> {
        let x = self.resolve(x, PHI_DEPTH);
        match *self.defs.get(&x)? {
            (b, None) if b == self.header => {
                let block = &self.p.blocks[b];
                let phi = block.phis.iter().find(|q| q.value == x)?;
                let mut step = None;
                for (n, q) in block.preds.iter().enumerate() {
                    if !self.body.contains(q) {
                        continue;
                    }
                    let back = self.resolve(*phi.args.get(n)?, PHI_DEPTH);
                    let (from, s) = self.added(back)?;
                    if from != x || *step.get_or_insert(s) != s {
                        return None;
                    }
                }
                step
            }
            (_, None) => None,
            (_, Some(_)) => {
                let (from, s) = self.added(x)?;
                (self.induction(from)? == s).then_some(s)
            }
        }
    }

    /// Whether a lane where `atom` equals `holds` has an induction variable
    /// that has not passed a loop-invariant bound.
    fn bounds(&self, atom: Atom, holds: bool) -> bool {
        let (Atom::Lane(v) | Atom::Uniform(v)) = atom else {
            return false;
        };
        let Some(inst) = self.def(v) else {
            return false;
        };
        let name = inst.name();
        let rest = ["v_cmpx_", "v_cmp_", "s_cmpk_", "s_cmp_"]
            .into_iter()
            .find_map(|prefix| name.strip_prefix(prefix));
        let Some((test, ty)) = rest.and_then(|r| r.split_once('_')) else {
            return false;
        };
        let Some(test) = Test::parse(test).filter(|_| matches!(ty, "u32" | "i32")) else {
            return false;
        };
        let ops = match &inst.fields {
            Fields::Valu(x) if x.neg == 0 && x.abs == 0 && x.sdwa.is_none() && x.dpp.is_none() => {
                match (x.src.first(), x.src.get(1)) {
                    (Some(a), Some(b)) => [term(inst, a), term(inst, b)],
                    _ => return false,
                }
            }
            Fields::Sop {
                src, imm: Some(k), ..
            } => {
                let k = if ty == "i32" {
                    i64::from(*k as i16)
                } else {
                    i64::from(*k)
                };
                [
                    src.first().and_then(|a| term(inst, a)),
                    Some(Term32::Const(k)),
                ]
            }
            Fields::Sop { src, imm: None, .. } => match (src.first(), src.get(1)) {
                (Some(a), Some(b)) => [term(inst, a), term(inst, b)],
                _ => return false,
            },
            _ => return false,
        };
        [(0, 1, test), (1, 0, test.swapped())]
            .into_iter()
            .any(|(at, other, test)| {
                let (Some(Term32::Value(i)), Some(n)) = (ops[at], ops[other]) else {
                    return false;
                };
                let fixed = match n {
                    Term32::Const(_) => true,
                    Term32::Value(n) => self.invariant(n),
                };
                let Some(step) = self.induction(i).filter(|_| fixed) else {
                    return false;
                };
                match if holds { test } else { test.negated() } {
                    Test::Lt | Test::Le => step > 0,
                    Test::Gt | Test::Ge => step < 0,
                    // Leaving only on equality (see `counted_loops`).
                    Test::Ne | Test::Eq => false,
                }
            })
    }

    /// Whether the branch ending block `b` keeps a lane in the loop only
    /// while a test of [`Self::bounds`] holds.
    fn exit_counts(&self, st: &Structured, b: usize) -> bool {
        let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = st.exits[b]
        else {
            return false;
        };
        let stay = match (self.body.contains(&taken), self.body.contains(&not_taken)) {
            (true, false) => true,
            (false, true) => false,
            _ => return false,
        };
        st.bools.support(when).into_iter().any(|atom| {
            let Some(a) = st.bools.var(atom) else {
                return false;
            };
            // Every lane that stays has the atom's value `holds`.
            [true, false].into_iter().any(|holds| {
                !st.bools.meet(&[(when, stay), (a, !holds)]) && self.bounds(atom, holds)
            })
        })
    }
}

impl<'a> Lower<'a> {
    pub(super) fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        // Where lanes that end drain: the barriers a lane has passed when
        // it leaves this block. The entry block first clears the counts of
        // ended threads, behind a barrier every thread passes before any
        // can end.
        let passed = if self.drains() {
            let mut cleared = 0;
            if self.rpo[b] == 0 {
                let ended = self.ended_counts();
                let c = self.cur;
                let z = self.s.m.i32(0);
                for k in [0, 1] {
                    let at = self.s.m.i32(k);
                    let count = self.s.m.element(c, ended, at);
                    self.s.m.store(c, count, z);
                }
                self.group_barrier(c);
                cleared = 1;
            }
            Some(self.passed(b, cleared))
        } else {
            None
        };
        // Where loops have a budget: the back edges a lane has taken (none
        // within a block); a back edge adds its own ([`Self::count_iteration`]).
        if self.budgeted() {
            let taken = self.entering(Count::BackEdges, b);
            self.budget.taken.exit.insert(b, taken);
        }
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, self.cur);
                let abort = self.abort_block(self.cur, passed);
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } if self.console_cf => {
                let cond = self.console_branch(b).map_err(|w| unsupported(offset, w))?;
                self.branch_term(b, cond, taken, not_taken, passed)
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                self.branch_term(b, cond, taken, not_taken, passed)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                match passed {
                    Some(n) => Term::Br(self.drain_from(self.cur, n)),
                    None => Term::Ret,
                }
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// The terminator of block `b` branching to `taken` where `cond` holds
    /// (`passed`: the barriers a lane has passed there, where lanes drain).
    fn branch_term(
        &mut self,
        b: usize,
        cond: V,
        taken: usize,
        not_taken: usize,
        passed: Option<V>,
    ) -> Term {
        let back = |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
        // Critical edges get a block of their own, so loops have
        // dedicated exits and preheaders as DXC emits them.
        let (mut t, mut f) = (self.edge(b, taken), self.edge(b, not_taken));
        // A back edge goes through a latch of its own, which goes on below
        // the iteration limit and leaves through the abort block at it.
        // (Folding the limit into the branch condition sent a lane at the
        // limit down the loop's exit, on through the program with the
        // loop's values unfinished: it stored them.)
        // Ported from Marvisak/VIctory#1483 (head 0daabcc) by spy0eye-beep.
        if back(self, taken) {
            t = self.limited_latch(b, taken, passed);
        }
        if back(self, not_taken) {
            f = if not_taken == taken {
                t
            } else {
                self.limited_latch(b, not_taken, passed)
            };
        }
        for (to, via) in [(taken, t), (not_taken, f)] {
            if !back(self, to) {
                // From an edge block, or from here.
                let from = if Some(via) == self.block[to] {
                    self.cur
                } else {
                    via
                };
                self.enter_loop(b, to, from);
            }
        }
        Term::CondBr(cond, t, f)
    }

    /// Console flow: whether block `b`'s `s_cbranch_*` is taken, the same
    /// in both halves (wave-wide mask tests through [`Self::any_lane`]).
    fn console_branch(&mut self, b: usize) -> Result<V, Why> {
        let inst = self.x.p.blocks[b].insts.last().ok_or("a branch")?;
        let name = inst.name();
        let c = self.cur;
        let (any, set) = match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let v = inst.read(Reg::Scc).ok_or("a branch's SCC")?;
                let scc = self.scc_bit(v)?;
                (scc, name == "s_cbranch_scc1")
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let reg = if name.contains("vcc") {
                    crate::ir::VCC
                } else {
                    EXEC
                };
                let lo = inst.read(Reg::S(reg)).ok_or("a branch's mask")?;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&lo)
                    .ok_or("a branch on an untracked mask")?;
                (self.any_lane(m)?, name.ends_with("nz"))
            }
            _ => return Err("a branch the console flow does not model"),
        };
        Ok(if set {
            any
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, any, t)
        })
    }

    /// Console flow: an SCC value as a bit. A mask operation's SCC (result
    /// not zero) is whether some lane of the console wave has its result's
    /// bit: for `s_and_saveexec` and the like, the new EXEC's, which is
    /// not the first mask the instruction writes (the saved EXEC is).
    /// Structurisation says which mask it is ([`Structured::sccs`]).
    ///
    /// [`Structured::sccs`]: crate::structure::Structured::sccs
    fn scc_bit(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&Val::Bit(x)) = self.vals.get(&v) {
            return Ok(x);
        }
        let &(mask, negated) = self
            .x
            .st
            .sccs
            .get(&v)
            .ok_or("an SCC the emitter did not compute")?;
        // (A compare either way, no `xor` of one: see `branch_term`.)
        let c = self.cur;
        let bit = self.boolean(mask, c)?;
        let (lo, hi) = self.console_ballot(c, bit);
        let or = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        let p = if negated { pred::EQ } else { pred::NE };
        Ok(self.s.m.cmp(c, p, or, z))
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    pub(super) fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// The iteration limit of the loop at header `h`:
    /// [`LOOP_LIMIT`](super::LOOP_LIMIT), the spin-wait limit for a loop
    /// that only waits for memory to change (`spin.rs`), or the higher
    /// limit of a counted loop ([`counted_loops`]).
    fn limit_of(&self, h: usize) -> u32 {
        if self.spin.contains(&h) {
            crate::emit::spin_limit()
        } else if self.counted.contains(&h) {
            crate::emit::counted_loop_limit()
        } else {
            crate::emit::loop_limit()
        }
    }

    /// A back edge from the current IR block to header `to`, entering it
    /// from DXIL block `from`: the count goes up (computed here); the
    /// result is whether it is below the loop's limit ([`Self::limit_of`];
    /// go on; at it: stop) and, where loops have a budget ([`Budget`]),
    /// the lane's back edges below it.
    pub(super) fn count_iteration(&mut self, to: usize, from: usize) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        let limit = self.limit_of(to);
        #[cfg(test)]
        self.limits.push(limit);
        let limit = self.s.m.i32(limit);
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, from));
        let ok = self.s.m.cmp(c, pred::IULT, next, limit);
        let b = self.cur_ir;
        let Some(&taken) = self.budget.taken.exit.get(&b) else {
            return ok;
        };
        let taken = self.s.m.bin(c, Bin::Add, taken, one);
        self.budget.taken.back.insert((b, to), taken);
        let budget = self.s.m.i32(crate::emit::loop_budget());
        let left = self.s.m.cmp(c, pred::IULT, taken, budget);
        self.s.m.bin(c, Bin::And, ok, left)
    }

    /// The latch of a conditional back edge from IR block `from` to header
    /// `to`: back to the header below the iteration limit, to the abort
    /// block at it. The header's phis take this edge's values from it.
    // Ported from Marvisak/VIctory#1483 (head 0daabcc) by spy0eye-beep.
    fn limited_latch(&mut self, from: usize, to: usize, passed: Option<V>) -> usize {
        let latch = self.s.m.new_block();
        let ok = self.count_iteration(to, latch);
        let abort = self.abort_block(latch, passed);
        let header = self.block[to].expect("reachable");
        self.s.m.terminate(latch, Term::CondBr(ok, header, abort));
        self.edges.insert((from, to), latch);
        latch
    }

    /// The block a lane stopped by [`LOOP_LIMIT`](super::LOOP_LIMIT) leaves
    /// through, from DXIL block `from`: where lanes drain (`passed`, the
    /// barriers it has passed), [`Self::drain_from`]; elsewhere the end of
    /// the program (a pixel discards).
    // shortcut: limit hits are not reported to the host.
    pub(super) fn abort_block(&mut self, from: usize, passed: Option<V>) -> usize {
        if let Some(n) = passed {
            return self.drain_from(from, n);
        }
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// Whether the program's lanes drain ([`Drain`]): a compute or mesh
    /// program where a group barrier can follow a loop. Not the console
    /// flow in halves: there both halves of the group's one console wave
    /// follow one flow (the limit stops them together), and their
    /// exchanges add barriers that no block counts. Nor a wave64 mesh
    /// program on a host without 64-lane waves: its first-lane exchanges
    /// (`mesh.rs`) add barriers that no block counts either.
    // shortcut: nor a program whose LDS leaves no room in DXIL's 32 KiB (a
    // mesh program's 28 KiB) for the two counts (6 compute programs of the
    // pack, of 32 KiB): a lane the limit stops there returns, as before,
    // rather than the program being refused.
    fn drains(&mut self) -> bool {
        if let Some(on) = self.drain.on {
            return on;
        }
        let on = self.s.threads.is_some()
            && !self.s.pixel
            && (!(self.console_cf && self.split) || self.s.m.paired_lanes())
            && !self.mesh_halves()
            && self.s.lds_dwords + 2
                <= if self.s.mesh.is_some() {
                    super::mesh::MESH_LDS_DWORDS
                } else {
                    8192
                }
            && barrier_after_loop(self.x.p, self.counters.keys().copied(), |b| {
                self.barriers_in(b) > 0
            });
        self.drain.on = Some(on);
        #[cfg(test)]
        DRAINS.set(Some(on));
        on
    }

    /// The group barriers IR block `b` passes: its `s_barrier`s, none in a
    /// mesh block that leaves them out, nor inside a whole-wave section
    /// (whose simulation stands for its instructions).
    fn barriers_in(&self, b: usize) -> u32 {
        if self.no_barrier.contains(&b) {
            return 0;
        }
        let insts = &self.x.p.blocks[b].insts;
        let (mut n, mut i) = (0, 0);
        while i < insts.len() {
            n += u32::from(insts[i].name() == "s_barrier");
            i = self
                .sections
                .get(&b)
                .and_then(|s| s.get(&i))
                .map_or(i + 1, |s| s.end);
        }
        n
    }

    /// The barriers a lane has passed when it leaves IR block `b`
    /// ([`Drain`]; `extra`: those its exit adds): what it entered with (a
    /// phi; 0 in the entry block), plus the block's.
    fn passed(&mut self, b: usize, extra: u32) -> V {
        let start = self.entering(Count::Barriers, b);
        let n = self.barriers_in(b) + extra;
        let v = if n == 0 {
            start
        } else {
            let k = self.s.m.i32(n);
            self.s.m.bin(self.cur, Bin::Add, start, k)
        };
        self.drain.passed.exit.insert(b, v);
        v
    }

    /// Whether the program's lanes count their back edges against the
    /// budget ([`Budget`]).
    fn budgeted(&mut self) -> bool {
        if let Some(on) = self.budget.on {
            return on;
        }
        let budget = crate::emit::loop_budget();
        let mut headers: Vec<usize> = self.counters.keys().copied().collect();
        headers.sort_unstable();
        let on = budget > 0
            && loops_can_pass(self.x.p, &self.rpo, &headers, |h| self.limit_of(h), budget);
        self.budget.on = Some(on);
        on
    }

    fn carried(&mut self, count: Count) -> &mut Carried {
        match count {
            Count::Barriers => &mut self.drain.passed,
            Count::BackEdges => &mut self.budget.taken,
        }
    }

    /// `count` where IR block `b` starts: a phi (0 in the entry block).
    fn entering(&mut self, count: Count, b: usize) -> V {
        if self.rpo[b] == 0 {
            return self.s.m.i32(0);
        }
        let at = self.block[b].expect("reachable");
        let phi = self.s.m.phi(at, self.s.t.i32);
        self.carried(count).entry.push((b, phi));
        phi
    }

    /// The phis of `count`: from each predecessor, its count where it ends,
    /// or the one its back edge carries.
    fn fill_carried(&mut self, count: Count) {
        let p = self.x.p;
        let carried = std::mem::take(self.carried(count));
        for (b, phi) in carried.entry {
            let mut incoming = Vec::new();
            for &q in &p.blocks[b].preds {
                let (Some(qb), Some(&n)) = (self.tail[q], carried.exit.get(&q)) else {
                    continue;
                };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let n = carried.back.get(&(q, b)).copied().unwrap_or(n);
                incoming.push((n, qb));
            }
            self.s.m.set_phi(phi, incoming);
        }
    }

    /// [`Drain`]'s two group-shared counts of ended threads.
    fn ended_counts(&mut self) -> V {
        if let Some(g) = self.drain.ended {
            return g;
        }
        let arr = self.s.m.array(2, self.s.t.i32);
        let g = self.s.m.global("ended", arr, 3);
        // A mesh program's group-shared size (its PSV) counts them.
        self.s.shared_extra += 8;
        self.drain.ended = Some(g);
        g
    }

    /// A group barrier in DXIL block `at`, with the fences of the
    /// program's own (`s_barrier`).
    fn group_barrier(&mut self, at: usize) {
        let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
        let mode = self.s.m.i32(mode);
        let void = self.s.t.void;
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// The block a lane that ends enters from DXIL block `from`, having
    /// passed `passed` barriers ([`Drain`]).
    fn drain_from(&mut self, from: usize, passed: V) -> usize {
        let (block, _) = self.drain_block();
        self.drain.incoming.push((passed, from));
        block
    }

    /// [`Drain`]'s blocks: the lane counts itself ended, by the parity of
    /// the barriers it has passed, then passes barriers until the counts it
    /// reads on either side of one say that every thread has ended.
    fn drain_block(&mut self) -> (usize, V) {
        if let Some(d) = self.drain.block {
            return d;
        }
        let threads: u32 = self.s.threads.expect("a group").iter().product();
        let ended = self.ended_counts();
        let i32t = self.s.t.i32;
        let (entry, wait, done) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        let passed = self.s.m.phi(entry, i32t);
        let one = self.s.m.i32(1);
        // Every thread that ends before the next barrier has passed as many
        // barriers as this one, so it counts in this parity too.
        let own = self.s.m.bin(entry, Bin::And, passed, one);
        let at = self.s.m.element(entry, ended, own);
        let _ = self.s.m.atomic(entry, Rmw::Add, at, one);
        // The other parity: no thread writes it before the next barrier.
        let other = self.s.m.bin(entry, Bin::Xor, own, one);
        let at = self.s.m.element(entry, ended, other);
        let first = self.s.m.load(entry, at);
        self.s.m.terminate(entry, Term::Br(wait));
        // Before the barrier: `n` barriers passed, and `before` threads
        // ended with a count of parity n + 1 (all of them, up to the next
        // barrier).
        let n = self.s.m.phi(wait, i32t);
        let before = self.s.m.phi(wait, i32t);
        self.group_barrier(wait);
        let next = self.s.m.bin(wait, Bin::Add, n, one);
        // Parity n: the last thread to count in it did so before this
        // barrier. Together, every thread that ended before it.
        let own = self.s.m.bin(wait, Bin::And, n, one);
        let at = self.s.m.element(wait, ended, own);
        let after = self.s.m.load(wait, at);
        let sum = self.s.m.bin(wait, Bin::Add, before, after);
        // Each physical lane counts both of its logical lanes when paired.
        let all = self.s.m.i32(if self.s.m.paired_lanes() {
            2 * threads
        } else {
            threads
        });
        let all_ended = self.s.m.cmp(wait, pred::EQ, sum, all);
        self.s
            .m
            .terminate(wait, Term::CondBr(all_ended, done, wait));
        self.s.m.set_phi(n, vec![(passed, entry), (next, wait)]);
        self.s
            .m
            .set_phi(before, vec![(first, entry), (after, wait)]);
        self.s.m.terminate(done, Term::Ret);
        self.drain.block = Some((entry, passed));
        (entry, passed)
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    pub(super) fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    pub(super) fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        // Where lanes drain: the barriers passed when entering each block,
        // and by each lane that ends; where loops have a budget, the back
        // edges taken.
        self.fill_carried(Count::Barriers);
        self.fill_carried(Count::BackEdges);
        if let Some((_, phi)) = self.drain.block {
            let incoming = std::mem::take(&mut self.drain.incoming);
            self.s.m.set_phi(phi, incoming);
        }
        // The halves' exchanges would add barriers no block counts.
        debug_assert!(self.drain.on != Some(true) || self.xchg.is_none());
        // A phi that only another read phi consumes is live too. Without this, a value with no DXIL form that
        // reaches a read through two joins is taken for dead at the first join and replaced by zero below, and the
        // lowering accepts what it refuses when the same value is read directly.
        let mut work: Vec<Value> = self.read_phis.iter().copied().collect();
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &arg in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&arg) && self.read_phis.insert(arg) {
                    work.push(arg);
                }
            }
        }
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            // Where the program reads the joined value, an edge whose value
            // has no DXIL form brings an undefined one only if no lane that
            // takes it reads the register before writing it again
            // (`unread_on_edge`). A join nothing reads takes any value.
            let read = self.read_phis.contains(&phi.value);
            // A join of computed numbers (`dead_at_join`): an SGPR, not a
            // mask itself, some of whose incoming values are numbers and
            // none a constant.
            let st = self.x.st;
            let numbers = !bit
                && !st.masks.contains_key(&phi.value)
                && !st.highs.contains_key(&phi.value)
                && phi
                    .args
                    .iter()
                    .any(|a| matches!(self.vals.get(a), Some(Val::I32(_))))
                && !phi
                    .args
                    .iter()
                    .any(|a| matches!(self.vals.get(a), Some(Val::I32(V::Const(_)))));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                if self.read_phis.contains(&phi.value) && !self.vals.contains_key(&arg) {
                    // Une constante déportée peut servir uniquement sur cette
                    // arête. Sa lecture à l'entrée domine tous les prédécesseurs.
                    if !self.x.res.spill_loads.is_empty()
                        && let Some(&(param, dword)) = self.params.get(&arg)
                    {
                        let val = self.user(param, dword);
                        self.vals.insert(arg, val);
                    } else {
                        let _ = self.lazy_barycentric(arg);
                    }
                }
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    // A descriptor's dwords reaching a join that also takes
                    // data: the registers are reused after the descriptor's
                    // last access, so the program reads the joined value
                    // only after the data edge (the console compiler's phi
                    // of a value defined on one edge only). Undefined on
                    // the descriptor's edge where no lane that takes it
                    // reads the register (`unread_on_edge`); a read there
                    // stops the program, as a direct read does. Only dwords
                    // that have no host value (addresses, formats): any
                    // other reason the descriptor load stopped with (a cube
                    // or MSAA image, a constant buffer's descriptor read as
                    // data) stops the program as it would without the join.
                    // shortcut: in the console flow (branches taken
                    // wave-wide) the proof follows every path, as no
                    // condition says which way the wave goes, so a dword
                    // that the program reads only behind a flag still fails
                    // it there; such a join keeps the undefined value
                    // unproven, as before the proof, and counts in
                    // `emit::UNPROVEN_DESCRIPTOR_JOINS`. The 12 ray-tracing
                    // resolve compute programs of the characters
                    // (`CS_RaytracingResolve` f21e9178, 842de0c32bc8dcc5 and
                    // its family) run so. Wrong if the program reads the
                    // descriptor's dwords as data on that edge.
                    (Some(Val::Missing(why)), false)
                        if why == super::smem::DESCRIPTOR_WORD
                            && self.descriptor_dword(arg)
                            && (!read
                                || self.unread_on_edge(phi.reg, q, b)
                                || self.console_cf && unproven_join()) =>
                    {
                        self.s.m.undef(self.s.t.i32)
                    }
                    (
                        Some(Val::Missing(super::mask::LANE_MASK | super::smem::ONLY_DATA)),
                        false,
                    ) if self.unused_pixel_sgpr(phi.reg, q, b) => self.s.m.i32(0),
                    (Some(Val::Missing(_)), false)
                        if numbers
                            && self.dead_at_join(arg, &mut Vec::new())
                            && (!read || self.unread_on_edge(phi.reg, q, b)) =>
                    {
                        self.s.m.undef(self.s.t.i32)
                    }
                    (Some(Val::Missing(super::mask::LANE_MASK)), false)
                        if self.x.console_cf
                            && self.emitted_double_count
                            && self.read_phis.contains(&phi.value)
                            && !self.bad.contains_key(&arg)
                            && (self.x.st.masks.contains_key(&arg)
                                || self.x.st.highs.contains_key(&arg)) =>
                    {
                        // Le vote doit être émis à la définition du masque,
                        // avant de résoudre les jonctions dont il dépend.
                        return Err(Stop::Retry(arg, super::mask::LANE_MASK));
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                // Console flow: every lane arrives by every edge, so a join
                // also takes masks that are constants or built from scalars.
                let arg = &phi.args[n];
                let others = self.console_cf.then(|| self.x.st.others.get(arg)).flatten();
                let m =
                    *self.x.st.masks.get(arg).or(others).ok_or_else(|| {
                        unsupported(offset, "a lane-mask join of an untracked mask")
                    })?;
                let x = self.boolean(m, qb).map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }

    /// Whether `v`, an incoming value of a join of computed numbers, is of
    /// a kind compiled code holds in a register it then reuses for a
    /// number: a descriptor's dword with no host value
    /// ([`super::smem::DESCRIPTOR_WORD`], in every stage, as the rule for a
    /// descriptor's dword joined directly), a lane mask's dword outside
    /// compute (where no ballot gives it as data), or a join of only such
    /// values. A physical register that holds a mask on one path and a
    /// number on the other holds two variables, and the program reads the
    /// joined value only where the number's path set it: behind a flag kept
    /// in a VGPR, 0 on the mask's edge (the skinned and shadow mesh programs
    /// 235e56869c1ea883 at 0x1980, c3ba22ec6fc5dd4a at 0x1b48). The join
    /// takes an undefined value on that edge only where
    /// [`Self::unread_on_edge`] proves it: a program that reads the mask's
    /// side after the join stops, as a direct read does. In compute
    /// (ballots are exact) such a join stays refused, as does one with a
    /// constant (`cond ? ballot : 5`).
    // Ported from fxceless's VIctory fork (commit 578d459) by fxceless
    // (`dead` and its arm in the joins). Here: undefined rather than 0, as
    // the descriptor-word rule above; only where no lane on the edge reads
    // the register; not for masks in compute; not where a number of the
    // join is a constant.
    fn dead_at_join(&self, v: Value, seen: &mut Vec<Value>) -> bool {
        let st = self.x.st;
        let word = matches!(self.vals.get(&v), Some(Val::Missing(w)) if *w == super::smem::DESCRIPTOR_WORD)
            && self.descriptor_dword(v);
        let mask = self.mask_data != super::mask::MaskData::Console
            && (st.masks.contains_key(&v) || st.highs.contains_key(&v));
        if word || mask {
            return true;
        }
        let Some(&(b, i)) = self.phi_at.get(&v) else {
            return false;
        };
        if seen.contains(&v) {
            return true;
        }
        seen.push(v);
        self.x.p.blocks[b].phis[i]
            .args
            .iter()
            .all(|&a| self.dead_at_join(a, seen))
    }

    /// Whether no lane that enters join `b` from `pred` makes what `reg`
    /// held on that edge matter: what the register held there is then no
    /// value the program uses, and the join may take an undefined one.
    ///
    /// Per lane, a path follows only the branches the lane can take with
    /// what it knows ([`Path`]): what holds on entering `pred` and its
    /// branch to the join, the branches taken since, the constants the
    /// join's phis take on that edge (a flag VGPR that is 0 on the mask's
    /// edge and that the program tests before reading the register), and a
    /// comparison already decided on the path made again of the same
    /// values. Vector instructions whose guard cannot hold do not run.
    ///
    /// The value may flow into scalar ALU operations: what they write is
    /// then tainted too, until written again. Taint ends the proof where
    /// anything else reads it (a vector instruction, memory, a message, an
    /// EXEC write), and at a branch whose way it can change: one whose
    /// condition, with what the lane knows, takes both ways and tests a
    /// tainted atom. The material and character pixel programs
    /// (f23ac5187c49056a, the `PedTextured` family) test the register at
    /// the join beside a flag (`flag && word != 0`) whose side alone
    /// decides the branch on the edge where the word is a descriptor's.
    ///
    /// Back at the join (a loop), the lane goes on without the join's
    /// constants. An indirect jump, or a search of more than
    /// [`UNREAD_STEPS`] blocks: no proof. In the console flow (branches
    /// taken wave-wide) every successor is followed, every instruction
    /// runs, any read ends the proof, and only a scalar instruction's write
    /// ends a path.
    fn unread_on_edge(&self, reg: Reg, pred: usize, b: usize) -> bool {
        let (p, st) = (self.x.p, self.x.st);
        let per_lane = !self.console_cf;
        let Some(n) = p.blocks[b].preds.iter().position(|&q| q == pred) else {
            return false;
        };
        let joined: HashMap<Value, u32> = p.blocks[b]
            .phis
            .iter()
            .filter_map(|phi| Some((phi.value, self.word_of(phi.args[n])?)))
            .collect();
        let none = HashMap::new();
        // What lanes entering `pred` know, and its branch to the join.
        // Over a back edge it holds for the iteration that ends.
        let mut start = Taint {
            known: Vec::new(),
            regs: vec![reg],
            values: Vec::new(),
            fresh: true,
        };
        if per_lane && self.rpo[b] > self.rpo[pred] {
            let entry = Path {
                joined: &none,
                at: pred,
                from: None,
                known: &[],
                tainted: &[],
            };
            start.known = self.learn(&entry, st.reach[pred], true);
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = st.exits[pred]
                && taken != not_taken
            {
                let more = self.learn(
                    &Path {
                        known: &start.known,
                        ..entry
                    },
                    when,
                    taken == b,
                );
                start.known.extend(more);
            }
        }
        // What a lane knows entering each block by each edge, and what is
        // tainted there, over all the paths found so far: the facts only
        // shrink and the taint only grows as the search goes on.
        let mut states: HashMap<(usize, usize), Taint> = HashMap::new();
        states.insert((pred, b), start);
        let mut work = vec![(pred, b)];
        let mut steps = 0;
        'paths: while let Some((from, at)) = work.pop() {
            steps += 1;
            if steps > UNREAD_STEPS {
                return false;
            }
            let mut t = states[&(from, at)].clone();
            let block = &p.blocks[at];
            // A phi of the register a path holds tainted is tainted on it
            // (the join's own was tainted at the start).
            if let Some(k) = block.preds.iter().position(|&q| q == from) {
                for phi in &block.phis {
                    if (t.regs.contains(&phi.reg) || t.values.contains(&phi.args[k]))
                        && !t.values.contains(&phi.value)
                    {
                        t.values.push(phi.value);
                    }
                }
            }
            let known = t.known.clone();
            let values = t.values.clone();
            let path = Path {
                joined: if t.fresh { &joined } else { &none },
                at,
                from: Some(from),
                known: &known,
                tainted: &values,
            };
            let last = block.insts.len().saturating_sub(1);
            for (i, inst) in block.insts.iter().enumerate() {
                let guard = if per_lane { st.guards[at][i] } else { None };
                let (runs, skips) = guard.map_or((true, false), |g| self.outcomes(g, &path));
                let scalar = matches!(inst.fields, Fields::Sop { .. } | Fields::Smem { .. });
                // A vector instruction writes its SGPRs (a lane read, a
                // compare's mask) in every lane, whatever its guard
                // (`write_placed`): they hold its result afterwards. A lane
                // read's value reaches them even where the guard fails (a
                // compare's bit is 0 there).
                let to_sgpr = matches!(inst.fields, Fields::Valu(_))
                    && inst.writes.iter().any(|w| matches!(w.0, Reg::S(_)));
                let lane_read = to_sgpr && !inst.name().starts_with("v_cmp");
                if (runs || per_lane && lane_read)
                    && inst
                        .reads
                        .iter()
                        .any(|(r, v)| t.regs.contains(r) || t.values.contains(v))
                {
                    let writes_exec = inst
                        .writes
                        .iter()
                        .any(|w| matches!(w.0, Reg::S(c) if c == EXEC || c == EXEC + 1));
                    let branch = i == last
                        && matches!(inst.fields, Fields::Sopp { .. })
                        && inst.name().starts_with("s_cbranch_");
                    if per_lane && branch {
                        // The block's exit decides with the condition.
                    } else if per_lane && matches!(inst.fields, Fields::Sop { .. }) && !writes_exec
                    {
                        for &(r, v) in &inst.writes {
                            if !t.regs.contains(&r) {
                                t.regs.push(r);
                            }
                            if !t.values.contains(&v) {
                                t.values.push(v);
                            }
                        }
                        continue;
                    } else {
                        return false;
                    }
                }
                if per_lane && to_sgpr {
                    t.regs
                        .retain(|r| !matches!(r, Reg::S(_)) || inst.write(*r).is_none());
                }
                if !skips && (per_lane || scalar) {
                    t.regs.retain(|r| inst.write(*r).is_none());
                }
            }
            if t.regs.is_empty() {
                continue 'paths;
            }
            let values = t.values.clone();
            let path = Path {
                tainted: &values,
                ..path
            };
            let next: Vec<(usize, Option<(Bool, bool)>)> = match (st.exits[at], &block.exit) {
                (_, Exit::Indirect) | (LaneExit::Indirect, _) => return false,
                (
                    LaneExit::Branch {
                        when,
                        taken,
                        not_taken,
                    },
                    _,
                ) if per_lane && taken != not_taken => {
                    let (can, cannot) = self.outcomes(when, &path);
                    if can && cannot && self.tests_taint(when, &values) {
                        return false;
                    }
                    let mut next = Vec::new();
                    if can {
                        next.push((taken, Some((when, true))));
                    }
                    if cannot {
                        next.push((not_taken, Some((when, false))));
                    }
                    next
                }
                (_, exit) => exit.successors().into_iter().map(|t| (t, None)).collect(),
            };
            for (to, branch) in next {
                let mut next = Taint {
                    known: Vec::new(),
                    regs: t.regs.clone(),
                    values: t.values.clone(),
                    // Back at the join, its phis are the next iteration's.
                    fresh: t.fresh && to != b,
                };
                // Over a back edge the loop redefines what the lane knew.
                if self.rpo[to] > self.rpo[at] {
                    next.known.clone_from(&known);
                    if let Some((when, holds)) = branch {
                        let more = self.learn(&path, when, holds);
                        next.known.extend(more);
                    }
                    // Knowing less only adds paths.
                    let over = next.known.len().saturating_sub(MAX_KNOWN);
                    next.known.drain(..over);
                }
                match states.get_mut(&(at, to)) {
                    None => {
                        states.insert((at, to), next);
                    }
                    Some(old) => {
                        if !old.merge(&next) {
                            continue;
                        }
                    }
                }
                work.push((at, to));
            }
        }
        true
    }

    /// Whether `b` tests an atom of a tainted value ([`Self::unread_on_edge`]).
    fn tests_taint(&self, b: Bool, tainted: &[Value]) -> bool {
        self.x
            .st
            .bools
            .support(b)
            .into_iter()
            .any(|a| atom_of(a).is_some_and(|v| tainted.contains(&v)))
    }

    /// The atoms that `b` holding as `holds` decides on `path`, which it
    /// does not know yet (none when `b` tests more than [`MAX_LEARN`]).
    fn learn(&self, path: &Path, b: Bool, holds: bool) -> Vec<(Atom, bool)> {
        let atoms = self.x.st.bools.support(b);
        if atoms.len() > MAX_LEARN {
            return Vec::new();
        }
        let mut out = Vec::new();
        for atom in atoms {
            if self.atom_value(atom, path).is_some()
                || atom_of(atom).is_some_and(|v| path.tainted.contains(&v))
            {
                continue;
            }
            for value in [false, true] {
                let mut known = path.known.to_vec();
                known.push((atom, value));
                let (t, f) = self.outcomes(
                    b,
                    &Path {
                        known: &known,
                        ..*path
                    },
                );
                if !(if holds { t } else { f }) {
                    out.push((atom, !value));
                    break;
                }
            }
        }
        out
    }

    /// Whether `b` can hold, and whether it can fail, in a lane on `path`.
    fn outcomes(&self, b: Bool, path: &Path) -> (bool, bool) {
        self.outcomes_in(b, path, &mut HashMap::new(), &mut HashMap::new())
    }

    fn outcomes_in(
        &self,
        b: Bool,
        path: &Path,
        memo: &mut HashMap<Bool, (bool, bool)>,
        atoms: &mut HashMap<Atom, Option<bool>>,
    ) -> (bool, bool) {
        if b == Bool::TRUE {
            return (true, false);
        }
        if b == Bool::FALSE {
            return (false, true);
        }
        if let Some(&r) = memo.get(&b) {
            return r;
        }
        let Some((atom, low, high)) = self.x.st.bools.node(b) else {
            return (true, true);
        };
        let value = match atoms.get(&atom) {
            Some(&v) => v,
            None => {
                let v = self.atom_value(atom, path);
                atoms.insert(atom, v);
                v
            }
        };
        let r = match value {
            Some(v) => self.outcomes_in(if v { high } else { low }, path, memo, atoms),
            None => {
                let (lt, lf) = self.outcomes_in(low, path, memo, atoms);
                let (ht, hf) = self.outcomes_in(high, path, memo, atoms);
                (lt || ht, lf || hf)
            }
        };
        memo.insert(b, r);
        r
    }

    /// An atom's value in a lane on `path`: known there, a comparison of
    /// two constants, or the same comparison of the same values as an atom
    /// known there.
    fn atom_value(&self, atom: Atom, path: &Path) -> Option<bool> {
        if atom_of(atom).is_some_and(|v| path.tainted.contains(&v)) {
            return None;
        }
        if let Some(&(_, v)) = path.known.iter().find(|k| k.0 == atom) {
            return Some(v);
        }
        let (rel, ops, negated) = match self.condition(atom, path)? {
            Cond::Fixed(v) => return Some(v),
            Cond::Compare(rel, ops, negated) => (rel, ops, negated),
        };
        if let [Op::Word(x), Op::Word(y)] = ops {
            return Some(rel.holds(x, y) != negated);
        }
        path.known
            .iter()
            .find_map(|&(other, v)| match self.condition(other, path) {
                Some(Cond::Compare(r, o, n)) if r == rel && o == ops => Some((v != n) != negated),
                _ => None,
            })
    }

    /// A lane's bit of a VALU compare or an SCC of a scalar compare, of
    /// 32-bit integers with no modifier, as a [`Cond`] of its operands on
    /// `path`.
    fn condition(&self, atom: Atom, path: &Path) -> Option<Cond> {
        let (Atom::Lane(v) | Atom::Uniform(v)) = atom else {
            return None;
        };
        let &cb = self.def_block.get(&v)?;
        let inst = self.x.p.blocks[cb]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))?;
        let name = inst.name();
        let (rest, ops) = match (&inst.fields, atom) {
            (Fields::Valu(alu), Atom::Lane(_))
                if matches!(inst.machine.encoding, Encoding::Vopc | Encoding::Vop3)
                    && alu.sdwa.is_none()
                    && alu.dpp.is_none()
                    && alu.neg == 0
                    && alu.abs == 0
                    && !alu.clamp
                    && alu.omod == 0
                    && alu.src.len() == 2 =>
            {
                let rest = name
                    .strip_prefix("v_cmp_")
                    .or_else(|| name.strip_prefix("v_cmpx_"))?;
                (rest, [alu.src[0], alu.src[1]])
            }
            (Fields::Sop { src, imm: None, .. }, Atom::Uniform(_)) if src.len() == 2 => {
                (name.strip_prefix("s_cmp_")?, [src[0], src[1]])
            }
            _ => return None,
        };
        let (rel, ty) = rest.split_once('_')?;
        let signed = match ty {
            "u32" => false,
            "i32" => true,
            _ => return None,
        };
        match rel {
            "f" => return Some(Cond::Fixed(false)),
            "t" => return Some(Cond::Fixed(true)),
            _ => {}
        }
        let x = self.compare_word(inst, cb, ops[0], path)?;
        let y = self.compare_word(inst, cb, ops[1], path)?;
        let lt = Rel::Lt { signed };
        Some(match rel {
            "eq" => Cond::Compare(Rel::Eq, [x.min(y), x.max(y)], false),
            "ne" | "lg" => Cond::Compare(Rel::Eq, [x.min(y), x.max(y)], true),
            "lt" => Cond::Compare(lt, [x, y], false),
            "ge" => Cond::Compare(lt, [x, y], true),
            "gt" => Cond::Compare(lt, [y, x], false),
            "le" => Cond::Compare(lt, [y, x], true),
            _ => return None,
        })
    }

    /// A compare's 32-bit operand on `path`: a constant where it is one
    /// there (a phi of the block the lane is in, by the edge it came from;
    /// one of the join's, by the join's edge), else the value it reads.
    fn compare_word(&self, inst: &Inst, cb: usize, op: Operand, path: &Path) -> Option<Op> {
        let reg = match op {
            Operand::Const(c) => return crate::resource::inline(c).map(Op::Word),
            Operand::Literal(k) => return Some(Op::Word(k)),
            Operand::Scalar { code: NULL, .. } => return Some(Op::Word(0)),
            Operand::Scalar { code, dwords: 1 } => Reg::S(code),
            Operand::Vector { reg, dwords: 1 } => Reg::V(reg),
            _ => return None,
        };
        let mut x = inst.read(reg)?;
        let block = &self.x.p.blocks[cb];
        if cb == path.at
            && let Some(from) = path.from
            && let Some(phi) = block.phis.iter().find(|phi| phi.value == x)
        {
            x = phi.args[block.preds.iter().position(|&q| q == from)?];
        }
        if path.tainted.contains(&x) {
            return None;
        }
        Some(
            match path.joined.get(&x).copied().or_else(|| self.word_of(x)) {
                Some(k) => Op::Word(k),
                None => Op::Value(x),
            },
        )
    }

    /// The DXIL constant an IR value is, as a dword.
    fn word_of(&self, v: Value) -> Option<u32> {
        match self.vals.get(&v)? {
            Val::I32(x) => self.s.m.int_value(*x).map(|k| k as u32),
            _ => None,
        }
    }
}

/// Counts a console-flow join that takes a descriptor's dword as undefined
/// without proof (see the descriptor-word rule of [`Lower::fill_phis`]).
fn unproven_join() -> bool {
    crate::emit::UNPROVEN_DESCRIPTOR_JOINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Blocks [`Lower::unread_on_edge`] searches at most (each a visit of a
/// block by an edge, again when what reaches it changes). 4096 ended the
/// search in the large character pixel programs (33 KiB of code), where
/// the taint goes far before every register holding it is written again.
const UNREAD_STEPS: usize = 1 << 16;
/// Atoms a path keeps at most (the latest).
const MAX_KNOWN: usize = 32;
/// Atoms of a condition beyond which a path learns nothing from it.
const MAX_LEARN: usize = 16;

/// What a lane knows on a path from a join it entered by one edge
/// ([`Lower::unread_on_edge`]).
#[derive(Clone, Copy)]
struct Path<'p> {
    /// The constants the join's phis take on that edge.
    joined: &'p HashMap<Value, u32>,
    /// The block the lane is in, and the one it came from (none: unknown).
    at: usize,
    from: Option<usize>,
    /// Atoms that what the lane passed decides, since the last back edge.
    known: &'p [(Atom, bool)],
    /// Values derived from the edge's (`unread_on_edge`): their atoms and
    /// operands are not known.
    tainted: &'p [Value],
}

/// The IR value an atom is of, if any.
fn atom_of(a: Atom) -> Option<Value> {
    match a {
        Atom::Lane(v) | Atom::Uniform(v) | Atom::Phi(v) | Atom::Quad(v) | Atom::Opaque(v) => {
            Some(v)
        }
        Atom::Constant(_) => None,
    }
}

/// A path's state in [`Lower::unread_on_edge`], merged over the paths that
/// enter a block by the same edge.
#[derive(Clone)]
struct Taint {
    /// What the lane knows ([`Path::known`]).
    known: Vec<(Atom, bool)>,
    /// Registers holding the edge's value or one derived from it.
    regs: Vec<Reg>,
    /// Those values.
    values: Vec<Value>,
    /// Whether the join's constants still hold (not back at the join).
    fresh: bool,
}

impl Taint {
    /// Adds what another path brings (fewer facts, more taint): whether it
    /// changed.
    fn merge(&mut self, other: &Taint) -> bool {
        let was = (
            self.known.len(),
            self.regs.len(),
            self.values.len(),
            self.fresh,
        );
        self.known.retain(|f| other.known.contains(f));
        for &r in &other.regs {
            if !self.regs.contains(&r) {
                self.regs.push(r);
            }
        }
        for &v in &other.values {
            if !self.values.contains(&v) {
                self.values.push(v);
            }
        }
        self.fresh &= other.fresh;
        was != (
            self.known.len(),
            self.regs.len(),
            self.values.len(),
            self.fresh,
        )
    }
}

/// A lane's bit or an SCC as a comparison ([`Lower::condition`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cond {
    Fixed(bool),
    /// The operands in that relation, negated where the flag is set.
    Compare(Rel, [Op; 2], bool),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rel {
    Eq,
    Lt { signed: bool },
}

impl Rel {
    fn holds(self, x: u32, y: u32) -> bool {
        match self {
            Rel::Eq => x == y,
            Rel::Lt { signed: false } => x < y,
            Rel::Lt { signed: true } => (x as i32) < (y as i32),
        }
    }
}

/// A compare's operand: a constant, or the value it reads.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Op {
    Word(u32),
    Value(Value),
}

#[cfg(test)]
mod tests {
    use super::super::{Parts, lower};
    use crate::emit::tests::{compute, emit_for};
    use crate::emit::{counted_loop_limit, loop_limit, spin_limit};
    use crate::native::{ComputeIo, ResourceType, RootParam, RootType, StageIo, UsedResource};
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// The limit each back edge of `words` compares its loop's count with:
    /// a wave32 compute program of 32 threads whose s0 holds a table of
    /// structured buffers of dwords and s1 a root constant, with 512 bytes
    /// of LDS.
    fn limits(words: &[u32]) -> Vec<u32> {
        limits_with_lds(words, 1)
    }

    /// [`limits`] with `granules` of 512 bytes of LDS.
    fn limits_with_lds(words: &[u32], granules: u32) -> Vec<u32> {
        let param = |root, user_data| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot: user_data,
        };
        let mut native = compute(words, false);
        native.layout = vec![param(RootType::Table, 0), param(RootType::Constants, 1)];
        native.used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        native.io = StageIo::Cs(ComputeIo {
            rsrc2: granules << 15 | 2 << 1,
            threads: [32, 1, 1],
        });
        let p = ir::build(&native.code, false, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
        .unwrap()
        .loop_limits
    }

    /// A uniform loop `for (i = 0; i < s1; i += step)` writing LDS each
    /// round, its test `test` (`s_cmp_*_u32 s2, s1` by default).
    fn uniform_loop(step: u32, test: u32) -> [u32; 8] {
        [
            0xBE82_0380, // s_mov_b32 s2, 0
            0x7E02_0202, // 4: v_mov_b32 v1, s2
            0xD834_0000,
            0x0000_0100,             // ds_write_b32 v0, v1
            0x8102_0002 | step << 8, // s_add_i32 s2, s2, step (inline constant)
            test,
            0xBF85_FFFA, // s_cbranch_scc1 -> 4
            S_ENDPGM,
        ]
    }

    /// Counted loops keep the higher limit: 2^20 unless
    /// `XTP_SHADER_LOOP_LIMIT` says otherwise. 2^14 cut short the loops of
    /// 16,385 to 2^20 iterations.
    #[test]
    fn counted_loops_keep_the_higher_limit() {
        if std::env::var_os("XTP_SHADER_LOOP_LIMIT").is_none() {
            assert_eq!(counted_loop_limit(), 1 << 20);
            assert_eq!(loop_limit(), 1 << 14);
        }
        // i < s1, i += 1 (and += 3).
        let lt = 0xBF0A_0102; // s_cmp_lt_u32 s2, s1
        assert_eq!(limits(&uniform_loop(0x81, lt)), [counted_loop_limit()]);
        assert_eq!(limits(&uniform_loop(0x83, lt)), [counted_loop_limit()]);
        // s1 > i, the operands the other way round.
        let gt = 0xBF08_0201; // s_cmp_gt_u32 s1, s2
        assert_eq!(limits(&uniform_loop(0x81, gt)), [counted_loop_limit()]);
        // A divergent loop: each lane counts to its own v0 and leaves once
        // v1 >= v0 (an exit mask, EXEC narrowed each round).
        let lanes = [
            0x7E02_0280, // v_mov_b32 v1, 0
            0xBE82_0380, // s_mov_b32 s2, 0
            0xD834_0000,
            0x0000_0100, // 8: ds_write_b32 v0, v1
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1
            0x7D8C_0101, // v_cmp_ge_u32 vcc_lo, v1, v0
            0x8802_026A, // s_or_b32 s2, vcc_lo, s2
            0x8A7E_027E, // s_andn2_b32 exec_lo, exec_lo, s2
            0xBF89_FFF9, // s_cbranch_execnz -> 8
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            S_ENDPGM,
        ];
        assert_eq!(limits(&lanes), [counted_loop_limit()]);
    }

    /// The IR of `words`, the reverse-postorder index of each block
    /// (`usize::MAX`: unreachable) and the loop headers (blocks a later
    /// block branches back to).
    fn loops_of(words: &[u32]) -> (ir::Program, Vec<usize>, Vec<usize>) {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, false, 8).unwrap();
        let order = ir::reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (n, &b) in order.iter().enumerate() {
            rpo[b] = n;
        }
        let headers = order
            .iter()
            .copied()
            .filter(|&h| {
                p.blocks[h]
                    .preds
                    .iter()
                    .any(|&q| rpo[q] != usize::MAX && rpo[h] <= rpo[q])
            })
            .collect();
        (p, rpo, headers)
    }

    /// Whether [`super::barrier_after_loop`] finds an `s_barrier` reachable
    /// from a loop header of `words`.
    fn barrier_after(words: &[u32]) -> bool {
        let (p, _, headers) = loops_of(words);
        super::barrier_after_loop(&p, headers, |b| {
            p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier")
        })
    }

    /// The budget counts where loops can pass it together: a loop inside a
    /// loop, or loops one after the other whose limits add up to it. One
    /// loop alone has its own limit.
    #[test]
    fn budgets_count_where_loops_can_pass_them() {
        let can_pass = |words: &[u32], limit: u32, budget: u32| {
            let (p, rpo, headers) = loops_of(words);
            super::loops_can_pass(&p, &rpo, &headers, |_| limit, budget)
        };
        let lt = 0xBF0A_0102; // s_cmp_lt_u32 s2, s1
        let one = uniform_loop(0x81, lt);
        assert!(!can_pass(&one, 1 << 14, 1 << 20));
        assert!(!can_pass(&one, 1 << 20, 1 << 20));
        // The same loop twice, one after the other.
        let twice = [&one[..7], &one[..7], &[S_ENDPGM]].concat();
        assert!(!can_pass(&twice, 1 << 14, 1 << 20));
        assert!(can_pass(&twice, 1 << 20, 1 << 20));
        // An inner loop whose latch also goes back to the outer header.
        let nested = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBF82_0003, // s_branch -> outer
            0x8001_8101, // inner: s_add_u32 s1, s1, 1
            0xBF06_8201, // s_cmp_eq_u32 s1, 2
            0xBF84_FFFD, // s_cbranch_scc0 -> inner; else on to outer
            0x8000_8100, // outer: s_add_u32 s0, s0, 1
            0xBF06_8200, // s_cmp_eq_u32 s0, 2
            0xBF85_0002, // s_cbranch_scc1 -> exit
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF82_FFF8, // s_branch -> inner
            S_ENDPGM,
        ];
        assert!(can_pass(&nested, 1 << 14, 1 << 20));
    }

    /// A group barrier after a loop or inside it is one a lane the limit
    /// stops would miss; one before the loop, or none, is not. The lowering
    /// drains the lanes of the first two only (the rest keep their form).
    #[test]
    fn barriers_a_stopped_lane_would_miss_are_found() {
        const S_BARRIER: u32 = 0xBF8A_0000;
        let lt = 0xBF0A_0102; // s_cmp_lt_u32 s2, s1
        let l = uniform_loop(0x81, lt);
        assert!(!barrier_after(&l));
        let after = [&l[..7], &[S_BARRIER, S_ENDPGM]].concat();
        assert!(barrier_after(&after));
        let before = [&l[..1], &[S_BARRIER], &l[1..]].concat();
        assert!(!barrier_after(&before));
        // In the body: the back edge one dword further.
        let inside = [&l[..4], &[S_BARRIER], &l[4..6], &[0xBF85_FFF9, S_ENDPGM]].concat();
        assert!(barrier_after(&inside));
        // The lowering drains the lanes of those two only.
        let drains = |words: &[u32]| {
            super::DRAINS.set(None);
            limits(words);
            super::DRAINS.get().expect("decided")
        };
        assert!(drains(&after));
        assert!(drains(&inside));
        assert!(!drains(&before));
        assert!(!drains(&l));
        // A barrier and no loop.
        assert!(!drains(&[0xBE82_0380, S_BARRIER, S_ENDPGM]));
        // 32 KiB of LDS leaves no room for the counts: the program still
        // translates, its lanes return at the limit.
        super::DRAINS.set(None);
        assert_eq!(limits_with_lds(&after, 64).len(), 1);
        assert_eq!(super::DRAINS.get(), Some(false));
    }

    /// Loops with no induction variable bounded by a loop-invariant value
    /// keep 2^14: one leaving only on equality with its bound (`!=`, by
    /// steps of 1 or 2: a variable that starts past the bound never meets
    /// it), one counting away from it, one whose bound moves, one that
    /// walks a buffer until it reads 0.
    #[test]
    fn other_loops_keep_the_general_limit() {
        let ne = 0xBF07_0102; // s_cmp_lg_u32 s2, s1
        assert_eq!(limits(&uniform_loop(0x81, ne)), [loop_limit()]);
        assert_eq!(limits(&uniform_loop(0x82, ne)), [loop_limit()]);
        // The same going down, from the other operand: s1 != i by -1.
        let ne_swapped = 0xBF07_0201; // s_cmp_lg_u32 s1, s2
        assert_eq!(limits(&uniform_loop(0xC1, ne_swapped)), [loop_limit()]);
        let lt = 0xBF0A_0102; // s_cmp_lt_u32 s2, s1
        assert_eq!(limits(&uniform_loop(0xC1, lt)), [loop_limit()]);
        let moving = [
            0xBE82_0380, // s_mov_b32 s2, 0
            0x7E02_0202, // 4: v_mov_b32 v1, s2
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0x8102_8102, // s_add_i32 s2, s2, 1
            0x8104_8304, // s_add_i32 s4, s4, 3
            0xBF0A_0402, // s_cmp_lt_u32 s2, s4
            0xBF85_FFF9, // s_cbranch_scc1 -> 4
            S_ENDPGM,
        ];
        assert_eq!(limits(&moving), [loop_limit()]);
        let walks = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xBE82_0380, // s_mov_b32 s2, 0
            0xE030_6000,
            0x8002_0100, // 0x14: buffer_load_dword v1, v0, s[8:11], 0 idxen glc
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x4A00_0081, // v_add_nc_u32 v0, 1, v0
            0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
            0x8802_026A, // s_or_b32 s2, vcc_lo, s2
            0x8A7E_027E, // s_andn2_b32 exec_lo, exec_lo, s2
            0xBF89_FFF8, // s_cbranch_execnz -> 0x14
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        assert_eq!(limits(&walks), [loop_limit()]);
    }

    /// A spin wait (rereading one flag until it is no longer 0) compares
    /// with the spin limit, 4096 under the default loop limit.
    #[test]
    fn spin_waits_emit_the_spin_limit() {
        if std::env::var_os("XTP_SHADER_LOOP_LIMIT").is_none() {
            assert_eq!(spin_limit(), 4096);
        }
        let spin = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xBE82_0380, // s_mov_b32 s2, 0
            0xE030_6000,
            0x8002_0100, // 0x14: buffer_load_dword v1, v0, s[8:11], 0 idxen glc
            0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
            0x8802_026A, // s_or_b32 s2, vcc_lo, s2
            0x8A7E_027E, // s_andn2_b32 exec_lo, exec_lo, s2
            0xBF89_FFFA, // s_cbranch_execnz -> 0x14
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        assert_eq!(limits(&spin), [spin_limit()]);
    }

    /// s8 holds a buffer descriptor's first dword on one edge (a root
    /// constant, s1, chose the load) and 5 on the other, and the program
    /// reads the joined s8 only where s1 chose the data edge. Lane t stores
    /// at dword t of the UAV at table entry 1: with s1 = 0, the 5 of the
    /// data edge; otherwise the dword t of the buffer at entry 0.
    const DESCRIPTOR_JOIN: [u32; 23] = [
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0009, // s_cbranch_scc1 -> 0x40
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2000,
        0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8003_0204, // buffer_store_dword v2, v4, s[12:15], 0 offen
        0xBF82_0001, // s_branch -> 0x44
        0xBE88_0385, // 0x40: s_mov_b32 s8, 5
        0xBF06_8001, // 0x44: s_cmp_eq_u32 s1, 0
        0xBF84_0003, // s_cbranch_scc0 -> 0x58
        0x7E06_0208, // v_mov_b32 v3, s8
        0xE070_1000,
        0x8003_0304, // buffer_store_dword v3, v4, s[12:15], 0 offen
        S_ENDPGM,    // 0x58
    ];

    /// [`DESCRIPTOR_JOIN`] in groups of 32, entry 0 of the table at s0 a
    /// buffer of `kind` (stride 4), entry 1 a raw UAV.
    fn descriptor_join(kind: ResourceType) -> crate::NativeProgram {
        let mut p = compute(&DESCRIPTOR_JOIN, false);
        let param = |root, user_data| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot: user_data,
        };
        let entry = |kind, table_offset_dwords, stride| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride,
            dwords: 4,
            user_data: 0,
            slot: 0,
        };
        p.layout = vec![param(RootType::Table, 0), param(RootType::Constants, 1)];
        p.used = vec![entry(kind, 0, 4), entry(ResourceType::Uav, 8, 0)];
        p.io = StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        });
        p
    }

    /// The descriptor edge of [`DESCRIPTOR_JOIN`] brings an undefined dword
    /// instead of stopping the program ("a descriptor's address or format
    /// read as data"), and both edges give what the console does (on WARP).
    #[test]
    fn descriptor_dwords_join_data_as_undefined_on_their_edge() {
        let cs = emit_for(
            &descriptor_join(ResourceType::Srv),
            crate::emit::Target::ANY,
        )
        .unwrap();
        #[cfg(windows)]
        {
            use crate::emit::host::{Dispatch, Host, Root, Stride, View};
            let Ok(host) = Host::new(true) else {
                eprintln!("no WARP device; skipped");
                return;
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let data: Vec<u32> = (0..32).map(|t| 1000 + t).collect();
            let views = [
                View::Buffer {
                    data: &data,
                    stride: Stride::Structured(4),
                },
                View::RawUav(32),
            ];
            for (choice, want) in [(0, vec![5; 32]), (1, data.clone())] {
                let mut user = vec![0; (cs.user_data_dwords() as usize).max(2)];
                user[1] = choice;
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(user))],
                            views: &views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap();
                assert_eq!(out[0], want, "s1 = {choice}");
            }
        }
    }

    /// [`DESCRIPTOR_JOIN`] without the join's test (its `s_cmp` and
    /// `s_cbranch_scc0` made `s_nop`): every lane reads the joined s8, so
    /// where s1 chose the load it reads the descriptor's first dword, an
    /// address. That stops the program, as a direct read of the dword does
    /// (the review of meshdata), instead of reading an undefined value.
    #[test]
    fn a_descriptor_dword_read_on_its_edge_after_a_join_stops() {
        const S_NOP: u32 = 0xBF80_0000;
        let mut words = DESCRIPTOR_JOIN;
        words[17] = S_NOP;
        words[18] = S_NOP;
        // [`FLAG_AND_WORD`] with the flag -1 on both edges: on the
        // descriptor's, whether the branch skips the store depends on s8.
        let mut beside = FLAG_AND_WORD;
        beside[5] = 0x7E0A_02C1; // v_mov_b32 v5, -1
        for words in [&words[..], &beside[..]] {
            let mut p = descriptor_join(ResourceType::Srv);
            p.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let r = emit_for(&p, crate::emit::Target::ANY);
            assert!(
                matches!(
                    r,
                    Err(crate::ShaderError::Unsupported {
                        what: "a descriptor's address or format read as data",
                        ..
                    })
                ),
                "{r:?}"
            );
        }
    }

    /// The console flow's shortcut of the descriptor-word rule: the join of
    /// [`a_descriptor_dword_read_on_its_edge_after_a_join_stops`] (s8 read
    /// on both edges) after a loop whose lane read makes the program run in
    /// the console flow (`lane_read_loop_tests`). There the join keeps the
    /// undefined value without proof and is counted; the same join in the
    /// per-lane form stops (the test above).
    #[test]
    fn the_console_flow_keeps_a_descriptor_join_unproven_and_counts_it() {
        use std::sync::atomic::Ordering;
        const S_NOP: u32 = 0xBF80_0000;
        let mut words = vec![
            0xBE89_0380, // s_mov_b32 s9, 0
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
            0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
            0xBF88_0006, // s_cbranch_execz -> end
            0xD760_0009,
            0x0001_0300, // loop: v_readlane_b32 s9, v0, 1
            0x4A04_0481, // v_add_nc_u32 v2, 1, v2
            0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
            0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
            0xBF89_FFFA, // s_cbranch_execnz -> loop
            0x887E_0A7E, // end: s_or_b32 exec_lo, exec_lo, s10
        ];
        let mut join = DESCRIPTOR_JOIN;
        join[17] = S_NOP;
        join[18] = S_NOP;
        words.extend(join);
        let mut p = descriptor_join(ResourceType::Srv);
        p.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let before = crate::emit::UNPROVEN_DESCRIPTOR_JOINS.load(Ordering::Relaxed);
        emit_for(&p, crate::emit::Target::ANY).expect("translated in the console flow");
        let flow = crate::emit::last_module_summary();
        assert!(flow.contains("console flow"), "{flow}");
        assert!(crate::emit::UNPROVEN_DESCRIPTOR_JOINS.load(Ordering::Relaxed) > before);
    }

    /// [`DESCRIPTOR_JOIN`] in the form of the material and character pixel
    /// programs (0xb9b18de at 0x1100): a flag v5 (0 on the descriptor's
    /// edge, -1 on the data's), and at the join the joined s8 tested beside
    /// it, `flag && s8 != 0` as masks (`s_xor`, `s_cmp` + `s_cselect`,
    /// `s_or`, then `s_cbranch_vccnz` past the store). On the descriptor's
    /// edge the flag alone takes the branch, whatever s8 holds: the dword
    /// is dead there and the program translates; both edges give what the
    /// console does (on WARP), as [`DESCRIPTOR_JOIN`] does.
    const FLAG_AND_WORD: [u32; 30] = [
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0009, // s_cbranch_scc1 -> 0x44
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2000,
        0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8003_0204, // buffer_store_dword v2, v4, s[12:15], 0 offen
        0xBF82_0002, // s_branch -> 0x4c
        0xBE88_0385, // 0x44: s_mov_b32 s8, 5
        0x7E0A_02C1, // v_mov_b32 v5, -1
        0x7D84_0AC1, // 0x4c: v_cmp_eq_u32 vcc_lo, -1, v5
        0x8906_C16A, // s_xor_b32 s6, vcc_lo, -1
        0xBF06_8008, // s_cmp_eq_u32 s8, 0
        0x8507_807E, // s_cselect_b32 s7, exec_lo, 0
        0x8806_0706, // s_or_b32 s6, s6, s7
        0x876A_067E, // s_and_b32 vcc_lo, exec_lo, s6
        0xBF87_0003, // s_cbranch_vccnz -> 0x74
        0x7E06_0208, // v_mov_b32 v3, s8
        0xE070_1000,
        0x8003_0304, // buffer_store_dword v3, v4, s[12:15], 0 offen
        S_ENDPGM,    // 0x74
    ];

    #[test]
    fn a_descriptor_dword_tested_beside_a_deciding_flag_is_dead_on_its_edge() {
        let mut p = descriptor_join(ResourceType::Srv);
        p.code = FLAG_AND_WORD.iter().flat_map(|w| w.to_le_bytes()).collect();
        let cs = emit_for(&p, crate::emit::Target::ANY).unwrap();
        #[cfg(windows)]
        {
            use crate::emit::host::{Dispatch, Host, Root, Stride, View};
            let Ok(host) = Host::new(true) else {
                eprintln!("no WARP device; skipped");
                return;
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let data: Vec<u32> = (0..32).map(|t| 1000 + t).collect();
            let views = [
                View::Buffer {
                    data: &data,
                    stride: Stride::Structured(4),
                },
                View::RawUav(32),
            ];
            for (choice, want) in [(0, vec![5; 32]), (1, data.clone())] {
                let mut user = vec![0; (cs.user_data_dwords() as usize).max(2)];
                user[1] = choice;
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(user))],
                            views: &views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap();
                assert_eq!(out[0], want, "s1 = {choice}");
            }
        }
    }

    /// The same join with entry 0 a constant buffer: its descriptor's
    /// dwords stop with their own reason, which the join keeps.
    #[test]
    fn a_join_keeps_the_refusal_of_a_constant_buffers_descriptor() {
        let r = emit_for(
            &descriptor_join(ResourceType::Cbv),
            crate::emit::Target::ANY,
        );
        assert!(
            matches!(
                r,
                Err(crate::ShaderError::Unsupported {
                    what: "a constant buffer's descriptor read as data",
                    ..
                })
            ),
            "{r:?}"
        );
    }
}
