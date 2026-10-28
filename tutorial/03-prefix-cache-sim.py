"""Lesson 3 companion: run with `.venv/bin/python tutorial/03-prefix-cache-sim.py`.

Drive vLLM v1's real KVCacheManager through simulated multi-turn agent loops.

Uses the helpers from tests/v1/core/test_prefix_caching.py so the manager is
built exactly the way the unit tests build it (single FullAttention group,
block_size == hash_block_size, sha256 + init_none_hash).

Each "turn" is one stateless /v1/chat/completions-style request, mirroring how
the v1 Scheduler drives the manager:
  1. get_computed_blocks(req)                -> prefix-cache hit (tokens)
  2. allocate_slots(req, prompt-hit, hit, b) -> prefill of the uncached suffix
  3. per decode step: append sampled token, allocate_slots(req, 1)
  4. free(req) when the request finishes (blocks go back to the free queue,
     but keep their hashes until they are actually reused = evicted)
"""

import sys
from pathlib import Path

# Run from anywhere: put the repo root on sys.path so `tests.v1.core...` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.v1.core.test_prefix_caching import (  # noqa: E402
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import init_none_hash  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

BS = 16
init_none_hash(sha256)  # same as the autouse fixture in the test file


def rng(start, n):
    return list(range(start, start + n))


class Sim:
    def __init__(self, num_blocks=400, enable_caching=True):
        self.mgr = make_kv_cache_manager(
            make_kv_cache_config(BS, num_blocks),
            max_model_len=8192,
            enable_caching=enable_caching,
            hash_block_size=BS,
        )
        self.pool = self.mgr.block_pool
        self.owner = {}
        self.n = 0
        self.total_prompt = 0
        self.total_prefill = 0

    def free_blocks(self):
        return self.pool.get_num_free_blocks()

    def cached_entries(self):
        return len(self.pool.cached_block_hash_to_block)

    def run(self, label, prompt, outputs, cache_salt=None, quiet=False):
        self.n += 1
        req = make_request(
            f"{label}#{self.n}", prompt, BS, sha256, cache_salt=cache_salt
        )
        free_before = self.free_blocks()
        hit_blocks, hit, _ = self.mgr.get_computed_blocks(req)
        n_prefill = req.num_tokens - hit
        new = self.mgr.allocate_slots(req, n_prefill, hit, hit_blocks)
        assert new is not None, f"{label}: cannot allocate prefill"
        for b in new.blocks[0]:
            self.owner[b.block_id] = label
        req.status = RequestStatus.RUNNING
        req.num_computed_tokens = hit + n_prefill
        # The prefill forward samples the 1st output token; every later output
        # token needs one decode step that writes the previous token's KV.
        req.append_output_token_ids(outputs[0])
        for tok in outputs[1:]:
            new = self.mgr.allocate_slots(req, 1)
            assert new is not None, f"{label}: cannot allocate decode slot"
            for b in new.blocks[0]:
                self.owner[b.block_id] = label
            req.num_computed_tokens += 1
            req.append_output_token_ids(tok)
        stm = self.mgr.coordinator.single_type_managers[0]
        cached_full = stm.num_cached_block.get(req.request_id, 0)
        n_alloc = len(self.mgr.get_block_ids(req.request_id)[0])
        self.mgr.free(req)
        self.total_prompt += len(prompt)
        self.total_prefill += n_prefill
        row = dict(
            label=label,
            prompt=len(prompt),
            hit=hit,
            prefill=n_prefill,
            out=len(outputs),
            kv_written=req.num_computed_tokens,
            cached_full=cached_full,
            n_alloc=n_alloc,
            free_before=free_before,
            free_after=self.free_blocks(),
            entries=self.cached_entries(),
        )
        if not quiet:
            print(
                f"  {label:<10} prompt={row['prompt']:>4}  hit={hit:>4}  "
                f"prefill={n_prefill:>4}  out={len(outputs):>3}  "
                f"kv_tokens_at_end={req.num_computed_tokens:>4}  "
                f"cached_full_blocks={cached_full:>3}  "
                f"free_q(before lookup)={free_before:>3}  "
                f"free_q(after free)={row['free_after']:>3}  "
                f"hash_map_entries={row['entries']:>3}"
            )
        return row

    def describe_free_queue(self):
        """Run-length view of the free queue, head (evicted first) -> tail."""
        runs = []
        for b in self.pool.free_block_queue.get_all_free_blocks():
            key = (self.owner.get(b.block_id, "never-used"), b.block_hash is not None)
            if runs and runs[-1][0] == key:
                runs[-1][1].append(b.block_id)
            else:
                runs.append([key, [b.block_id]])
        parts = []
        for (own, hashed), ids in runs:
            tag = "hashed" if hashed else "no-hash"
            span = f"{ids[0]}" if len(ids) == 1 else f"{ids[0]}->{ids[-1]}"
            parts.append(f"[{own}/{tag} x{len(ids)} id {span}]")
        return "HEAD " + " ".join(parts) + " TAIL"


# ---------------------------------------------------------------- content
SYS = rng(1000, 300)  # system prompt + tool schemas
USER = rng(2000, 37)  # user message
TPL_TOOL = [90001, 90002, 90003]  # "<|im_start|>tool\n" style template
TPL_ASST = [90011, 90012, 90013, 90014, 90015]  # "<|im_end|>\n<|im_start|>assistant\n"
OUT_LENS = [45, 38, 52, 60]
TOOL_LENS = [120, 260, 80]


def out_tokens(turn, n):
    return rng(10000 + turn * 1000, n)


def tool_tokens(turn, n):
    return rng(50000 + turn * 1000, n)


def agent_loop(
    sim, rewrite_turn=None, drop=0, dynamic_pos=None, between=None, show_queue=False
):
    """4-turn tool-calling loop. Client resends the full history every turn."""
    history_outputs = []  # what the client will send back for each turn
    prompt = SYS + USER
    rows = []
    for t in range(4):
        if dynamic_pos is not None:
            prompt = list(prompt)
            prompt[dynamic_pos] = 70000 + t  # e.g. "Current time: ..." token
        rows.append(sim.run(f"agent-T{t + 1}", prompt, out_tokens(t, OUT_LENS[t])))
        if show_queue:
            print("     free queue:", sim.describe_free_queue())
        if t == 3:
            break
        history_outputs.append(out_tokens(t, OUT_LENS[t]))
        if between is not None:
            between(sim, t)
        # Rebuild the next prompt from the (possibly rewritten) history.
        sent = []
        for i, o in enumerate(history_outputs):
            if rewrite_turn is not None and t + 2 >= rewrite_turn and i == 0:
                o = o[drop:]
            sent.append(o)
        prompt = list(SYS + USER)
        if dynamic_pos is not None:
            prompt[dynamic_pos] = 70000 + t + 1
        for i, o in enumerate(sent):
            prompt += o + TPL_TOOL + tool_tokens(i, TOOL_LENS[i]) + TPL_ASST
    return rows


def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def summary(sim, rows):
    hits = sum(r["hit"] for r in rows)
    print(
        f"  TOTAL prompt tokens={sim.total_prompt}  prefilled={sim.total_prefill}  "
        f"cache-hit={hits}  ({100.0 * hits / max(sim.total_prompt, 1):.1f}% of prompt)"
    )


# ---------------------------------------------------------------- S1
header("S1 agent loop, caching ON (block_size=16, 400 blocks)")
s1 = Sim()
c = s1.mgr.coordinator
print(
    f"  coordinator={type(c).__name__}  enable_partial_hash_hits="
    f"{c.enable_partial_hash_hits}  hash_block_size={s1.pool.hash_block_size}"
)
rows1 = agent_loop(s1)
summary(s1, rows1)
prev = 0
for r in rows1:
    if prev:
        print(
            f"  {r['label']}: hit {r['hit']} = prev prompt {prev} + "
            f"{r['hit'] - prev} tokens of prev turn's generated output/decode KV"
        )
    prev = r["prompt"]

# ---------------------------------------------------------------- S2
header("S2 same loop, caching OFF (KVCacheManager(enable_caching=False))")
s2 = Sim(enable_caching=False)
c = s2.mgr.coordinator
print(f"  coordinator={type(c).__name__}")
rows2 = agent_loop(s2)
summary(s2, rows2)
print(
    f"  S1 prefilled {s1.total_prefill} vs S2 prefilled {s2.total_prefill} "
    f"-> {s2.total_prefill / s1.total_prefill:.2f}x more prefill without cache"
)

# ---------------------------------------------------------------- S3
header(
    "S3 history rewritten: from turn 3 on, client drops first 20 tokens of "
    "turn-1 assistant output"
)
s3 = Sim()
first_changed = len(SYS + USER)
print(
    f"  first changed token position in turn 3 = {first_changed} "
    f"-> block {first_changed // BS} (block covers "
    f"{first_changed // BS * BS}..{first_changed // BS * BS + BS - 1})"
)
rows3 = agent_loop(s3, rewrite_turn=3, drop=20)
summary(s3, rows3)
print(
    f"  turn-3 hit with rewrite={rows3[2]['hit']} vs S1={rows1[2]['hit']}; "
    f"floor({first_changed}/16)*16={first_changed // BS * BS}"
)

# ---------------------------------------------------------------- S4
header("S4 dynamic system prompt: token at position 5 changes every turn")
s4 = Sim()
rows4 = agent_loop(s4, dynamic_pos=5)
summary(s4, rows4)
header("S4b same, but the dynamic token moved to position 299 (end of system prompt)")
s4b = Sim()
rows4b = agent_loop(s4b, dynamic_pos=299)
summary(s4b, rows4b)

# ---------------------------------------------------------------- S5
header(
    "S5 eviction pressure: num_blocks=70 (69 usable, block 0 = null block); "
    "unrelated requests run between agent turns"
)
s5 = Sim(num_blocks=70)
U_SIZES = {0: 880, 1: 400, 2: 1090}
print(
    "  free-queue legend: owner = request that WROTE the block's KV; "
    "HEAD is popped first by get_new_blocks (evicted first)"
)


def unrelated(sim, t):
    n = U_SIZES[t]
    sim.run(f"other-U{t + 1}", rng(200000 + t * 10000, n), rng(300000 + t * 1000, 8))
    print("     free queue:", sim.describe_free_queue())


rows5 = agent_loop(s5, between=unrelated, show_queue=True)
summary(s5, rows5)
for a, b in zip(rows1, rows5):
    print(f"  {a['label']}: hit no-pressure={a['hit']:>4}  with-pressure={b['hit']:>4}")

# ---------------------------------------------------------------- S6
header("S6 cache_salt: same 337-token prompt, different salts")
s6 = Sim()
P = SYS + USER
for lbl, salt in [
    ("salt=A", "tenant-A"),
    ("salt=B", "tenant-B"),
    ("salt=A", "tenant-A"),
    ("no-salt", None),
    ("salt=B", "tenant-B"),
]:
    s6.run(lbl, P, rng(400000, 5), cache_salt=salt)

# ---------------------------------------------------------------- S7
header("S7 exact repeat: 64-token prompt (4 full blocks) sent twice")
s7 = Sim()
P64 = rng(500000, 64)


def physical_hashed(sim):
    return sum(1 for b in sim.pool.blocks if b.block_hash is not None)


s7.run("first", P64, [1])
print(f"     physical blocks carrying a hash: {physical_hashed(s7)}")
s7.run("repeat", P64, [1])
print(
    f"     physical blocks carrying a hash: {physical_hashed(s7)} "
    "(4th block recomputed into a new block -> duplicate under the same key)"
)
s7.run("repeat+1", P64 + [7], [1])
print("  (repeat: max_cache_hit_length = 64-1 = 63 -> 63//16 = 3 blocks = 48)")

# ---------------------------------------------------------------- S8
header(
    "S8 tail granularity: prompts of length 47/48/49, each sent twice; "
    "then a 40-token shared prefix"
)
for L in (47, 48, 49):
    s8 = Sim()
    p = rng(600000, L)
    s8.run(f"L={L} cold", p, [1])
    s8.run(f"L={L} warm", p, [1])
s8 = Sim()
base = rng(700000, 49)
s8.run("base49", base, [1])
s8.run("share40", base[:40] + rng(710000, 9), [1])
c = s8.mgr.coordinator
print(
    f"  coordinator={type(c).__name__}  enable_partial_hash_hits="
    f"{c.enable_partial_hash_hits}"
)
