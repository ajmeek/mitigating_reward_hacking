"""Short stories from dilemma prompts (prompts.json) under activation addition of +/-S * u at L35 (plus an unsteered base).
usage: uv run --with safetensors --with numpy python experiments/story-dilemma-steering/scripts/generate_stories.py <UPSTREAM> [n_samples]
Env: ONLY="<prompt keys>", ARMS="base addp16 addn16 ...", THINK=0, MAXC (client concurrency, keep <= server max_num_seqs), MAXTOK, JSONL, TAG.
Streams straight from the vLLM server with the payload the repo proxy would stamp (single-turn prompt, so
steer_positions = [] and only decoded tokens are steered; per-arm cache_salt). Live markdown per prompt in
$(present --dir)/stories_<prompt>.md, rewritten every 2 s; final records in $OUT/$JSONL."""
import json, os, subprocess, sys, threading, time, urllib.request
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))   # paths below are repo-relative
from steering_reasoning.steering import build_steer_payload, with_positions
UP = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ART = "src/steering_reasoning/canonical/u_L35_steering.safetensors"
MODEL = "Qwen/Qwen3.5-27B"
OUT = "experiments/story-dilemma-steering/data/stories"; os.makedirs(OUT, exist_ok=True)
try: LIVE = subprocess.check_output(["present", "--dir"], text=True).strip()   # live markdown view, if a viewer exists
except Exception: LIVE = f"{OUT}/live"; os.makedirs(LIVE, exist_ok=True)
SUFFIX = " Keep it under 600 words."
THINK = os.environ.get("THINK", "1") != "0"   # THINK=0: chat_template_kwargs enable_thinking=false
TAG = os.environ.get("TAG", "" if THINK else "nothink_")
PROMPTS = json.load(open("experiments/story-dilemma-steering/prompts.json"))   # the paper's dilemma prompts
if os.environ.get("ONLY"): PROMPTS = {k: PROMPTS[k] for k in os.environ["ONLY"].split()}
ARMS = {"base": None,
        "addp8": with_positions(build_steer_payload(ART, mode="add", scale=8.0), []),
        "addn8": with_positions(build_steer_payload(ART, mode="add", scale=-8.0), [])}
def _arm(a):   # base | addp<S> | addn<S>: activation addition of +/-S * u at L35
    if a in ARMS: return ARMS[a]
    sc = float(a[4:]) * (1 if a.startswith("addp") else -1)
    return with_positions(build_steer_payload(ART, mode="add", scale=sc), [])
if os.environ.get("ARMS"): ARMS = {a: _arm(a) for a in os.environ["ARMS"].split()}
state = {}   # (arm, prompt, i) -> dict(reasoning, story, status)
lock = threading.Lock()

def run(arm, key, i):
    with SEM: _run(arm, key, i)

def _run(arm, key, i):
    st = state[(arm, key, i)]
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[key] + SUFFIX}], "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.7, "top_p": 0.95, "seed": 1000 + i,
            "max_tokens": int(os.environ.get("MAXTOK", "8192")), "cache_salt": f"stories-{TAG}{arm}"}
    if not THINK: body["chat_template_kwargs"] = {"enable_thinking": False}
    if ARMS[arm]: body["vllm_xargs"] = {"apply_steering_vectors": ARMS[arm]}
    t0 = time.time(); st["status"] = "streaming"
    try:
        req = urllib.request.Request(UP + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
                c = json.loads(line[5:])
                if c.get("usage"): st["tokens"] = c["usage"]["completion_tokens"]
                for ch in c.get("choices", []):
                    d = ch.get("delta", {})
                    st["reasoning"] += d.get("reasoning_content") or d.get("reasoning") or ""
                    st["story"] += d.get("content") or ""
                    if ch.get("finish_reason"): st["finish"] = ch["finish_reason"]
        st["status"] = "done"
    except Exception as e:
        st["status"] = f"error {e!r}"
    st["secs"] = round(time.time() - t0)
    rec = {"arm": arm, "thinking": THINK, "prompt": key, "prompt_text": PROMPTS[key] + SUFFIX, "sample": i, "seed": 1000 + i, "steer_payload": ARMS[arm], **st}
    with lock, open(f"{OUT}/" + os.environ.get("JSONL", "stories.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")
    print(f"{arm:6s} {key:8s} s{i} {st.get('tokens')} tok {st['secs']}s {st['status']} {st.get('finish', '')}", flush=True)

def render():
    for key, p in PROMPTS.items():
        md = [f"# {key}{'' if THINK else ' (thinking off)'}\n\n> {p}{SUFFIX}\n"]
        for arm in ARMS:
            for i in range(N):
                st = state[(arm, key, i)]
                n = st.get("tokens") or len(st["reasoning"] + st["story"]) // 4
                md.append(f"\n---\n\n## {arm} · sample {i} · {st['status']} · ~{n} tok\n")
                if st["story"]: md.append(st["story"] + "\n")
                elif st["reasoning"]: md.append("*(thinking)* " + " ".join(st["reasoning"][-600:].split()) + "\n")
        tmp = f"{LIVE}/.stories_{TAG}{key}.md"; open(tmp, "w").write("\n".join(md)); os.replace(tmp, f"{LIVE}/stories_{TAG}{key}.md")

threads = []
DONE_KEYS = set()
_jl = f"{OUT}/" + os.environ.get("JSONL", "stories.jsonl")
if os.path.exists(_jl):
    for _l in open(_jl):
        _r = json.loads(_l)
        if _r.get("status") == "done" and _r.get("thinking", True) == THINK: DONE_KEYS.add((_r["arm"], _r["prompt"], _r["sample"]))
SEM = threading.Semaphore(int(os.environ.get("MAXC", "32")))   # some providers' proxies (RunPod's) drop connections that sit idle in vLLM's queue
for i in range(N):
    for key in PROMPTS:
        for arm in ARMS:
            state[(arm, key, i)] = {"reasoning": "", "story": "", "status": "queued"}
            if (arm, key, i) in DONE_KEYS: state[(arm, key, i)]["status"] = "done (earlier)"; continue
            threads.append(threading.Thread(target=run, args=(arm, key, i), daemon=True))
render()
for t in threads: t.start()
while any(t.is_alive() for t in threads): render(); time.sleep(2)
render(); print("DONE", f"{OUT}/stories.jsonl")
