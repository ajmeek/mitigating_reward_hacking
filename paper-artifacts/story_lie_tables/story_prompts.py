"""The six story prompts of the appendix (artifacts/story_prompt_<key>.tex in the paper repo), written from
experiments/story-dilemma-steering/prompts.json plus the length suffix that generate_stories.py
appends to every prompt, with LaTeX escaping (dollar signs, double quotes to ``'', accented letters).
Writes story_prompt_{resume,package,late,rye,dent,reddot}.tex beside this script.

    uv run python paper-artifacts/story_lie_tables/story_prompts.py
"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
EXP = os.path.join(ROOT, "experiments/story-dilemma-steering")
SUFFIX = " Keep it under 600 words."   # generate_stories.py: prompt_text = PROMPTS[key] + SUFFIX
KEYS = ["resume", "package", "late", "rye", "dent", "reddot"]   # the paper's six; caddy*, homework excluded (see story_lie_bars.py)
ACCENTS = {"é": r"\'e", "è": r"\`e", "á": r"\'a", "ó": r"\'o", "ï": r'\"i', "ç": r"\c{c}"}

def tex(s):
    s = s.replace("\\", r"\textbackslash{}").replace("$", r"\$").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    for ch, rep in ACCENTS.items(): s = s.replace(ch, rep)
    assert s.isascii(), [c for c in s if not c.isascii()]
    return s

P = json.load(open(os.path.join(EXP, "prompts.json")))
for k in KEYS:
    out = os.path.join(HERE, f"story_prompt_{k}.tex")
    open(out, "w").write(tex(P[k] + SUFFIX) + "\n")
    print("wrote", out)
