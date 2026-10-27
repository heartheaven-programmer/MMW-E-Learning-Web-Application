"""WHSG-DRA engine (thesis Ch. 1.4). Pure functions: no database, no I/O."""
import math
from collections import deque

WEIGHTS = {"F": 0.30, "P": 0.40, "C": 0.30}
MASTERY_CUT = 80
DEVELOPING_CUT = 60
TARGET_LIMIT = 3

# A1 / A2 parameters
TAU = 0.30
GAMMA = 0.8
H_MAX = 3


def band(p):
    if p >= MASTERY_CUT:
        return "Mastered"
    if p >= DEVELOPING_CUT:
        return "Developing"
    return "Needs Review"


def is_valid_attempt(answered, administered):
    return administered > 0 and 2 * answered >= administered


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _downstream_influence(k, deficit, children):
    """Legacy downstream-decay helper.

    Still used to build the `need` vector that feeds `rank_resources()` so
    retest targeting keeps its current behavior. The paper's Phase 3 algorithm
    (prerequisite walk with gamma and H) is added separately as
    `prerequisite_influence()` and exposed through `prereq_causes`.
    """
    total, seen, queue = 0.0, set(), deque([(k, 0)])
    while queue:
        node, depth = queue.popleft()
        for child in children.get(node, ()):
            if child in seen:
                continue
            seen.add(child)
            total += deficit[child] * 0.5 ** (depth + 1)
            queue.append((child, depth + 1))
    return total


def _evidence_for_response(item_index, chosen, dimap, tau=TAU):
    """A1: formal evidence layer.

    Returns (scores, selected_dimension, is_unclassified).

    The admin tagger writes one feature per (item, option). Choosing that
    option on a wrong response sets observed_value=1.0, kappa=1.0. If the
    option has no tag, all dimension scores stay 0 and the response is
    Unclassified. If a tag exists but its aggregate evidence is below tau,
    the response is also Unclassified. With a single feature, the only way
    to be Unclassified is to have no tag.
    """
    scores = {"F": 0.0, "P": 0.0, "C": 0.0}
    if chosen is None or chosen == "-":
        return scores, None, True
    tag = dimap.get((item_index, chosen)) if dimap else None
    if tag in ("F", "P", "C"):
        scores[tag] = 1.0
    top = max(scores, key=scores.get)
    if scores[top] >= tau:
        return scores, top, False
    return scores, None, True


def prerequisite_influence(target_k, prof, edges, H=H_MAX, gamma=GAMMA):
    """A2: paper Phase 3.

    For a target skill t with P_t < 60, walk prerequisites up to H edges.
    Each candidate prerequisite u at hop distance h contributes
    I = gamma^(h-1) * D_u, where D_u = max(0, 80 - P_u).

    Returns a list of (prerequisite_skill_index, h, influence) sorted by
    influence descending.
    """
    if prof[target_k] >= DEVELOPING_CUT:
        return []
    prereqs_of = {}
    for a, b in edges:          # a is prerequisite, b is dependent
        prereqs_of.setdefault(b, []).append(a)
    results = []
    seen = {target_k}
    queue = deque([(target_k, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= H:
            continue
        for p in prereqs_of.get(node, ()):
            if p in seen:
                continue
            seen.add(p)
            h = depth + 1
            deficit = max(0.0, MASTERY_CUT - prof[p])
            influence = (gamma ** (h - 1)) * deficit
            if influence > 0:
                results.append((p, h, influence))
            queue.append((p, h))
    results.sort(key=lambda x: -x[2])
    return results


def analyze(answers, key, cats, skills, edges, modules, weights=None, distractor_map=None):
    """answers: one letter per item, '-' for blank. key: correct letters.
    cats: F/P/C per item. skills: [(name, module, subtopic, [item indices])].
    edges: [(prereq, dependent)] skill indices. distractor_map: {(item, letter): 'F'|'P'|'C'}.
    """
    if weights is None:
        weights = WEIGHTS

    n = len(key)
    if len(answers) != n:
        raise ValueError("answers must have one entry per item")
    corr = [answers[i] == key[i] for i in range(n)]
    sk = []

    for name, mod, sub, items in skills:
        num = den = 0.0
        err = {"F": 0, "P": 0, "C": 0, "U": 0}
        blank = 0
        for i in items:
            w = weights[cats[i]]
            den += w
            if corr[i]:
                num += w
            elif answers[i] == "-":
                blank += 1
            else:
                # A1: formal evidence layer replaces the CATS[i] shortcut.
                _, dim, unclassified = _evidence_for_response(i, answers[i], distractor_map)
                if unclassified:
                    err["U"] += 1
                else:
                    err[dim] += 1
        sk.append({"name": name, "mod": mod, "sub": sub, "P": 100 * num / den,
                   "err": err, "blank": blank})

    mods = [{"name": m, "P": _mean(s["P"] for s in sk if s["mod"] == j)}
            for j, m in enumerate(modules)]
    course = _mean(s["P"] for s in sk)
    children = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
    deficit = [max(0.0, MASTERY_CUT - s["P"]) for s in sk]
    influence = [_downstream_influence(k, deficit, children) for k in range(len(sk))]
    need = [deficit[k] + 0.5 * influence[k] for k in range(len(sk))]
    root = [deficit[k] + influence[k] for k in range(len(sk))]

    # A2: paper Phase 3 outputs
    prof = [s["P"] for s in sk]
    prereq_causes = []
    for k in range(len(sk)):
        if prof[k] >= DEVELOPING_CUT:
            continue
        for prereq, h, inf in prerequisite_influence(k, prof, edges):
            prereq_causes.append({
                "target_skill_id": k,
                "target_P": prof[k],
                "candidate_prerequisite_skill_id": prereq,
                "candidate_P": prof[prereq],
                "h": h,
                "influence": inf,
            })
    prereq_causes.sort(key=lambda c: -c["influence"])

    total_err = {c: sum(s["err"][c] for s in sk) for c in ("F", "P", "C", "U")}
    return {
        "raw": sum(corr),
        "sk": sk,
        "mods": mods,
        "course": course,
        "need": need,
        "root": root,
        "err": total_err,
        "blank": sum(s["blank"] for s in sk),
        "prereq_causes": prereq_causes,    # A2 output
    }


def targets(result, limit=TARGET_LIMIT):
    idx = [k for k, v in enumerate(result["need"]) if v > 0]
    return sorted(idx, key=lambda k: -result["need"][k])[:limit]


def retest_items(tgt, skills_list):
    return sorted(i for k in tgt for i in skills_list[k][3])


def apply_retest(pre, retest, tgt):
    post = [retest["sk"][k]["P"] if k in tgt else s["P"] for k, s in enumerate(pre["sk"])]
    before = _mean(pre["sk"][k]["P"] for k in tgt)
    after = _mean(retest["sk"][k]["P"] for k in tgt)
    return {"post": post, "course": _mean(post), "before": before,
            "after": after, "gain": after - before}


def rank_resources(need, resources, top_n=4):
    nn = math.sqrt(sum(x * x for x in need))
    out = []
    for rid, title, cov in resources:
        rn = math.sqrt(sum(w * w for w in cov.values()))
        if nn == 0 or rn == 0:
            continue
        cosine = sum(need[k] * w for k, w in cov.items()) / (nn * rn)
        if cosine > 0:
            out.append({"id": rid, "title": title, "cosine": cosine,
                        "covers": max(cov, key=lambda k: need[k])})
    out.sort(key=lambda r: -r["cosine"])
    return out[:top_n]