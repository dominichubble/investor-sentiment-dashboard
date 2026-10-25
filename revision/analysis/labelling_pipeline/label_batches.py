#!/usr/bin/env python3
"""Label the corpus_v2 candidate pool in batches, resumably, never overwriting a labels file.

Subcommands
  next     show the next unlabelled batch (read-only)
  export   write the blind batch file (if not already present) and a prompt file for Codex
  import   validate a pasted model reply (inbox JSON) and save it as a NEW labels_NNNN.json
  api      OpenAI API backend (stub): refuses to run unless --confirm-api and OPENAI_API_KEY set

Global options: --batch-size (default 50), --out DIR (write everything there instead of the real
folders; used for dry runs/tests), --dry-run (print what would be written, write nothing).
"""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import urllib.request

import common as C


# ---------- batch selection ----------
def next_batch(pool, labelled, size):
    """First unlabelled item in pool order, then up to `size` contiguous unlabelled items."""
    start = next((i for i, r in enumerate(pool) if r['item'] not in labelled), None)
    if start is None:
        return None, []
    batch = []
    for r in pool[start:]:
        if r['item'] in labelled or len(batch) == size:
            break
        batch.append(r)
    return start, batch


def blind_view(batch):
    return [{k: r[k] for k in C.FIELDS_SHOWN} for r in batch]


def blind_bytes(batch):
    return (json.dumps(blind_view(batch), indent=1) + '\n').encode()   # same format as build_corpus_candidates.py


def blind_path(start, batch, out_dir):
    """Reuse an existing blind_NNNN.json if it holds exactly this batch; otherwise a size-tagged new name."""
    want = blind_bytes(batch)
    names = [f'blind_{start:04d}.json', f'blind_{start:04d}_n{len(batch)}.json']
    for d in {C.CORPUS, Path(out_dir)}:
        for name in names:
            p = Path(d) / name
            if p.exists() and p.read_bytes() == want:
                return p, False
    for name in names:
        if not (C.CORPUS / name).exists() and not (Path(out_dir) / name).exists():
            return Path(out_dir) / name, True
    raise SystemExit(f'{names} exist with different content; refusing to overwrite. Inspect them first.')


def tag(start, size):
    return f'{start:04d}' if size == 50 else f'{start:04d}_n{size}'


# ---------- prompt ----------
PROMPT_HEAD = """You are labelling financial social-media/news items for the DSFE-635 training corpus.
Apply rubric v2 below exactly (sha256 {sha}). Judge each item independently from its text, ticker and channel only.

Key rules (summary; the full rubric text follows and wins on any conflict):
- Label = likely effect on the value of the asset the text is actually about, from an investor's point of view
  (Financial PhraseBank taxonomy). The ticker can be a spurious match; judge the asset discussed.
- R1 position gains follow the asset direction: profit on puts/shorts = negative; on calls/longs = positive;
  bare P&L with no view on the asset = neutral.
- R2 chart calls and quoted price moves take their stated direction; neutral only if no direction or balanced.
- R3 net-skeptical / risk-framed text = negative; a question with no lean = neutral.
- R4 new initiatives / favourable policy / naming favourite holdings = positive.
- R5 mixed analyst or rotation lists and bare launch headlines = neutral unless one side clearly dominates.
- R6 factual filler (holdings lists, transcripts, event notices, how-to, roundups) = neutral with
  "low_information": true. low_information may be true ONLY for neutral.
- R7 exclude ONLY spam/advert/SEO boilerplate, bare mention lists/thread headers, non-English, or too fragmentary.

Reply with ONE JSON object and nothing else, covering every item below exactly once:
{{"labels": {{"<item>": {{"label": "positive|negative|neutral|exclude", "low_information": false, "rationale": "<one short sentence>"}}, ...}}}}

----- RUBRIC v2 (verbatim) -----
{rubric}
----- ITEMS ({n}: {first} .. {last}) -----
"""


def build_prompt(batch):
    head = PROMPT_HEAD.format(sha=C.sha256(C.RUBRIC), rubric=C.RUBRIC.read_text().strip(), n=len(batch),
                              first=batch[0]['item'], last=batch[-1]['item'])
    lines = [f"[{r['item']}] ({r['channel']}, {r['ticker']}) {r['text']}" for r in batch]
    return head + '\n'.join(lines) + '\n'


# ---------- validation ----------
def _no_dupes(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f'duplicate key in reply: {k}')
        seen[k] = v
    return seen


def parse_reply(text):
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    if m:
        text = m.group(1).strip()
    data = json.loads(text, object_pairs_hook=_no_dupes)
    return data['labels'] if isinstance(data, dict) and 'labels' in data else data


def validate(labels, batch, block_ids, block_texts):
    errs = []
    want = [r['item'] for r in batch]
    if not isinstance(labels, dict):
        return ['reply is not a JSON object of item -> judgement']
    missing, extra = set(want) - set(labels), set(labels) - set(want)
    if missing:
        errs.append(f'missing ids: {sorted(missing)}')
    if extra:
        errs.append(f'unexpected ids: {sorted(extra)}')
    for item, rec in labels.items():
        if not isinstance(rec, dict):
            errs.append(f'{item}: judgement is not an object'); continue
        if rec.get('label') not in C.ALLOWED:
            errs.append(f'{item}: label {rec.get("label")!r} not in {C.ALLOWED}')
        if not isinstance(rec.get('low_information'), bool):
            errs.append(f'{item}: low_information must be true/false')
        elif rec['low_information'] and rec.get('label') != 'neutral':
            errs.append(f'{item}: low_information only allowed with neutral')
        if not isinstance(rec.get('rationale'), str) or not rec['rationale'].strip():
            errs.append(f'{item}: empty rationale')
        if set(rec) - {'label', 'low_information', 'rationale'}:
            errs.append(f'{item}: unexpected fields {sorted(set(rec) - {"label", "low_information", "rationale"})}')
    for r in batch:
        if r['id'] in block_ids or r['item'] in block_ids or C.normalise(C.clean(r['text'])) in block_texts:
            errs.append(f'{r["item"]}: overlaps the benchmark (id or normalised text)')
    return errs


def save_labels(labels, batch, start, src, out_dir, labeller, effort, method, dry):
    target = Path(out_dir) / f'labels_{start:04d}.json'
    for d in {C.CORPUS, Path(out_dir)}:
        if (Path(d) / target.name).exists():
            print(f'{target.name} already exists in {d}; nothing written (idempotent).')
            return None
    ordered = {r['item']: {'label': labels[r['item']]['label'],
                           'low_information': labels[r['item']]['low_information'],
                           'rationale': labels[r['item']]['rationale'].strip()} for r in batch}
    out = dict(meta=dict(labeller=labeller, reasoning_effort=effort, method=method, rubric='v2',
                         rubric_sha256=C.sha256(C.RUBRIC),
                         date=datetime.now(timezone.utc).date().isoformat(),
                         input_sha256=C.sha256(src), fields_shown=C.FIELDS_SHOWN, blindness=C.BLINDNESS),
               labels=ordered)
    if dry:
        print(f'[dry-run] would write {target} ({len(ordered)} labels, labeller={labeller!r})')
        return target
    with target.open('x') as f:                 # 'x' = fail rather than overwrite
        json.dump(out, f, indent=1)
        f.write('\n')
    target.chmod(0o444)
    print(f'Saved {len(ordered)} labels to {target} (read-only). labeller={labeller!r}')
    return target


# ---------- commands ----------
def context(a):
    out_dir = Path(a.out) if a.out else C.CORPUS
    work = Path(a.out) if a.out else C.PIPE / 'batches'
    pool = C.load_candidates()
    labelled, _ = C.load_labels([out_dir])
    start, batch = next_batch(pool, labelled, a.batch_size)
    return out_dir, work, pool, labelled, start, batch


def cmd_next(a):
    out_dir, work, pool, labelled, start, batch = context(a)
    print(f'pool {len(pool)} | labelled {len(labelled)} | remaining {len(pool) - len(labelled)}')
    if not batch:
        print('Nothing left to label.'); return 0
    bp, new = blind_path(start, batch, out_dir)
    print(f'next batch: start index {start}, {len(batch)} items, {batch[0]["item"]} .. {batch[-1]["item"]}')
    print(f'blind file: {bp} ({"to be created" if new else "existing, byte-identical, reused"})')
    print(f'labels file: {out_dir / f"labels_{start:04d}.json"}')
    print('ids:', ' '.join(r['item'] for r in batch))
    return 0


def cmd_export(a):
    out_dir, work, pool, labelled, start, batch = context(a)
    if not batch:
        print('Nothing left to label.'); return 0
    bp, new = blind_path(start, batch, out_dir)
    prompt = work / f'prompt_{tag(start, len(batch))}.md'
    inbox = work / f'inbox_{tag(start, len(batch))}.json'
    if a.dry_run:
        print(f'[dry-run] blind: {bp} ({"new" if new else "reuse existing"}); prompt: {prompt}; '
              f'paste reply into: {inbox}')
        return 0
    work.mkdir(parents=True, exist_ok=True)
    if new:
        with bp.open('xb') as f:
            f.write(blind_bytes(batch))
    if prompt.exists():
        print(f'{prompt} already exists; left unchanged.')
    else:
        prompt.write_text(build_prompt(batch))
    print(f'blind file: {bp}\nprompt:     {prompt}\nNext: paste the prompt into Codex (GPT-6 Astra, high), save the '
          f'JSON reply as {inbox}, then run: [--batch-size {a.batch_size}] import --start {start}')
    return 0


def batch_for_start(a, pool, labelled):
    start = a.start
    batch = []
    for r in pool[start:]:
        if r['item'] in labelled or len(batch) == a.batch_size:
            break
        batch.append(r)
    return batch


def cmd_import(a, reply_text=None, labeller=None, method=None):
    out_dir = Path(a.out) if a.out else C.CORPUS
    work = Path(a.out) if a.out else C.PIPE / 'batches'
    if (C.CORPUS / f'labels_{a.start:04d}.json').exists() or (out_dir / f'labels_{a.start:04d}.json').exists():
        print(f'labels_{a.start:04d}.json already exists; nothing to do (idempotent).'); return 0
    pool = C.load_candidates()
    labelled, _ = C.load_labels([out_dir])
    batch = batch_for_start(a, pool, labelled)
    if not batch:
        print('No unlabelled items at that start index.'); return 1
    bp, new = blind_path(a.start, batch, out_dir)
    if new:
        print(f'Blind file for this batch not found ({bp}); run export first.'); return 1
    if reply_text is None:
        inbox = Path(a.inbox) if a.inbox else work / f'inbox_{tag(a.start, len(batch))}.json'
        reply_text = inbox.read_text()
    try:
        labels = parse_reply(reply_text)
    except (ValueError, KeyError) as e:
        print(f'INVALID reply: {e}'); return 1
    ids, texts = C.benchmark_blocklist()
    errs = validate(labels, batch, ids, texts)
    if errs:
        print('INVALID reply, nothing saved:'); [print('  -', e) for e in errs]; return 1
    save_labels(labels, batch, a.start, bp, out_dir, labeller or a.labeller, a.reasoning_effort,
                method or a.method, a.dry_run)
    return 0


def cmd_api(a):
    """Disabled-by-default OpenAI call. Untested against the live API (no calls were made while building)."""
    key_set = bool(os.environ.get('OPENAI_API_KEY'))
    if not a.confirm_api:
        print(f'API mode is disabled. Re-run with --confirm-api --model NAME. OPENAI_API_KEY set: {key_set}')
        return 2
    if not key_set:
        print('OPENAI_API_KEY is not set; refusing.'); return 2
    if not a.model:
        print('--model is required in API mode.'); return 2
    out_dir, work, pool, labelled, start, batch = context(a)
    if not batch:
        print('Nothing left to label.'); return 0
    bp, new = blind_path(start, batch, out_dir)
    if a.dry_run:
        print(f'[dry-run] would POST {len(batch)} items ({batch[0]["item"]}..) to {a.api_url} model={a.model}'); return 0
    work.mkdir(parents=True, exist_ok=True)
    if new:
        with bp.open('xb') as f:
            f.write(blind_bytes(batch))
    body = {'model': a.model, 'input': build_prompt(batch)}
    if a.reasoning_effort:
        body['reasoning'] = {'effort': a.reasoning_effort}
    req = urllib.request.Request(a.api_url, data=json.dumps(body).encode(), method='POST', headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']})
    with urllib.request.urlopen(req, timeout=a.timeout) as resp:
        payload = json.loads(resp.read())
    text = payload.get('output_text') or ''.join(
        c.get('text', '') for o in payload.get('output', []) for c in o.get('content', []) or [])
    raw = work / f'api_reply_{tag(start, len(batch))}.json'
    with raw.open('x') as f:
        f.write(text)
    a.start = start
    return cmd_import(a, reply_text=text, labeller=f'OpenAI API ({a.model})',
                      method='single API request per batch, full rubric v2 in prompt')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch-size', type=int, default=50)
    ap.add_argument('--out', help='write blind/prompt/inbox/labels here instead of the real folders (tests)')
    ap.add_argument('--dry-run', action='store_true')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('next')
    sub.add_parser('export')
    im = sub.add_parser('import')
    im.add_argument('--start', type=int, required=True, help='pool index of the batch (e.g. 150)')
    im.add_argument('--inbox', help='reply file (default batches/inbox_NNNN.json)')
    for p in (im,):
        p.add_argument('--labeller', default=C.DEFAULT_LABELLER,
                       help='record exactly the model shown in Codex for this turn')
        p.add_argument('--method', default=C.DEFAULT_METHOD)
    api = sub.add_parser('api')
    api.add_argument('--model')
    api.add_argument('--confirm-api', action='store_true')
    api.add_argument('--api-url', default='https://api.openai.com/v1/responses')
    api.add_argument('--timeout', type=int, default=600)
    for p in (im, api):
        p.add_argument('--reasoning-effort', default='high')
    a = ap.parse_args(argv)
    if a.batch_size < 1:
        ap.error('--batch-size must be >= 1')
    return {'next': cmd_next, 'export': cmd_export, 'import': cmd_import, 'api': cmd_api}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
