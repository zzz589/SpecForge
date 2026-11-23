#!/usr/bin/env python3
import argparse, ast, json, pathlib, re, statistics
from PIL import Image, ImageDraw

KEYS = ["loss", "acc", "lr", "dflash2/selector/loss", "dflash2/selector/greedy_accepted_length"]

def parse(path):
    rows = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"step (\d+): (\{.*\})$", line)
        if not m:
            continue
        try:
            raw = ast.literal_eval(m.group(2))
        except Exception:
            continue
        row = {"step": int(m.group(1))}
        row.update({k: float(raw[k]) for k in KEYS if k in raw})
        rows.append(row)
    return rows

def smooth(values, n=15):
    return [statistics.fmean(values[max(0, i-n+1):i+1]) for i in range(len(values))]

def chart(rows, keys, output, title):
    w, h, pad = 1600, 900, 90
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im)
    d.text((pad, 25), title, fill="black")
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for idx, key in enumerate(keys):
        pts = [(r["step"], r[key]) for r in rows if key in r]
        if not pts: continue
        vals = smooth([v for _, v in pts]); lo, hi = min(vals), max(vals)
        if hi == lo: hi = lo + 1
        xy = [(pad + (w-2*pad)*i/max(1, len(vals)-1), h-pad-(h-2*pad)*(v-lo)/(hi-lo)) for i,v in enumerate(vals)]
        d.line(xy, fill=colors[idx], width=4)
        d.text((pad + idx*420, h-55), f"{key}  last={vals[-1]:.5g}", fill=colors[idx])
    d.rectangle((pad, pad, w-pad, h-pad), outline="#666666", width=2)
    im.save(output)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--log", required=True); p.add_argument("--out", required=True); a=p.parse_args()
    out=pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); rows=parse(a.log)
    if not rows: raise SystemExit("no step metrics found")
    (out/"metrics.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
    summary={"logged_points":len(rows),"first_step":rows[0]["step"],"last_step":rows[-1]["step"]}
    for k in KEYS:
        vals=[r[k] for r in rows if k in r]
        if vals: summary[k]={"first":vals[0],"last":vals[-1],"last_10_mean":statistics.fmean(vals[-10:])}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    chart(rows,["loss","dflash2/selector/loss"],out/"loss_function.png","DFlash2 D-PACE training loss (15-point moving average)")
    chart(rows,["acc","dflash2/selector/greedy_accepted_length"],out/"training_metrics.png","DFlash2 D-PACE training metrics (15-point moving average)")
    (out/"README.md").write_text("# Training curves\n\nGenerated from train.log. Curves use a 15-point moving average; raw values are in metrics.jsonl.\n")
if __name__ == "__main__": main()
