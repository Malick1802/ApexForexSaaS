import json, os
base = 'models/specialist'
for p in sorted(os.listdir(base)):
    pdir = os.path.join(base, p)
    if not os.path.isdir(pdir):
        continue
    for d in ['BUY', 'SELL']:
        cfg = os.path.join(pdir, d, 'config.json')
        if os.path.exists(cfg):
            with open(cfg) as f:
                c = json.load(f)
            print(f"{p}/{d}: threshold={c.get('threshold','N/A')}, win_rate={c.get('win_rate','N/A'):.2%}, trades={c.get('trades','N/A')}")
