"""Pure-python repeat, grid and MC-uncertainty protocol (no trainer imports)."""
from __future__ import annotations
import hashlib,math

def get_grid(text,largest):
    if text.strip().lower()=='auto':
        values=[64,256,1024,4096,8192,16384]
        return sorted(set([s for s in values if s<=largest]+[largest]))
    try:values=[int(s.strip()) for s in text.split(',') if s.strip()]
    except ValueError as e:raise ValueError('sample-grid requires comma-separated integer values') from e
    if len(values)!=len(set(values)) or any(v<2 or v>largest for v in values) or largest not in values:
        raise ValueError('grid must contain unique S >= 2 and include --samples')
    return sorted(values)


def deterministic_seed(dataset,mode,seed,split,batch,repeat):
    data=f'stage1-v3|{dataset}|{mode}|{seed}|{split}|{batch}|{repeat}'
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:7], 'big')


def mean_se(rows):
    """MC-SE is the between-independent-repeat standard error for fixed checkpoint."""
    keys=set(rows[0])
    out={}
    for key in keys:
        values=[v[key] for v in rows]
        if not all(isinstance(v,(int,float)) and math.isfinite(float(v)) for v in values):
            continue
        import statistics
        center=statistics.mean(values)
        sd=statistics.stdev(values) if len(values)>1 else None
        se=sd/math.sqrt(len(values)) if sd is not None else None
        out[key]={'mean':center,'sd_between_repeats':sd,'mc_se':se,
                  'ci95_normal_halfwidth':(1.96*se if se is not None else None)}
    return out


