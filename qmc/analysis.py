from pathlib import Path

def get_stat():
    stat = dict()
    for name in Path("out").iterdir():
        with open(f"{name}/model.py", "rt") as f:
            model = f.read()
        if model[:3] == "'''" or model[:3] == '"""':
            desc = model[3:3+model[3:].index(model[:3])]
        else:
            desc = ''
        with open(f"{name}/history", "rt") as f:
            hist = f.read()
        T = [0]
        E = []
        info = []
        for line in hist.strip().split("\n"):
            if line[0] == '#':
                d = eval(line[1:])
                d['step'] = len(E)
                info.append(d)
            else:
                t,e = map(float, line.split())
                T.append(T[-1]+t)
                E.append(e)
        stat[name.stem] = {'desc'   : desc,
                           "time"   : T[1:],
                           "energy" : E,
                           "info"   : info}
    return stat


