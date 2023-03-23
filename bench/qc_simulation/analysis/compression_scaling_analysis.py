import glob
import pandas as pd
import json
import numpy as np
import sys

def fmt_unit(x, unit):
    return str(np.round(x, 2)) + " " + unit

def main():
    glob_pat = sys.argv[1]
    filenames = glob.glob(glob_pat)
    filenames = sorted(filenames)

    for file in filenames:
        data = json.load(open(file))
        stats = {}
        for atr in ["compress", "decompress"]:
            items = data["compression"][atr]
            if len(items)==0:
                continue
            df = pd.DataFrame(items)
            df["CR"] = df["size_in"]/df["size_out"]
            df["T"] = df["size_in"]/df["time"]
            stats["mean " + atr+" CR"] = df["CR"].mean()
            stats["mean " + atr+" Throughput"] = fmt_unit(df["T"].mean( )/1e9, "GB/s")
            stats[atr+" Count"] = len(df)

        _res = data["result"]
        stats["result"] = (_res["Re"] , _res["Im"])
        stats["Time"] = fmt_unit(data["time"],'s')
        stats["Memory"] = str(data["memory"]/1024/1024) + " MB"
        print(file)
        _prefix = "  "
        last = lambda x: x==len(stats.items())-1
        char = lambda i: "⎬ " if not last(i) else "┕ "
        print("\n".join([
            _prefix+char(i) + " = ".join(map(str, items))
            for i, items in enumerate(stats.items())
        ]))


if __name__=="__main__":
    main()
