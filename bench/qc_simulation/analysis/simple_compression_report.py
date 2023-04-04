import pandas as pd
import json
import sys

def main():
    file = sys.argv[1]
    data = json.load(open(file))
    rows = []
    for item in data['compression']['compress']:
        k = item.copy()
        k['type']='compress'
        rows.append(k)

    for item in data['compression']['decompress']:
        k = item.copy()
        k['type']='decompress'
        rows.append(k)

    if len(rows) == 0:
        print("Rows:\n", rows)
        return
    df = pd.DataFrame(rows)
    dfc = df[df['type'] == 'compress']
    dfd = df[df['type'] == 'decompress']

    for d in [dfc, dfd]:
        d['Throughput'] = d['size_in'] / d['time']
        d['CR'] = d['size_in'] / d['size_out']

    print("Compression:")
    print(dfc.describe([0.5]))
    print("Decompression:")
    print(dfd.describe([0.5]))

if __name__=="__main__":
    main()
