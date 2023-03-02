# Scripts

These are example and helper scripts

## Examples

### Download via http, unpack on the fly

```
╰─λ ./scripts/http_unzip_on_the_fly.sh
[main.py]  bris_5_24_0.txt -> [echo] -> circuits/bris/bris_5_24_0.txt_dummy1.circ
[main.py]  bris_5_24_0.txt -> [echo] -> circuits/bris/bris_5_24_0.txt_dummy2.circ
[main.py]  bris_5_28_0.txt -> [echo] -> circuits/bris/bris_5_28_0.txt_dummy1.circ
[main.py]  bris_5_28_0.txt -> [echo] -> circuits/bris/bris_5_28_0.txt_dummy2.circ
[main.py]  bris_5_32_0.txt -> [echo] -> circuits/bris/bris_5_32_0.txt_dummy1.circ
[main.py]  bris_5_32_0.txt -> [echo] -> circuits/bris/bris_5_32_0.txt_dummy2.circ
[main.py]  bris_5_36_0.txt -> [echo] -> circuits/bris/bris_5_36_0.txt_dummy1.circ
[main.py]  bris_5_36_0.txt -> [echo] -> circuits/bris/bris_5_36_0.txt_dummy2.circ
[main.py]  bris_5_40_0.txt -> [echo] -> circuits/bris/bris_5_40_0.txt_dummy1.circ
[main.py]  bris_5_40_0.txt -> [echo] -> circuits/bris/bris_5_40_0.txt_dummy2.circ
╰─λ tree circuits/
circuits/
└── bris
    ├── bris_5_24_0.txt_dummy1.circ
    ├── bris_5_24_0.txt_dummy2.circ
    ├── bris_5_28_0.txt_dummy1.circ
    ├── bris_5_28_0.txt_dummy2.circ
    ├── bris_5_32_0.txt_dummy1.circ
    ├── bris_5_32_0.txt_dummy2.circ
    ├── bris_5_36_0.txt_dummy1.circ
    ├── bris_5_36_0.txt_dummy2.circ
    ├── bris_5_40_0.txt_dummy1.circ
    └── bris_5_40_0.txt_dummy2.circ

2 directories, 10 file
```
