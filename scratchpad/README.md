# Usage

`python expr.py 30 |mongocat -W -d tensim problem_graphs`

## Search for a tensor expression

```bash
⟩ mongocat -F -d tensim expr_graphs --query '{"n_qubits":100}'
[...bson....]
```

## Optimize a tensor expression

```bash
ptyhon contr_sheme.py --ordering <ordering>
```

where ordering is one of 'qbb' or 'nghs'


## Compositions

The `contr_sheme.py` script accepts input in json, which means you can combine the 
commands. Feel the power of UNIX philosophy!

### Create tensor expression and pipe it to ordering

```bash
⟩ python expr.py 47 --qaoa-layers 1 | python contr_sheme.py --ordering nghs \
  | mongocat -W -d tensim contr_schemes
```


### Optimize straight from db and store the result

for example, take 100 qubits and p=1 tensor expression, and run QuickBB on it

```bash
⟩ mongocat -F -d tensim expr_graphs --query '{"n_qubits":100, "extra.p":1}' |\
    python contr_sheme.py --ordering qbb \|
    mongocat -W -d tensim contr_schemes
```


## Count objects in DB

```bash

⟩ mongocat -F -d tensim expr_graphs --query '{"n_qubits":100}' | wc
      3  259087 1991556

⟩ mongocat -F -d tensim expr_graphs --query '{"n_qubits":100, "extra.p":1}' | wc
      1   31179  232108

```

# Test

`pytest`
