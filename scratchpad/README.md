# Usage

`python problem_graph_json.py 30 |mongocat -W -d tensim problem_graphs`

## Count objects in DB

```bash

⟩ mongocat -F -d tensim expr_graphs --query '{"n_qubits":100}' | wc
      3  259087 1991556

⟩ mongocat -F -d tensim expr_graphs --query '{"n_qubits":100, "extra.p":1}' | wc
      1   31179  232108

```

# Test

`pytest`
