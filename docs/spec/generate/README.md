
## Usage

```bash
    » echo '{"foo":[1,2], "bar":[10,4]}' | ./matrix_output.py 'python -c "print($0 + $1)"'
    | foo\bar | 10 | 4 |
    | ------- | -- | - |
    | 1       | 11 | 5 |
    | 2       | 12 | 6 |
```
