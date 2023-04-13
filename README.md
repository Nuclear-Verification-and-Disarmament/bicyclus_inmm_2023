# Reconstructing Nuclear Fuel Cycle Operations with Nuclear Archaeology

## Contributing: Usage of pre-commit hooks
We follow the [`Black`](https://black.readthedocs.io/en/stable/) code style.
Run the following command from the root directory to enable use of the
pre-commit hook.
This will automatically run `black` when comitting and thus will ensure proper
formatting of the committed code.
```bash
$ git config --local core.hooksPath .githooks/
```
