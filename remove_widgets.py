import nbformat
from pathlib import Path

NOTEBOOK = Path("Day_1_Neural_Networks.ipynb")
nb = nbformat.read(str(NOTEBOOK), as_version=4)

nb.metadata.pop("widgets", None)
for cell in nb.cells:
    cell.metadata.pop("widgets", None)

nbformat.write(nb, str(NOTEBOOK))
