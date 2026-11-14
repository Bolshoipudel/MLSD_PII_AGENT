import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .sources import ROOT


def main():
    notebook_path = ROOT / "notebooks/eda.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="mlsd-jupyter-") as temporary:
        os.environ.setdefault("IPYTHONDIR", str(Path(temporary) / "ipython"))
        os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(Path(temporary) / "runtime"))
        os.environ.setdefault("MPLCONFIGDIR", str(Path(temporary) / "matplotlib"))
        client = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Executed {len(notebook.cells)} cells: {notebook_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
