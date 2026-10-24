"""Verifie que requirements.txt couvre les imports tiers reels du projet."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "biasdetect", ROOT / "modules", ROOT / "experiments"]
SCAN_FILES = [ROOT / "main.py"]

LOCAL = {"biasdetect", "modules", "experiments", "tests"}
STDLIB = set(sys.stdlib_module_names) | {"__future__"}

IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "sentence_transformers": "sentence-transformers",
}
TRANSITIVE = {"torch": "sentence-transformers"}

EXPECTED = {
    "streamlit": "1.66.0",
    "spacy": "3.8.16",
    "sentence-transformers": "6.1.0",
    "transformers": "5.20.2",
    "numpy": "2.5.4",
    "pandas": "3.0.6",
    "scikit-learn": "1.9.1",
    "scipy": "1.18.1",
    "rdflib": "7.6.0",
    "PyYAML": "6.0.3",
    "openai": "3.35.1",
    "datasets": "5.1.0",
    "tqdm": "4.70.1",
}


def _normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _python_files():
    files = list(SCAN_FILES)
    for base in SCAN_DIRS:
        files.extend(sorted(base.rglob("*.py")))
    return files


def _third_party_imports():
    found = {}
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                modules = [node.module] if node.module else []
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                if root in STDLIB or root in LOCAL:
                    continue
                found.setdefault(root, set()).add(path.relative_to(ROOT).as_posix())
    return found


def _declared_names():
    names = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        names.add(_normalize(match.group(1)))
    return names


def test_all_third_party_imports_are_declared():
    declared = _declared_names()
    missing = {}
    for root, files in _third_party_imports().items():
        dist = TRANSITIVE.get(root) or IMPORT_TO_DIST.get(root, root)
        if _normalize(dist) not in declared:
            missing[root] = sorted(files)
    assert not missing, f"imports non couverts par requirements.txt : {missing}"


def test_requirements_are_pinned_to_expected_versions():
    text = (ROOT / "requirements.txt").read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert len(lines) == 13
    assert text.count("==") == 13
    declared = {}
    for line in lines:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==(\S+)", line)
        assert match, f"ligne non epinglee avec == : {line!r}"
        declared[_normalize(match.group(1))] = match.group(2)
    assert declared == {_normalize(name): version for name, version in EXPECTED.items()}
    assert "torch" not in declared


def test_dev_requirements_and_no_train_file():
    dev = [line.strip() for line in (ROOT / "requirements-dev.txt").read_text().splitlines() if line.strip()]
    assert dev == ["pytest==9.1.1"]
    assert not (ROOT / "requirements-train.txt").exists()


def test_readme_documents_install():
    readme = (ROOT / "README.md").read_text()
    assert "python -m spacy download en_core_web_sm" in readme
    assert "requirements-dev.txt" in readme
    assert "requirements-train.txt" not in readme
