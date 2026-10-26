# Contributing to TPClass

Thank you for your interest in TPClass! We welcome bug reports, feature requests, documentation improvements, and code contributions.

## How to contribute

### Reporting bugs

Open a GitHub issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs actual behavior
- Environment details (Python version, PyTorch version, CUDA version, OS, GPU model)
- The full error traceback if applicable
- A minimal FASTA input that reproduces the issue, if possible

### Requesting features

Open a GitHub issue with the "enhancement" label. For new substrate-specific transporter mining support, please provide:
- Substrate name + literature references for known transporters
- TC classes likely involved (link to TCDB hierarchy)
- Description keywords (TCDB-curated preferred)
- Initial reference set (5-25 known accessions, or 0 if no literature exists)
- Expected use case (academic / industrial / clinical)

### Submitting code changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Make your changes
4. Add or update tests
5. Run the test suite: `pytest tests/`
6. Update documentation if you changed user-facing behavior
7. Commit with a clear message: `git commit -m "Add support for X"`
8. Push to your fork: `git push origin feature/my-change`
9. Open a Pull Request with a description of the changes

### Improving documentation

Documentation improvements are highly valued. Open a PR directly with your changes — no need to file an issue first for typos, clarifications, or new examples.

---

## Development setup

```bash
# Clone the repository
git clone https://github.com/your-lab/TPClass.git
cd TPClass

# Create a development environment
conda create -n tpclass-dev python=3.10
conda activate tpclass-dev

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest tests/

# Run the linter
flake8 tpclass/ tests/
black tpclass/ tests/
```

---

## Code style

- Python 3.10+
- Black formatter (line length 100)
- flake8 linter
- Type hints for all public functions
- Docstrings (Google style) for all public functions and classes
- Module-level docstring describing the purpose

---

## Testing

- All new features must include unit tests
- Run `pytest tests/` before submitting a PR
- Aim for >80% code coverage on new code

---

## Adding a new case study

To add support for a new substrate (e.g., a new metabolite or drug):

1. Curate 5-25 reference protein accessions from literature
2. Create `data/refs/<substrate>_refs.fasta` (UniProt FASTA)
3. Compute reference embeddings: `python scripts/compute_ref_embeddings.py --fasta data/refs/<substrate>_refs.fasta --out models/ref_embeddings_<substrate>.npz`
4. Document the case study in `examples/<substrate>/README.md` with:
   - Substrate biology
   - Reference set curation rationale (with literature citations)
   - TC class whitelist + description keywords
   - Expected output (top candidates with TC families)
5. Add a tutorial in `docs/tutorials/<substrate>_tutorial.md`

---

## Releasing

TPClass uses semantic versioning. To cut a release:

1. Update version in `tpclass/__init__.py` and `pyproject.toml`
2. Update `CHANGELOG.md` with the release notes
3. Tag the release: `git tag -a v1.x.y -m "Release v1.x.y"`
4. Push the tag: `git push origin v1.x.y`
5. GitHub Actions will automatically build and publish to PyPI

---

## Code of conduct

This project follows the Contributor Covenant Code of Conduct. See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for details.

---

## Questions?

Open an issue or contact the maintainers at [email].
