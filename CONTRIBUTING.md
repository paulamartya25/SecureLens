# Contributing to SecureLens

Thank you for your interest in contributing to SecureLens! This document provides guidelines and instructions for contributing.

---

## Code of Conduct

- Be respectful and professional
- Focus on medical ethics and patient privacy
- Follow Python best practices
- Maintain code quality and test coverage

---

## Getting Started

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/securelens.git
cd securelens

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Development Guidelines

### Code Style

- Use **Black** for formatting: `black .`
- Use **Flake8** for linting: `flake8 .`
- Use **Pylint** for analysis: `pylint cloud_server/`
- Use **isort** for imports: `isort .`

### Type Hints

All functions should have type hints:

```python
def predict(image: np.ndarray, context: ts.Context) -> dict:
    """
    Run encrypted inference on X-ray image.
    
    Args:
        image: Preprocessed X-ray image
        context: CKKS encryption context
        
    Returns:
        dict: {"prediction": "NORMAL/PNEUMONIA", "confidence": 0.95}
    """
```

### Testing

- Write tests for new features
- Run full test suite: `pytest tests/ -v --cov`
- Maintain >80% code coverage
- Include edge cases and error conditions

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=cloud_server --cov=crypto_layer

# Run specific test
pytest tests/test_api.py::test_predict
```

### Documentation

- Add docstrings to all functions
- Update README if behavior changes
- Document CKKS parameter changes
- Include examples for new features

---

## Commit Guidelines

Use conventional commits:

```
feat: Add differential privacy option
fix: Handle corrupted image uploads
docs: Update encryption parameter docs
test: Add security test cases
refactor: Optimize inference latency
```

---

## Pull Request Process

1. **Update your branch** with latest main:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run all checks locally**:
   ```bash
   pytest tests/ -v
   black --check .
   flake8 .
   ```

3. **Create pull request** with:
   - Clear description of changes
   - Motivation/problem solved
   - Testing performed
   - Backwards compatibility notes

4. **Link issues**: Reference any related issues (#123)

5. **Code review**: Address reviewer feedback

---

## Areas for Contribution

### Priority Areas (High Impact)

- [ ] **ReLU Approximation** — Polynomial approximations for deeper HE
- [ ] **GPU Acceleration** — CUDA-based SEAL integration
- [ ] **Multi-disease** — Extend to ChestX-ray14 dataset
- [ ] **HIPAA Audit** — Formal security audit
- [ ] **Federated Learning** — Hospital network training

### Medium Priority

- [ ] Performance optimization
- [ ] Better error messages
- [ ] Enhanced logging
- [ ] API authentication
- [ ] Web UI improvements

### Low Priority

- [ ] Documentation improvements
- [ ] Code style cleanup
- [ ] Test coverage increase
- [ ] Dependency updates

---

## Testing Requirements

All submissions must:

1. Pass unit tests: `pytest tests/`
2. Pass type checking: `mypy cloud_server/`
3. Pass linting: `flake8 .`
4. Have >80% code coverage
5. Include docstrings
6. Have at least one test case per function

---

## Medical/Regulatory Considerations

When contributing medical-related features:

1. **Privacy-First**: Always assume patient data could be leaked
2. **Accuracy**: Medical accuracy is non-negotiable
3. **Compliance**: Consider HIPAA/GDPR/DPDP implications
4. **Testing**: Include both accuracy and edge case tests
5. **Documentation**: Clearly mark limitations and disclaimers

---

## Reporting Issues

When reporting bugs:

1. Use clear title describing the issue
2. Include steps to reproduce
3. Provide expected vs actual behavior
4. Include relevant logs/error messages
5. Specify Python version and OS

---

## Security

If you discover a security vulnerability:

1. **Do NOT** create a public GitHub issue
2. Email: securelens-security@example.com
3. Include description and proof of concept
4. Allow 90 days for fix before disclosure

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

- Open a GitHub issue with `[question]` prefix
- Check existing documentation and FAQs
- Review discussion threads

---

**Thank you for contributing to SecureLens! 🎉**

Your work helps advance privacy-preserving medical AI for everyone.
