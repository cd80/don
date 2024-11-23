# Contributing to Bitcoin Trading RL

Thank you for your interest in contributing to Bitcoin Trading RL! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Style Guide](#style-guide)
9. [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:

```bash
git clone https://github.com/yourusername/don.git
cd don
```

3. Add upstream remote:

```bash
git remote add upstream https://github.com/cd80/don.git
```

4. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

5. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development Setup

1. Install pre-commit hooks:

```bash
pre-commit install
```

2. Configure git:

```bash
git config --local core.autocrlf input
git config --local core.eol lf
```

3. Set up your IDE:

- Use Python 3.8+
- Enable type checking
- Configure black formatter
- Enable flake8 linting

## Making Changes

1. Create a new branch:

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our [style guide](#style-guide)

3. Commit your changes:

```bash
git add .
git commit -m "Description of changes"
```

4. Keep your branch updated:

```bash
git fetch upstream
git rebase upstream/main
```

## Testing

1. Run tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=src tests/
```

2. Add tests for new features:

- Place tests in the `tests/` directory
- Follow existing test patterns
- Ensure good test coverage
- Include edge cases

## Documentation

1. Update documentation for changes:

- Update relevant guides in `docs/guides/`
- Add API documentation for new features
- Include docstrings for new functions/classes
- Update example notebooks if needed

2. Build documentation:

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Pull Request Process

1. Update your branch:

```bash
git fetch upstream
git rebase upstream/main
```

2. Run quality checks:

```bash
# Format code
black src/ tests/

# Run linter
flake8 src/ tests/

# Run type checking
mypy src/

# Run tests
pytest
```

3. Push changes:

```bash
git push origin feature/your-feature-name
```

4. Create Pull Request:

- Use a clear title
- Describe changes in detail
- Reference any related issues
- Include test results
- Add screenshots if relevant

5. Review Process:

- Address review comments
- Update PR as needed
- Maintain clean commit history

## Style Guide

### Python Code Style

1. Follow PEP 8 with these modifications:

- Line length: 88 characters (black default)
- Use double quotes for strings
- Use trailing commas in multi-line structures

2. Type Hints:

```python
def function_name(param1: str, param2: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Function description.

    Args:
        param1: Parameter description
        param2: Parameter description

    Returns:
        Description of return value
    """
    pass
```

3. Class Structure:

```python
class ClassName:
    """Class description."""

    def __init__(self, param1: str):
        """Initialize class."""
        self.param1 = param1

    def method_name(self) -> None:
        """Method description."""
        pass
```

### Documentation Style

1. Docstrings:

- Use Google style
- Include types
- Describe parameters and returns
- Add examples for complex functions

2. Markdown:

- Use ATX headers (#)
- Wrap at 80 characters
- Include code blocks with syntax highlighting
- Use relative links

### Commit Messages

1. Format:

```
type(scope): description

[optional body]
[optional footer]
```

2. Types:

- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

Example:

```
feat(arbitrage): add statistical arbitrage strategy

- Implement pairs trading
- Add cointegration analysis
- Include risk management
- Add unit tests

Closes #123
```

## Community

### Getting Help

1. Check existing documentation
2. Search issues
3. Ask in discussions
4. Contact maintainers

### Reporting Issues

1. Use issue templates
2. Include reproducible examples
3. Provide system information
4. Add relevant logs

### Feature Requests

1. Check existing issues/discussions
2. Use feature request template
3. Provide clear use cases
4. Include implementation ideas

## Recognition

Contributors will be:

1. Added to CONTRIBUTORS.md
2. Mentioned in release notes
3. Recognized in documentation

## Contact

- **Maintainer**: Kim, Sungwoo
- **Email**: rkwk0112@gmail.com
- **GitHub**: [cd80](https://github.com/cd80)
- **Project Repository**: https://github.com/cd80/don

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

Feel free to:

- Open an issue: https://github.com/cd80/don/issues
- Contact maintainers: rkwk0112@gmail.com

Thank you for contributing to Bitcoin Trading RL!
