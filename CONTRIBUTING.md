# Contributing Guidelines

Thank you for your interest in contributing to Webcam Coordinate Tracker.

## Development Workflow

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/webcam_coordinate_tracker.git
   cd webcam_coordinate_tracker
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
5. Create a topic branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Quality and Standards

- Code Style: Adhere to PEP 8 standards with a line-length limit of 100 characters.
- Type Annotations: Use type hints for all public functions, classes, and methods.
- Testing: Add unit tests under the `tests/` directory for any new logic or bug fixes.
- Verification: Run the test suite before submitting:
  ```bash
  python -m unittest discover tests
  python webcam_coordinate_tracker.py --self-test
  ```

## Submitting Pull Requests

- Keep pull requests focused on a single logical change or feature.
- Write clear, imperative commit messages (for example: "Add focal length validation check").
- Ensure all existing and new unit tests pass cleanly without errors.
- Update documentation in `README.md` if command-line flags or configuration change.
