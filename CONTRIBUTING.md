# Contributing to TURBINE-X 2.0

Thank you for your interest in contributing to **TURBINE-X 2.0**! We welcome bug reports, feature enhancements, documentation improvements, and algorithmic extensions.

---

## Code of Conduct
Please ensure a respectful, professional, and collaborative environment. Be constructive in issues and pull request discussions.

---

## Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/tanishaaaaa9/TURBINE-X-2.0.git
   cd TURBINE-X-2.0
   ```

2. **Create a Dedicated Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Running the Automated Test Suite

Before submitting any code changes, ensure all automated tests pass with 100% success rate:
```bash
python -m pytest tests -v
```

---

## Contribution Guidelines

1. **Scientific Integrity:** Do not introduce synthetic or fabricated evaluation numbers. All benchmark metrics must be generated through reproducible scripts on the official temporal partitions.
2. **Preserve Chronological Split:** Any new model or pipeline must strictly fit scalers, thresholds, and parameters on the Development partition (2011–2013). Never leak Calibration (2014) or Locked Test (2015) data into training.
3. **Coding Standards:** Follow PEP 8 guidelines. Write clean docstrings and type annotations.
4. **Pull Requests:** Provide a clear description of changes, link related issues, and attach pytest output.

