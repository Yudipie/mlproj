"""Smoke-test: import every project dependency and print its version."""
import importlib
import sys

# (display name, importable module name)
PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("torch", "torch"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("pytest", "pytest"),
    ("httpx", "httpx"),
    ("jupyter", "jupyter_core"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]


def main():
    failures = []
    for display_name, module_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{display_name:<15} {version}")
        except ImportError as e:
            print(f"{display_name:<15} FAILED TO IMPORT ({e})")
            failures.append(display_name)

    if failures:
        print(f"\n{len(failures)} package(s) failed to import: {', '.join(failures)}")
        sys.exit(1)

    print(f"\nAll {len(PACKAGES)} packages imported successfully.")


if __name__ == "__main__":
    main()
