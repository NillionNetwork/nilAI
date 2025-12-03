#!/usr/bin/env python3
"""
Script to automatically increment Test PyPI package versions.

This script fetches the latest version from Test PyPI, increments the alpha version,
and updates the pyproject.toml file accordingly.
"""

import requests
import re
import sys
from pathlib import Path


def get_latest_version(package_name="nilai-py"):
    """
    Fetch the latest version from Test PyPI.

    Args:
        package_name: Name of the package to check

    Returns:
        str: Latest version string, or "0.0.0a0" if package doesn't exist
    """
    try:
        response = requests.get(
            f"https://test.pypi.org/pypi/{package_name}/json", timeout=10
        )
        if response.status_code == 404:
            # Package doesn't exist yet, start with 0.0.0a1
            print(f"Package {package_name} not found on Test PyPI, starting fresh")
            return "0.0.0a0"

        response.raise_for_status()
        data = response.json()
        versions = list(data["releases"].keys())

        if not versions:
            print("No versions found, starting fresh")
            return "0.0.0a0"

        # Filter for alpha versions and find the latest
        alpha_versions = [v for v in versions if "a" in v]
        if not alpha_versions:
            print("No alpha versions found, starting fresh")
            return "0.0.0a0"

        # Sort versions and get the latest
        alpha_versions.sort(key=lambda x: [int(i) for i in re.findall(r"\d+", x)])
        latest = alpha_versions[-1]
        print(f"Found latest alpha version: {latest}")
        return latest

    except Exception as e:
        print(f"Error fetching version: {e}")
        return "0.0.0a0"


def increment_version(version):
    """
    Increment the alpha version number.

    Args:
        version: Version string like "0.0.0a1"

    Returns:
        str: Incremented version string like "0.0.0a2"
    """
    # Parse version like "0.0.0a1" or "0.1.0a5"
    match = re.match(r"(\d+)\.(\d+)\.(\d+)a(\d+)", version)
    if match:
        major, minor, patch, alpha = match.groups()
        new_alpha = int(alpha) + 1
        new_version = f"{major}.{minor}.{patch}a{new_alpha}"
        print(f"Incrementing {version} -> {new_version}")
        return new_version
    else:
        # If no match, start with a1
        print(f"Could not parse version {version}, defaulting to 0.0.0a1")
        return "0.0.0a1"


def update_pyproject_version(new_version, pyproject_path="pyproject.toml"):
    """
    Update the version in pyproject.toml file.

    Args:
        new_version: New version string to set
        pyproject_path: Path to pyproject.toml file

    Returns:
        str: The new version that was set
    """
    pyproject_file = Path(pyproject_path)

    if not pyproject_file.exists():
        raise FileNotFoundError(f"Could not find {pyproject_path}")

    content = pyproject_file.read_text()

    # Update version line
    updated_content = re.sub(
        r'^version = ".*"', f'version = "{new_version}"', content, flags=re.MULTILINE
    )

    if content == updated_content:
        print("Warning: No version line found to update in pyproject.toml")

    pyproject_file.write_text(updated_content)
    print(f"Updated {pyproject_path} with version {new_version}")
    return new_version


def main():
    """Main function to orchestrate version update."""
    print("=== Updating package version ===")

    # Get latest version from Test PyPI
    latest_version = get_latest_version()
    print(f"Latest version from Test PyPI: {latest_version}")

    # Increment version
    new_version = increment_version(latest_version)
    print(f"New version: {new_version}")

    # Update pyproject.toml
    update_pyproject_version(new_version)

    # Output for GitHub Actions (using newer syntax)
    print(f"NEW_VERSION={new_version}")

    return new_version


if __name__ == "__main__":
    try:
        version = main()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
