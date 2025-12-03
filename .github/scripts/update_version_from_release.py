#!/usr/bin/env python3
"""
Script to update pyproject.toml version based on GitHub release tag.

This script takes a release tag (like 'v1.0.0' or '1.0.0') and updates
the version field in pyproject.toml accordingly.
"""

import re
import sys
import argparse
from pathlib import Path


def normalize_version(tag_version):
    """
    Normalize a version tag to a clean version string.

    Args:
        tag_version: Version from GitHub release tag (e.g., 'v1.0.0', '1.0.0', 'v1.0.0-beta.1')

    Returns:
        str: Clean version string (e.g., '1.0.0', '1.0.0b1')
    """
    # Remove 'v' prefix if present
    version = tag_version.lstrip("v")

    # Convert beta/alpha/rc notation to PEP 440 format
    # v1.0.0-beta.1 -> 1.0.0b1
    # v1.0.0-alpha.2 -> 1.0.0a2
    # v1.0.0-rc.1 -> 1.0.0rc1
    version = re.sub(r"-beta\.?(\d+)", r"b\1", version)
    version = re.sub(r"-alpha\.?(\d+)", r"a\1", version)
    version = re.sub(r"-rc\.?(\d+)", r"rc\1", version)

    print(f"Normalized version: {tag_version} -> {version}")
    return version


def validate_version(version):
    """
    Validate that the version follows PEP 440 format.

    Args:
        version: Version string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    # Basic PEP 440 version pattern
    pattern = r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$"

    if re.match(pattern, version):
        print(f"Version {version} is valid")
        return True
    else:
        print(f"Warning: Version {version} may not be PEP 440 compliant")
        return False


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
    original_content = content

    # Update version line
    updated_content = re.sub(
        r'^version = ".*"', f'version = "{new_version}"', content, flags=re.MULTILINE
    )

    if content == updated_content:
        raise ValueError("No version line found to update in pyproject.toml")

    pyproject_file.write_text(updated_content)
    print(f"Updated {pyproject_path} with version {new_version}")

    # Show the change
    old_version_match = re.search(r'^version = "(.*)"', original_content, re.MULTILINE)
    if old_version_match:
        old_version = old_version_match.group(1)
        print(f"Version changed: {old_version} -> {new_version}")

    return new_version


def main():
    """Main function to orchestrate version update from release tag."""
    parser = argparse.ArgumentParser(
        description="Update pyproject.toml version from GitHub release tag"
    )
    parser.add_argument(
        "tag_version", help="The release tag version (e.g., 'v1.0.0' or '1.0.0')"
    )
    parser.add_argument(
        "--pyproject", default="pyproject.toml", help="Path to pyproject.toml file"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate version format"
    )

    args = parser.parse_args()

    print("=== Updating version from release tag ===")
    print(f"Release tag: {args.tag_version}")

    # Normalize the version
    normalized_version = normalize_version(args.tag_version)

    # Validate if requested
    if args.validate:
        validate_version(normalized_version)

    # Update pyproject.toml
    try:
        update_pyproject_version(normalized_version, args.pyproject)
        print(f"SUCCESS: Updated version to {normalized_version}")

        # Output for GitHub Actions
        print(f"RELEASE_VERSION={normalized_version}")

        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
