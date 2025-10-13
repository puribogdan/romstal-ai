#!/usr/bin/env python3
"""
CI guard script to prevent lookaside URL regressions.

This script searches for forbidden lookaside URL patterns in Python files
and fails the build if any are found.
"""

import os
import re
import sys
from pathlib import Path


# Forbidden lookaside URL patterns
FORBIDDEN_PATTERNS = [
    r'lookaside\.facebook\.com/whatsapp_business',
    r'lookaside\.fbsbx\.com/whatsapp_business',
    r'lookaside\.whatsapp_business/attachments'
]

# Regex pattern that combines all forbidden patterns
COMBINED_PATTERN = re.compile('|'.join(FORBIDDEN_PATTERNS), re.IGNORECASE)


def find_python_files(directory):
    """Recursively find all Python files in the given directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common directories that shouldn't be checked
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'venv', '.venv', 'node_modules'}]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files


def check_file_for_lookaside_urls(file_path):
    """Check a single file for forbidden lookaside URL patterns."""
    violations = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            # Check if line contains any forbidden pattern
            if COMBINED_PATTERN.search(line):
                violations.append({
                    'file': file_path,
                    'line': line_num,
                    'content': line.strip()
                })

    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")

    return violations


def main():
    """Main function to check all Python files for lookaside URLs."""
    # Get the project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"[INFO] Checking for lookaside URLs in Python files in: {project_root}")

    # Find all Python files
    python_files = find_python_files(project_root)

    if not python_files:
        print("[ERROR] No Python files found to check.")
        sys.exit(1)

    print(f"[INFO] Found {len(python_files)} Python files to check")

    # Check each file for violations
    all_violations = []
    files_with_violations = set()

    for file_path in python_files:
        violations = check_file_for_lookaside_urls(file_path)
        if violations:
            all_violations.extend(violations)
            files_with_violations.add(file_path)

    # Report results
    if all_violations:
        print("\n[SECURITY VIOLATION] Found lookaside URLs in the following files:\n")

        # Group violations by file for better readability
        violations_by_file = {}
        for violation in all_violations:
            file_path = violation['file']
            if file_path not in violations_by_file:
                violations_by_file[file_path] = []
            violations_by_file[file_path].append(violation)

        for file_path, violations in violations_by_file.items():
            print(f"[FILE] {file_path}:")
            for violation in violations:
                print(f"   Line {violation['line']}: {violation['content']}")
            print()

        print("[ERROR] These URLs bypass the Graph API security model. Use get_media_download_url_from_id() instead.")
        print("[INFO] See WhatsApp Business API documentation for secure media URL handling.")

        sys.exit(1)  # Fail the build
    else:
        print("[SUCCESS] No lookaside URLs found. Security check passed!")
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()