#!/usr/bin/env python3
"""
Script to update project status and documentation.
This script helps maintain project documentation and status files.
"""

import os
import sys
import yaml
import json
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import re

def get_git_root() -> str:
    """Get git repository root directory."""
    return subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel']
    ).decode('utf-8').strip()

def load_yaml(file_path: str) -> Dict:
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict, file_path: str) -> None:
    """Save data to YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def get_components() -> List[Dict]:
    """Get list of project components."""
    components = []
    root_dir = get_git_root()
    src_dir = os.path.join(root_dir, 'src')
    
    for root, dirs, files in os.walk(src_dir):
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                rel_path = os.path.relpath(
                    os.path.join(root, file),
                    src_dir
                )
                components.append({
                    'path': rel_path,
                    'module': rel_path.replace('/', '.').replace('.py', '')
                })
    
    return components

def get_test_coverage() -> Dict:
    """Get test coverage information."""
    try:
        output = subprocess.check_output(
            ['pytest', '--cov=src', '--cov-report=json'],
            stderr=subprocess.DEVNULL
        )
        with open('coverage.json', 'r') as f:
            coverage = json.load(f)
        os.remove('coverage.json')
        return coverage
    except:
        return {}

def get_documentation_status() -> Dict:
    """Get documentation status."""
    root_dir = get_git_root()
    docs_dir = os.path.join(root_dir, 'docs')
    guides_dir = os.path.join(docs_dir, 'guides')
    
    guides = []
    if os.path.exists(guides_dir):
        for file in os.listdir(guides_dir):
            if file.endswith('.md'):
                guides.append(file[:-3])
    
    return {
        'guides': guides,
        'api_docs': os.path.exists(os.path.join(docs_dir, 'api')),
        'examples': len(os.listdir(os.path.join(root_dir, 'notebooks')))
    }

def update_project_status() -> None:
    """Update PROJECT_STATUS.md file."""
    root_dir = get_git_root()
    status_file = os.path.join(root_dir, 'PROJECT_STATUS.md')
    
    # Get current status
    with open(status_file, 'r') as f:
        current_status = f.read()
    
    # Update last modified date
    current_status = re.sub(
        r'Last updated: \[.*\]',
        f'Last updated: [{datetime.now().strftime("%Y-%m-%d")}]',
        current_status
    )
    
    # Save updated status
    with open(status_file, 'w') as f:
        f.write(current_status)

def update_changelog() -> None:
    """Update CHANGELOG.md file."""
    root_dir = get_git_root()
    changelog_file = os.path.join(root_dir, 'CHANGELOG.md')
    
    # Get current changelog
    with open(changelog_file, 'r') as f:
        current_changelog = f.read()
    
    # Get latest git changes
    try:
        latest_changes = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=format:%s'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8')
        
        # Add to unreleased section if exists
        if '## [Unreleased]' in current_changelog:
            current_changelog = current_changelog.replace(
                '## [Unreleased]',
                f'## [Unreleased]\n\n### Added\n- {latest_changes}'
            )
            
            # Save updated changelog
            with open(changelog_file, 'w') as f:
                f.write(current_changelog)
    except:
        pass

def update_documentation() -> None:
    """Update documentation files."""
    root_dir = get_git_root()
    
    # Update mkdocs.yml
    mkdocs_file = os.path.join(root_dir, 'mkdocs.yml')
    if os.path.exists(mkdocs_file):
        config = load_yaml(mkdocs_file)
        
        # Update nav section
        nav = []
        docs_dir = os.path.join(root_dir, 'docs')
        guides_dir = os.path.join(docs_dir, 'guides')
        
        if os.path.exists(guides_dir):
            guides = []
            for file in sorted(os.listdir(guides_dir)):
                if file.endswith('.md'):
                    name = file[:-3].replace('-', ' ').title()
                    guides.append({name: f'guides/{file}'})
            if guides:
                nav.append({'Guides': guides})
        
        api_dir = os.path.join(docs_dir, 'api')
        if os.path.exists(api_dir):
            nav.append({'API Reference': 'api/'})
        
        config['nav'] = nav
        save_yaml(config, mkdocs_file)

def main() -> None:
    """Main function."""
    try:
        print("Updating project status...")
        update_project_status()
        print("Updated PROJECT_STATUS.md")
        
        print("\nUpdating changelog...")
        update_changelog()
        print("Updated CHANGELOG.md")
        
        print("\nUpdating documentation...")
        update_documentation()
        print("Updated documentation")
        
        print("\nGetting test coverage...")
        coverage = get_test_coverage()
        if coverage:
            total = coverage.get('totals', {}).get('percent_covered', 0)
            print(f"Test coverage: {total:.1f}%")
        
        print("\nGetting documentation status...")
        docs = get_documentation_status()
        print(f"Guides: {len(docs['guides'])}")
        print(f"API docs: {'Yes' if docs['api_docs'] else 'No'}")
        print(f"Example notebooks: {docs['examples']}")
        
        print("\nUpdate completed successfully!")
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
