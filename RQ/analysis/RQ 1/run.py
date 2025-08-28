#!/usr/bin/env python3
"""
RQ1 Analysis Runner
Simple script to run all RQ1 analysis scripts with YAML configuration.
"""

import yaml
import subprocess
import sys
from pathlib import Path

def load_config():
    """Load YAML configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_script(script_dir, project_root, script_config):
    """Run any script."""
    script_path = script_dir / script_config['script']
    cmd = [sys.executable, str(script_path)]
    
    # Add input files if they exist
    if 'input_files' in script_config:
        input_files = [str(project_root / f) for f in script_config['input_files']]
        cmd.extend(input_files)
    
    result = subprocess.run(cmd, cwd=script_dir)
    return result.returncode == 0

def main():
    """Main execution."""
    script_dir = Path(__file__).parent
    config = load_config()
    project_root = (script_dir / '../../../').resolve()
    
    print("🎯 Running RQ1 Analysis Scripts")
    print("=" * 40)
    
    # Get execution order
    execution_order = config.get('execution_order', list(config['scripts'].keys()))
    
    # Run each script
    for script_name in execution_order:
        script_config = config['scripts'][script_name]
        print(f"\n🚀 Running {script_name}...")
        
        try:
            success = run_script(script_dir, project_root, script_config)
            
            if success:
                print(f"✅ {script_name} completed successfully")
            else:
                print(f"❌ {script_name} failed")
                
        except Exception as e:
            print(f"❌ {script_name} failed with error: {e}")
    
    print("\n🎉 All scripts executed!")

if __name__ == "__main__":
    main()