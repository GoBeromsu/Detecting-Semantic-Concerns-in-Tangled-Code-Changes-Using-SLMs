#!/usr/bin/env python3
"""
RQ1 Analysis Runner
Simple script to run all RQ1 analysis scripts with YAML configuration.
"""

import yaml
import subprocess
import sys
import json
from pathlib import Path

def load_config():
    """Load YAML configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_msg_impact_analysis(script_dir, project_root, script_config):
    """Run msg_impact_analysis with config."""
    script_path = script_dir / script_config['script']
    
    # Create temp config for this script
    temp_config = {
        'models': script_config['models'],
        'project_root': str(project_root)
    }
    
    temp_config_path = script_dir / "temp_msg_config.json"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        json.dump(temp_config, f, indent=2)
    
    try:
        cmd = [sys.executable, str(script_path), '--config', str(temp_config_path)]
        result = subprocess.run(cmd, cwd=script_dir)
        return result.returncode == 0
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()

def run_script_with_files(script_dir, project_root, script_config):
    """Run script with input files."""
    script_path = script_dir / script_config['script']
    input_files = [str(project_root / f) for f in script_config['input_files']]
    
    cmd = [sys.executable, str(script_path)] + input_files
    result = subprocess.run(cmd, cwd=script_dir)
    return result.returncode == 0

def main():
    """Main execution."""
    script_dir = Path(__file__).parent
    config = load_config()
    project_root = (script_dir / config.get('project_root', '../../../')).resolve()
    
    print("🎯 Running RQ1 Analysis Scripts")
    print("=" * 40)
    
    # Get execution order
    execution_order = config.get('execution_order', list(config['scripts'].keys()))
    
    # Run each script
    for script_name in execution_order:
        script_config = config['scripts'][script_name]
        print(f"\n🚀 Running {script_name}...")
        
        try:
            if script_name == 'msg_impact_analysis':
                success = run_msg_impact_analysis(script_dir, project_root, script_config)
            else:
                success = run_script_with_files(script_dir, project_root, script_config)
            
            if success:
                print(f"✅ {script_name} completed successfully")
            else:
                print(f"❌ {script_name} failed")
                
        except Exception as e:
            print(f"❌ {script_name} failed with error: {e}")
    
    print("\n🎉 All scripts executed!")

if __name__ == "__main__":
    main()