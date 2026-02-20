#!/usr/bin/env python3
"""
Execute visualization notebook for jury presentation
"""
import json
import subprocess
import sys
import os

# Change to correct directory
os.chdir("/home/abhishek30/Projects/ai_assisted_drug_design/ai_assisted_drug_design")

# Run using nbconvert if available, otherwise use a direct execution
try:
    result = subprocess.run([
        sys.executable, "-m", "nbconvert", 
        "--to", "notebook",
        "--execute", 
        "--ExecutePreprocessor.timeout=300",
        "notebooks/jury_presentation.ipynb",
        "--output", "notebooks/jury_presentation_executed.ipynb"
    ], capture_output=True, text=True, timeout=600)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"⚠️ nbconvert failed with code {result.returncode}")
        raise Exception("nbconvert execution failed")
        
except Exception as e:
    print(f"Using alternative execution method...")
    
    # Alternative: Execute notebook using IPython
    try:
        from nbconvert import exporters
        from nbconvert.preprocessors import ExecutePreprocessor
        import nbformat
        
        # Load notebook
        with open("notebooks/jury_presentation.ipynb") as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute
        ep = ExecutePreprocessor(timeout=300)
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
        
        # Save executed notebook
        with open("notebooks/jury_presentation_executed.ipynb", 'w') as f:
            nbformat.write(nb, f)
        
        print("✅ Notebook executed successfully using IPython preprocessor")
        
    except Exception as e2:
        print(f"❌ Both methods failed: {e2}")
        sys.exit(1)

print("\n✅ Visualization notebook execution complete!")
print("📁 Output location: notebooks/")
