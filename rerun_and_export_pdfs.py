"""
Script to:
1. Rerun all 3 notebooks from top to bottom:
   - Linear_Regression_and_Data_Normalization_Advertising.ipynb
   - Linear_Regression_and_Data_Normalization_Practice.ipynb
   - 2_1_Supervised_Learning.ipynb
2. Verify all executions complete with 0 errors and save the executed notebooks.
3. Export each executed notebook to HTML using nbconvert HTMLExporter.
4. Convert each HTML to high-quality PDF using headless Chrome/Edge and save in 'PDF results'.
"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import subprocess
import os
import sys
import time

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure nbconvert finds template directories
os.environ['JUPYTER_PATH'] = r'C:\Users\ADMIN\AppData\Local\Programs\Python\Python314\share\jupyter'

NOTEBOOKS = [
    'Linear_Regression_and_Data_Normalization_Advertising.ipynb',
    'Linear_Regression_and_Data_Normalization_Practice.ipynb',
    '2_1_Supervised_Learning.ipynb'
]

OUTPUT_DIR = 'PDF results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detect Chrome or Edge executable
CHROME_PATHS = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
]

browser_exe = None
for p in CHROME_PATHS:
    if os.path.exists(p):
        browser_exe = p
        break

if not browser_exe:
    raise FileNotFoundError("Could not find Chrome or Edge executable for PDF conversion.")
print(f"Using browser for PDF export: {browser_exe}")

def rerun_and_export():
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'lab'

    for nb_name in NOTEBOOKS:
        print(f"\n{'='*60}")
        print(f"Processing: {nb_name}")
        print(f"{'='*60}")
        
        # 1. Read notebook
        with open(nb_name, 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)

        # 2. Execute all cells from top to bottom
        print("Executing all cells from top to bottom...")
        start_time = time.time()
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': '.'}})
        except Exception as e:
            print(f"Error executing {nb_name}: {e}")
            # Save whatever executed
            with open(nb_name, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
            raise e

        elapsed = time.time() - start_time
        print(f"Execution completed in {elapsed:.2f} seconds.")

        # Check for errors in outputs
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        errors = []
        for idx, c in enumerate(code_cells, 1):
            for o in c.outputs:
                if o.output_type == 'error':
                    errors.append((idx, o.ename, o.evalue))
        
        if errors:
            print(f"Found {len(errors)} errors in {nb_name}:")
            for err in errors:
                print(f"  Code cell {err[0]} Error ({err[1]}): {err[2]}")
            sys.exit(1)
        else:
            print(f"All {len(code_cells)} code cells executed successfully with 0 errors!")

        # 3. Save fully executed notebook
        with open(nb_name, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"Saved executed notebook to {nb_name}.")

        # 4. Export to HTML
        html_data, _ = html_exporter.from_notebook_node(nb)
        base_name = os.path.splitext(nb_name)[0]
        temp_html = os.path.join(OUTPUT_DIR, f"{base_name}.html")
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_data)
        print(f"Exported HTML to {temp_html}.")

        # 5. Convert HTML to PDF using headless Chrome/Edge
        pdf_name = os.path.join(OUTPUT_DIR, f"{base_name}.pdf")
        abs_html = os.path.abspath(temp_html)
        abs_pdf = os.path.abspath(pdf_name)

        cmd = [
            browser_exe,
            '--headless=new',
            '--disable-gpu',
            '--no-pdf-header-footer',
            '--run-all-compositor-stages-before-draw',
            '--virtual-time-budget=5000',
            f'--print-to-pdf={abs_pdf}',
            abs_html
        ]
        
        print(f"Generating PDF: {pdf_name}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Browser stderr: {res.stderr}")
            cmd[1] = '--headless'
            res = subprocess.run(cmd, capture_output=True, text=True)

        if os.path.exists(pdf_name):
            size_kb = os.path.getsize(pdf_name) / 1024
            print(f"Successfully generated: {pdf_name} ({size_kb:.1f} KB)")
        else:
            print(f"Failed to generate PDF for {nb_name}!")
            sys.exit(1)
            
        # Clean up temporary html
        if os.path.exists(temp_html):
            os.remove(temp_html)

    print("\n" + "="*60)
    print("ALL 3 NOTEBOOKS RERUN AND EXPORTED TO PDF SUCCESSFULLY!")
    print("="*60)

if __name__ == '__main__':
    rerun_and_export()
