import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import time
import sys

# Configure UTF-8 for console output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def execute_notebook(notebook_path='Linear_Regression_and_Data_Normalization_Practice.ipynb'):
    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    print(f"Executing all cells in {notebook_path} from top to bottom...")
    start_time = time.time()
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
    except Exception as e:
        print(f"Execution Error: {e}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"Execution completed successfully in {elapsed:.2f} seconds.")

    # Count cells and verify status
    total_cells = len(nb.cells)
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    md_cells = [c for c in nb.cells if c.cell_type == 'markdown']

    successful_cells = 0
    failed_cells = 0

    for idx, cell in enumerate(code_cells, 1):
        has_error = False
        for output in cell.outputs:
            if output.output_type == 'error':
                has_error = True
                print(f"Error in Code Cell {idx} ({output.ename}): {output.evalue}")
                failed_cells += 1
                break
        if not has_error:
            successful_cells += 1

    print(f"\nExecution Summary:")
    print(f"Total cells: {total_cells}")
    print(f"Markdown cells: {len(md_cells)}")
    print(f"Total code cells: {len(code_cells)}")
    print(f"Successful code cells: {successful_cells}")
    print(f"Failed code cells: {failed_cells}")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully saved fully executed notebook with all outputs to {notebook_path}.")

if __name__ == '__main__':
    execute_notebook()
