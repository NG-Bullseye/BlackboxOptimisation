# generate_docs.py
import os
import webbrowser
import subprocess

def create_rst_files(package_path, source_path):
    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('_'):
                module_name = os.path.splitext(file)[0]
                module_path = os.path.join(root, file)
                relative_path = os.path.relpath(module_path, package_path)
                relative_path_no_ext = os.path.splitext(relative_path)[0]
                import_path = relative_path_no_ext.replace(os.sep, '.')

                rst_content = f"{module_name} module\n"
                rst_content += "=" * (len(module_name) + 7) + "\n\n"
                rst_content += f".. automodule:: {import_path}\n"
                rst_content += "   :members:\n"

                with open(os.path.join(source_path, f"{module_name}.rst"), 'w') as rst_file:
                    rst_file.write(rst_content)
def update_index_rst(source_path):
    rst_files = [os.path.splitext(f)[0] for f in os.listdir(source_path) if f.endswith('.rst') and not f.startswith('_')]
    with open(os.path.join(source_path, 'index.rst'), 'a') as index_file:
        for rst in rst_files:
            index_file.write(f'   {rst}\n')
def main():
    package_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    docs_path = os.path.join(package_path, "docs")
    source_path = os.path.join(docs_path, "source")
    os.chdir(docs_path)
    create_rst_files(package_path, source_path)

    subprocess.run(["make", "html"])
    webbrowser.open(os.path.join(docs_path, 'build', 'html', 'index.html'))
    update_index_rst(source_path)

if __name__ == "__main__":
    main()