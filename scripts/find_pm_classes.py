"""
Find all PyMC3 classes used in script.

This'll find both call of

    pymc3.Categorical(...

and
    
    from pymc3 import Categorical
    Categorical
"""
import ast
import sys


class ImportVisitor(ast.NodeVisitor):
    def __init__(self, file):
        self.imports = set()

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module.split(".")[0] == "pymc3":
            for name in node.names:
                if name.name[0].isupper():
                    self.imports.add(name.name)


class CallVisitor(ast.NodeVisitor):
    def __init__(self, file, imports):
        self.file = file
        self.imports = imports
        self.classes_used = set()

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in {"pm", "pymc3"}:
                    if node.func.attr[0].isupper():
                        self.classes_used.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            if node.func.id in self.imports:
                self.classes_used.add(node.func.id)


if __name__ == "__main__":
    for file in sys.argv[1:]:
        with open(file) as fd:
            content = fd.read()
        tree = ast.parse(content)

        import_visitor = ImportVisitor(file)
        import_visitor.visit(tree)

        visitor = CallVisitor(file, import_visitor.imports)
        visitor.visit(tree)
        for i in visitor.classes_used:
            print(i)
