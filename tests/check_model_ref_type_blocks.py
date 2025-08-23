import ast
import sys
from pathlib import Path


def extract_block_nodes(
    tree: ast.AST,
) -> tuple[list[ast.stmt] | None, list[ast.stmt] | None]:
    """Locate and extract the two code blocks under a Python version check in the AST.

    This function searches for an `if` statement that checks `sys.version_info` (or similar)
    and returns the body and orelse blocks. This is useful for ensuring that code paths
    for different Python versions remain consistent, which is important for maintainability
    and cross-version compatibility.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # Look for a version check: if sys.version_info ... (tuple compare)
        if (
            (
                isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Attribute)
                and node.test.left.attr == "version_info"
                and isinstance(node.test.comparators[0], ast.Tuple)
            )
            and hasattr(node, "body")
            and hasattr(node, "orelse")
        ):
            return node.body, node.orelse
    return None, None


def normalize_block(block: list[ast.stmt]) -> str:
    """Normalize a block of AST statements for comparison.

    This function removes docstrings and type alias annotations, then unparses the
    remaining statements to source code. This normalization is necessary to compare
    code blocks for logical equivalence, ignoring superficial differences such as
    docstrings or type aliasing, which do not affect runtime behavior.
    """
    lines: list[str] = []
    for stmt in block:
        # Skip docstrings, which are represented as Expr nodes with a string constant.
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue

        # Skip type alias annotations, which are not a factor in our comparison.
        if isinstance(stmt, ast.AnnAssign) and (
            isinstance(stmt.target, ast.Name)
            and stmt.annotation
            and getattr(stmt.annotation, "id", None) == "TypeAlias"
        ):
            continue
        # Unparse the AST node back to source code for comparison.
        src: str = ast.unparse(stmt) if hasattr(ast, "unparse") else ""
        # Remove 'type ' prefix if present, as removing it should yield equivalent code in the other code path.
        src = src.replace("type ", "")
        lines.append(src.strip())
    return "\n".join(lines)


def main() -> None:
    """Entry point: checks that the two code blocks under the version check are consistent.

    This script is intended to be run as a CI or pre-commit check to ensure that
    the code paths for different Python versions do not diverge unintentionally.
    If the blocks differ, it prints both for manual inspection and exits with an error.
    """
    file_path: Path = Path(__file__).parent.parent / "src" / "horde_model_reference" / "model_reference_records.py"
    with open(file_path, encoding="utf-8") as f:
        source: str = f.read()
    tree: ast.AST = ast.parse(source)
    body: list[ast.stmt] | None
    orelse: list[ast.stmt] | None
    body, orelse = extract_block_nodes(tree)
    if not body or not orelse:
        print("Could not find version check blocks.")
        sys.exit(1)
    norm_body: str = normalize_block(body)
    norm_orelse: str = normalize_block(orelse)
    if norm_body != norm_orelse:
        print("The code blocks under the version check are not consistent!")
        print("Block 1:\n", norm_body)
        print("Block 2:\n", norm_orelse)
        sys.exit(1)
    print("Blocks are consistent.")


if __name__ == "__main__":
    main()
