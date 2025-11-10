"""
Demo script showing how to use the THRML Sudoku Solver.
"""

import jax
import jax.numpy as jnp
from src.sudoku_model import SudokuModel
from src.solver import SudokuSolver
from src.puzzle_generator import PuzzleGenerator


def demo_9x9():
    """Demo solving a 9×9 Sudoku puzzle."""
    print("=" * 60)
    print("Demo: 9×9 Sudoku")
    print("=" * 60)
    print()
    
    # Create generator and generate a puzzle
    generator = PuzzleGenerator(grid_size=9)
    key = jax.random.PRNGKey(42)
    
    print("Generating medium difficulty puzzle...")
    puzzle, solution = generator.generate_puzzle(difficulty="medium", key=key)
    
    print(f"Generated puzzle with {jnp.sum(puzzle != 0)} clues")
    print()
    print("Puzzle:")
    generator.print_grid(puzzle)
    print()
    
    # Create model and solver
    model = SudokuModel(grid_size=9)
    solver = SudokuSolver(model, beta=2.0)
    
    # Initialize state
    key, subkey = jax.random.split(key)
    initial_state, clamped_cells = solver.initialize_state(puzzle, subkey)
    
    print(f"Initial energy: {model.compute_energy(initial_state):.1f}")
    print()
    
    # Solve
    print("Solving with block Gibbs sampling...")
    key, subkey = jax.random.split(key)
    final_state, stats = solver.solve(
        initial_state=initial_state,
        clamped_cells=clamped_cells,
        n_iterations=3000,
        block_strategy="checkerboard",
        key=subkey,
        verbose=True
    )
    
    print()
    print("Solution:")
    generator.print_grid(final_state)
    print()
    
    if stats['converged']:
        print("✓ Valid solution found!")
        # Verify against known solution
        if jnp.all(final_state == solution):
            print("✓ Solution matches the original!")
    else:
        print("✗ No valid solution found (try more iterations)")
    
    print()


def demo_16x16():
    """Demo solving a 16×16 Sudoku puzzle."""
    print("=" * 60)
    print("Demo: 16×16 Sudoku")
    print("=" * 60)
    print()
    
    # Create generator for 16×16 grid
    generator = PuzzleGenerator(grid_size=16)
    key = jax.random.PRNGKey(123)
    
    print("Generating medium difficulty 16×16 puzzle...")
    puzzle, solution = generator.generate_puzzle(difficulty="medium", key=key)
    
    print(f"Generated puzzle with {jnp.sum(puzzle != 0)} clues")
    print()
    print("Puzzle:")
    generator.print_grid(puzzle)
    print()
    
    # Create model and solver
    model = SudokuModel(grid_size=16)
    solver = SudokuSolver(model, beta=2.5)
    
    # Initialize state
    key, subkey = jax.random.split(key)
    initial_state, clamped_cells = solver.initialize_state(puzzle, subkey)
    
    print(f"Initial energy: {model.compute_energy(initial_state):.1f}")
    print()
    
    # Solve (may need more iterations for larger grid)
    print("Solving with block Gibbs sampling...")
    key, subkey = jax.random.split(key)
    final_state, stats = solver.solve(
        initial_state=initial_state,
        clamped_cells=clamped_cells,
        n_iterations=5000,
        block_strategy="checkerboard",
        key=subkey,
        verbose=True
    )
    
    print()
    print("Solution:")
    generator.print_grid(final_state)
    print()
    
    if stats['converged']:
        print("✓ Valid solution found!")
    else:
        print("✗ No valid solution found (try more iterations)")
    
    print()


def demo_custom_puzzle():
    """Demo solving a custom puzzle."""
    print("=" * 60)
    print("Demo: Custom 9×9 Sudoku Puzzle")
    print("=" * 60)
    print()
    
    # Define a custom puzzle (0 for empty cells)
    # This is a well-known puzzle
    custom_puzzle = jnp.array([
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9
    ])
    
    generator = PuzzleGenerator(grid_size=9)
    
    print("Custom puzzle:")
    generator.print_grid(custom_puzzle)
    print()
    
    # Create model and solver
    model = SudokuModel(grid_size=9)
    solver = SudokuSolver(model, beta=2.0)
    
    # Initialize state
    key = jax.random.PRNGKey(999)
    initial_state, clamped_cells = solver.initialize_state(custom_puzzle, key)
    
    print(f"Initial energy: {model.compute_energy(initial_state):.1f}")
    print()
    
    # Solve
    print("Solving...")
    key, subkey = jax.random.split(key)
    final_state, stats = solver.solve(
        initial_state=initial_state,
        clamped_cells=clamped_cells,
        n_iterations=3000,
        block_strategy="checkerboard",
        key=subkey,
        verbose=False
    )
    
    print()
    print("Solution:")
    generator.print_grid(final_state)
    print()
    
    if stats['converged']:
        print("✓ Valid solution found!")
    else:
        print("✗ No valid solution found")
    
    print()


if __name__ == "__main__":
    # Run demos
    demo_9x9()
    print("\n" + "=" * 60 + "\n")
    
    demo_custom_puzzle()
    print("\n" + "=" * 60 + "\n")
    
    # Uncomment to run 16x16 demo (takes longer)
    # demo_16x16()

