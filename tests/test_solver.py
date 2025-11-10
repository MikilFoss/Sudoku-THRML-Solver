"""
Tests for the Sudoku solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
from src.sudoku_model import SudokuModel
from src.solver import SudokuSolver
from src.puzzle_generator import PuzzleGenerator


def test_sudoku_model_creation():
    """Test creating Sudoku models of different sizes."""
    # Test 9×9
    model = SudokuModel(9)
    assert model.grid_size == 9
    assert model.box_size == 3
    assert model.n_cells == 81
    assert len(model.nodes) == 81
    
    # Test 16×16
    model = SudokuModel(16)
    assert model.grid_size == 16
    assert model.box_size == 4
    assert model.n_cells == 256
    
    # Test 25×25
    model = SudokuModel(25)
    assert model.grid_size == 25
    assert model.box_size == 5
    assert model.n_cells == 625
    
    print("✓ Model creation test passed")


def test_constraint_creation():
    """Test that constraints are created correctly."""
    model = SudokuModel(9)
    
    # Check row constraints
    assert len(model.row_constraints) == 9
    for row in model.row_constraints:
        assert len(row) == 9
    
    # Check column constraints
    assert len(model.col_constraints) == 9
    for col in model.col_constraints:
        assert len(col) == 9
    
    # Check box constraints
    assert len(model.box_constraints) == 9
    for box in model.box_constraints:
        assert len(box) == 9
    
    print("✓ Constraint creation test passed")


def test_energy_computation():
    """Test energy computation."""
    model = SudokuModel(9)
    
    # Perfect solution (no violations)
    perfect_state = jnp.array([
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9
    ])
    
    energy = model.compute_energy(perfect_state)
    assert energy == 0, f"Expected energy 0 for perfect solution, got {energy}"
    
    # State with violations
    bad_state = jnp.ones(81, dtype=int)  # All cells = 1 (many violations)
    energy = model.compute_energy(bad_state)
    assert energy > 0, f"Expected energy > 0 for bad state, got {energy}"
    
    print("✓ Energy computation test passed")


def test_puzzle_generation():
    """Test puzzle generation."""
    generator = PuzzleGenerator(9)
    key = jax.random.PRNGKey(0)
    
    # Generate complete grid
    complete_grid = generator.generate_complete_grid(key)
    assert len(complete_grid) == 81
    assert jnp.all(complete_grid >= 1) and jnp.all(complete_grid <= 9)
    
    # Check if it's a valid solution
    model = SudokuModel(9)
    energy = model.compute_energy(complete_grid)
    assert energy == 0, f"Generated grid should be valid, got energy {energy}"
    
    # Generate puzzle
    key, subkey = jax.random.split(key)
    puzzle, solution = generator.generate_puzzle("medium", subkey)
    assert len(puzzle) == 81
    assert len(solution) == 81
    
    # Check that puzzle has some empty cells
    n_clues = jnp.sum(puzzle != 0)
    assert n_clues > 0 and n_clues < 81
    
    print("✓ Puzzle generation test passed")


def test_solver_initialization():
    """Test solver initialization."""
    model = SudokuModel(9)
    solver = SudokuSolver(model, beta=2.0)
    
    # Create a simple puzzle
    puzzle = jnp.zeros(81, dtype=int)
    puzzle = puzzle.at[0].set(5)
    puzzle = puzzle.at[10].set(3)
    
    key = jax.random.PRNGKey(0)
    initial_state, clamped_cells = solver.initialize_state(puzzle, key)
    
    # Check that clamped cells are preserved
    assert initial_state[0] == 5
    assert initial_state[10] == 3
    assert clamped_cells[0] == True
    assert clamped_cells[10] == True
    
    # Check that empty cells are filled
    assert jnp.all(initial_state >= 1) and jnp.all(initial_state <= 9)
    
    print("✓ Solver initialization test passed")


def test_block_creation():
    """Test block creation strategies."""
    model = SudokuModel(9)
    solver = SudokuSolver(model)
    
    # Test row strategy
    blocks = solver.create_blocks("row")
    assert len(blocks) == 2
    assert len(blocks[0]) + len(blocks[1]) == 81
    
    # Test column strategy
    blocks = solver.create_blocks("col")
    assert len(blocks) == 2
    assert len(blocks[0]) + len(blocks[1]) == 81
    
    # Test checkerboard strategy
    blocks = solver.create_blocks("checkerboard")
    assert len(blocks) == 2
    assert len(blocks[0]) + len(blocks[1]) == 81
    
    print("✓ Block creation test passed")


def test_simple_solve():
    """Test solving a simple puzzle."""
    # Create a nearly complete puzzle
    puzzle = jnp.array([
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 0  # Last cell empty
    ])
    
    model = SudokuModel(9)
    solver = SudokuSolver(model, beta=3.0)
    
    key = jax.random.PRNGKey(0)
    initial_state, clamped_cells = solver.initialize_state(puzzle, key)
    
    key, subkey = jax.random.split(key)
    final_state, stats = solver.solve(
        initial_state=initial_state,
        clamped_cells=clamped_cells,
        n_iterations=500,
        block_strategy="checkerboard",
        key=subkey,
        verbose=False
    )
    
    # Should find the solution (last cell should be 9)
    assert stats['converged'], "Should solve a nearly complete puzzle"
    assert final_state[-1] == 9, "Last cell should be 9"
    
    print("✓ Simple solve test passed")


def test_scalability():
    """Test that the solver works with different grid sizes."""
    for grid_size in [4, 9, 16]:
        model = SudokuModel(grid_size)
        assert model.grid_size == grid_size
        assert len(model.nodes) == grid_size * grid_size
        
        generator = PuzzleGenerator(grid_size)
        key = jax.random.PRNGKey(42)
        complete_grid = generator.generate_complete_grid(key)
        
        # Verify it's a valid solution
        energy = model.compute_energy(complete_grid)
        assert energy == 0, f"Grid size {grid_size}: expected energy 0, got {energy}"
    
    print("✓ Scalability test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Tests")
    print("=" * 60)
    print()
    
    test_sudoku_model_creation()
    test_constraint_creation()
    test_energy_computation()
    test_puzzle_generation()
    test_solver_initialization()
    test_block_creation()
    test_simple_solve()
    test_scalability()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

