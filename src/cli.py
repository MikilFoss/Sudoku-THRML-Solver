"""
Command-line interface for the Sudoku solver.
"""

import argparse
import time
import jax
import jax.numpy as jnp
from .sudoku_model import SudokuModel
from .solver import SudokuSolver
from .puzzle_generator import PuzzleGenerator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="THRML Probabilistic Sudoku Solver"
    )
    
    # Grid configuration
    parser.add_argument(
        "--size",
        type=int,
        default=9,
        help="Size of Sudoku grid (must be a perfect square, default: 9)"
    )
    
    # Puzzle input
    parser.add_argument(
        "--puzzle",
        type=str,
        help="Path to puzzle file (CSV format, 0 for empty cells)"
    )
    
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a new puzzle"
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Difficulty level for generated puzzles (default: medium)"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of warmup iterations (default: 1000)"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of sampling iterations (default: 5000)"
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Inverse temperature parameter (default: 2.0, higher = more deterministic)"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["row", "col", "checkerboard"],
        default="checkerboard",
        help="Block sampling strategy (default: checkerboard)"
    )
    
    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and convergence information"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save solution to file (CSV format)"
    )
    
    args = parser.parse_args()
    
    # Initialize random key
    key = jax.random.PRNGKey(args.seed)
    
    # Create model and generator
    generator = PuzzleGenerator(args.size)
    model = SudokuModel(args.size)
    solver = SudokuSolver(model, beta=args.beta)
    
    print("=" * 60)
    print(f"THRML Probabilistic Sudoku Solver ({args.size}×{args.size})")
    print("=" * 60)
    print()
    
    # Get or generate puzzle
    if args.puzzle:
        print(f"Loading puzzle from {args.puzzle}...")
        puzzle = generator.load_puzzle_from_file(args.puzzle)
        solution = None
    elif args.generate:
        print(f"Generating {args.difficulty} puzzle...")
        key, subkey = jax.random.split(key)
        puzzle, solution = generator.generate_puzzle(args.difficulty, subkey)
        print(f"Generated puzzle with {jnp.sum(puzzle != 0)} clues")
    else:
        print("Error: Either --puzzle or --generate must be specified")
        return
    
    print()
    print("Initial puzzle:")
    print()
    generator.print_grid(puzzle)
    print()
    
    # Initialize state
    print("Initializing solver...")
    key, subkey = jax.random.split(key)
    initial_state, clamped_cells = solver.initialize_state(puzzle, subkey)
    
    initial_energy = model.compute_energy(initial_state)
    print(f"Initial energy: {initial_energy:.1f}")
    print(f"Clamped cells: {jnp.sum(clamped_cells)}/{args.size * args.size}")
    print()
    
    # Solve puzzle
    print("Solving...")
    print(f"Parameters: warmup={args.warmup}, samples={args.samples}, beta={args.beta}, strategy={args.strategy}")
    print()
    
    start_time = time.time()
    
    key, subkey = jax.random.split(key)
    final_state, stats = solver.solve(
        initial_state=initial_state,
        clamped_cells=clamped_cells,
        n_iterations=args.warmup + args.samples,
        block_strategy=args.strategy,
        key=subkey,
        verbose=args.verbose
    )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Iterations: {stats['iterations']}")
    print(f"Final energy: {stats['final_energy']:.1f}")
    print(f"Converged: {'Yes' if stats['converged'] else 'No'}")
    print()
    
    if stats['converged']:
        print("✓ Solution found!")
    else:
        print("✗ No valid solution found (energy > 0)")
        print("  Try increasing --samples or adjusting --beta")
    
    print()
    print("Final state:")
    print()
    generator.print_grid(final_state)
    print()
    
    # Compare with known solution if available
    if solution is not None:
        matches = jnp.sum(final_state == solution)
        total = args.size * args.size
        accuracy = 100.0 * matches / total
        print(f"Accuracy vs. known solution: {accuracy:.1f}% ({matches}/{total} cells)")
        print()
    
    # Save solution if requested
    if args.output:
        generator.save_puzzle_to_file(final_state, args.output)
        print(f"Solution saved to {args.output}")
        print()
    
    # Print energy convergence summary
    if args.verbose and len(stats['energies']) > 0:
        print("=" * 60)
        print("Energy Convergence Summary")
        print("=" * 60)
        print()
        
        energies = stats['energies']
        print(f"Initial energy: {energies[0]:.1f}")
        print(f"Final energy: {energies[-1]:.1f}")
        print(f"Minimum energy: {min(energies):.1f}")
        print(f"Energy reduction: {energies[0] - energies[-1]:.1f}")
        
        # Find when best energy was reached
        best_iter = energies.index(min(energies))
        print(f"Best energy reached at iteration: {best_iter}")
        print()


if __name__ == "__main__":
    main()

