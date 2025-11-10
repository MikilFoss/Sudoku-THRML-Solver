"""
Sudoku puzzle generator.

This module generates valid Sudoku puzzles of arbitrary size.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple
from .sudoku_model import SudokuModel
from .solver import SudokuSolver


class PuzzleGenerator:
    """
    Generator for Sudoku puzzles of arbitrary size.
    
    Args:
        grid_size: Size of the Sudoku grid
    """
    
    def __init__(self, grid_size: int = 9):
        self.model = SudokuModel(grid_size)
        self.grid_size = grid_size
    
    def generate_complete_grid(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Generate a valid complete Sudoku grid.
        
        Args:
            key: JAX random key
            
        Returns:
            Complete valid Sudoku grid as flat array
        """
        # Start with a simple valid pattern and permute it
        grid = self._create_base_solution()
        
        # Shuffle rows within box groups
        key, subkey = jax.random.split(key)
        grid = self._shuffle_rows(grid, subkey)
        
        # Shuffle columns within box groups
        key, subkey = jax.random.split(key)
        grid = self._shuffle_cols(grid, subkey)
        
        # Shuffle box rows
        key, subkey = jax.random.split(key)
        grid = self._shuffle_box_rows(grid, subkey)
        
        # Shuffle box columns
        key, subkey = jax.random.split(key)
        grid = self._shuffle_box_cols(grid, subkey)
        
        # Shuffle numbers
        key, subkey = jax.random.split(key)
        grid = self._shuffle_numbers(grid, subkey)
        
        return grid.flatten()
    
    def _create_base_solution(self) -> np.ndarray:
        """Create a basic valid Sudoku solution using a simple pattern."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Fill using a shifted pattern
        # For standard 9x9: row i has pattern starting at (i * box_size + i // box_size) % grid_size
        box_size = self.model.box_size
        
        for row in range(self.grid_size):
            shift = (row % box_size) * box_size + row // box_size
            for col in range(self.grid_size):
                value = (col + shift) % self.grid_size + 1
                grid[row, col] = value
        
        return grid
    
    def _shuffle_rows(self, grid: np.ndarray, key: jax.random.PRNGKey) -> np.ndarray:
        """Shuffle rows within their box groups."""
        grid = grid.copy()
        box_size = self.model.box_size
        
        for box_row in range(box_size):
            start_row = box_row * box_size
            rows = list(range(start_row, start_row + box_size))
            key, subkey = jax.random.split(key)
            shuffled_rows = jax.random.permutation(subkey, jnp.array(rows))
            grid[start_row:start_row + box_size] = grid[shuffled_rows]
        
        return grid
    
    def _shuffle_cols(self, grid: np.ndarray, key: jax.random.PRNGKey) -> np.ndarray:
        """Shuffle columns within their box groups."""
        grid = grid.copy()
        box_size = self.model.box_size
        
        for box_col in range(box_size):
            start_col = box_col * box_size
            cols = list(range(start_col, start_col + box_size))
            key, subkey = jax.random.split(key)
            shuffled_cols = jax.random.permutation(subkey, jnp.array(cols))
            grid[:, start_col:start_col + box_size] = grid[:, shuffled_cols]
        
        return grid
    
    def _shuffle_box_rows(self, grid: np.ndarray, key: jax.random.PRNGKey) -> np.ndarray:
        """Shuffle box rows (groups of rows)."""
        grid = grid.copy()
        box_size = self.model.box_size
        
        box_rows = [grid[i*box_size:(i+1)*box_size] for i in range(box_size)]
        key, subkey = jax.random.split(key)
        shuffled_indices = jax.random.permutation(subkey, jnp.arange(box_size))
        
        result = np.zeros_like(grid)
        for i, idx in enumerate(shuffled_indices):
            result[i*box_size:(i+1)*box_size] = box_rows[idx]
        
        return result
    
    def _shuffle_box_cols(self, grid: np.ndarray, key: jax.random.PRNGKey) -> np.ndarray:
        """Shuffle box columns (groups of columns)."""
        grid = grid.copy()
        box_size = self.model.box_size
        
        box_cols = [grid[:, i*box_size:(i+1)*box_size] for i in range(box_size)]
        key, subkey = jax.random.split(key)
        shuffled_indices = jax.random.permutation(subkey, jnp.arange(box_size))
        
        result = np.zeros_like(grid)
        for i, idx in enumerate(shuffled_indices):
            result[:, i*box_size:(i+1)*box_size] = box_cols[idx]
        
        return result
    
    def _shuffle_numbers(self, grid: np.ndarray, key: jax.random.PRNGKey) -> np.ndarray:
        """Permute the number assignments."""
        grid = grid.copy()
        
        # Create a random permutation of numbers 1 to grid_size
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, jnp.arange(1, self.grid_size + 1))
        
        # Apply permutation
        result = np.zeros_like(grid)
        for old_val in range(1, self.grid_size + 1):
            new_val = perm[old_val - 1]
            result[grid == old_val] = new_val
        
        return result
    
    def create_puzzle(self, 
                     complete_grid: jnp.ndarray,
                     n_clues: Optional[int] = None,
                     key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """
        Create a puzzle by removing cells from a complete grid.
        
        Args:
            complete_grid: Complete valid Sudoku grid
            n_clues: Number of clues to leave (if None, use default based on grid size)
            key: JAX random key
            
        Returns:
            Puzzle array with 0 for empty cells
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Default number of clues based on grid size
        if n_clues is None:
            # For 9x9, typically 25-35 clues
            # Scale proportionally with grid size
            n_clues = int(self.grid_size * self.grid_size * 0.35)
        
        # Select random cells to keep as clues
        n_cells = self.grid_size * self.grid_size
        key, subkey = jax.random.split(key)
        clue_indices = jax.random.choice(subkey, n_cells, shape=(n_clues,), replace=False)
        
        # Create puzzle with zeros for empty cells
        puzzle = jnp.zeros(n_cells, dtype=int)
        puzzle = puzzle.at[clue_indices].set(complete_grid[clue_indices])
        
        return puzzle
    
    def generate_puzzle(self, 
                       difficulty: str = "medium",
                       key: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate a complete puzzle with specified difficulty.
        
        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            key: JAX random key
            
        Returns:
            Tuple of (puzzle, solution)
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Generate complete grid
        key, subkey = jax.random.split(key)
        complete_grid = self.generate_complete_grid(subkey)
        
        # Determine number of clues based on difficulty
        n_cells = self.grid_size * self.grid_size
        if difficulty == "easy":
            n_clues = int(n_cells * 0.45)  # More clues = easier
        elif difficulty == "medium":
            n_clues = int(n_cells * 0.35)
        elif difficulty == "hard":
            n_clues = int(n_cells * 0.25)  # Fewer clues = harder
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        
        # Create puzzle
        key, subkey = jax.random.split(key)
        puzzle = self.create_puzzle(complete_grid, n_clues, subkey)
        
        return puzzle, complete_grid
    
    def load_puzzle_from_file(self, filename: str) -> jnp.ndarray:
        """
        Load a puzzle from a CSV file.
        
        Args:
            filename: Path to CSV file (0 for empty cells)
            
        Returns:
            Puzzle array
        """
        grid = np.loadtxt(filename, delimiter=',', dtype=int)
        
        if grid.shape[0] != self.grid_size or grid.shape[1] != self.grid_size:
            raise ValueError(f"Expected {self.grid_size}×{self.grid_size} grid, got {grid.shape}")
        
        return jnp.array(grid.flatten())
    
    def save_puzzle_to_file(self, puzzle: jnp.ndarray, filename: str):
        """
        Save a puzzle to a CSV file.
        
        Args:
            puzzle: Puzzle array
            filename: Path to output CSV file
        """
        grid = puzzle.reshape(self.grid_size, self.grid_size)
        np.savetxt(filename, grid, delimiter=',', fmt='%d')
    
    def print_grid(self, grid: jnp.ndarray):
        """
        Pretty-print a Sudoku grid.
        
        Args:
            grid: Flat array representing the grid
        """
        grid_2d = grid.reshape(self.grid_size, self.grid_size)
        box_size = self.model.box_size
        
        # Determine width for each cell (to handle larger numbers)
        cell_width = len(str(self.grid_size))
        
        # Print grid with box separators
        for row in range(self.grid_size):
            if row > 0 and row % box_size == 0:
                # Print horizontal separator
                sep = '+'.join(['-' * ((cell_width + 1) * box_size + 1)] * box_size)
                print(sep)
            
            row_str = ""
            for col in range(self.grid_size):
                if col > 0 and col % box_size == 0:
                    row_str += "| "
                
                value = grid_2d[row, col]
                if value == 0:
                    row_str += "." + " " * cell_width
                else:
                    row_str += str(value).rjust(cell_width) + " "
            
            print(row_str)

