"""
Block Gibbs Sampler for Sudoku.

This module implements efficient block Gibbs sampling for solving Sudoku puzzles.
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Optional
import numpy as np
from .sudoku_model import SudokuModel


class SudokuSolver:
    """
    Block Gibbs sampler for solving Sudoku puzzles.
    
    Args:
        model: SudokuModel instance
        beta: Inverse temperature parameter (higher = more deterministic)
    """
    
    def __init__(self, model: SudokuModel, beta: float = 2.0):
        self.model = model
        self.beta = beta
        
    def create_blocks(self, strategy: str = "row") -> List[List[int]]:
        """
        Create sampling blocks for block Gibbs sampling.
        
        Args:
            strategy: Block creation strategy ("row", "col", "box", or "checkerboard")
            
        Returns:
            List of blocks, where each block is a list of cell indices
        """
        if strategy == "row":
            # Alternate rows: even rows in one block, odd rows in another
            even_rows = []
            odd_rows = []
            for row in range(self.model.grid_size):
                for col in range(self.model.grid_size):
                    cell_idx = self.model.coords_to_cell(row, col)
                    if row % 2 == 0:
                        even_rows.append(cell_idx)
                    else:
                        odd_rows.append(cell_idx)
            return [even_rows, odd_rows]
        
        elif strategy == "col":
            # Alternate columns
            even_cols = []
            odd_cols = []
            for row in range(self.model.grid_size):
                for col in range(self.model.grid_size):
                    cell_idx = self.model.coords_to_cell(row, col)
                    if col % 2 == 0:
                        even_cols.append(cell_idx)
                    else:
                        odd_cols.append(cell_idx)
            return [even_cols, odd_cols]
        
        elif strategy == "checkerboard":
            # Checkerboard pattern (like a chess board)
            white = []
            black = []
            for row in range(self.model.grid_size):
                for col in range(self.model.grid_size):
                    cell_idx = self.model.coords_to_cell(row, col)
                    if (row + col) % 2 == 0:
                        white.append(cell_idx)
                    else:
                        black.append(cell_idx)
            return [white, black]
        
        else:
            raise ValueError(f"Unknown block strategy: {strategy}")
    
    def compute_conditional_probabilities(self, 
                                         state: jnp.ndarray, 
                                         cell_idx: int,
                                         clamped_cells: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute conditional probability distribution for a cell given its neighbors.
        
        Args:
            state: Current state array
            cell_idx: Index of the cell to compute probabilities for
            clamped_cells: Boolean mask of clamped cells (given clues)
            
        Returns:
            Probability distribution over possible values (1 to grid_size)
        """
        # If cell is clamped, return delta distribution at current value
        if clamped_cells is not None and clamped_cells[cell_idx]:
            probs = jnp.zeros(self.model.grid_size)
            current_value = int(state[cell_idx])
            probs = probs.at[current_value - 1].set(1.0)
            return probs
        
        # Compute energy for each possible value
        energies = jnp.zeros(self.model.grid_size)
        
        for value in range(1, self.model.grid_size + 1):
            # Create candidate state with this value
            candidate_state = state.at[cell_idx].set(value)
            # Compute energy (only need to check constraints involving this cell)
            energy = self._compute_local_energy(candidate_state, cell_idx)
            energies = energies.at[value - 1].set(energy)
        
        # Convert energies to probabilities using Boltzmann distribution
        # P(value) ∝ exp(-beta * energy)
        log_probs = -self.beta * energies
        log_probs = log_probs - jnp.max(log_probs)  # Numerical stability
        probs = jnp.exp(log_probs)
        probs = probs / jnp.sum(probs)
        
        return probs
    
    def _compute_local_energy(self, state: jnp.ndarray, cell_idx: int) -> float:
        """
        Compute energy contribution from constraints involving a specific cell.
        
        Args:
            state: Current state array
            cell_idx: Index of the cell
            
        Returns:
            Local energy value
        """
        energy = 0.0
        
        # Check all constraint groups that include this cell
        for constraint_group in self.model.get_constraint_groups():
            if cell_idx in constraint_group:
                values = state[jnp.array(constraint_group)]
                violations = self.model._count_duplicates(values)
                energy += violations
        
        return energy
    
    def sample_block(self, 
                    key: jax.random.PRNGKey,
                    state: jnp.ndarray,
                    block: List[int],
                    clamped_cells: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Sample new values for all cells in a block.
        
        Args:
            key: JAX random key
            state: Current state array
            block: List of cell indices to sample
            clamped_cells: Boolean mask of clamped cells
            
        Returns:
            Updated state array
        """
        new_state = state.copy()
        
        for i, cell_idx in enumerate(block):
            # Skip clamped cells
            if clamped_cells is not None and clamped_cells[cell_idx]:
                continue
            
            # Get conditional probabilities
            probs = self.compute_conditional_probabilities(new_state, cell_idx, clamped_cells)
            
            # Sample new value
            key, subkey = jax.random.split(key)
            new_value = jax.random.choice(subkey, self.model.grid_size, p=probs) + 1
            new_state = new_state.at[cell_idx].set(new_value)
        
        return new_state
    
    def solve(self,
             initial_state: jnp.ndarray,
             clamped_cells: jnp.ndarray,
             n_iterations: int = 5000,
             block_strategy: str = "checkerboard",
             key: Optional[jax.random.PRNGKey] = None,
             verbose: bool = False) -> Tuple[jnp.ndarray, Dict]:
        """
        Solve a Sudoku puzzle using block Gibbs sampling.
        
        Args:
            initial_state: Initial state array
            clamped_cells: Boolean mask indicating which cells are given (fixed)
            n_iterations: Number of sampling iterations
            block_strategy: Block creation strategy
            key: JAX random key
            verbose: Whether to print progress
            
        Returns:
            Tuple of (final_state, stats_dict)
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Create sampling blocks
        blocks = self.create_blocks(block_strategy)
        
        # Initialize state
        state = initial_state.copy()
        
        # Track statistics
        energies = []
        best_state = state.copy()
        best_energy = self.model.compute_energy(state)
        
        # Sampling loop
        for iteration in range(n_iterations):
            # Sample each block in sequence
            for block_idx, block in enumerate(blocks):
                key, subkey = jax.random.split(key)
                state = self.sample_block(subkey, state, block, clamped_cells)
            
            # Compute energy
            energy = self.model.compute_energy(state)
            energies.append(float(energy))
            
            # Track best state
            if energy < best_energy:
                best_energy = energy
                best_state = state.copy()
            
            # Check for solution
            if energy == 0:
                if verbose:
                    print(f"Solution found at iteration {iteration}!")
                break
            
            # Print progress
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.1f}, Best = {best_energy:.1f}")
        
        # Return best state found
        stats = {
            "final_energy": best_energy,
            "energies": energies,
            "converged": best_energy == 0,
            "iterations": len(energies)
        }
        
        return best_state, stats
    
    def initialize_state(self, 
                        puzzle: jnp.ndarray,
                        key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize state from a puzzle (0 indicates empty cells).
        
        Args:
            puzzle: Puzzle array with 0 for empty cells
            key: JAX random key
            
        Returns:
            Tuple of (initial_state, clamped_cells)
        """
        state = puzzle.copy()
        clamped_cells = puzzle != 0
        
        # Fill empty cells with random values
        empty_cells = jnp.where(puzzle == 0)[0]
        for cell_idx in empty_cells:
            key, subkey = jax.random.split(key)
            random_value = jax.random.randint(subkey, (), 1, self.model.grid_size + 1)
            state = state.at[cell_idx].set(random_value)
        
        return state, clamped_cells

