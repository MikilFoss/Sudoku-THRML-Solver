"""
Sudoku Model using THRML's energy-based framework.

This module defines a scalable Sudoku model that can handle grids of any size (N×N where N is a perfect square).
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple
import math


class DiscreteNode:
    """A discrete node that can take on N different states (1 to N)."""
    
    def __init__(self, node_id: int, n_states: int):
        self.node_id = node_id
        self.n_states = n_states
        
    def __repr__(self):
        return f"DiscreteNode(id={self.node_id}, states={self.n_states})"


class SudokuModel:
    """
    Energy-based model for Sudoku puzzles of arbitrary size.
    
    Args:
        grid_size: Size of the Sudoku grid (must be a perfect square, e.g., 9, 16, 25)
    """
    
    def __init__(self, grid_size: int = 9):
        if not self._is_perfect_square(grid_size):
            raise ValueError(f"Grid size {grid_size} must be a perfect square")
        
        self.grid_size = grid_size
        self.box_size = int(math.sqrt(grid_size))
        self.n_cells = grid_size * grid_size
        
        # Create nodes for each cell
        self.nodes = [DiscreteNode(i, grid_size) for i in range(self.n_cells)]
        
        # Create constraint edges
        self.row_constraints = self._create_row_constraints()
        self.col_constraints = self._create_col_constraints()
        self.box_constraints = self._create_box_constraints()
        
    @staticmethod
    def _is_perfect_square(n: int) -> bool:
        """Check if n is a perfect square."""
        root = int(math.sqrt(n))
        return root * root == n
    
    def _create_row_constraints(self) -> List[List[int]]:
        """Create constraint groups for rows."""
        constraints = []
        for row in range(self.grid_size):
            row_cells = [row * self.grid_size + col for col in range(self.grid_size)]
            constraints.append(row_cells)
        return constraints
    
    def _create_col_constraints(self) -> List[List[int]]:
        """Create constraint groups for columns."""
        constraints = []
        for col in range(self.grid_size):
            col_cells = [row * self.grid_size + col for row in range(self.grid_size)]
            constraints.append(col_cells)
        return constraints
    
    def _create_box_constraints(self) -> List[List[int]]:
        """Create constraint groups for boxes (sub-grids)."""
        constraints = []
        for box_row in range(self.box_size):
            for box_col in range(self.box_size):
                box_cells = []
                for r in range(self.box_size):
                    for c in range(self.box_size):
                        row = box_row * self.box_size + r
                        col = box_col * self.box_size + c
                        cell_idx = row * self.grid_size + col
                        box_cells.append(cell_idx)
                constraints.append(box_cells)
        return constraints
    
    def compute_energy(self, state: jnp.ndarray) -> float:
        """
        Compute the energy of a given state.
        
        Energy = sum of constraint violations (lower is better, 0 is perfect solution)
        
        Args:
            state: Array of shape (n_cells,) with values from 1 to grid_size
            
        Returns:
            Energy value (number of constraint violations)
        """
        energy = 0.0
        
        # Check row constraints
        for row_cells in self.row_constraints:
            values = state[jnp.array(row_cells)]
            # Count violations: number of duplicate values
            violations = self._count_duplicates(values)
            energy += violations
        
        # Check column constraints
        for col_cells in self.col_constraints:
            values = state[jnp.array(col_cells)]
            violations = self._count_duplicates(values)
            energy += violations
        
        # Check box constraints
        for box_cells in self.box_constraints:
            values = state[jnp.array(box_cells)]
            violations = self._count_duplicates(values)
            energy += violations
        
        return energy
    
    @staticmethod
    def _count_duplicates(values: jnp.ndarray) -> int:
        """Count the number of duplicate values in an array."""
        # JIT-friendly implementation using sort
        sorted_values = jnp.sort(values)
        # Check adjacent elements
        duplicates = sorted_values[1:] == sorted_values[:-1]
        return jnp.sum(duplicates)
    
    def get_constraint_groups(self) -> List[List[int]]:
        """Get all constraint groups (rows, columns, boxes)."""
        return self.row_constraints + self.col_constraints + self.box_constraints
    
    def cell_to_coords(self, cell_idx: int) -> Tuple[int, int]:
        """Convert cell index to (row, col) coordinates."""
        row = cell_idx // self.grid_size
        col = cell_idx % self.grid_size
        return row, col
    
    def coords_to_cell(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to cell index."""
        return row * self.grid_size + col
    
    def state_to_grid(self, state: jnp.ndarray) -> jnp.ndarray:
        """Convert flat state array to 2D grid."""
        return state.reshape(self.grid_size, self.grid_size)
    
    def grid_to_state(self, grid: jnp.ndarray) -> jnp.ndarray:
        """Convert 2D grid to flat state array."""
        return grid.flatten()
    
    def is_valid_solution(self, state: jnp.ndarray) -> bool:
        """Check if the state is a valid Sudoku solution."""
        return self.compute_energy(state) == 0
    
    def get_neighbors(self, cell_idx: int) -> List[int]:
        """Get all cells that share a constraint with the given cell."""
        neighbors = set()
        
        # Add cells from same row, column, and box
        for constraint_group in self.get_constraint_groups():
            if cell_idx in constraint_group:
                neighbors.update(constraint_group)
        
        # Remove self
        neighbors.discard(cell_idx)
        return list(neighbors)

