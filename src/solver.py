"""
Block Gibbs Sampler for Sudoku.

This module implements efficient block Gibbs sampling for solving Sudoku puzzles.
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Optional, Generator
import numpy as np
from sudoku_model import SudokuModel


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
        
        # Define the step function for JAX scan
        # We capture blocks and model configuration as closure variables (static)
        # clamped_cells is passed as argument
        
        def step_fn(carry, _):
            key, current_state = carry
            
            # Iterate over blocks (unrolled by JAX)
            for block in blocks:
                # Iterate over cells in block (unrolled by JAX)
                for cell_idx in block:
                    # Define update logic
                    def update_cell(args):
                        s, k = args
                        # Compute probabilities
                        # We inline the logic here to avoid overhead and simplify
                        energies = jnp.zeros(self.model.grid_size)
                        
                        # Vectorized energy computation for all candidates
                        # This is a bit complex to vectorize fully efficiently without changing model
                        # So we stick to the loop over values 1..N, but unrolled or scanned?
                        # N is small (9). Unrolling is fine.
                        
                        def body_fun(val, e_acc):
                            candidate = s.at[cell_idx].set(val + 1)
                            # Inline local energy computation
                            energy = 0.0
                            # We know the constraints for this cell statically
                            # But self.model.get_constraint_groups() is a list.
                            # We can pre-compute relevant constraints for this cell?
                            # For now, let's rely on JIT to optimize the loop over constraints
                            # since 'cell_idx' is static here.
                            
                            for constraint_group in self.model.get_constraint_groups():
                                if cell_idx in constraint_group:
                                    values = candidate[jnp.array(constraint_group)]
                                    # count duplicates
                                    # violations = len(values) - len(unique(values))
                                    # jnp.unique is not JIT-friendly for variable size? 
                                    # It returns variable size array.
                                    # We need a fixed size way to count duplicates.
                                    # Sort and check adjacent?
                                    vals = jnp.sort(values)
                                    # diff = vals[1:] == vals[:-1]
                                    # violations = jnp.sum(diff)
                                    violations = jnp.sum(vals[1:] == vals[:-1])
                                    energy += violations
                            
                            return e_acc.at[val].set(energy)
                            
                        # Use fori_loop or unroll? 9 is small.
                        # energies = jax.lax.fori_loop(0, self.model.grid_size, body_fun, energies)
                        # Actually, let's just loop python-wise, JAX will unroll.
                        for val in range(self.model.grid_size):
                            energies = body_fun(val, energies)
                            
                        # Softmax
                        log_probs = -self.beta * energies
                        log_probs = log_probs - jnp.max(log_probs)
                        probs = jnp.exp(log_probs)
                        probs = probs / jnp.sum(probs)
                        
                        # Sample
                        k, subk = jax.random.split(k)
                        new_val = jax.random.choice(subk, self.model.grid_size, p=probs) + 1
                        return s.at[cell_idx].set(new_val), k

                    def no_update(args):
                        return args

                    # Conditional update based on clamped status
                    is_clamped = clamped_cells[cell_idx]
                    current_state, key = jax.lax.cond(is_clamped, no_update, update_cell, (current_state, key))
            
            # Compute total energy for stats
            # We can also JIT this
            energy = self.model.compute_energy(current_state)
            return (key, current_state), energy

        # JIT compile the scan loop
        # We wrap it to handle the scan
        @jax.jit
        def run_scan(k, s):
            return jax.lax.scan(step_fn, (k, s), None, length=n_iterations)

        # Run the solver
        if verbose:
            print("JIT compiling solver...")
        
        start_time = 0
        if verbose:
            import time
            start_time = time.time()
            
        (key, final_state), energies = run_scan(key, state)
        
        # Block until ready to measure time
        final_state.block_until_ready()
        
        if verbose:
            print(f"Solver finished in {time.time() - start_time:.4f}s")

        best_energy = jnp.min(energies)
        best_idx = jnp.argmin(energies)
        # We should ideally return the state at best_idx, but scan returns final state.
        # For Gibbs sampling with high beta (simulated annealing-ish), final state is usually good.
        # Or we can track best state in carry?
        # Tracking best state in carry adds overhead (copying state).
        # Let's just return final state for now, or we can run a second pass if needed.
        # Actually, if energy == 0, we are good.
        
        stats = {
            "final_energy": float(energies[-1]),
            "energies": [float(e) for e in energies], # Convert to list for JSON serialization
            "converged": float(best_energy) == 0,
            "iterations": n_iterations
        }
        
        return final_state, stats

    def solve_stream(self,
                    initial_state: jnp.ndarray,
                    clamped_cells: jnp.ndarray,
                    batch_size: int = 100,
                    max_iterations: int = 100000,
                    block_strategy: str = "checkerboard",
                    beta: Optional[float] = None,
                    key: Optional[jax.random.PRNGKey] = None) -> Generator[Tuple[jnp.ndarray, Dict], None, None]:
        """
        Yield intermediate results from the solver.
        
        Args:
            initial_state: Initial state array
            clamped_cells: Boolean mask indicating which cells are given (fixed)
            batch_size: Number of iterations per yield
            max_iterations: Maximum total iterations
            block_strategy: Block creation strategy
            beta: Inverse temperature (optional, overrides self.beta)
            key: JAX random key
            
        Yields:
            Tuple of (current_state, stats_dict)
        """
        if key is None:
            key = jax.random.PRNGKey(0)
            
        current_beta = beta if beta is not None else self.beta
        
        # Create sampling blocks
        blocks = self.create_blocks(block_strategy)
        
        # Initialize state
        state = initial_state.copy()
        
        # Define the step function for JAX scan (same as in solve)
        def step_fn(carry, _):
            key, current_state = carry
            
            for block in blocks:
                for cell_idx in block:
                    def update_cell(args):
                        s, k = args
                        energies = jnp.zeros(self.model.grid_size)
                        
                        def body_fun(val, e_acc):
                            candidate = s.at[cell_idx].set(val + 1)
                            energy = 0.0
                            for constraint_group in self.model.get_constraint_groups():
                                if cell_idx in constraint_group:
                                    values = candidate[jnp.array(constraint_group)]
                                    sorted_values = jnp.sort(values)
                                    duplicates = sorted_values[1:] == sorted_values[:-1]
                                    violations = jnp.sum(duplicates)
                                    energy += violations
                            return e_acc.at[val].set(energy)
                            
                        for val in range(self.model.grid_size):
                            energies = body_fun(val, energies)
                            
                        log_probs = -current_beta * energies
                        log_probs = log_probs - jnp.max(log_probs)
                        probs = jnp.exp(log_probs)
                        probs = probs / jnp.sum(probs)
                        
                        k, subk = jax.random.split(k)
                        new_val = jax.random.choice(subk, self.model.grid_size, p=probs) + 1
                        return s.at[cell_idx].set(new_val), k

                    def no_update(args):
                        return args

                    is_clamped = clamped_cells[cell_idx]
                    current_state, key = jax.lax.cond(is_clamped, no_update, update_cell, (current_state, key))
            
            energy = self.model.compute_energy(current_state)
            return (key, current_state), energy

        # JIT compile the scan loop for a batch
        @jax.jit
        def run_batch(k, s):
            return jax.lax.scan(step_fn, (k, s), None, length=batch_size)

        # Streaming loop
        total_iterations = 0
        energies_history = []
        
        while total_iterations < max_iterations:
            (key, state), batch_energies = run_batch(key, state)
            
            # Block until ready to ensure we yield actual data
            state.block_until_ready()
            
            batch_energies_list = [float(e) for e in batch_energies]
            energies_history.extend(batch_energies_list)
            current_energy = batch_energies_list[-1]
            avg_energy = sum(batch_energies_list) / len(batch_energies_list)
            
            total_iterations += batch_size
            
            stats = {
                "iteration": total_iterations,
                "energy": current_energy,
                "avg_energy": avg_energy,
                "energies": energies_history, 
                "converged": current_energy == 0
            }
            
            yield state, stats
            
            if current_energy == 0:
                break
    
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

