# THRML Probabilistic Sudoku Solver

A scalable Sudoku solver built using [THRML](https://github.com/extropic-ai/thrml)'s energy-based models and block Gibbs sampling. This project demonstrates how to use probabilistic graphical models to solve constraint satisfaction problems.

## Features

- **Scalable Grid Sizes**: Supports standard 9×9 Sudoku as well as larger grids (16×16, 25×25, etc.)
- **Energy-Based Modeling**: Uses THRML's energy-based framework to model Sudoku constraints
- **Block Gibbs Sampling**: Efficient sampling strategy for finding solutions
- **CLI Interface**: Clean command-line interface for solving and generating puzzles
- **Performance Tracking**: Monitors convergence metrics and solution quality

## Installation

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv pip install -r requirements.txt
```

## Usage

### Solve a puzzle from a file

```bash
python -m src.cli --puzzle puzzle.csv --size 9
```

### Generate and solve a new puzzle

```bash
python -m src.cli --generate --size 9 --warmup 1000 --samples 5000
```

### Verbose output with convergence tracking

```bash
python -m src.cli --generate --size 9 --verbose
```

## Project Structure

- `src/sudoku_model.py` - THRML model definition for Sudoku
- `src/solver.py` - Block Gibbs sampling implementation
- `src/puzzle_generator.py` - Puzzle generation and validation
- `src/cli.py` - Command-line interface
- `examples/demo.py` - Example usage scripts

## How It Works

The solver models Sudoku as a probabilistic graphical model where:
- Each cell is a node with N possible states (1 to N)
- Constraints (rows, columns, boxes) are represented as edges
- An energy function penalizes constraint violations
- Block Gibbs sampling explores the solution space efficiently

## License

MIT License
