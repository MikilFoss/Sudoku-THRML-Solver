# THRML Probabilistic Sudoku Solver

A scalable Sudoku solver built using [THRML](https://github.com/extropic-ai/thrml)'s energy-based models and systematic scan Gibbs sampling. This project demonstrates how to use probabilistic graphical models to solve constraint satisfaction problems.

## Installation

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv pip install -r requirements.txt
```

## Usage

Start the Flask web server:

```bash
python src/app.py
```

Then open your browser to `http://localhost:5000` to access the interactive Sudoku solver interface.

You can:

- Generate new puzzles of varying difficulty
- Watch the solver work in real-time as it finds solutions
- Adjust solver parameters (beta, max iterations)

## How It Works

The solver models Sudoku as a probabilistic graphical model where:

- Each cell is a node with N possible states (1 to N)
- Constraints (rows, columns, boxes) are represented as edges
- An energy function penalizes constraint violations
- Systematic scan Gibbs sampling explores the solution space efficiently
