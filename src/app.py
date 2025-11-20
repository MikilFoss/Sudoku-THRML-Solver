from flask import Flask, render_template, request, jsonify
import time
import jax
import jax.numpy as jnp
import numpy as np
from sudoku_model import SudokuModel
from solver import SudokuSolver
from puzzle_generator import PuzzleGenerator

app = Flask(__name__)

# Initialize model and solver
GRID_SIZE = 9
model = SudokuModel(GRID_SIZE)
solver = SudokuSolver(model)
generator = PuzzleGenerator(GRID_SIZE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['GET'])
def generate():
    difficulty = request.args.get('difficulty', 'medium')
    
    # Generate puzzle
    key = jax.random.PRNGKey(np.random.randint(0, 100000))
    puzzle, solution = generator.generate_puzzle(difficulty, key)
    
    return jsonify({
        'puzzle': np.array(puzzle).tolist(),
        'solution': np.array(solution).tolist()
    })

@app.route('/api/solve', methods=['GET'])
def solve():
    # Get puzzle from query params (since EventSource uses GET)
    # We need to pass the puzzle data somehow. 
    # EventSource doesn't support POST body.
    # We can pass it as a query parameter string (JSON encoded)
    import json
    puzzle_str = request.args.get('puzzle')
    
    if not puzzle_str:
        return jsonify({'error': 'No puzzle data provided'}), 400
    
    try:
        puzzle_data = json.loads(puzzle_str)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid puzzle data'}), 400
    
    # Get params
    beta = request.args.get('beta', type=float)
    max_iterations = request.args.get('max_iterations', type=int, default=100000)
    
    # Convert to JAX array
    initial_state = jnp.array(puzzle_data)
    clamped_cells = initial_state != 0
    
    # Solve
    key = jax.random.PRNGKey(np.random.randint(0, 100000))
    
    # Fill empty cells with random values for initial state
    state, _ = solver.initialize_state(initial_state, key)
    
    def generate():
        # Use solve_stream
        # Batch size 100 iterations per yield
        stream = solver.solve_stream(
            state, 
            clamped_cells, 
            batch_size=100,
            max_iterations=max_iterations,
            beta=beta,
            key=key
        )
        
        for current_state, stats in stream:
            # Yield data
            data = {
                'type': 'update',
                'solution': np.array(current_state).tolist(),
                'stats': stats
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            # Check convergence
            if stats['converged']:
                yield f"data: {json.dumps({'type': 'complete', 'solution': np.array(current_state).tolist(), 'stats': stats})}\n\n"
                break
        else:
            # Loop finished without convergence (max iterations reached)
            yield f"data: {json.dumps({'type': 'timeout', 'message': f'Max iterations ({max_iterations}) reached'})}\n\n"
                
    from flask import Response, stream_with_context
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
