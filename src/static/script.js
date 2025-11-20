document.addEventListener('DOMContentLoaded', () => {
    const board = document.getElementById('sudoku-board');
    const difficultySelect = document.getElementById('difficulty');
    const newGameBtn = document.getElementById('new-game-btn');
    const solveBtn = document.getElementById('solve-btn');
    const resetBtn = document.getElementById('reset-btn');
    const statusMessage = document.getElementById('status-message');
    const statIterations = document.getElementById('stat-iterations');
    const statEnergy = document.getElementById('stat-energy');
    const statStatus = document.getElementById('stat-status');
    
    let currentPuzzle = [];
    let currentSolution = []; // Full solution if available
    let userGrid = []; // Current state of the grid
    let selectedCellIndex = null;
    let originalPuzzle = []; // To track clues
    let energyChart = null;
    
    const GRID_SIZE = 9;
    
    // Initialize grid
    createGrid();
    initChart();
    startNewGame();
    
    // Event Listeners
    newGameBtn.addEventListener('click', startNewGame);
    solveBtn.addEventListener('click', solvePuzzle);
    resetBtn.addEventListener('click', resetBoard);
    
    document.addEventListener('keydown', handleKeyPress);
    
    function createGrid() {
        board.innerHTML = '';
        for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.index = i;
            cell.addEventListener('click', () => selectCell(i));
            board.appendChild(cell);
        }
    }
    
    function selectCell(index) {
        if (selectedCellIndex !== null) {
            board.children[selectedCellIndex].classList.remove('selected');
            // Remove related highlighting
            const cells = document.querySelectorAll('.cell');
            cells.forEach(c => c.classList.remove('related'));
        }
        
        selectedCellIndex = index;
        const cell = board.children[index];
        cell.classList.add('selected');
        
        // Highlight related cells (row, col, box)
        highlightRelated(index);
    }
    
    function highlightRelated(index) {
        const row = Math.floor(index / GRID_SIZE);
        const col = index % GRID_SIZE;
        const boxRow = Math.floor(row / 3);
        const boxCol = Math.floor(col / 3);
        
        const cells = document.querySelectorAll('.cell');
        
        for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            if (i === index) continue;
            
            const r = Math.floor(i / GRID_SIZE);
            const c = i % GRID_SIZE;
            const br = Math.floor(r / 3);
            const bc = Math.floor(c / 3);
            
            if (r === row || c === col || (br === boxRow && bc === boxCol)) {
                cells[i].classList.add('related');
            }
        }
    }
    
    function handleKeyPress(e) {
        if (selectedCellIndex === null) return;
        
        // If it's a clue, don't allow editing
        if (originalPuzzle[selectedCellIndex] !== 0) return;
        
        const key = e.key;
        if (key >= '1' && key <= '9') {
            updateCell(selectedCellIndex, parseInt(key));
        } else if (key === 'Backspace' || key === 'Delete' || key === '0') {
            updateCell(selectedCellIndex, 0);
        } else if (key === 'ArrowUp') {
            e.preventDefault();
            moveSelection(-9);
        } else if (key === 'ArrowDown') {
            e.preventDefault();
            moveSelection(9);
        } else if (key === 'ArrowLeft') {
            e.preventDefault();
            moveSelection(-1);
        } else if (key === 'ArrowRight') {
            e.preventDefault();
            moveSelection(1);
        }
    }
    
    function moveSelection(delta) {
        let newIndex = selectedCellIndex + delta;
        if (newIndex >= 0 && newIndex < GRID_SIZE * GRID_SIZE) {
            selectCell(newIndex);
        }
    }
    
    function updateCell(index, value) {
        userGrid[index] = value;
        const cell = board.children[index];
        cell.textContent = value === 0 ? '' : value;
        
        if (value !== 0) {
            cell.classList.add('user-input');
        } else {
            cell.classList.remove('user-input');
        }
    }
    
    async function startNewGame() {
        const difficulty = difficultySelect.value;
        statusMessage.textContent = 'Generating new puzzle...';
        newGameBtn.disabled = true;
        
        try {
            const response = await fetch(`/api/generate?difficulty=${difficulty}`);
            const data = await response.json();
            
            currentPuzzle = data.puzzle; // Flat array
            originalPuzzle = [...currentPuzzle];
            userGrid = [...currentPuzzle];
            
            renderBoard();
            statusMessage.textContent = 'New game started!';
            statusMessage.className = 'status-message';
            
            // Reset stats
            statIterations.textContent = '0';
            statEnergy.textContent = '-';
            statStatus.textContent = 'Idle';
            resetChart();
            
        } catch (error) {
            console.error('Error starting new game:', error);
            statusMessage.textContent = 'Error generating puzzle.';
            statusMessage.className = 'status-message error';
        } finally {
            newGameBtn.disabled = false;
        }
    }
    
    function renderBoard() {
        const cells = document.querySelectorAll('.cell');
        cells.forEach((cell, i) => {
            const value = userGrid[i];
            cell.textContent = value === 0 ? '' : value;
            
            // Reset classes
            cell.className = 'cell';
            
            if (originalPuzzle[i] !== 0) {
                cell.classList.add('clue');
            } else if (value !== 0) {
                cell.classList.add('user-input');
            }
        });
        
        selectedCellIndex = null;
    }
    
    function resetBoard() {
        userGrid = [...originalPuzzle];
        renderBoard();
        statusMessage.textContent = 'Board reset.';
        statusMessage.className = 'status-message';
        
        resetChart();
        statIterations.textContent = '0';
        statEnergy.textContent = '-';
        statStatus.textContent = 'Idle';
    }

    // Initialize Chart
    // Initialize Chart
    function initChart() {
        const ctx = document.getElementById('energyChart').getContext('2d');
        energyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Current Energy',
                        data: [],
                        borderColor: '#e78f81', // Primary color
                        backgroundColor: 'rgba(231, 143, 129, 0.2)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Avg Energy (Batch)',
                        data: [],
                        borderColor: '#2a9d8f', // Teal
                        backgroundColor: 'rgba(42, 157, 143, 0.0)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false, // Disable animation for performance
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Iterations'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Energy'
                        },
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    }
                }
            }
        });
    }

    function updateChart(stats) {
        if (!energyChart) return;
        
        const iteration = stats.iteration;
        const energy = stats.energy;
        const avgEnergy = stats.avg_energy;
        
        energyChart.data.labels.push(iteration);
        energyChart.data.datasets[0].data.push(energy);
        energyChart.data.datasets[1].data.push(avgEnergy);
        
        // Limit chart data points to avoid performance issues
        if (energyChart.data.labels.length > 200) {
             energyChart.data.labels.shift();
             energyChart.data.datasets[0].data.shift();
             energyChart.data.datasets[1].data.shift();
        }
        
        energyChart.update();
    }

    function resetChart() {
        if (energyChart) {
            energyChart.data.labels = [];
            energyChart.data.datasets[0].data = [];
            energyChart.data.datasets[1].data = [];
            energyChart.update();
        }
    }

    function getGridValues() {
        return [...userGrid];
    }

    async function solvePuzzle() {
        const currentGrid = getGridValues();
        const flatGrid = currentGrid.flat();
        const beta = document.getElementById('beta').value;
        const maxIterations = document.getElementById('max-iterations').value;
        
        solveBtn.disabled = true;
        newGameBtn.disabled = true;
        resetBtn.disabled = true;
        statusMessage.textContent = 'Solving...';
        statusMessage.className = 'status-message';
        
        resetChart();
        
        // Use EventSource for streaming
        const puzzleJson = JSON.stringify(flatGrid);
        const eventSource = new EventSource(`/api/solve?puzzle=${encodeURIComponent(puzzleJson)}&beta=${beta}&max_iterations=${maxIterations}`);
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'update') {
                // Update grid
                updateGrid(data.solution);
                
                // Update stats
                statIterations.textContent = data.stats.iteration;
                statEnergy.textContent = data.stats.energy.toFixed(2);
                statStatus.textContent = 'Solving...';
                
                // Update chart
                updateChart(data.stats);
                
            } else if (data.type === 'complete') {
                updateGrid(data.solution);
                statStatus.textContent = 'Solved!';
                statusMessage.textContent = 'Puzzle Solved!';
                statusMessage.className = 'status-message success';
                
                eventSource.close();
                solveBtn.disabled = false;
                newGameBtn.disabled = false;
                resetBtn.disabled = false;
                
            } else if (data.type === 'timeout') {
                statusMessage.textContent = data.message;
                statusMessage.className = 'status-message error';
                statStatus.textContent = 'Timeout';
                
                eventSource.close();
                solveBtn.disabled = false;
                newGameBtn.disabled = false;
                resetBtn.disabled = false;
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('EventSource failed:', error);
            statusMessage.textContent = 'Connection error.';
            statusMessage.className = 'status-message error';
            eventSource.close();
            solveBtn.disabled = false;
            newGameBtn.disabled = false;
            resetBtn.disabled = false;
        };
    }

    function updateGrid(flatSolution) {
        userGrid = [...flatSolution];
        const cells = document.querySelectorAll('.cell');
        cells.forEach((cell, index) => {
            const val = flatSolution[index];
            cell.textContent = val === 0 ? '' : val;
            
            // Keep clue styling
            if (originalPuzzle[index] !== 0) {
                cell.classList.add('clue');
            } else if (val !== 0) {
                cell.classList.add('user-input');
            }
        });
    }
});
