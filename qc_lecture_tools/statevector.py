import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import ipywidgets
from ipywidgets import interact
from IPython.display import display, Math
import sympy
from sympy import sqrt, sin, cos, pi, E, N

def format_complex(v:float):
    """
    Function for return a latex expression of a value

    Args:
        v (float): Number that is converted to latex

    Returns:
        Latex expression 
    """
    
    # Use sympy to simplify and detect sqrt/rational expressions

    constants = [sqrt(2), sqrt(3), sqrt(5), pi, E]
    trig_constants = [
    sin(pi/6), cos(pi/6),   # 1/2, sqrt(3)/2
    sin(pi/4), cos(pi/4),   # sqrt(2)/2
    sin(pi/3), cos(pi/3),   # sqrt(3)/2, 1/2 
    ]

    constants += trig_constants
    expr = N(v, 15, chop=1e-10)
    expr = sympy.nsimplify(expr, constants=constants, tolerance=1e-15)
    
    # Try to further simplify sqrt expressions
    expr = sympy.simplify(expr)   

    # Avoid huge denominators for rational numbers 
    if any(isinstance(r, sympy.Rational) and r.q > 50 for r in expr.atoms(sympy.Rational)):
        expr = expr.evalf()  # Display as float

    if isinstance(expr, sympy.core.numbers.Float):
        expr = round(expr, 15)

    # If expr is a sympy number, format as LaTeX
    if isinstance(expr, sympy.Basic):
        latex = sympy.latex(expr)
    else:
        # Fallback to float formatting
        latex = f"{v.real:.2f}"
        if not np.isclose(v.imag, 0):
            sign = "+" if v.imag > 0 else ""
            latex += f"{sign}{v.imag:.2f}i"
    return latex

def matrix_to_latex(matrix, label=None):
    """Convert a matrix to LaTeX format and display it with analytic expressions for typical numbers, including sqrt."""
 
    latex_str = "\\begin{pmatrix}"
    rows = []
    for row in matrix:
        row_str = " & ".join([format_complex(v) for v in row])
        rows.append(row_str)
    latex_str += " \\\\ ".join(rows)
    latex_str += "\\end{pmatrix}"
    if label is None:
        display(Math(f"{latex_str}"))
    else:
        display(Math(f"{label}{latex_str}"))

def sv_compute(quantum_circuit: QuantumCircuit) -> Statevector:
    """Compute the statevector of a quantum circuit.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to simulate.

    Returns:
        Statevector: The statevector of the quantum circuit.
    """

    statevector = Statevector.from_instruction(quantum_circuit.reverse_bits())
    return statevector

def sv_array(quantum_circuit: QuantumCircuit) -> np.ndarray:
    """Compute the statevector of a quantum circuit and return it as a list.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to simulate.
    Returns:
        np.ndarray: The statevector of the quantum circuit as a list.
    """
    statevector = sv_compute(quantum_circuit)
    return statevector.data

def sv_vector(quantum_circuit: QuantumCircuit, label:str=None):
    """ Prints the statevector as a latex vector
    
    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to simulate
        name (str): Optional name that is added to the output
    """
    statevector = sv_compute(quantum_circuit)
    latex_str = "\\begin{pmatrix}"
    rows = []
    for value in statevector:
        latex_str += format_complex(value)
        latex_str += "\\\\"
    latex_str += "\\end{pmatrix}"
    if label is None:
        return Math(latex_str)
    else:
        return Math(f"{label}{latex_str}")

def sv_state(quantum_circuit: QuantumCircuit, label:str=None):
    """Print the statevector of a quantum circuit.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to simulate.
        name (str): Optional name that is added to the output
    """
    statevector = sv_compute(quantum_circuit)
    if label is None:
        return Math(statevector.draw('latex_source'))
    else:
        return Math(f"{label}{statevector.draw('latex_source')}")

def sv_probs(quantum_circuit: QuantumCircuit, label:str=None):
    """Computes and prints the probabilies to measure the basis states for the given quantum circuit """
    vector = sv_compute(quantum_circuit)
    probs = np.square(np.abs(vector))

    if label is not None:
        print(label)

    latex_str = "\\begin{array}{rcl}"
    for i,p in enumerate(probs):
        latex_str += f"p_{{{i}}}" + r" & = & " 
        latex_str += format_complex(p) + " \\\\[1mm]"
    latex_str += "\\end{array}"
    return Math(latex_str)

def sv_latex(quantum_circuit: QuantumCircuit):
    """Print the statevector of a quantum circuit.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to simulate.
    """
    return sv_state(quantum_circuit)


def sv_blochsphere(quantum_circuit: QuantumCircuit):
    """Draw the statevector of a quantum circuit.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to simulate.
    """

    if quantum_circuit.num_qubits != 1:
        raise ValueError("Statevector drawing in the Bloch-Sphere is only supported for single-qubit circuits.")
    
    statevector = sv_compute(quantum_circuit)
    return statevector.draw('bloch')

def sv_interactive_blochsphere(quantum_circuit: QuantumCircuit):
    """Draw the statevector of a quantum circuit interactively.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to simulate.
    """

    if quantum_circuit.num_qubits != 1:
        raise ValueError("Statevector drawing in the Bloch-Sphere is only supported for single-qubit circuits.")
    
    if quantum_circuit.num_parameters == 0:
        return sv_blochsphere(quantum_circuit)
    

    sliders = {}
    for param in quantum_circuit.parameters:
        sliders[param.name] = ipywidgets.FloatSlider(
            value=0,
            min=-2 * np.pi,
            max=2 * np.pi,
            step=0.1,
            description=param.name
        )

    def plot_ry(**parameter_values):
        param_dict = {param: parameter_values[param.name] for param in quantum_circuit.parameters}
        return sv_blochsphere(quantum_circuit.assign_parameters(param_dict, inplace=False))

    frame = interact(plot_ry, **sliders)

    return frame
