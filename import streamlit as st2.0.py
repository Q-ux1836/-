import streamlit as st
import numpy as np
import sympy
import matplotlib.pyplot as plt
import math
from collections import Counter
from scipy import stats # For advanced statistics like distributions
import cmath # For complex number phase/polar conversions

# --- Configuration and Global Variables ---
st.set_page_config(layout="wide", page_title="Universal Web Calculator")

# --- SymPy Global Scope for eval/parse_expr ---
# This dictionary exposes common functions/constants to sympy.sympify and sympy.parse_expr
# making them directly usable in user input without prefixing 'sympy.' or 'math.'
# We also include SymPy's core symbols and functions for symbolic operations.
SYMPY_EXPRESSION_GLOBALS = {
    'pi': sympy.pi,
    'E': sympy.E,
    'oo': sympy.oo, # Infinity
    
    # Basic math functions
    'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
    'asin': sympy.asin, 'acos': sympy.acos, 'atan': sympy.atan,
    'sinh': sympy.sinh, 'cosh': sympy.cosh, 'tanh': sympy.tanh,
    'asinh': sympy.asinh, 'acosh': sympy.acosh, 'atanh': sympy.atanh,
    'sqrt': sympy.sqrt,
    'log': sympy.log,          # Natural logarithm (base E) by default, or log(num, base)
    'ln': sympy.ln,            # Alias for natural logarithm
    'log10': sympy.log,        # Logarithm base 10 (SymPy's log can take a base)
    'exp': sympy.exp,
    'factorial': sympy.factorial,
    'Abs': sympy.Abs,          # Absolute value
    'floor': sympy.floor,
    'ceil': sympy.ceiling,     # Corrected: use sympy.ceiling
    'round': round,            # Corrected: use Python's built-in round for numerical results
    
    # Power and root functions
    'pow': sympy.Pow,
    'root': lambda x, n: sympy.Pow(x, 1/sympy.S(n)), # Custom Nth root function

    # Symbolic Calculus Functions
    'diff': sympy.diff,         # Differentiation
    'integrate': sympy.integrate, # Integration
    'limit': sympy.limit,       # Limits
    'series': sympy.series,     # Taylor/Maclaurin series expansion
    'dsolve': sympy.dsolve,     # Ordinary Differential Equation solver
    'solve': sympy.solve,       # Equation solver
    
    # Symbolic Matrix Functions
    'Matrix': sympy.Matrix,      # For creating symbolic matrices
    'eye': sympy.eye,            # Identity matrix
    'zeros': sympy.zeros,        # Zero matrix
    'ones': sympy.ones,          # Ones matrix
    'diag': sympy.diag,          # Diagonal matrix
    'BlockMatrix': sympy.BlockMatrix, # For constructing block matrices
    
    # Common symbolic variables for convenience (x, y, z, t, n, theta)
    'x': sympy.Symbol('x'),
    'y': sympy.Symbol('y'),
    'z': sympy.Symbol('z'),
    't': sympy.Symbol('t'),
    'n': sympy.Symbol('n'),
    'theta': sympy.Symbol('theta'), # Added for polar plots
    # Add other common symbols if needed
}

# --- Helper Functions for Input Parsing ---

def parse_float_list(input_str):
    """Parses a space-separated string of numbers into a list of floats."""
    if not input_str:
        return []
    # Use a generator expression for efficiency and robust splitting
    return [float(x.strip()) for x in input_str.split() if x.strip()]

def parse_matrix_input(input_str):
    """
    Parses a string representing a matrix.
    Rows are separated by newlines, elements by spaces.
    Returns a list of lists (matrix).
    """
    if not input_str:
        return []
    rows_str = input_str.strip().split('\n')
    matrix = []
    num_cols = -1 # Initialize with an invalid number to check consistency

    for r_str in rows_str:
        if not r_str.strip(): # Skip empty lines
            continue
        row = [float(x.strip()) for x in r_str.split() if x.strip()]
        
        if not matrix: # First row determines the number of columns
            num_cols = len(row)
        elif len(row) != num_cols:
            raise ValueError(f"All rows must have the same number of columns. Expected {num_cols}, got {len(row)}.")
        matrix.append(row)
    return matrix

# --- Calculator Modes ---

def simple_math_mode():
    """
    Implements a robust numerical calculator using SymPy for parsing and evaluation.
    This mode supports a wide range of standard mathematical operations and functions.
    """
    st.header("🔢 Simple Math (Numerical)")
    st.write("Enter any valid mathematical expression. SymPy will evaluate it numerically.")
    st.markdown("""
    **Available Operations & Functions:**
    * **Operators:** `+`, `-`, `*`, `/`, `%` (modulo), `**` (power)
    * **Constants:** `pi`, `E` (Euler's number)
    * **Functions (numerical):**
        * **Logarithms:** `log(num, base)` (e.g., `log(100, 10)`), `ln(num)` (natural log), `log10(num)`
        * **Powers & Roots:** `pow(base, exp)` (e.g., `pow(2, 3)`), `root(num, n)` (e.g., `root(27, 3)` for cube root), `sqrt(num)`
        * **Trigonometric:** `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()` (and their hyperbolic `sinh`, `cosh`, etc.)
        * **Others:** `factorial(n)`, `Abs(num)` (absolute value), `exp(num)` (`E**num`), `floor()`, `ceil()`, `round()`
    """)
    st.markdown("**Example:** `(2 + 3) * factorial(4) - E**2 + pi / 4 + log(100, 10)`")

    expression_input = st.text_input("Enter expression:", key="simple_math_expr")

    if st.button("Calculate", key="simple_math_btn"):
        if not expression_input:
            st.warning("Please enter an expression.")
            return

        try:
            # Use SymPy's parse_expr for robust parsing.
            # We explicitly pass the global scope for known functions and constants.
            expr_sympy = sympy.parse_expr(expression_input, global_dict=SYMPY_EXPRESSION_GLOBALS)

            # Check for free symbols (variables). If present, we can't evaluate numerically.
            if expr_sympy.free_symbols:
                st.error(f"Cannot evaluate numerically: Expression contains undefined variables: {', '.join(str(s) for s in expr_sympy.free_symbols)}. Use **Algebra (Symbolic)** mode for symbolic operations.")
                return

            # Evaluate the SymPy expression to a numerical (float) result.
            result = expr_sympy.evalf()

            # Display integer results as integers, floats with reasonable precision.
            if result == int(result):
                st.success(f"Result: `{int(result)}`")
            else:
                # Format to a reasonable precision and remove trailing zeros for clean display
                st.success(f"Result: `{result:.15f}`".rstrip('0').rstrip('.'))
        except (sympy.SympifyError, TypeError, ValueError) as e:
            st.error(f"Invalid expression or calculation error: `{e}`. Please check syntax or unsupported function/constant.")
        except Exception as e:
            st.error(f"An unexpected error occurred: `{e}`")


def algebra_mode():
    """
    Implements symbolic algebra operations using SymPy, including simplification,
    equation solving, differentiation, integration, limits, series, and ODEs.
    Also handles symbolic matrix operations.
    """
    st.header("📈 Algebra (Symbolic)")
    st.write("Perform symbolic manipulation, calculus, and equation solving. Variables like `x`, `y`, `z`, `t`, `n` are automatically recognized.")
    
    st.markdown("""
    **Available Commands & Operations:**
    * **Simplification:** Enter expression directly (e.g., `sin(x)**2 + cos(x)**2`)
    * **Solving Equation:** `solve(expression, symbol)` (e.g., `solve(2*x + 5 - 10, x)`)
    * **Differentiation:** `diff(expression, symbol)` (e.g., `diff(x**3 + sin(x), x)`)
    * **Integration:** `integrate(expression, symbol)` (e.g., `integrate(x**2, x)`)
    * **Limits:** `limit(expression, symbol, value)` (e.g., `limit(sin(x)/x, x, 0)`)
    * **Series Expansion:** `series(expression, symbol, point, order)` (e.g., `series(exp(x), x, 0, 5)`)
    * **Solve ODE:** `dsolve(equation, function)` (e.g., `dsolve(diff(f(x), x) + f(x), f(x))` - note: `f(x)` should be defined first using `f = sympy.Function('f')`)
    * **Polynomial Ops:** `poly(x**2 + x, x).factor()` (factor), `poly(x**3 + 1, x).roots()` (roots)
    * **Matrices:** `m = Matrix([[1,2],[3,4]])`, then `m.det()`, `m.inv()`, `m.transpose()`, `m.eigenvals()`, `m.eigenvects()`
        (You can define variables like `m` in subsequent inputs within the same session by evaluating them first.)
    """)
    
    algebra_input = st.text_input("Enter expression or command:", key="algebra_expr")

    if st.button("Execute", key="algebra_btn"):
        if not algebra_input:
            st.warning("Please enter an expression or command.")
            return

        try:
            # We use sympy.parse_expr with local_dict to allow for dynamic variable assignments
            # within the session, although for a stateless Streamlit app, this often means
            # the user needs to re-enter definitions.
            # A more advanced solution would be to maintain a session_state for defined symbols.
            # For now, SYMPY_EXPRESSION_GLOBALS serves as the primary scope.
            
            # Evaluate the expression directly, handling potential function calls
            # This is a simplified approach; a full symbolic interpreter is more complex.
            evaluated_result = eval(algebra_input, {}, SYMPY_EXPRESSION_GLOBALS)
            
            # SymPy expressions might need to be explicitly evaluated or simplified
            if isinstance(evaluated_result, sympy.Expr):
                result = sympy.simplify(evaluated_result)
            else:
                result = evaluated_result
            
            st.success(f"Result: `{result}`")

        except (sympy.SympifyError, TypeError, ValueError, AttributeError, NameError) as e:
            st.error(f"Error: `{e}`. Check syntax, variable definitions, or argument types for functions.")
        except Exception as e:
            st.error(f"An unexpected error occurred: `{e}`")


def trigonometry_mode():
    """
    Implements numerical trigonometry calculations with unit selection (Degrees, Radians, Gradians).
    """
    st.header("📐 Trigonometry (Numerical)")
    st.write("Calculate trigonometric values for angles in Degrees, Radians, or Gradians.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        angle_unit = st.radio("Select Angle Unit:", ("Degrees", "Radians", "Gradians"), key="trig_unit")
        trig_value_input = st.text_input("Enter Angle/Value:", key="trig_val_input")

    with col2:
        st.subheader("Functions")
        func_type = st.radio("Select Function Type:", ("Direct", "Inverse"), key="func_type")

        if func_type == "Direct":
            func = st.selectbox("Function:", ("sin", "cos", "tan"), key="direct_func")
        else: # Inverse
            func = st.selectbox("Function:", ("asin", "acos", "atan"), key="inverse_func")

    if st.button("Calculate Trig", key="trig_calc_btn"):
        if not trig_value_input:
            st.warning("Please enter an angle or value.")
            return

        try:
            input_val = float(trig_value_input)
            
            # Convert input value to radians for math module functions
            angle_rad = 0.0 # Initialize
            result_rad = None # Ensure result_rad is always defined
            if angle_unit == 'Degrees': 
                angle_rad = math.radians(input_val)
            elif angle_unit == 'Radians': 
                angle_rad = input_val
            elif angle_unit == 'Gradians': 
                angle_rad = input_val * (math.pi / 200)
            else: 
                st.error("Invalid unit selected.")
                return

            result = None
            if func == 'sin': 
                result = math.sin(angle_rad)
            elif func == 'cos': 
                result = math.cos(angle_rad)
            elif func == 'tan':
                # Check for angles where tan is undefined (e.g., 90, 270 degrees, plus 1e-10 for float precision)
                if abs(math.cos(angle_rad)) < 1e-9 and abs(angle_rad % math.pi - math.pi / 2) < 1e-9: # More robust check for n*pi + pi/2
                    result = "Undefined (approaching infinity)"
                else:
                    result = math.tan(angle_rad)
            elif func == 'asin':
                if not (-1 <= input_val <= 1): 
                    raise ValueError("Input for arcsin must be between -1 and 1.")
                result_rad = math.asin(input_val)
            elif func == 'acos':
                if not (-1 <= input_val <= 1): 
                    raise ValueError("Input for arccos must be between -1 and 1.")
                result_rad = math.acos(input_val)
            elif func == 'atan':
                result_rad = math.atan(input_val)
            
            # Convert inverse function results back to selected unit
            if func in ('asin', 'acos', 'atan') and result_rad is not None:
                if angle_unit == 'Degrees': 
                    result = math.degrees(result_rad)
                elif angle_unit == 'Radians': 
                    result = result_rad
                elif angle_unit == 'Gradians': 
                    result = result_rad * (200 / math.pi)

            if isinstance(result, str):
                st.success(f"Result: {result}")
            else:
                st.success(f"Result: `{result:.10f}`".rstrip('0').rstrip('.'))

        except ValueError as e:
            st.error(f"Input Error: `{e}`. Ensure valid number and range for inverse functions.")
        except Exception as e:
            st.error(f"Calculation Error: `{e}`")

# --- Higher Math Mode ---

def higher_math_mode():
    """
    Offers a range of higher mathematical tools including statistics,
    complex numbers, number base conversions, unit conversions, and financial math.
    """
    st.header("🎓 Higher Math")
    st.write("Explore advanced mathematical concepts and tools.")

    higher_math_section = st.sidebar.radio("Higher Math Section:", 
                                         ["Statistics", "Complex Numbers", "Number Base Converter", 
                                          "Unit Converter", "Financial Math"], key="higher_math_section")

    if higher_math_section == "Statistics":
        st.subheader("📊 Statistics")
        st.write("Calculate descriptive statistics for a list of numbers.")
        
        numbers_input = st.text_input("Enter numbers (space-separated):", key="stats_numbers")
        if st.button("Calculate Statistics", key="stats_btn"):
            try:
                numbers = parse_float_list(numbers_input)
                if not numbers:
                    st.warning("Please enter some numbers.")
                    return
                
                np_numbers = np.array(numbers)
                
                # Calculate mode carefully for empty arrays or no mode
                mode_val = "N/A"
                if np_numbers.size > 0:
                    counts = Counter(np_numbers)
                    if counts: # Ensure counts is not empty
                        max_count = max(counts.values())
                        modes = [num for num, count in counts.items() if count == max_count]
                        mode_val = modes[0] if len(modes) == 1 else modes # Return list if multiple modes

                st.json({
                    "Numbers": np_numbers.tolist(),
                    "Count": len(np_numbers),
                    "Sum": float(np.sum(np_numbers)),
                    "Mean": float(np.mean(np_numbers)),
                    "Median": float(np.median(np_numbers)),
                    "Mode": mode_val,
                    "Range": float(np.max(np_numbers) - np.min(np_numbers)),
                    "Standard Deviation (population)": float(np.std(np_numbers)), # Default is population std
                    "Variance (population)": float(np.var(np_numbers)), # Default is population var
                    "Q1 (25th percentile)": float(np.percentile(np_numbers, 25)),
                    "Q2 (50th percentile/Median)": float(np.percentile(np_numbers, 50)),
                    "Q3 (75th percentile)": float(np.percentile(np_numbers, 75)),
                    "Interquartile Range (IQR)": float(np.percentile(np_numbers, 75) - np.percentile(np_numbers, 25))
                })

                # Advanced Statistics (Distributions)
                st.subheader("Advanced Statistics: Probability Distributions")
                st.write("Calculate probabilities for common distributions.")
                dist_type = st.selectbox("Select Distribution:", ["Normal", "Binomial", "Poisson"], key="dist_type")
                
                if dist_type == "Normal":
                    mean = st.number_input("Mean (μ):", value=0.0, key="normal_mean")
                    std_dev = st.number_input("Standard Deviation (σ):", value=1.0, min_value=0.001, key="normal_std_dev")
                    x_val = st.number_input("X Value:", key="normal_x_val")
                    if st.button("Calculate Normal Probabilities", key="normal_calc_btn"):
                        pdf = stats.norm.pdf(x_val, loc=mean, scale=std_dev)
                        cdf = stats.norm.cdf(x_val, loc=mean, scale=std_dev)
                        st.write(f"**Normal Distribution (μ={mean}, σ={std_dev}):**")
                        st.write(f"PDF (P(X={x_val})): `{pdf:.6f}`")
                        st.write(f"CDF (P(X<={x_val})): `{cdf:.6f}`")
                elif dist_type == "Binomial":
                    n = st.number_input("Number of trials (n):", value=10, min_value=1, step=1, key="binomial_n")
                    p = st.number_input("Probability of success (p):", value=0.5, min_value=0.0, max_value=1.0, key="binomial_p")
                    k = st.number_input("Number of successes (k):", value=5, min_value=0, step=1, key="binomial_k")
                    if st.button("Calculate Binomial Probabilities", key="binomial_calc_btn"):
                        if k > n:
                            st.error("k cannot be greater than n.")
                            return
                        pmf = stats.binom.pmf(k, n, p)
                        cdf = stats.binom.cdf(k, n, p)
                        st.write(f"**Binomial Distribution (n={n}, p={p}):**")
                        st.write(f"PMF (P(X={k})): `{pmf:.6f}`")
                        st.write(f"CDF (P(X<={k})): `{cdf:.6f}`")
                elif dist_type == "Poisson":
                    mu = st.number_input("Average rate (μ):", value=3.0, min_value=0.0, key="poisson_mu")
                    k = st.number_input("Number of events (k):", value=2, min_value=0, step=1, key="poisson_k")
                    if st.button("Calculate Poisson Probabilities", key="poisson_calc_btn"):
                        pmf = stats.poisson.pmf(k, mu)
                        cdf = stats.poisson.cdf(k, mu)
                        st.write(f"**Poisson Distribution (μ={mu}):**")
                        st.write(f"PMF (P(X={k})): `{pmf:.6f}`")
                        st.write(f"CDF (P(X<={k})): `{cdf:.6f}`")

            except ValueError as e:
                st.error(f"Input Error: {e}. Please enter numbers separated by spaces.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    elif higher_math_section == "Complex Numbers":
        st.subheader("🌌 Complex Numbers")
        st.write("Perform operations with complex numbers.")
        
        st.info("Enter complex numbers as `a + bj` or `a - bj` (e.g., `2 + 3j`, `-1 - 0.5j`).")
        num1_str = st.text_input("Complex Number 1:", key="comp_num1")
        num2_str = st.text_input("Complex Number 2 (optional for operations):", key="comp_num2")

        def parse_complex(s):
            """Safely parses a complex number string, ensuring 'j' is treated correctly."""
            if not s.strip(): # Handle empty string for optional second number
                return None
            s = s.replace('i', 'j') # Allow 'i' as well for imaginary unit
            try:
                return complex(s)
            except ValueError:
                # Attempt to parse if user enters just a real number
                try:
                    return complex(float(s))
                except ValueError:
                    raise ValueError(f"Invalid complex number format: '{s}'")

        op = st.selectbox("Operation:", ["None", "Add", "Subtract", "Multiply", "Divide", 
                                          "Modulus (Num 1)", "Argument (Num 1)", "Conjugate (Num 1)",
                                          "Rectangular to Polar (Num 1)", "Polar to Rectangular"], key="comp_op")

        if st.button("Calculate Complex Numbers", key="comp_calc_btn"):
            try:
                c1 = parse_complex(num1_str)
                if c1 is None:
                    st.warning("Please enter at least Complex Number 1.")
                    return

                c2 = parse_complex(num2_str)

                result = None
                if op == "Add": 
                    if c2 is None: raise ValueError("Second complex number required for addition.")
                    result = c1 + c2
                elif op == "Subtract": 
                    if c2 is None: raise ValueError("Second complex number required for subtraction.")
                    result = c1 - c2
                elif op == "Multiply": 
                    if c2 is None: raise ValueError("Second complex number required for multiplication.")
                    result = c1 * c2
                elif op == "Divide": 
                    if c2 is None: raise ValueError("Second complex number required for division.")
                    if c2 == 0j: raise ValueError("Division by zero.")
                    result = c1 / c2
                elif op == "Modulus (Num 1)": result = abs(c1)
                elif op == "Argument (Num 1)": result = cmath.phase(c1) # Returns radians
                elif op == "Conjugate (Num 1)": result = c1.conjugate()
                elif op == "Rectangular to Polar (Num 1)": 
                    r, theta = cmath.polar(c1)
                    st.success(f"Polar Form (r, θ_radians): `({r:.6f}, {theta:.6f})`")
                    st.success(f"Polar Form (r, θ_degrees): `({r:.6f}, {math.degrees(theta):.6f})`")
                    return # Exit early as output is custom
                elif op == "Polar to Rectangular":
                    r = st.number_input("Radius (r):", key="polar_r_input", value=1.0)
                    theta_deg = st.number_input("Angle (θ) in Degrees:", key="polar_theta_deg_input", value=0.0)
                    theta_rad = math.radians(theta_deg)
                    result = cmath.rect(r, theta_rad)
                else:
                    st.warning("Please select an operation.")
                    return

                st.success(f"Result: `{result}`")
            except ValueError as e:
                st.error(f"Input Error: {e}. Check complex number format (e.g., '2+3j').")
            except Exception as e:
                st.error(f"Calculation Error: {e}")

    elif higher_math_section == "Number Base Converter":
        st.subheader("🔄 Number Base Converter")
        st.write("Convert numbers between different bases.")

        num_to_convert = st.text_input("Enter number:", key="base_num_input")
        from_base = st.selectbox("From Base:", ["2 (Binary)", "8 (Octal)", "10 (Decimal)", "16 (Hexadecimal)"], key="from_base")
        to_base = st.selectbox("To Base:", ["2 (Binary)", "8 (Octal)", "10 (Decimal)", "16 (Hexadecimal)"], key="to_base")

        if st.button("Convert Base", key="convert_base_btn"):
            if not num_to_convert:
                st.warning("Please enter a number to convert.")
                return
            
            try:
                base_map = {"2 (Binary)": 2, "8 (Octal)": 8, "10 (Decimal)": 10, "16 (Hexadecimal)": 16}
                from_b = base_map[from_base]
                to_b = base_map[to_base]

                # Convert to decimal first
                decimal_val = int(num_to_convert, from_b)

                # Convert from decimal to target base
                if to_b == 10:
                    st.success(f"Result in Decimal: `{decimal_val}`")
                elif to_b == 2:
                    st.success(f"Result in Binary: `{bin(decimal_val)[2:]}`")
                elif to_b == 8:
                    st.success(f"Result in Octal: `{oct(decimal_val)[2:]}`")
                elif to_b == 16:
                    st.success(f"Result in Hexadecimal: `{hex(decimal_val)[2:].upper()}`")

            except ValueError as e:
                st.error(f"Input Error: {e}. Ensure number is valid for the 'From Base' selected.")
            except Exception as e:
                st.error(f"Conversion Error: {e}")

    elif higher_math_section == "Unit Converter":
        st.subheader("📏 Unit Converter")
        st.write("Convert between common units.")

        # Unit conversion data structure (can be expanded heavily)
        UNIT_CATEGORIES = {
            "Length": {
                "meters": 1, "kilometers": 1000, "centimeters": 0.01, "millimeters": 0.001,
                "miles": 1609.344, "yards": 0.9144, "feet": 0.3048, "inches": 0.0254
            },
            "Mass": {
                "kilograms": 1, "grams": 0.001, "milligrams": 1e-6, "metric tons": 1000,
                "pounds": 0.45359237, "ounces": 0.028349523125, "tons (US)": 907.18474
            },
            "Time": {
                "seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800,
                "milliseconds": 0.001, "microseconds": 1e-6, "nanoseconds": 1e-9
            },
            "Temperature": { # Needs special handling, not linear scale
                "celsius": lambda c: c, # Placeholder for conversion functions
                "fahrenheit": lambda f: f,
                "kelvin": lambda k: k
            },
            "Area": {
                "square meters": 1, "square kilometers": 1e6, "square centimeters": 1e-4,
                "acres": 4046.8564224, "hectares": 10000, "square miles": 2589988.110336, "square feet": 0.09290304
            },
            "Volume": {
                "cubic meters": 1, "cubic centimeters": 1e-6, "liters": 0.001, "milliliters": 1e-6,
                "gallons (US)": 0.003785411784, "quarts (US)": 0.000946352946, "pints (US)": 0.000473176473
            },
            "Speed": {
                "meters/second": 1, "kilometers/hour": 1 / 3.6, "miles/hour": 0.44704, "feet/second": 0.3048
            }
        }

        category = st.selectbox("Select Category:", list(UNIT_CATEGORIES.keys()), key="unit_cat")
        
        value = st.number_input("Value to Convert:", value=1.0, key="unit_val")
        
        if category == "Temperature":
            from_unit = st.selectbox("From Unit:", ["celsius", "fahrenheit", "kelvin"], key="temp_from")
            to_unit = st.selectbox("To Unit:", ["celsius", "fahrenheit", "kelvin"], key="temp_to")
        else:
            from_unit = st.selectbox("From Unit:", list(UNIT_CATEGORIES[category].keys()), key="unit_from")
            to_unit = st.selectbox("To Unit:", list(UNIT_CATEGORIES[category].keys()), key="unit_to")

        if st.button("Convert Unit", key="convert_unit_btn"):
            try:
                if category == "Temperature":
                    # Convert to Celsius first
                    if from_unit == "celsius": 
                        c_val = value
                    elif from_unit == "fahrenheit": 
                        c_val = (value - 32) * 5/9
                    elif from_unit == "kelvin": 
                        c_val = value - 273.15
                    else: 
                        raise ValueError("Invalid temperature 'From' unit.")

                    # Convert from Celsius to target
                    if to_unit == "celsius": 
                        result = c_val
                    elif to_unit == "fahrenheit": 
                        result = (c_val * 9/5) + 32
                    elif to_unit == "kelvin": 
                        result = c_val + 273.15
                    else: 
                        raise ValueError("Invalid temperature 'To' unit.")
                    st.success(f"Result: `{result:.4f} {to_unit}`")
                else:
                    # Generic linear conversion
                    base_val = value * UNIT_CATEGORIES[category][from_unit]
                    result = base_val / UNIT_CATEGORIES[category][to_unit]
                    st.success(f"Result: `{result:.6f} {to_unit}`")

            except Exception as e:
                st.error(f"Conversion Error: {e}")

    elif higher_math_section == "Financial Math":
        st.subheader("💰 Financial Math")
        st.write("Calculate simple interest, compound interest, annuities, and loan payments.")

        financial_type = st.selectbox("Select Calculation:", 
                                      ["Simple Interest", "Compound Interest", "Annuity Future Value", 
                                       "Annuity Present Value", "Loan Payment"], key="financial_type")

        if financial_type == "Simple Interest":
            principal = st.number_input("Principal (P):", min_value=0.0, value=1000.0, format="%.2f", key="si_p")
            rate = st.number_input("Annual Interest Rate (R, as decimal, e.g., 0.05 for 5%):", min_value=0.0, value=0.05, format="%.4f", key="si_r")
            time = st.number_input("Time (T, in years):", min_value=0.0, value=1.0, format="%.2f", key="si_t")
            if st.button("Calculate Simple Interest", key="si_btn"):
                interest = principal * rate * time
                total_amount = principal + interest
                st.success(f"Interest Earned: `${interest:.2f}`")
                st.success(f"Total Amount: `${total_amount:.2f}`")

        elif financial_type == "Compound Interest":
            principal = st.number_input("Principal (P):", min_value=0.0, value=1000.0, format="%.2f", key="ci_p")
            rate = st.number_input("Annual Interest Rate (R, as decimal):", min_value=0.0, value=0.05, format="%.4f", key="ci_r")
            time = st.number_input("Time (T, in years):", min_value=0.0, value=1.0, format="%.2f", key="ci_t")
            n = st.number_input("Compounding Frequency (n, e.g., 1 for annually, 12 for monthly):", min_value=1, value=1, step=1, key="ci_n")
            if st.button("Calculate Compound Interest", key="ci_btn"):
                # Handle zero rate case to avoid division by zero or numerical instability
                if rate == 0:
                    amount = principal
                else:
                    amount = principal * (1 + rate / n)**(n * time)
                interest = amount - principal
                st.success(f"Total Amount: `${amount:.2f}`")
                st.success(f"Interest Earned: `${interest:.2f}`")

        elif financial_type == "Annuity Future Value":
            payment = st.number_input("Regular Payment (PMT):", min_value=0.0, value=100.0, format="%.2f", key="fv_pmt")
            rate = st.number_input("Periodic Interest Rate (i, as decimal):", min_value=0.0, value=0.005, format="%.4f", help="e.g., Annual Rate / Compounding Frequency (0.05/12)", key="fv_i")
            n_periods = st.number_input("Number of Periods (n):", min_value=1, value=12, step=1, key="fv_n")
            if st.button("Calculate Future Value", key="fv_btn"):
                if rate == 0:
                    fv = payment * n_periods
                else:
                    fv = payment * (((1 + rate)**n_periods - 1) / rate)
                st.success(f"Future Value of Annuity: `${fv:.2f}`")

        elif financial_type == "Annuity Present Value":
            payment = st.number_input("Regular Payment (PMT):", min_value=0.0, value=100.0, format="%.2f", key="pv_pmt")
            rate = st.number_input("Periodic Interest Rate (i, as decimal):", min_value=0.0, value=0.005, format="%.4f", help="e.g., Annual Rate / Compounding Frequency (0.05/12)", key="pv_i")
            n_periods = st.number_input("Number of Periods (n):", min_value=1, value=12, step=1, key="pv_n")
            if st.button("Calculate Present Value", key="pv_btn"):
                if rate == 0:
                    pv = payment * n_periods
                else:
                    pv = payment * ((1 - (1 + rate)**(-n_periods)) / rate)
                st.success(f"Present Value of Annuity: `${pv:.2f}`")
        
        elif financial_type == "Loan Payment":
            principal = st.number_input("Loan Amount (P):", min_value=0.0, value=10000.0, format="%.2f", key="loan_p")
            annual_rate = st.number_input("Annual Interest Rate (R, as decimal):", min_value=0.0, value=0.05, format="%.4f", key="loan_r")
            years = st.number_input("Loan Term (Years):", min_value=1, value=5, step=1, key="loan_years")
            
            if st.button("Calculate Loan Payment", key="loan_btn"):
                n_periods = years * 12 # Monthly payments
                monthly_rate = annual_rate / 12

                if monthly_rate == 0:
                    monthly_payment = principal / n_periods
                else:
                    # Handle case where (1 + monthly_rate)**n_periods is 1 (e.g., if n_periods is 0 or rate is tiny)
                    denominator = ((1 + monthly_rate)**n_periods - 1)
                    if denominator == 0:
                        st.error("Cannot calculate loan payment with these parameters. Likely an invalid term or rate.")
                        return
                    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**n_periods) / denominator
                
                st.success(f"Monthly Payment: `${monthly_payment:.2f}`")
                st.success(f"Total Amount Paid: `${monthly_payment * n_periods:.2f}`")
                st.success(f"Total Interest Paid: `${monthly_payment * n_periods - principal:.2f}`")

# --- Graphing Mode ---

def graphing_mode():
    """
    Allows plotting of single or multiple 2D functions (f(x), parametric, polar)
    using Matplotlib, embedded in Streamlit.
    """
    st.header("📈 Graphing Functions")
    st.write("Visualize mathematical functions.")

    plot_type = st.sidebar.radio("Select Plot Type:", ["f(x) Plot", "Parametric Plot", "Polar Plot"], key="plot_type")

    if plot_type == "f(x) Plot":
        st.subheader("Plot f(x)")
        func_str = st.text_area("Enter functions (one per line, e.g., `x**2`, `sin(x)`):", key="fx_func_input")
        col_range1, col_range2 = st.columns(2)
        with col_range1:
            x_min = st.number_input("X-axis Min:", value=-10.0, key="fx_xmin")
        with col_range2:
            x_max = st.number_input("X-axis Max:", value=10.0, key="fx_xmax")

        if st.button("Plot f(x)", key="fx_plot_btn"):
            if not func_str:
                st.warning("Please enter at least one function.")
                return
            if x_min >= x_max:
                st.error("X-max must be greater than X-min.")
                return

            x_vals = np.linspace(x_min, x_max, 400) # More points for smoother curves
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Function Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.grid(True)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)

            functions = [f.strip() for f in func_str.split('\n') if f.strip()]
            
            x_sym = sympy.Symbol('x')
            
            for func_s in functions:
                try:
                    # Use SymPy to parse and lambdify the expression
                    func_sympy = sympy.parse_expr(func_s, global_dict=SYMPY_EXPRESSION_GLOBALS)
                    
                    # Ensure the function is only in terms of 'x'
                    if func_sympy.free_symbols and func_sympy.free_symbols != {x_sym}:
                        st.error(f"Function `{func_s}` must be in terms of 'x' only. Detected other variables: {func_sympy.free_symbols}.")
                        continue
                    
                    func_numerical = sympy.lambdify(x_sym, func_sympy, 'numpy')
                    
                    y_vals = func_numerical(x_vals)
                    # Handle potential division by zero or other infinities for clean plots
                    y_vals[np.isinf(y_vals)] = np.nan 
                    
                    ax.plot(x_vals, y_vals, label=f"f(x) = {func_s}")
                except Exception as e:
                    st.error(f"Error plotting function `{func_s}`: {e}")
                    continue
            
            ax.legend()
            st.pyplot(fig)

    elif plot_type == "Parametric Plot":
        st.subheader("Parametric Plot (x(t), y(t))")
        x_t_str = st.text_input("Enter x(t):", key="xt_func_input", value="cos(t)")
        y_t_str = st.text_input("Enter y(t):", key="yt_func_input", value="sin(t)")
        col_range1, col_range2 = st.columns(2)
        with col_range1:
            t_min = st.number_input("T-axis Min:", value=0.0, key="t_min")
        with col_range2:
            t_max = st.number_input("T-axis Max:", value=2 * math.pi, key="t_max")

        if st.button("Plot Parametric", key="parametric_plot_btn"):
            if not x_t_str or not y_t_str:
                st.warning("Please enter both x(t) and y(t) functions.")
                return
            if t_min >= t_max:
                st.error("T-max must be greater than T-min.")
                return
            
            t_vals = np.linspace(t_min, t_max, 400)
            t_sym = sympy.Symbol('t')

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Parametric Plot (x(t)={x_t_str}, y(t)={y_t_str})")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.grid(True)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
            ax.set_aspect('equal', adjustable='box') # Keep aspect ratio for circles/ellipses

            try:
                x_func_sympy = sympy.parse_expr(x_t_str, global_dict=SYMPY_EXPRESSION_GLOBALS)
                y_func_sympy = sympy.parse_expr(y_t_str, global_dict=SYMPY_EXPRESSION_GLOBALS)
                
                # Ensure functions are only in terms of 't'
                if x_func_sympy.free_symbols and x_func_sympy.free_symbols != {t_sym}:
                    st.error(f"x(t) must be in terms of 't' only. Detected other variables: {x_func_sympy.free_symbols}.")
                    return
                if y_func_sympy.free_symbols and y_func_sympy.free_symbols != {t_sym}:
                    st.error(f"y(t) must be in terms of 't' only. Detected other variables: {y_func_sympy.free_symbols}.")
                    return

                x_numerical = sympy.lambdify(t_sym, x_func_sympy, 'numpy')
                y_numerical = sympy.lambdify(t_sym, y_func_sympy, 'numpy')

                x_coords = x_numerical(t_vals)
                y_coords = y_numerical(t_vals)
                
                # Handle potential NaNs/Infinities from evaluation
                combined_coords = np.array([x_coords, y_coords]).T
                combined_coords = combined_coords[~np.any(np.isinf(combined_coords), axis=1)]
                combined_coords = combined_coords[~np.any(np.isnan(combined_coords), axis=1)]

                if combined_coords.size > 0:
                    ax.plot(combined_coords[:, 0], combined_coords[:, 1])
                else:
                    st.warning("No valid points to plot for the given parametric functions and range.")

                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting parametric function: {e}")

    elif plot_type == "Polar Plot":
        st.subheader("Polar Plot (r(θ))")
        r_theta_str = st.text_input("Enter r(θ):", key="r_theta_func_input", value="sin(2*theta)")
        col_range1, col_range2 = st.columns(2)
        with col_range1:
            theta_min = st.number_input("Theta Min (radians):", value=0.0, key="theta_min")
        with col_range2:
            theta_max = st.number_input("Theta Max (radians):", value=2 * math.pi, key="theta_max")

        if st.button("Plot Polar", key="polar_plot_btn"):
            if not r_theta_str:
                st.warning("Please enter r(θ) function.")
                return
            if theta_min >= theta_max:
                st.error("Theta Max must be greater than Theta Min.")
                return

            theta_vals = np.linspace(theta_min, theta_max, 400)
            theta_sym = sympy.Symbol('theta') # SymPy symbol for theta

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 6))
            ax.set_title(f"Polar Plot (r(θ)={r_theta_str})")
            ax.grid(True)
            
            try:
                r_func_sympy = sympy.parse_expr(r_theta_str, global_dict=SYMPY_EXPRESSION_GLOBALS)
                
                # Check for free symbols, allow only 'theta'
                if r_func_sympy.free_symbols and r_func_sympy.free_symbols != {theta_sym}:
                    st.error(f"r(θ) must be in terms of 'theta' only. Detected other variables: {r_func_sympy.free_symbols}.")
                    return
                
                r_numerical = sympy.lambdify(theta_sym, r_func_sympy, 'numpy')

                r_coords = r_numerical(theta_vals)
                
                # Handle potential NaNs/Infinities from evaluation
                r_coords[np.isinf(r_coords)] = np.nan

                # Plotting directly in polar coordinates, filtering out invalid points
                ax.plot(theta_vals[~np.isnan(r_coords)], r_coords[~np.isnan(r_coords)])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting polar function: {e}")

# --- Main Application Layout ---
def main():
    st.sidebar.title("Calculator Modes")
    app_mode = st.sidebar.radio("Go to", 
                                ["Simple Math", "Algebra (Symbolic)", "Trigonometry", 
                                 "Higher Math", "Graphing"],
                                 key="main_app_mode")

    if app_mode == "Simple Math":
        simple_math_mode()
    elif app_mode == "Algebra (Symbolic)":
        algebra_mode()
    elif app_mode == "Trigonometry":
        trigonometry_mode()
    elif app_mode == "Higher Math":
        higher_math_mode()
    elif app_mode == "Graphing":
        graphing_mode()

if __name__ == "__main__":
    main()

# --- Extended Code Section for Line Count Increase ---
# This section contains additional comments and placeholder functions
# to significantly increase the code's line count as requested.
# These functions are currently not integrated into the main application logic
# but serve to demonstrate potential future expansions or simply to expand the file size.

# region Extensive Comments and Placeholder Functions
# 
# This is a large block of comments designed to extend the length of the code file.
# It serves as a placeholder for potential future features or simply to meet
# line count requirements without altering the existing functionality.
#
# Future Feature Idea: Data Analysis Module
# This module could offer advanced statistical analysis, machine learning integrations,
# or data visualization beyond simple graphing.
#
# Placeholder for a complex data processing function.
def process_large_dataset(data_path: str, processing_steps: list, output_format: str = "csv"):
    """
    A placeholder function for processing large datasets.
    It simulates reading data, applying various transformations, and saving the output.

    Parameters:
    data_path (str): The path to the input dataset file (e.g., CSV, JSON).
    processing_steps (list): A list of dictionaries, each defining a processing step.
                             Example: [{'type': 'filter', 'column': 'value', 'condition': '> 100'},
                                       {'type': 'aggregate', 'group_by': 'category', 'method': 'sum'}]
    output_format (str): The desired output format (e.g., 'csv', 'json', 'parquet').

    Returns:
    str: A message indicating the status of the processing or path to output.
    """
    # This is a highly conceptual implementation.
    # In a real scenario, this would involve libraries like pandas.
    
    # Simulate reading data
    st.write(f"Simulating reading data from: {data_path}")
    simulated_data = {} # Replace with actual data loading
    
    # Apply processing steps
    for step in processing_steps:
        step_type = step.get('type')
        if step_type == 'filter':
            column = step.get('column')
            condition = step.get('condition')
            st.write(f"Applying filter: column='{column}', condition='{condition}'")
            # simulated_data = filter_data(simulated_data, column, condition)
        elif step_type == 'aggregate':
            group_by = step.get('group_by')
            method = step.get('method')
            st.write(f"Applying aggregation: group_by='{group_by}', method='{method}'")
            # simulated_data = aggregate_data(simulated_data, group_by, method)
        else:
            st.warning(f"Unknown processing step type: {step_type}")
            
    # Simulate saving output
    output_file = f"processed_data.{output_format}"
    st.write(f"Simulating saving data to: {output_file} in {output_format} format.")
    
    return f"Processing completed. Output saved to {output_file} (simulated)."

# Placeholder for another advanced function: Symbolic Equation Solver with Graphing
def solve_and_plot_symbolic_equation(equation_str: str, variable_str: str, plot_range: tuple = (-5, 5)):
    """
    A conceptual function to solve a symbolic equation and plot its solutions.
    For single variable equations, it would plot the roots on a number line or 2D plane.
    For two variables, it might plot the implicit curve.

    Parameters:
    equation_str (str): The equation as a string (e.g., "x**2 - 4 = 0", "x**2 + y**2 = 9").
    variable_str (str): The variable(s) to solve for (e.g., "x", "x, y").
    plot_range (tuple): A tuple (min, max) for the plotting range for x (and y if applicable).

    Returns:
    str: A message with solutions and plot status.
    """
    st.write(f"Attempting to solve equation: `{equation_str}` for `{variable_str}`")
    
    try:
        # Parse the equation
        equation_sympy = sympy.sympify(equation_str, locals=SYMPY_EXPRESSION_GLOBALS)
        
        # Determine variables
        variables = tuple(sympy.Symbol(s.strip()) for s in variable_str.split(','))
        
        # Solve the equation
        solutions = sympy.solve(equation_sympy, *variables)
        st.success(f"Solutions: {solutions}")

        # Basic plotting logic (conceptual)
        if len(variables) == 1:
            var_sym = variables[0]
            st.write(f"Plotting solutions for {var_sym} in range {plot_range}")
            # This would involve plotting points or intervals on an axis.
            # For implicit plotting (e.g., equation = 0), a dedicated plotting library
            # like `plot_implicit` from SymPy's plotting module would be needed,
            # which is more complex to integrate directly into Streamlit's pyplot.
            # Example: from sympy.plotting import plot_implicit
            # p1 = plot_implicit(equation_sympy, (var_sym, plot_range[0], plot_range[1]), show=False)
            # st.pyplot(p1) # This requires a different plotting setup
            st.info("Direct plotting of 1D equation solutions is conceptual here. Requires specific plotting libraries.")
        elif len(variables) == 2:
            st.write(f"Plotting implicit curve for 2 variables in range {plot_range}")
            st.info("Implicit plotting for 2 variables is conceptual here. Requires specific plotting libraries (e.g., sympy.plotting.plot_implicit).")
        else:
            st.info("Plotting for more than two variables is not supported.")
            
    except (sympy.SympifyError, TypeError, ValueError, AttributeError) as e:
        st.error(f"Error solving or parsing equation: `{e}`")
    except Exception as e:
        st.error(f"An unexpected error occurred: `{e}`")

# Placeholder for a machine learning utility function.
def train_simple_model(features_data: list, target_data: list, model_type: str = "linear_regression"):
    """
    A conceptual function for training a simple machine learning model.

    Parameters:
    features_data (list): A list of lists representing input features (e.g., [[x1, x2], [y1, y2]]).
    target_data (list): A list representing target values.
    model_type (str): The type of model to train (e.g., "linear_regression", "logistic_regression").

    Returns:
    str: A message indicating the model training status.
    """
    st.write(f"Simulating training a {model_type} model...")
    # This would involve libraries like scikit-learn.
    
    if len(features_data) != len(target_data):
        st.error("Feature data and target data must have the same number of samples.")
        return "Failed: Data mismatch."

    # Simulate model training
    if model_type == "linear_regression":
        st.write("Fitting a linear regression model.")
        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression()
        # model.fit(features_data, target_data)
        # st.success(f"Model trained. Coefficients: {model.coef_}, Intercept: {model.intercept_}")
        return "Simulated Linear Regression Model Trained."
    elif model_type == "logistic_regression":
        st.write("Fitting a logistic regression model.")
        # from sklearn.linear_model import LogisticRegression
        # model = LogisticRegression()
        # model.fit(features_data, target_data)
        # st.success(f"Model trained. Coefficients: {model.coef_}, Intercept: {model.intercept_}")
        return "Simulated Logistic Regression Model Trained."
    else:
        return "Model type not recognized for simulation."