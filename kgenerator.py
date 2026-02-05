import numpy as np
import pandas as pd


def calculate_arrhenius(A, n, Ea, T_start, T_end, step):
    """
    Calculate the rate constant k based on the three-parameter Arrhenius equation.
    (This function remains unchanged from the previous version)

    Parameters:
    A (float): Pre-exponential factor (Unit must be consistent with k, e.g., s^-1)
    n (float): Temperature exponent (Dimensionless)
    Ea (float): Activation energy (Unit: J/mol)
    T_start (float): Start temperature (Unit: K)
    T_end (float): End temperature (Unit: K)
    step (float): Temperature sampling step (Unit: K)

    Returns:
    pandas.DataFrame: DataFrame containing Temperature, Rate Constant k, and ln(k)
    """
    R = 8.314  # Ideal gas constant (J/(mol·K))
    temperatures = np.arange(T_start, T_end + step, step)
    results = []

    for T in temperatures:
        if T <= 0:
            continue
        k = A * (T ** n) * np.exp(-Ea / (R * T))
        ln_k = np.log(k)
        results.append({
            'Temperature (K)': T,
            'Rate Constant k': k,
            'ln(k)': ln_k
        })

    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    # --- User Input Parameters ---
    # Define multiple sets of parameters in the list below
    parameter_sets = [
        {"A": 2.06e12, "n": 0.04, "Ea": 1472, "description": "No. 1"},
        {"A": 1.73e12, "n": 0.00, "Ea": 2420, "description": "No. 2"},
        {"A": 4.58e12, "n": 0.03, "Ea": 1382, "description": "No. 3"},
        {"A": 1.86e12, "n": 0.01, "Ea": 2072, "description": "No. 4"},
        {"A": 1.06e14, "n": -0.13, "Ea": 8671, "description": "No. 5"},
        {"A": 4.66e13, "n": -0.07, "Ea": 8224, "description": "No. 6"},
        {"A": 2.90e13, "n": -0.24, "Ea": 4349, "description": "No. 7"},
        {"A": 2.21e13, "n": -0.01, "Ea": 15737, "description": "No. 8"},
        {"A": 5.95e13, "n": -0.24, "Ea": 23189, "description": "No. 9"},
        {"A": 1.84e14, "n": -0.24, "Ea": 20051, "description": "No. 10"},
        {"A": 6.10e15, "n": -0.24, "Ea": 27302, "description": "No. 11"},
        {"A": 2.03e11, "n": 0.31, "Ea": 45903, "description": "No. 12"},
        {"A": 5.48e13, "n": 0.39, "Ea": 47298, "description": "No. 13"},
        {"A": 4.79e13, "n": 0.28, "Ea": 43791, "description": "No. 14"},
        {"A": 1.97e13, "n": 0.44, "Ea": 46048, "description": "No. 15"},
        {"A": 1.76e13, "n": 0.46, "Ea": 49351, "description": "No. 16"},
        {"A": 1.54e2, "n": 3.40, "Ea": 36959, "description": "No. 17"},
        {"A": 1.48e15, "n": -0.03, "Ea": 39110, "description": "No. 18"},
        #{"A": 1.48e15, "n": -0.03, "Ea": 47298, "description": "No. 19"},
        {"A": 5.43e13, "n": 0.42, "Ea": 36223, "description": "No. 20"},
        #{"A": 1.00e12, "n": 0.28, "Ea": 36223, "description": "No. 21"},
        {"A": 1.02e14, "n": 0.28, "Ea": 33981, "description": "No. 22"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 23"},
        {"A": 9.82e15, "n": -0.34, "Ea": 33975, "description": "No. 24"},
        # {"A": 5.85e11, "n": 0.00, "Ea": 35324, "description": "No. 25"},
        {"A": 5.85e12, "n": 1.07, "Ea": 35324, "description": "No. 26"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 27"},
        {"A": 3.36e11, "n": 0.01, "Ea": -60, "description": "No. 28"},
        {"A": 2.27e10, "n": 0.78, "Ea": 12828, "description": "No. 29"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 30"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 31"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 32"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 33"},
        {"A": 2.47, "n": 3.44, "Ea": 25748, "description": "No. 34"},
        {"A": 5.20e12, "n": 0.10, "Ea": 3332, "description": "No. 35"},
        {"A": 6.79e11, "n": 0.07, "Ea": 20683, "description": "No. 36"},
        {"A": 3.36e10, "n": 1.08, "Ea": 4725, "description": "No. 37"},
        {"A": 4.33e10, "n": 0.06, "Ea": 19141, "description": "No. 38"},
        {"A": 2.34e-3, "n": 3.56, "Ea": 5885, "description": "No. 39"},
        {"A": 5.16e-3, "n": 3.61, "Ea": -2098, "description": "No. 40"},
        {"A": 1.35e12, "n": 0.33, "Ea": 36500, "description": "No. 41"},
        {"A": 1.24e12, "n": 0.34, "Ea": 36681, "description": "No. 42"},
        {"A": 1.24e12, "n": 0.38, "Ea": 35652, "description": "No. 43"},
        {"A": 9.07e9, "n": 1.69, "Ea": 38416, "description": "No. 44"},
        # {"A": 1.00e12, "n": 0.06, "Ea": 5885, "description": "No. 45"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 46"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 47"},
        # {"A": 1.00e12, "n": 0.00, "Ea": 0, "description": "No. 48"},
        # {"A": 3.98e13, "n": 0.08, "Ea": 36013, "description": "No. 49"},
        {"A": 3.98e13, "n": 0.08, "Ea": 36013, "description": "No. 50"},
        {"A": 1.19e12, "n": 1.23, "Ea": 31147, "description": "No. 51"},
        {"A": 3.22e24, "n": -2.55, "Ea": 37480, "description": "No. 52"},
        {"A": 7.31e13, "n": 1.07, "Ea": 31205, "description": "No. 53"},
        {"A": 1.34e14, "n": 0.34, "Ea": 31525, "description": "No. 54"},
        {"A": 3.61e-1, "n": 3.68, "Ea": 20628, "description": "No. 55"},
        {"A": 4.15e1, "n": 2.80, "Ea": 10005, "description": "No. 56"},
        {"A": 3.88e12, "n": 0.43, "Ea": 3996, "description": "No. 57"},
        {"A": 4.38, "n": 2.89, "Ea": 1120, "description": "No. 58"},
        {"A": 3.91e13, "n": -0.06, "Ea": 5864, "description": "No. 59"},
        {"A": 6.55e13, "n": 0.30, "Ea": 4095, "description": "No. 60"},
        {"A": 8.55, "n": 2.99, "Ea": -5374, "description": "No. 61"},
        {"A": 3.42e12, "n": 0.37, "Ea": 22261, "description": "No. 62"},
        {"A": 1.64e13, "n": 0.28, "Ea": 14412, "description": "No. 63"},
        {"A": 1.35e13, "n": -0.02, "Ea": 364, "description": "No. 64"},
        {"A": 1.07e12, "n": 1.29, "Ea": 46994, "description": "No. 70"},
        {"A": 6.86e14, "n": -0.09, "Ea": 41309, "description": "No. 71"},
        {"A": 1.25e1, "n": 3.12, "Ea": 1720, "description": "No. 73"},
        {"A": 1.41e12, "n":0.24, "Ea": 17862, "description": "No. 74"},
        {"A": 5.75e1, "n": 2.85, "Ea": -2643, "description": "No. 75"},
        {"A": 6.20e12, "n": 0.22, "Ea": 6679, "description": "No. 76"},
        {"A": 6.70, "n": 3.07, "Ea": 706, "description": "No. 77"},
        {"A": 3.89e3, "n": 2.48, "Ea": -1780, "description": "No. 78"},
        {"A": 7.44e12, "n": 0.29, "Ea": 20452, "description": "No. 79"},
    ]

    # --- General Parameters ---
    T_start_temp = 303.0
    T_end_temp = 683.0
    sampling_step = 5.0

    # --- Execute batch calculation and merge all results ---

    # Create an empty list to store the DataFrame generated from each calculation
    all_results_dfs = []

    print("Starting batch calculation...")

    # Iterate through each set of parameters
    for params in parameter_sets:
        description = params.get('description', 'N/A')
        print(f"--- Calculating: {description} ---")

        # Call function to calculate, getting a partial result DataFrame
        results_df = calculate_arrhenius(
            A=params["A"],
            n=params["n"],
            Ea=params["Ea"],
            T_start=T_start_temp,
            T_end=T_end_temp,
            step=sampling_step
        )

        # Add identifier columns to this partial result to indicate its source
        results_df['Parameter Set Description'] = description
        results_df['A (Pre-exponential factor)'] = params["A"]
        results_df['n (Temperature exponent)'] = params["n"]
        results_df['Ea (J/mol)'] = params["Ea"]

        # Add the processed DataFrame to the list
        all_results_dfs.append(results_df)

    # Use pd.concat to vertically concatenate all DataFrames in the list into one large DataFrame
    final_df = pd.concat(all_results_dfs, ignore_index=True)

    # (Optional) Adjust column order to put identifier columns first for easier viewing
    column_order = [
        'Parameter Set Description',
        'A (Pre-exponential factor)',
        'n (Temperature exponent)',
        'Ea (J/mol)',
        'Temperature (K)',
        'Rate Constant k',
        'ln(k)'
    ]
    final_df = final_df[column_order]

    # --- Save to Excel ---
    output_filename = "FILE_PATH"
    try:
        # Save the final merged DataFrame to a single Excel worksheet
        final_df.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"\nAll calculations completed! Results successfully merged and exported to file: {output_filename}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")