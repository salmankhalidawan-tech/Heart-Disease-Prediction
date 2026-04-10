import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Import Themed Tkinter for modern widgets
import numpy as np
import pickle
import sys
import warnings
import os
BASE_DIR = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(BASE_DIR,"heart_disease_model.pkl"),"rb"))

# Suppress non-critical warnings from ML libraries
warnings.filterwarnings("ignore")

# ---------------- 1. LOAD MODEL ----------------
MODEL_FILE = r"C:\Users\Salman Khalid\Desktop\Heart_Disease_Prediction\heart_disease_model.pkl"
model = None

try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    messagebox.showerror("Initialization Error", f"Failed to load model:\n{str(e)}")
    sys.exit()

# ---------------- 2. VALIDATION (ULTRA-RELIABLE) ----------------
# The 'P' parameter is the entry's value if the change is allowed.

def is_int(P):
    """Robust integer validation. Allows empty string and valid integer representation."""
    if P == "": return True
    try:
        int(P)
        return True
    except ValueError:
        return False

def is_float(P):
    """Robust float validation. Allows partial input like '.', '-', or '+'."""
    if P == "": return True
    try:
        float(P)
        return True
    except ValueError:
        # Allow partial input characters (., -, +) only if they are the entire string
        return P in ['.', '-', '+']

# ---------------- 3. PREDICT ----------------
def predict():
    # Ensure variables are accessible (They are global due to the GUI setup structure)
    global entry_age, entry_trestbps, entry_chol, entry_thalach, entry_oldpeak
    global var_sex, var_cp, var_fbs, var_restecg, var_exang, var_slope, var_ca, var_thal
    
    try:
        # Check if all numerical entry fields are filled
        numeric_values = [e.get().strip() for e in numeric_entries]
        if not all(numeric_values):
             raise ValueError("Please fill all numerical fields.")

        # Collect features in the correct order and convert types
        features = [
            float(entry_age.get()),
            int(var_sex.get()),
            int(var_cp.get()),
            float(entry_trestbps.get()),
            float(entry_chol.get()),
            int(var_fbs.get()),
            int(var_restecg.get()),
            float(entry_thalach.get()),
            int(var_exang.get()),
            float(entry_oldpeak.get()),
            int(var_slope.get()),
            int(var_ca.get()),
            int(var_thal.get())
        ]

        result = model.predict(np.array([features]))
        msg = "⚠ Heart Disease Detected" if result[0] == 1 else "✔ Patient is Healthy"
        messagebox.showinfo("Prediction Result", msg)

    except ValueError as ve:
        # Catch errors from conversion (e.g., trying to convert '.' to float) or empty fields
        messagebox.showerror("Input Error", f"Invalid data or missing field: {str(ve)}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An unexpected error occurred: {str(e)}")

# ---------------- 4. RESET (Consolidated) ----------------
def reset():
    # Clear all text entries
    for e in numeric_entries:
        e.delete(0, tk.END)
        
    # Reset all categorical variables to a default state
    categorical_defaults = {
        var_sex: "0", var_cp: "0", var_fbs: "0", var_restecg: "0",
        var_exang: "0", var_slope: "0", var_ca: "0", var_thal: "1"
    }
    for var, default in categorical_defaults.items():
        var.set(default)
        
# ---------------- 5. GUI SETUP (Modern ttk layout) ----------------

root = tk.Tk()
root.title("CardioPredict: Heart Disease Prediction")
root.geometry("800x600")
root.resizable(False, False)

# Apply a modern theme
style = ttk.Style(root)
style.theme_use('clam')

# Global validation commands
vcmd_i = root.register(is_int)
vcmd_f = root.register(is_float)

# List to hold Entry widgets for the reset function
numeric_entries = []

# Main container for padding
main_container = ttk.Frame(root, padding="20 20 20 20")
main_container.pack(fill='both', expand=True)

# Title
ttk.Label(main_container, text="Heart Disease Prediction", font=('Arial', 18, 'bold')).pack(pady=10)
ttk.Label(main_container, text="Please enter all 13 required features:", font=('Arial', 10)).pack(pady=5)

# --- A. FRAME FOR DEMOGRAPHICS & VITALS ---
vitals_frame = ttk.LabelFrame(main_container, text="Demographics & Vitals (Continuous Values)", padding="10 10 10 10")
vitals_frame.pack(fill='x', pady=10)

continuous_fields = [
    ("Age", "entry_age", vcmd_i, "years"),
    ("Resting BP (trestbps)", "entry_trestbps", vcmd_i, "mm Hg"),
    ("Cholesterol (chol)", "entry_chol", vcmd_i, "mg/dl"),
    ("Max Heart Rate (thalach)", "entry_thalach", vcmd_i, "bpm"),
    ("Oldpeak (ST depression)", "entry_oldpeak", vcmd_f, "(0.0 - 6.2)"),
]

for i, (label_text, var_name, vcmd, unit) in enumerate(continuous_fields):
    row = i // 2
    col_start = (i % 2) * 3

    ttk.Label(vitals_frame, text=f"{label_text}:", font=('Arial', 10, 'bold')).grid(row=row, column=col_start, sticky='w', padx=5, pady=5)
    
    # Use ttk.Entry for modern look
    entry = ttk.Entry(vitals_frame, validate='key', validatecommand=(vcmd, '%P'), width=15)
    entry.grid(row=row, column=col_start + 1, sticky='ew', padx=(0, 5), pady=5)
    
    ttk.Label(vitals_frame, text=unit).grid(row=row, column=col_start + 2, sticky='w')
    
    globals()[var_name] = entry
    numeric_entries.append(entry)

vitals_frame.grid_columnconfigure(1, weight=1)
vitals_frame.grid_columnconfigure(4, weight=1)

# --- B. FRAME FOR CATEGORICAL DATA ---
categorical_frame = ttk.LabelFrame(main_container, text="Categorical & Binary Features", padding="10 10 10 10")
categorical_frame.pack(fill='x', pady=10)

# Define variables for OptionMenus/Comboboxes
var_sex = tk.StringVar(value="0")
var_cp = tk.StringVar(value="0")
var_fbs = tk.StringVar(value="0")
var_restecg = tk.StringVar(value="0")
var_exang = tk.StringVar(value="0")
var_slope = tk.StringVar(value="0")
var_ca = tk.StringVar(value="0")
var_thal = tk.StringVar(value="1")

categorical_fields = [
    ("Sex (1=Male, 0=Female)", var_sex, ["0", "1"]),
    ("Chest Pain Type (cp)", var_cp, ["0", "1", "2", "3"]),
    ("Fasting Blood Sugar (>120)", var_fbs, ["0", "1"]),
    ("Resting ECG (restecg)", var_restecg, ["0", "1", "2"]),
    ("Exercise Angina (exang)", var_exang, ["0", "1"]),
    ("ST Slope (slope)", var_slope, ["0", "1", "2"]),
    ("Major Vessels (ca)", var_ca, ["0", "1", "2", "3", "4"]),
    ("Thal (Thalassemia)", var_thal, ["1", "2", "3"]),
]

for i, (label_text, variable, options) in enumerate(categorical_fields):
    row = i // 2
    col_start = (i % 2) * 2

    ttk.Label(categorical_frame, text=f"{label_text}:", font=('Arial', 10)).grid(row=row, column=col_start, sticky='w', padx=5, pady=5)
    
    # Use ttk.Combobox for a modern dropdown look
    combo = ttk.Combobox(categorical_frame, textvariable=variable, values=options, state="readonly", width=12)
    combo.grid(row=row, column=col_start + 1, sticky='ew', padx=(0, 20), pady=5)
    combo.set(variable.get()) # Set to the default value

categorical_frame.grid_columnconfigure(1, weight=1)
categorical_frame.grid_columnconfigure(3, weight=1)

# --- C. BUTTONS ---
btn_frame = ttk.Frame(main_container)
btn_frame.pack(pady=20)

style.configure('TButton', font=('Arial', 12, 'bold'), padding=6)
style.configure('Predict.TButton', background='#007bff', foreground='white')
style.configure('Reset.TButton', background='#6c757d', foreground='white')

ttk.Button(btn_frame, text="PREDICT RISK", command=predict, style='Predict.TButton', width=18).pack(side='left', padx=20)
ttk.Button(btn_frame, text="RESET FORM", command=reset, style='Reset.TButton', width=18).pack(side='left', padx=20)

root.mainloop()