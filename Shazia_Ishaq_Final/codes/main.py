# ================================================================
# File:        main.py
# Author:      Shazia Ishaq
# Course:      Introduction to Credit Risk — University of Freiburg
# Description: MAIN RUN FILE — runs the complete analysis pipeline
#
# This is the single entry point required by the assignment.
# Run this file to execute all four parts in order:
#   Part 1: Data Cleaning
#   Part 2: Parameter Estimation
#   Part 3: Simulation and Risk Measurement
#   Part 4: Analysis and Plots
#
# All outputs are saved automatically to the results/ folder.
#
# HOW TO RUN:
#   Click the green play button on this file in PyCharm
#   OR: python main.py
#
# DATA DIRECTORY:
#   Change DIR_DATA below if data folder location changes.
#   This is the only line you need to modify.
# ================================================================

import os
import importlib.util

# ================================================================
# DATA DIRECTORY — change only this line if data moves
# ================================================================
DIR_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'Final Assignment', 'data'
)

print("=" * 65)
print("   CREDIT PORTFOLIO RISK ANALYSIS — FULL PIPELINE")
print("   Shazia Ishaq | University of Freiburg | 2026")
print("=" * 65)
print(f"\nData: {os.path.normpath(DIR_DATA)}")
print("\nRunning all 4 parts in order...")

# ── Helper: run one script file ───────────────────────────────
def run_script(script_name):
    """
    Load and execute a Python script file.
    This allows main.py to run each part sequentially.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        script_name
    )
    spec   = importlib.util.spec_from_file_location(
        script_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

# ── Run all parts in order ────────────────────────────────────
print("\n" + "=" * 65)
print("PART 1: DATA CLEANING")
print("=" * 65)
run_script('01_data_cleaning.py')

print("\n" + "=" * 65)
print("PART 2: PARAMETER ESTIMATION")
print("=" * 65)
run_script('02_parameter_estimation.py')

print("\n" + "=" * 65)
print("PART 3: SIMULATION")
print("=" * 65)
run_script('03_simulation.py')

print("\n" + "=" * 65)
print("PART 4: ANALYSIS AND PLOTS")
print("=" * 65)
run_script('04_analysis_plots.py')

print("\n" + "=" * 65)
print("PIPELINE COMPLETE!")
print("All outputs saved in results/ folder")
print("=" * 65)