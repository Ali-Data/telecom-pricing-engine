"""
MLOps Pipeline: Expected Value (EV) Optimization Visualization
Generates a high-resolution, publication-ready plot demonstrating
the Prescriptive Pricing Engine's behavior for low-risk vs. high-risk customers.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def generate_ev_plot():
    print("[*] Generating EV Optimization Plot...")
    
    # 1. Define simulated data for the paper
    discounts = np.array([0, 5, 10, 15, 20, 25, 30])
    
    # Scenario A: Low-risk customer (Optimal discount is 0%)
    ev_low_risk = np.array([100, 95, 88, 75, 60, 45, 30])
    
    # Scenario B: High-risk customer (Optimal discount is 15%)
    ev_high_risk = np.array([20, 40, 65, 80, 70, 55, 35])

    # 2. Initialize plot with high resolution for Q1 Journals (300 DPI)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 3. Plot the scenarios
    ax.plot(discounts, ev_low_risk, marker='o', linestyle='-', color='#1f77b4', 
            linewidth=2.5, markersize=8, label='Customer A (Low Risk -> Optimal: 0%)')
    
    ax.plot(discounts, ev_high_risk, marker='s', linestyle='--', color='#d62728', 
            linewidth=2.5, markersize=8, label='Customer B (High Risk -> Optimal: 15%)')

    # 4. Highlight the optimal (Max EV) points
    ax.scatter(0, 100, s=250, facecolors='none', edgecolors='#1f77b4', linewidth=2.5, zorder=5)
    ax.scatter(15, 80, s=250, facecolors='none', edgecolors='#d62728', linewidth=2.5, zorder=5)

    # 5. Format axes and labels (Adding padding to prevent text cutoff)
    ax.set_title('Expected Value (EV) vs. Offered Discount Level', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Offered Discount (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Expected Value (Simulated Profit Index)', fontsize=12, fontweight='bold', labelpad=10)
    
    ax.set_xticks(discounts)
    ax.set_xticklabels([f"{d}%" for d in discounts])
    
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Place legend inside the plot cleanly
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9, borderpad=1)

    # 6. Save the figure with tight bounding box to ensure nothing is clipped
    output_filename = 'academic_ev_optimization_plot.png'
    plt.tight_layout(pad=2.0)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"[+] Success: High-resolution plot saved as '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    generate_ev_plot()