import numpy as np
import matplotlib.pyplot as plt
import os

ENV_NAME = "HalfCheetah-v4"  

SEED = 0

file_name = f"./results/TD3_{ENV_NAME}_{SEED}.npy"
save_name = f"{ENV_NAME}_curve.png"

try:
    results = np.load(file_name)
    print(f"Loading: {file_name}")
    print(f"Number of evaluation points: {len(results)}")

    eval_freq = 5000
    steps = np.arange(1, len(results) + 1) * eval_freq

    plt.figure(figsize=(10, 6), dpi=120)

    if 'Ant' in ENV_NAME:
        line_color = '#D95319' 
    elif 'Hopper' in ENV_NAME:
        line_color = '#77AC30' 
    else:
        line_color = '#0072BD' 

    plt.plot(steps, results, label=f'TD3 ({ENV_NAME})', color=line_color, linewidth=2)

    plt.title(f"Training Performance: TD3 on {ENV_NAME}", fontsize=16, fontweight='bold')
    plt.xlabel("Timesteps", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')

    max_reward = np.max(results)
    max_step = steps[np.argmax(results)]
    
    plt.plot(max_step, max_reward, 'o', color='red')

    plt.annotate(f'Max: {max_reward:.0f}', 
                 xy=(max_step, max_reward), 
                 xytext=(max_step, max_reward + (max_reward * 0.1)), 
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=11)

    plt.tight_layout() 
    plt.savefig(save_name)
    print("-" * 30)
    print(f"Saved plot to: {save_name}")
    print(f"Best score on {ENV_NAME}: {max_reward:.2f}")
    print("-" * 30)

except FileNotFoundError:
    print(f"Error: file not found — {file_name}")
    print(f"Make sure results/TD3_{ENV_NAME}_{SEED}.npy exists.")
    print("Hint: run training first, or check the environment name for typos.")