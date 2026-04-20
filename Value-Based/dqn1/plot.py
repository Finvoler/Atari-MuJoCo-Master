import pandas as pd
import matplotlib.pyplot as plt
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'training_log_rainbow_adam.csv')


try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"错误：无法找到文件: {csv_path}")
    exit()


plt.figure(figsize=(12, 6))




plt.plot(df['step'], df['reward'], color='blue', label='Raw Reward', linewidth=1)

plt.title('Training Process: Reward per Step')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)


save_path = os.path.join(current_dir, 'raw_reward_plot.png')
plt.savefig(save_path)
print(f"绘图完成，图片已保存至: {save_path}")

plt.show()