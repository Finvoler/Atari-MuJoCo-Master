import pandas as pd
import matplotlib.pyplot as plt
import os


file_name = 'training_log_rainbow_adam.csv'


current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, file_name)


try:
    df = pd.read_csv(csv_path)
    print(f"成功读取文件，共包含 {len(df)} 条数据。")
except FileNotFoundError:
    print(f"错误：找不到文件 {csv_path}")
    exit()


plt.figure(figsize=(12, 6))




plt.plot(df['step'], df['reward'], color='blue', label='Raw Reward', linewidth=1)


plt.title(f'Training Process: {file_name}')
plt.xlabel('Step')
plt.ylabel('Reward')


plt.legend()
plt.grid(True)


output_name = file_name.replace('.csv', '.png')
save_path = os.path.join(current_dir, output_name)
plt.savefig(save_path)
print(f"图片已保存至: {save_path}")


plt.show()