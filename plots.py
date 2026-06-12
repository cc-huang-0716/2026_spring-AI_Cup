import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

url_sample_submission = f'https://drive.google.com/uc?export=download&id=1jKK0XxAaRF1Cg7m1Y_XQaqgj9UKgLFFD'
url_test = f"https://drive.google.com/uc?export=download&id=1_WRaGLX0-hQO2zoUqHZMRGXvUePRotTq"
url_train = f"https://drive.google.com/uc?export=download&id=1YgWvLaesMqtXlC3k-Js3XdnSINON5n6P"

df_ss = pd.read_csv(url_sample_submission)
df_test = pd.read_csv(url_test)
df_train = pd.read_csv(url_train)

print(df_ss.head())
print(df_test.head())
print(df_train.head())

print(df_ss.info())
print(df_test.info())
print(df_train.info())

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 
sns.set_theme(style="whitegrid", font='Microsoft JhengHei')

output_dir = 'eda_plots'
os.makedirs(output_dir, exist_ok=True)
print(f"確認資料夾已建立：{output_dir}")

num_cols = df_train.select_dtypes(include=['number']).columns

for col in num_cols:
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_train[col], kde=True, color='skyblue')
    plt.title(f'{col} - Distribution')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_train[col], color='salmon')
    plt.title(f'{col} - Outliers')
    
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/num_{col}.png', dpi=300) 
    plt.close()
    print(f"已存檔:num_{col}.png")

plt.figure(figsize=(12, 10))
corr = df_train.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
