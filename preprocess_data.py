"""
preprocess_data.py - 数据预处理用于 AI 训练
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

def load_and_merge_data(data_dir="collected_data"):
    """加载并合并所有采集的数据文件"""
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return None
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not all_files:
        print(f"No CSV files found in {data_dir}")
        return None
    
    dfs = []
    for file in all_files:
        filepath = os.path.join(data_dir, file)
        df = pd.read_csv(filepath)
        df['source_file'] = file
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(all_files)} files, total {len(combined)} records")
    
    return combined

def create_features(df):
    """
    创建用于训练的特征
    
    特征包括:
    - 当前命令位置
    - 前1-3个时刻的命令位置（历史）
    - 运动方向
    - 速度（变化率）
    """
    # 添加历史特征
    df['cmd_prev_1'] = df['cmd_x'].shift(1)
    df['cmd_prev_2'] = df['cmd_x'].shift(2)
    df['cmd_prev_3'] = df['cmd_x'].shift(3)
    
    # 添加方向特征（+1 增加, -1 减少, 0 不变）
    df['direction'] = np.sign(df['cmd_x'] - df['cmd_prev_1'])
    df['direction'] = df['direction'].fillna(0)
    
    # 添加速度特征（每步的变化量）
    df['velocity'] = df['cmd_x'] - df['cmd_prev_1']
    df['velocity'] = df['velocity'].fillna(0)
    
    # 删除包含 NaN 的行（前几行）
    df = df.dropna().reset_index(drop=True)
    
    return df

def prepare_training_data(df, target='cmd_x', feature_cols=None):
    """
    准备训练数据
    
    默认特征: ['cmd_x', 'cmd_prev_1', 'cmd_prev_2', 'direction', 'velocity']
    目标: 'actual_x' (正向模型) 或 'cmd_x' (逆模型)
    """
    if feature_cols is None:
        feature_cols = ['cmd_x', 'cmd_prev_1', 'cmd_prev_2', 'direction', 'velocity']
    
    X = df[feature_cols].values
    y = df[target].values
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': feature_cols
    }

def save_preprocessed_data(data_dict, filename="preprocessed_data.npz"):
    """保存预处理后的数据"""
    np.savez(
        filename,
        X_train=data_dict['X_train'],
        X_test=data_dict['X_test'],
        y_train=data_dict['y_train'],
        y_test=data_dict['y_test'],
        feature_names=data_dict['feature_names']
    )
    print(f"Preprocessed data saved to {filename}")

def plot_data_distribution(df):
    """可视化数据分布"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 命令位置分布
    axes[0, 0].hist(df['cmd_x'], bins=50, alpha=0.7)
    axes[0, 0].set_xlabel('Command (um)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Command Distribution')
    
    # 误差分布
    axes[0, 1].hist(df['error'], bins=50, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Error (um)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    
    # 命令 vs 实际
    axes[1, 0].scatter(df['cmd_x'], df['actual_x'], s=1, alpha=0.3)
    axes[1, 0].plot([df['cmd_x'].min(), df['cmd_x'].max()], 
                    [df['cmd_x'].min(), df['cmd_x'].max()], 'r--')
    axes[1, 0].set_xlabel('Command (um)')
    axes[1, 0].set_ylabel('Actual (um)')
    axes[1, 0].set_title('Command vs Actual')
    
    # 按方向分组
    for direction, group in df.groupby('direction'):
        label = {1: 'Increasing', -1: 'Decreasing', 0: 'Stationary'}.get(direction, 'Unknown')
        axes[1, 1].plot(group['cmd_x'], group['error'], '.', alpha=0.3, label=label)
    axes[1, 1].set_xlabel('Command (um)')
    axes[1, 1].set_ylabel('Error (um)')
    axes[1, 1].set_title('Error by Direction')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("="*50)
    print("Data Preprocessing")
    print("="*50)
    
    # 加载数据
    df = load_and_merge_data("collected_data")
    
    if df is not None:
        # 创建特征
        df = create_features(df)
        
        # 可视化
        plot_data_distribution(df)
        
        # 准备训练数据（逆模型：输入实际位置，输出命令）
        # 注意：这里我们用 actual_x 作为输入，cmd_x 作为输出（逆模型）
        # 也可以训练正向模型，只需交换 X 和 y
        # 这里先准备逆模型数据，稍后训练时会使用
        # 但 prepare_training_data 函数需要特征列，我们使用当前特征列（基于命令）
        # 如果要训练逆模型，特征应该是实际位置及其历史
        # 这里我们提供两个版本，实际使用时可以按需选择
        
        # 版本1：正向模型（命令 -> 实际）
        train_data_forward = prepare_training_data(df, target='actual_x')
        save_preprocessed_data(train_data_forward, "preprocessed_data_forward.npz")
        
        # 版本2：逆模型（实际 -> 命令）需要重新构造特征
        # 构造基于实际位置的特征
        df_inv = df.copy()
        df_inv['actual_prev_1'] = df_inv['actual_x'].shift(1)
        df_inv['actual_prev_2'] = df_inv['actual_x'].shift(2)
        df_inv['direction_actual'] = np.sign(df_inv['actual_x'] - df_inv['actual_prev_1'])
        df_inv['velocity_actual'] = df_inv['actual_x'] - df_inv['actual_prev_1']
        df_inv = df_inv.dropna().reset_index(drop=True)
        
        feature_cols_inv = ['actual_x', 'actual_prev_1', 'actual_prev_2', 'direction_actual', 'velocity_actual']
        X_inv = df_inv[feature_cols_inv].values
        y_inv = df_inv['cmd_x'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X_inv, y_inv, test_size=0.2, random_state=42)
        scaler_X_inv = StandardScaler()
        scaler_y_inv = StandardScaler()
        X_train = scaler_X_inv.fit_transform(X_train)
        X_test = scaler_X_inv.transform(X_test)
        y_train = scaler_y_inv.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = scaler_y_inv.transform(y_test.reshape(-1, 1)).ravel()
        
        inv_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler_X': scaler_X_inv,
            'scaler_y': scaler_y_inv,
            'feature_names': feature_cols_inv
        }
        save_preprocessed_data(inv_data, "preprocessed_data_inverse.npz")
        
        print(f"\nTraining set size (forward): {len(train_data_forward['X_train'])}")
        print(f"Test set size (forward): {len(train_data_forward['X_test'])}")
        print(f"Features (forward): {train_data_forward['feature_names']}")
        print(f"\nTraining set size (inverse): {len(inv_data['X_train'])}")
        print(f"Test set size (inverse): {len(inv_data['X_test'])}")
        print(f"Features (inverse): {inv_data['feature_names']}")
        # 在 save_preprocessed_data 函数后添加
def save_model_data(data_dict, filename):
    """保存模型训练所需的所有数据（包括标准化器）"""
    joblib.dump(data_dict, filename)

# 在 main 中修改
import joblib

# 正向模型
train_data_forward = prepare_training_data(df, target='actual_x')
joblib.dump({
    'X_train': train_data_forward['X_train'],
    'X_test': train_data_forward['X_test'],
    'y_train': train_data_forward['y_train'],
    'y_test': train_data_forward['y_test'],
    'scaler_X': train_data_forward['scaler_X'],
    'scaler_y': train_data_forward['scaler_y'],
    'feature_names': train_data_forward['feature_names']
}, "forward_model_data.pkl")

# 逆模型
train_data_inverse = prepare_training_data(df_inv, target='cmd_x', 
                                           feature_cols=feature_cols_inv)
joblib.dump({
    'X_train': train_data_inverse['X_train'],
    'X_test': train_data_inverse['X_test'],
    'y_train': train_data_inverse['y_train'],
    'y_test': train_data_inverse['y_test'],
    'scaler_X': train_data_inverse['scaler_X'],
    'scaler_y': train_data_inverse['scaler_y'],
    'feature_names': train_data_inverse['feature_names']
}, "inverse_model_data.pkl")