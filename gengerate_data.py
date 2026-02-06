import numpy as np


def generate_dating_data(filename='dating.txt', num_samples=300):
    """
    生成模拟的海伦约会数据，共 num_samples 条
    """
    # 初始化数据矩阵和标签
    # 标签 1: 最想约会 (里程中高，游戏少，冰激凌适中)
    # 标签 2: 一般 (里程少，游戏适中，冰激凌不限)
    # 标签 3: 不想约会 (里程极高 或 游戏极多)

    data = []
    labels = []

    # --- 生成 1/3 的数据为 标签1 (Like) ---
    # 逻辑：生活丰富(里程2w-6w)，比较顾家(游戏<10%)
    for _ in range(num_samples // 3):
        fly = np.random.randint(20000, 60000)
        game = np.random.uniform(0, 10)
        ice = np.random.uniform(0.2, 1.2)
        data.append([fly, game, ice])
        labels.append(1)

    # --- 生成 1/3 的数据为 标签2 (Small Doses) ---
    # 逻辑：宅(里程<2w)，或者比较平庸
    for _ in range(num_samples // 3):
        fly = np.random.randint(0, 20000)
        game = np.random.uniform(5, 15)
        ice = np.random.uniform(0, 1.5)
        data.append([fly, game, ice])
        labels.append(2)

    # --- 生成 1/3 的数据为 标签3 (Didnt Like) ---
    # 逻辑：要么太忙(里程>6w)，要么沉迷游戏(游戏>15%)
    for _ in range(num_samples // 3):
        if np.random.random() > 0.5:
            # 极忙型
            fly = np.random.randint(60000, 90000)
            game = np.random.uniform(0, 15)
        else:
            # 沉迷游戏型
            fly = np.random.randint(10000, 50000)
            game = np.random.uniform(15, 30)

        ice = np.random.uniform(0.1, 1.8)
        data.append([fly, game, ice])
        labels.append(3)

    # 转换为 NumPy 数组
    data = np.array(data)
    labels = np.array(labels)

    # 打乱数据顺序 (Shuffle)，这对机器学习很重要，避免标签扎堆
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # 写入文件
    with open(filename, 'w') as f:
        for i in range(num_samples):
            # 格式: 里程 游戏 冰激凌 标签
            # 使用 \t 分隔，保留2位小数
            line = f"{int(data[i][0])}\t{data[i][1]:.2f}\t{data[i][2]:.2f}\t{labels[i]}\n"
            f.write(line)

    print(f"成功生成 {num_samples} 条数据，已保存至 {filename}")


# --- 运行生成 ---
generate_dating_data('dating.txt', 300)