import os

import matplotlib
matplotlib.use('Agg')  # 使用非交互式的 Agg 后端

import numpy as np
import matplotlib.pyplot as plt

# def save_heatmap(data, save_path='./res/vis_temp.png', cmap='hot'):
def save_heatmap(image, save_path='./res/vis_save_heatmap.png', cmap=None):# TODO：rgb格式：matplotlib.pyplot as plt和PIL.Image；bgr格式：cv2
    """
    生成热图并保存为 PNG 文件。

    参数:
    - data: 输入的二维数据 (numpy ndarray)
    - save_path: 保存图像的路径，默认为 './res/vis_temp.png'
    - cmap: 使用的颜色映射，默认为 'hot'

    # 示例调用
    h, w = 10, 10
    data = np.random.random((h, w))  # 生成示例数据
    save_heatmap(data, save_path='./res/vis_temp.png')
    """
    # 创建并保存热图
    plt.imshow(image, cmap=cmap, interpolation='nearest')
    plt.colorbar()  # 显示颜色条
    plt.savefig(save_path)  # 保存图像
    plt.close()  # 关闭图像，释放资源

def visualize_rgb_image(image: np.ndarray, title: str = "RGB Image", save_path='./res/vis_visualize_rgb_image.png'):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是 RGB 图像，即形状为 (h, w, 3)。")

    # 显示图像
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')  # 不显示坐标轴

    if save_path:
        plt.savefig(save_path)  # 保存图像到指定路径
    else:
        plt.show()  # 显示图像

    plt.close()


def save_points_on_image(image, points, labels, save_path='./res/vis_save_points_on_image.png', figsize=(10, 10), point_size=100):
    """
    在图像上标记多个点并保存到指定路径。

    参数:
    - image: ndarray，RGB 图像或灰度图像，作为背景。
    - points: list 或 ndarray，形状为 (N, 2)，每行是一个点的 (x, y) 坐标。
    - labels: list 或 ndarray，形状为 (N,)，每个点的标签，1 表示前景点，0 表示背景点。
    - save_path: str，保存标记图像的路径，默认为 './res/vis_temp.png'。
    - figsize: tuple，可选，图像的显示尺寸。
    - point_size: int，可选，标记点的大小。
    """
    # 确保 points 和 labels 是 ndarray 类型
    points = np.array(points) if not isinstance(points, np.ndarray) else points
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

    # 检查输入的合法性
    if points.shape[1] != 2:
        raise ValueError("points 应该是形状为 (N, 2) 的 ndarray 或 list，表示每个点的 (x, y) 坐标。")
    if len(points) != len(labels):
        raise ValueError("points 和 labels 的长度必须相同。")

    # 检查保存路径的目录是否存在，不存在则创建
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建绘图
    plt.figure(figsize=figsize)
    plt.imshow(image)

    # 获取前景点和背景点
    foreground_points = points[labels == 1]
    background_points = points[labels == 0]

    # 绘制前景点（红色星形标记）
    if len(foreground_points) > 0:
        plt.scatter(foreground_points[:, 0], foreground_points[:, 1],
                    c='red', label='Foreground', s=point_size, marker='*')

    # 绘制背景点（蓝色圆形标记）
    if len(background_points) > 0:
        plt.scatter(background_points[:, 0], background_points[:, 1],
                    c='blue', label='Background', s=point_size, marker='o')

    # 添加图例和关闭坐标轴
    plt.legend()
    plt.axis('off')  # 不显示坐标轴

    # 保存图像到指定路径
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭绘图，释放内存



def plot_image_with_bboxes_and_points(cur_image, visualize_save_path, bbox=None, points=None, labels=None, word_fg=None, word_bg=None, mask=None):
    """
    在RGB图像上绘制边界框及前景/背景点，并显示mask，最终保存图像，并在左下角添加前景和背景的描述文字。

    Parameters:
    - cur_image (ndarray): RGB图像，形状为 (H, W, 3)
    - bbox (ndarray): 边界框，格式为 [x_min, y_min, x_max, y_max]
    - points (list): 样本点列表，每个点为 [x, y]，表示样本的坐标
    - labels (list): 标签列表，与 points 对应，1 表示前景，0 表示背景
    - visualize_save_path (str): 图像保存路径的文件夹
    - word_fg (str): 前景描述文本
    - word_bg (str): 背景描述文本
    - mask (ndarray): 二值化掩膜，形状为 (H, W)，表示感兴趣的区域
    """
    # 显示图像
    plt.imshow(cur_image)
    ax = plt.gca()

    if bbox is not None:
        # 绘制边界框
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

    if points is not None and labels is not None:
        # 绘制前景和背景点
        for pt, label in zip(points, labels):
            x, y = pt
            color = 'red' if label == 1 else 'blue'
            plt.scatter(x, y, color=color, s=10, marker='o')  # 10是点的大小，marker='o'表示圆点

    # 如果提供了mask，将其显示在图像上
    if mask is not None:
        # 使用imshow叠加mask，设置alpha透明度
        plt.imshow(mask, cmap='jet', alpha=0.5)  # alpha控制透明度，0.5表示半透明

    if word_bg and word_fg:
        # 在左下角添加前景和背景描述文本
        plt.text(10, cur_image.shape[0] - 30, f"Foreground: {word_fg}", color='lime', fontsize=12, weight='bold')
        plt.text(10, cur_image.shape[0] - 60, f"Background: {word_bg}", color='cyan', fontsize=12, weight='bold')

    # 不显示坐标轴
    plt.axis('off')

    # 保存图像
    plt.savefig(visualize_save_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前图形，避免内存溢出

    print(f"Image saved to: {visualize_save_path}")

