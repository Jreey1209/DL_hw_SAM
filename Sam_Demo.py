import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def onclick(event):

    #使用全局变量x，y
    global x,y
    # 检查是否是鼠标左键点击事件
    if event.button == 1:
        # 获取点击点的坐标
        x = event.xdata
        y = event.ydata
        # 在控制台打印点击点的坐标
        print(f"Clicked at ({x}, {y})")

        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def model_instantiation(check_point_path, device_model, type):
        # 确定使用的权重文件位置和是否使用cuda等：
    sam_checkpoint = check_point_path  #"./model/sam_vit_h_4b8939.pth"
    device = device_model  #有GPU版本的就用CUDA，没有就用CPU  e.g. ”cpu“
    model_type = type            #"default"

    # 模型实例化
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return predictor

def find_point_in_image(img_path):
    # 读取图像并选择抠图点：
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(image)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return x, y, image

def show_point_in_image(x,y):
    input_point = np.array([[x, y]])
    input_label = np.array([1])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    return input_point, input_label

def find_max_square(mask):
    # 获取 True 值的索引坐标
    true_indices = np.argwhere(mask)

    # 计算最小外接矩形的左上角和右下角坐标
    min_row = np.min(true_indices[:, 0])   #左上角坐标x
    max_row = np.max(true_indices[:, 0])   #右下角坐标x
    min_col = np.min(true_indices[:, 1])   #左上角坐标y
    max_col = np.max(true_indices[:, 1])   #右下角坐标y

    return min_row, max_row, min_col, max_col

if __name__ == "__main__":

    x, y, image = find_point_in_image("./image/car8.jpg")
    predictor = model_instantiation(check_point_path="./model/sam_vit_h_4b8939.pth", device_model="cpu", type="default")
    predictor.set_image(image)

    input_point, input_label = show_point_in_image(x,y)
    # 同时扣取多个结果
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # 遍历读取每个扣出的结果，会输出三组不同score的图片。
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        #获取mask为True矩阵中最小外接矩的坐标
        min_row, max_row, min_col, max_col = find_max_square(mask)
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

        #展示不同得分下切割后的图像
        cropped_image = image[min_row:max_row, min_col:max_col]
        plt.imshow(cropped_image)
        plt.show()

    #最后输出效果最好的图片
    best_scores = np.array([])
    for score in scores:
        best_scores = np.append(best_scores,abs(score-1))

    best_score = np.min(best_scores)
    best_index = np.argmin(best_scores)
    min_row, max_row, min_col, max_col = find_max_square(masks[best_index])
    plt.title(f"Mask {best_index + 1}, Loss: {best_score:.3f}", fontsize=18)
    plt.axis('off')
    cropped_image = image[min_row:max_row, min_col:max_col]
    plt.imshow(cropped_image)
    plt.show()