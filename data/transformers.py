import cv2
import multiprocessing
from PIL import Image
from math import ceil
import numpy as np


def lower_resolution(img, scale):
    W, H, _ = img.shape
    img = cv2.resize(img, (int(H * scale), int(W * scale)))
    return img


def JPEG_compression(img, score, path="/tmp"):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), score]
    result, encimg = cv2.imencode(".jpg", img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def Gaussian_blur(img, sigma):
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return img

def random_lower_resolution(img, scale, p=0.5):
    W, H, _ = img.shape
    if np.random.random() <= p:
        scale = scale + np.random.rand() * (1 - scale)
        img = cv2.resize(img, (int(H * scale), int(W * scale)))
    return img

def random_JPEG_compression(img, low, high, p=0.5, path="/tmp"):
    if np.random.random() <= p:
        score = np.random.randint(low, high)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), score]
        result, encimg = cv2.imencode(".jpg", img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    else:
        return img

def random_Gaussian_blur(img, sigma, p=0.5):
    if np.random.random() <= p:
        sigma = np.random.randint(1, sigma)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return img

def random_flip(img, p=0.5, mode=1):
    if np.random.random() <= p:
        img = cv2.flip(img, mode)
    return img


def random_sample_aug(img, p1, scale, p2, low, high, p3, sigma):
    r = np.random.random()
    if r <= p1:
        return random_lower_resolution(img, scale, 1)
    elif r <= p1 + p2:
        return random_JPEG_compression(img, low, high, 1)
    elif r <= p1 + p2 + p3:
        return random_Gaussian_blur(img, sigma, 1)
    else:
        return img


if __name__ == "__main__":
    img = Image.open("/hd1/mengzheling/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val/ai/500_sdv4_00020.png").convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img_r = lower_resolution(img, 0.5)
    img_c = JPEG_compression(img, 30)
    img_g = Gaussian_blur(img, 3)

    # cv2.imwrite("ai.png", img)
    # cv2.imwrite("r.png", img_r)
    # cv2.imwrite("c.png", img_c)
    # cv2.imwrite("g.png", img_g)
