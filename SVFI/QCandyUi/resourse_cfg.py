import os

# 这些图片路径是相对main.py的, 而不是相对本py文件的
# 下面这些图片资源是TitleBar的
IMG_MIN_NORM = "minsize.png"
IMG_MIN_HOVER = "minsizehover.png"
IMG_MIN_PRESS = "minsizepress.png"
IMG_MAX_NORM = "maxsize.png"
IMG_MAX_HOVER = "maxsizehover.png"
IMG_MAX_PRESS = "maxsizepress.png"
IMG_RESIZE_NORM = "resize.png"
IMG_RESIZE_HOVER = "resizehover.png"
IMG_RESIZE_PRESS = "resizepress.png"
IMG_CLOSE_NORM = "close.png"
IMG_CLOSE_HOVER = "closehover.png"
IMG_CLOSE_PRESS = "closepress.png"

# 资源路径: 相对路径没找到就去安装路径下找
IMAGE_ROOT = './candyUi/'
if not os.path.exists(IMAGE_ROOT):
    IMAGE_ROOT = (os.path.split(__file__)[0] + "\\candyUi\\").replace('\\', '/')
