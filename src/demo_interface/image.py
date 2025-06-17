import cv2

Image = cv2.typing.MatLike


def convert_rgb_to_bgr(image: Image) -> Image:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def convert_bgr_to_rgb(image: Image) -> Image:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
