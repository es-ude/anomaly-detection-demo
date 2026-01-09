from argparse import ArgumentParser, Namespace
from pathlib import Path

from demo.camera.image import Image
from demo.dataset_recording.utils import IMG_EXT, load_image, save_image


def _center_crop(image: Image, image_size: tuple[int, int]) -> Image:
    h_input, w_input, _ = image.shape
    h_output, w_output = image_size

    x_mid, y_mid = w_input // 2, h_input // 2
    x_delta, y_delta = w_output // 2, h_output // 2
    x_min = x_mid - x_delta
    x_max = x_mid + x_delta
    y_min = y_mid - y_delta
    y_max = y_mid + y_delta

    cropped_frame = image[y_min:y_max, x_min:x_max]

    return cropped_frame


def main(args: Namespace) -> None:
    root_src_dir: Path = args.source
    root_dest_dir: Path = args.destination

    src_image_files = filter(lambda p: p.is_file(), root_src_dir.rglob(f"*.{IMG_EXT}"))

    for src_image_file in src_image_files:
        image = load_image(src_image_file)
        image = _center_crop(image, image_size=(args.height, args.width))

        dest_image_file = root_dest_dir / src_image_file.relative_to(root_src_dir)
        dest_image_file.parent.mkdir(parents=True, exist_ok=True)

        save_image(image, dest_image_file)


if __name__ == "__main__":
    argparser = ArgumentParser("Crop Recorded Images")
    argparser.add_argument("-s", "--source", type=Path)
    argparser.add_argument("-d", "--destination", type=Path)
    argparser.add_argument("--width", default=800, type=int)
    argparser.add_argument("--height", default=800, type=int)
    args = argparser.parse_args()

    main(args)
