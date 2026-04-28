#!/usr/bin/env python3
"""
Generate a collage of images from a given directory.

This script creates a collage of images from a specified directory and saves it as a single image file.
The collage can be customized with various parameters such as the number of rows and columns,
the spacing between images, and the background color.
"""

import os
from typing import List, Optional, Tuple

import typer
from PIL import Image


def load_images_from_directory(
    directory: str, extensions: Optional[List[str]] = None
) -> List[Image.Image]:
    """
    Load images from a directory.

    Args:
        directory: Path to the directory containing images.
        extensions: List of file extensions to include (e.g., ['.jpg', '.png']).
                    If None, includes common image extensions.

    Returns:
        List of PIL.Image objects.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

    images: List[Image.Image] = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            try:
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {filename}: {e}")

    return images


def create_collage(
    images: List[Image.Image],
    rows: int,
    cols: int,
    spacing: int = 10,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Create a collage of images.

    Args:
        images: List of PIL.Image objects.
        rows: Number of rows in the collage.
        cols: Number of columns in the collage.
        spacing: Spacing between images in pixels.
        background_color: RGB tuple for background color.

    Returns:
        Collage image.
    """
    if not images:
        raise ValueError("No images provided")

    # Calculate the size of each cell in the grid
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Calculate the total size of the collage
    total_width = cols * max_width + (cols - 1) * spacing
    total_height = rows * max_height + (rows - 1) * spacing

    # Create a new image with the background color
    collage = Image.new("RGB", (total_width, total_height), background_color)

    # Paste each image into the collage
    for i, img in enumerate(images[: rows * cols]):
        row = i // cols
        col = i % cols

        x = col * (max_width + spacing)
        y = row * (max_height + spacing)

        # Resize the image to fit the cell while maintaining aspect ratio
        img.thumbnail((max_width, max_height))

        # Calculate the position to center the image in the cell
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2

        collage.paste(img, (x + x_offset, y + y_offset))

    return collage


def parse_background_color(background_str: str) -> Tuple[int, int, int]:
    """
    Parse background color string into RGB tuple.

    Args:
        background_str: String in format "R,G,B"

    Returns:
        Tuple of (R, G, B) values

    Raises:
        ValueError: If format is invalid
    """
    background_color = tuple(int(x.strip()) for x in background_str.split(","))
    if len(background_color) != 3:
        raise ValueError("Background color must have exactly 3 components (R, G, B)")
    else:
        return background_color


def main(
    directory: str = typer.Argument(..., help="Directory containing images"),
    rows: int = typer.Option(3, "--rows", help="Number of rows in the collage"),
    cols: int = typer.Option(3, "--cols", help="Number of columns in the collage"),
    spacing: int = typer.Option(
        10, "--spacing", help="Spacing between images in pixels"
    ),
    background: str = typer.Option(
        "0,0,0",
        "--background",
        help='Background color as RGB tuple (e.g., "255,255,255" for white)',
    ),
    output: str = typer.Option(
        "collage.png", "--output", help="Output file name for the collage"
    ),
) -> None:
    """
    Main function to parse arguments and create the collage.
    """
    # Parse background color
    try:
        background_color: Tuple[int, int, int] = parse_background_color(background)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Load images
    images: List[Image.Image] = load_images_from_directory(directory)
    if not images:
        typer.echo("Error: No valid images found in the directory", err=True)
        raise typer.Exit(code=1)

    # Create collage
    try:
        collage: Image.Image = create_collage(
            images, rows, cols, spacing, background_color
        )
        collage.save(output)
        typer.echo(f"Collage saved as {output}")
    except Exception as e:
        typer.echo(f"Error creating collage: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
