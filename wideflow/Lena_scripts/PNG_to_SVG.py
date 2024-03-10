import cv2
import numpy as np
import svgwrite

def convert_png_to_svg(input_path, output_path):
    # Read the PNG image using OpenCV
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to create a binary mask
    _, mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an SVG drawing
    dwg = svgwrite.Drawing(output_path, profile='tiny')

    # Iterate through contours and convert to SVG paths
    for contour in contours:
        # Convert each point in the contour to a tuple
        points = [tuple(point[0]) for point in contour]
        points_str = ' '.join([f'{x[0]},{x[1]}' for x in points])
        dwg.add(dwg.path(d=f'M {points_str} Z', fill='white'))

    # Save the SVG file
    dwg.save()

# Example usage
input_png_path = '/data/Lena/WideFlow_prj/Figs_for_paper/life_of_a_frame.png'
output_svg_path = '/data/Lena/WideFlow_prj/Figs_for_paper/life_of_a_frame_try1.svg'
convert_png_to_svg(input_png_path, output_svg_path)
