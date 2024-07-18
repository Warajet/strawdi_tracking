import os
import cv2

from cycler import cycler as cy
from collections import defaultdict

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from tracker.data_obj_detect import sort_files

import numpy as np

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

def apply_mask(image, mask):
    """
    Applies a binary mask to an image, overlaying the masked area with a colored overlay.

    Parameters
    ----------
    image: numpy.ndarray with shape (H, W, 3)
      - The input image to which the mask will be applied. 
      - H is the height, 
      - W is the width, 
      - the color channels (RGB)

    mask: (numpy.ndarray) with the shape (H, W)
      - A binary mask where the non-zero values indicate the regions to be masked.

    Returns
    -------
    numpy.ndarray: 
      - Masked Image
    """
    # Ensure the mask is 3-channel to match the image dimensions
    mask_3channel = np.stack([mask]*3, axis=-1)

    # Alternatively, to visualize the mask with a color overlay
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = mask * 255  # Red channel, adjust color as needed
    overlay_image = np.where(mask_3channel, colored_mask, image)

    return overlay_image


def visualize_segmentation_mask(img, seg_mask, separate_bb = False):
    """
    Visualizes segmentation masks on an image

    - This function can either overlay all masks on the original image 
    or display each mask separately.

    Parameters
    ----------
    img: numpy.ndarray with shape (H, W, 3)
      - The input image on which the segmentation masks will be visualized.
      - H is the height.
      - W is the width.
      - The color channels (RGB).

    seg_mask: numpy.ndarray with shape (H, W, N)
      - The segmentation mask array.
      - H is the height.
      - W is the width.
      - N is the number of segmentation masks.

    separate_bb: bool, optional
      - If True, displays each segmentation mask separately in a grid layout.
      - If False, overlays all segmentation masks on the original image.
      - Default is False.

    Returns
    -------
    fig: matplotlib.figure.Figure
      - The matplotlib figure object containing the visualization.

    ax: matplotlib.axes._subplots.AxesSubplot or numpy.ndarray of them
      - The matplotlib axes object(s) containing the visualization.
    """
    num_objs = seg_mask.shape[2]
    if not separate_bb:
      fig, ax = plt.subplots(figsize = (10, 10))
      ax.imshow(img)
    else:
      fig, ax = plt.subplots(figsize = (20, 15))
      rows, columns = int(num_objs / 3) + 1, 3
    ax.set_axis_off()
    for i in range(num_objs):
      if separate_bb:
        ax = fig.add_subplot(rows, columns, i + 1)
        masked_img = apply_mask(img, seg_mask[:, :, i])
        ax.imshow(masked_img)
        ax.axis('off')
        ax.set_title(f'Segmentation Mask {i}')
        
    return fig, ax


def visualize_bounding_boxes(img, bounding_boxes, probs, separate_bb = False):
    """
    Visualizes bounding boxes on an image

    - This function can either overlay all bouding boxes on the original image 
    or display each bounding box separately.

    Parameters
    ----------
    img: numpy.ndarray with shape (H, W, 3)
      - The input image on which the bounding boxes will be visualized.
      - H is the height.
      - W is the width.
      - The color channels (RGB).

    bounding_boxes: torch.tensor([N, 4])
      - N bounding boxes on an image
      - Each bounding box has 4 entries (x_min, y_min, x_max, y_max)

    separate_bb: bool, optional
      - If True, displays each bounding box separately in a grid layout.
      - If False, overlays all bounding boxes on the original image.
      - Default is False.

    Returns
    -------
    fig: matplotlib.figure.Figure
      - The matplotlib figure object containing the visualization.

    ax: matplotlib.axes._subplots.AxesSubplot or numpy.ndarray of them
      - The matplotlib axes object(s) containing the visualization.
    """
    num_objs = len(bounding_boxes)

    if not separate_bb:
      fig, ax = plt.subplots(figsize = (10, 10))
      ax.imshow(img)
    else:
      fig, ax = plt.subplots(figsize = (20, 15))
      rows, columns = int(num_objs / 3) + 1, 3
    ax.set_axis_off()

    for i in range(num_objs):
      if separate_bb:
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Bounding Box {i + 1}')
      
      # Get the current bounding box
      x_min, y_min = bounding_boxes[i, 0], bounding_boxes[i, 1]
      width, height  = bounding_boxes[i,2] - bounding_boxes[i, 0], bounding_boxes[i,3] - bounding_boxes[i,1]

      # Create a Rectangle patch
      rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
      # Add the patch to the Axes
      ax.add_patch(rect)
        
    return fig, ax



def draw_frame(i, img, tracks, styles):
    """
    Draws a frame with bounding boxes and annotations for tracked objects.

    Parameters
    ----------
    i: int
      - The index of the current frame.
    
    img: numpy.ndarray with shape (H, W, 3)
      - The image on which the bounding boxes and annotations will be drawn.
      - H is the height.
      - W is the width.
      - The color channels (RGB).

    tracks: dict
      - A dictionary containing the track dictionaries in the form tracks[track_id][frame] = bounding_box.
      - Each bounding box is expected to be a list or tuple with four elements [x_min, y_min, x_max, y_max].

    styles: collections.defaultdict
      - A defaultdict containing the styling information for each track. 
      - Expected to provide style properties such as edge color (ec) for each track ID.

    Returns
    -------
    fig: matplotlib.figure.Figure
      - The matplotlib figure object containing the drawn frame.

    ax: matplotlib.axes._subplots.AxesSubplot
      - The matplotlib axes object containing the drawn frame.
    """

    width, height, _ = img.shape
    dpi = 96
    fig, ax = plt.subplots(1, dpi=dpi)
    fig.set_size_inches(width / dpi, height / dpi)
    ax.set_axis_off()
    ax.imshow(img)

    for j, t in tracks.items():
        if i in t.keys():
            t_i = t[i]
            ax.add_patch(
                plt.Rectangle(
                    (t_i[0], t_i[1]),
                    t_i[2] - t_i[0],
                    t_i[3] - t_i[1],
                    fill=False,
                    linewidth=1.0, **styles[j]
                ))
            ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                        color=styles[j]['ec'], weight='bold', fontsize=6, ha='center', va='center')
    return fig, ax


def plot_sequence(tracks, db, first_n_frames=None):
    """
    Plots a sequence of images with bounding boxes and annotations for tracked objects.

    Parameters
    ----------
    tracks: dict
      - The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb.
      - Each bounding box (bb) is expected to be a list or tuple with four elements [x_min, y_min, x_max, y_max].
    
    db: torch.utils.data.Dataset
      - The dataset with the images belonging to the tracks (e.g., MOT_Sequence object).
      - Each item in the dataset should be a tensor representing an image with shape (C, H, W) where C is the number of color channels, H is the height, and W is the width.

    first_n_frames: int, optional
      - The number of frames to plot from the sequence.
      - If None, plots the entire sequence.
      - Default is None.

    Returns
    -------
    None
    
    """
    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        img = v.mul(255).permute(1, 2, 0).byte().numpy()
        fig, ax = draw_frame(i, img, tracks, styles)
        plt.show()

        if first_n_frames is not None and first_n_frames - 1 == i:
            break


def generate_tracking_imseq(tracks, db, output_dir, fps=30):
    """
    Generates and saves a sequence of images with bounding boxes and annotations for tracked objects.

    Parameters
    ----------
    tracks: dict
      - The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb.
      - Each bounding box (bb) is expected to be a list or tuple with four elements [x_min, y_min, x_max, y_max].
    
    db: torch.utils.data.Dataset
      - The dataset with the images belonging to the tracks (e.g., MOT_Sequence object).
      - Each item in the dataset should be a tensor representing an image with shape (C, H, W) where C is the number of color channels, H is the height, and W is the width.

    output_dir: str
      - The directory where the output images will be saved.

    fps: int, optional
      - The frames per second for the output video.
      - Default is 30.

    Returns
    -------
    None

    """

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))
    
    # Initialize variables for video writing
    frame_list = []
    img_shape = None
    
    for i, v in enumerate(db):
        img = v.mul(255).permute(1, 2, 0).byte().numpy()
        width, height, _ = img.shape
        if img_shape is None:
            img_shape = (width, height)
        
        fig, ax = draw_frame(i, img, tracks, styles)
        
        image_path = os.path.join(output_dir, f"{i}.png")
        fig.savefig(image_path)
        plt.close(fig)  # Close the figure to avoid memory issues
    
    print(f"Video sequence images has been saved at: {output_dir}")



def imseq_to_video(image_folder, output_video_path, fps = 30):
    """
    Converts a sequence of images from a specified folder into a video file.

    Parameters
    ----------
    image_folder: str
      - The directory containing the sequence of images to be converted into a video.
    
    output_video_path: str
      - The file path where the output video will be saved.
    
    fps: int, optional
      - The frames per second for the output video.
      - Default is 30.

    Returns
    -------
    None

    """
     # Get list of image files in the directory
    image_files = sort_files([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    print(f"num output image frames: {len(image_files)}")
    
    if not image_files:
        print("No image files found in the specified folder.")
        return
    
    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    frame_size = (width, height)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)
    
    out.release()
    print(f"The output video has been saved at: {output_video_path}")

  
# Function to create a directory if it doesn't exist
def video_to_imseq(video_path, output_path):
    """
    Extracts frames from a video file and saves them as individual image files in a specified directory.

    Parameters
    ----------
    video_path: str
      - The path to the input video file.
    
    output_path: str
      - The directory where the extracted image frames will be saved.
      - The function will create the directory if it doesn't exist.

    Returns
    -------
    None
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        
        # If the frame was not read successfully, break the loop
        if not ret:
            break
        
        # Save the frame as an image file
        frame_filename = os.path.join(output_path, f'{frame_count}.png')
        
        cv2.imwrite(frame_filename, frame)
        frame = cv2.imread(frame_filename)
        target_width, target_height = 756, 1008
        frame = cv2.resize(frame, (target_height, target_width), interpolation= cv2.INTER_LINEAR)
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    video.release()

    print(f"Extracted {frame_count} frames from {video_path} and saved to {output_path}")

