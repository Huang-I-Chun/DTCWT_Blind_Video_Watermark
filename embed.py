import cv2
import os
import random
import numpy as np
import dtcwt
import multiprocessing
import heapq
import argparse

# global parameter
video_path = "/mnt/ssd1/H264_dirty_detect/video/life_300.mp4"
output_path = "video/life_300_wm.mp4"
key = 9999
code_length = 500
bit_to_pixel = 2
step = 5
highpass_str = 5
lowpass_str = 4
threads = 16
wm_level = 4
random_placement_key = 1984

# python embed.py -i /mnt/ssd1/H264_dirty_detect/video/life_300.mp4 -o video/life_300_wm.mp4 -k 66666 -cl 60 -t 16 -wl 4
# python embed.py -i /mnt/ssd1/H264_dirty_detect/video/park_joy_300.mp4 -o video/park_joy_300_wm.mp4 -k 66666 -cl 60 -t 16 -wl 4
# python embed.py -i /mnt/ssd1/H264_dirty_detect/video/pedestrian_area_300.mp4 -o video/pedestrian_area_300_wm.mp4 -k 66666 -cl 60 -t 16 -wl 4
# python embed.py -i /mnt/ssd1/H264_dirty_detect/video/speed_bag_300.mp4 -o video/speed_bag_300_wm.mp4 -k 66666 -cl 60 -t 16 -wl 4

# python embed.py -i /mnt/ssd1/H264_dirty_detect/video/speed_bag_300.mp4 -o video/speed_bag_300_wm_wl4.mp4 -k 66666 -cl 60 -t 16 -wl 4

# python embed.py -i /mnt/ssd1/H264_dirty_detect/video/life_300.mp4 -o video/life_300_wm_rpk1984.mp4 -k 66666 -cl 60 -t 16 -wl 4 -rpk 1984


parser = argparse.ArgumentParser(
    description="Blind Video Watermarking in DT CWT Domain"
)
parser.add_argument(
    "-i", dest="video_path", type=str, help="Set input video", required=True
)
parser.add_argument(
    "-o", dest="output_path", type=str, help="Set output path", required=True
)
parser.add_argument("-k", dest="key", type=int, help="Set key", required=True)

parser.add_argument(
    "-cl", dest="code_length", type=int, help="Set code length", required=True
)

parser.add_argument(
    "-rpk",
    dest="random_placement_key",
    type=int,
    default=1984,
    help="Set random_placement_key",
)

parser.add_argument(
    "-bp",
    dest="bit_to_pixel",
    type=int,
    default=2,
    help="Set 1 bit equal to how many pixels",
)

parser.add_argument(
    "-step",
    dest="step",
    type=int,
    default=5,
    help="Set step",
)

parser.add_argument(
    "-wl",
    dest="wm_level",
    type=int,
    default=4,
    help="Set watermark dtcwt level",
)

parser.add_argument(
    "-hstr",
    dest="highpass_str",
    type=int,
    default=5,
    help="Set highpass strength",
)

parser.add_argument(
    "-lstr",
    dest="lowpass_str",
    type=int,
    default=4,
    help="Set lowpass strength",
)


parser.add_argument("-t", dest="threads", type=int, default=4, help="Set thread nums")

args = parser.parse_args()

video_path = args.video_path
output_path = args.output_path
key = args.key
code_length = args.code_length
bit_to_pixel = args.bit_to_pixel
step = args.step
highpass_str = args.highpass_str
lowpass_str = args.lowpass_str
threads = args.threads
wm_level = args.wm_level
random_placement_key = args.random_placement_key


# utility functions ----------------------------------------------------------------------
def rebin(a, shape):
    if a.shape[0] % 2 == 1:
        a = np.vstack((a, np.zeros((1, a.shape[1]))))
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    # print(sh)
    return a.reshape(sh).mean(-1).mean(1)


def recover_string_from_image(
    n,
    original_length,
    gray_image,
    consider_black_threshold=3,
    black_appear_threshold=-1,
    avg_color_threshold=40,
):
    # Initialize variables for position tracking
    row, col = 0, 0

    # Use a list to store the detected strings in multiple passes
    detected_strings = []

    while row + n <= gray_image.shape[0]:  # Ensure we don't go beyond image boundaries
        # Initialize an empty string for the current pass
        current_string = ""

        while len(current_string) < original_length and row + n <= gray_image.shape[0]:
            # Extract the n x n block
            block = gray_image[row : row + n, col : col + n]

            avg_color = np.mean(block)

            # current_string += '0' if avg_color < 127.5 else '1'

            black_appear = 0
            look_all_black = True
            for i in range(n):
                for j in range(n):
                    if block[i, j] < consider_black_threshold:
                        black_appear += 1
                    # if block[i, j] > 50:
                    #     look_all_black = False

            if (
                black_appear > black_appear_threshold
                and avg_color < avg_color_threshold
            ):
                current_string += "0"
            else:
                current_string += "1"

            # Move to the next position
            col += n
            if col + n > gray_image.shape[1]:  # If we are at the end of a row
                col = 0
                row += n

        while len(current_string) < original_length:
            current_string += "5"

        detected_strings.append(current_string)

    # Calculate the final binary string using voting for each position
    final_string = ""
    for i in range(original_length):
        ones = sum([1 for s in detected_strings if s[i] == "1"])
        zeros = sum([1 for s in detected_strings if s[i] == "0"])

        # If there are more ones than zeros, append '1', else append '0'.
        # If there's a tie, append a random choice
        if ones > zeros:
            final_string += "1"
        elif zeros > ones:
            final_string += "0"
        else:
            final_string += np.random.choice(["0", "1"])

    return final_string


# generate wm through key
def generate_random_binary_string(length, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    binary_string = "".join(random.choice("10") for _ in range(length))
    return binary_string


def generate_image(random_binary_string, width, height, n):
    # Create a black image
    image = np.zeros((height, width), dtype=np.uint8)

    # Initialize variables for tracking the position in the binary string
    index = 0
    row = 0
    col = 0

    # Loop until the image is completely filled
    while row + n <= height:
        # Check if we have scanned all bits; if so, reset the index
        if index >= len(random_binary_string):
            index = 0

        # Determine the color based on the current bit (0 or 1)
        color = 0 if random_binary_string[index] == "0" else 255

        # Move to the next position
        if col + n <= width:
            # Place the nxn block
            for i in range(n):
                for j in range(n):
                    x = col + j
                    y = row + i
                    image[y, x] = color
            col += n
            index += 1
        else:
            # print(index)
            col = 0
            row += n

    # cv2.imwrite("wm.png", image)
    # Save the generated image using OpenCV
    return image


def get_random_pos(big_matrix_shape, small_matrices, seed=None):
    # Set seed for reproducibility
    random.seed(seed)
    # Keep track of occupied positions
    occupied = np.zeros(big_matrix_shape, dtype=bool)

    random_position_list = []

    for small_matrix in small_matrices:
        small_height, small_width = small_matrix.shape

        # Get all possible positions for this small matrix
        possible_positions = [
            (i, j)
            for i in range(big_matrix_shape[0] - small_height + 1)
            for j in range(big_matrix_shape[1] - small_width + 1)
            if not np.any(occupied[i : i + small_height, j : j + small_width])
        ]

        # If no possible position, return an error
        if not possible_positions:
            raise ValueError(
                "No space left for the matrix of shape {}".format(small_matrix.shape)
            )

        # Randomly select one position
        chosen_position = random.choice(possible_positions)

        # Place the small matrix at the chosen position
        # big_matrix[chosen_position[0]:chosen_position[0]+small_height,
        #            chosen_position[1]:chosen_position[1]+small_width] = small_matrix

        random_position_list.append(
            (
                chosen_position[0],
                chosen_position[0] + small_height,
                chosen_position[1],
                chosen_position[1] + small_width,
            )
        )

        # Mark the position as occupied
        occupied[
            chosen_position[0] : chosen_position[0] + small_height,
            chosen_position[1] : chosen_position[1] + small_width,
        ] = True

    return random_position_list


def place_random_pos(big_matrix, small_matrices, random_pos):
    for idx, small_matrix in enumerate(small_matrices):
        small_height, small_width = small_matrix.shape

        # Place the small matrix at the chosen position
        big_matrix[
            random_pos[idx][0] : random_pos[idx][1],
            random_pos[idx][2] : random_pos[idx][3],
        ] = small_matrix

    return big_matrix


def embed_frame(frame, wm_coeffs):
    img = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2YUV)

    # # DTCWT for V channel
    v_transform = dtcwt.Transform2d()
    v_coeffs = v_transform.forward(img[:, :, 2], nlevels=3)

    # DTCWT for U channel
    img_transform = dtcwt.Transform2d()
    img_coeffs = img_transform.forward(img[:, :, 1], nlevels=3)

    # DTCWT for Y Channel
    y_transform = dtcwt.Transform2d()
    y_coeffs = y_transform.forward(img[:, :, 0], nlevels=3)

    # # Masks for the level 3 subbands
    masks3 = [0 for i in range(6)]
    shape3 = y_coeffs.highpasses[2][:, :, 0].shape

    # embed watermark highpass into U channel's highpass
    for i in range(6):
        masks3[i] = cv2.filter2D(
            np.abs(y_coeffs.highpasses[1][:, :, i]),
            -1,
            np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]]),
        )
        masks3[i] = np.ceil(rebin(masks3[i], shape3) * (1 / step))
        # print(masks3[i])
        masks3[i] *= 1.0 / max(12.0, np.amax(masks3[i]))
        # print(masks3[i].shape) (135, 240)

    small_matrixs = []
    # 4 times redudant in each level
    small_matrixs.append(wm_coeffs.lowpass)
    small_matrixs.append(wm_coeffs.lowpass)
    small_matrixs.append(wm_coeffs.lowpass)
    small_matrixs.append(wm_coeffs.lowpass)

    for lv in range(wm_level):
        # 4 times redudant in each level
        small_matrixs.append(wm_coeffs.highpasses[lv][:, :, 0])
        small_matrixs.append(wm_coeffs.highpasses[lv][:, :, 0])
        small_matrixs.append(wm_coeffs.highpasses[lv][:, :, 0])
        small_matrixs.append(wm_coeffs.highpasses[lv][:, :, 0])

    # Note that first 4 position is for lowpass
    random_positions = get_random_pos(
        masks3[0].shape, small_matrixs, random_placement_key
    )

    for i in range(6):
        coeffs = np.zeros(masks3[i].shape, dtype="complex_")

        small_matrixs = []
        for lv in range(wm_level):
            small_matrixs.append(wm_coeffs.highpasses[lv][:, :, i])
            small_matrixs.append(wm_coeffs.highpasses[lv][:, :, i])
            small_matrixs.append(wm_coeffs.highpasses[lv][:, :, i])
            small_matrixs.append(wm_coeffs.highpasses[lv][:, :, i])

        coeffs = place_random_pos(coeffs, small_matrixs, random_positions[4:])
        img_coeffs.highpasses[2][:, :, i] += highpass_str * (masks3[i] * coeffs)

    ### embed watermark lowpass into V channel's highpass[2] (highpass as mask)
    lowpass_masks = [0 for i in range(6)]

    for i in range(4):
        lowpass_masks[i] = cv2.filter2D(
            np.abs(
                np.abs(y_coeffs.highpasses[1][:, :, i]),
            ),
            -1,
            np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]]),
        )
        lowpass_masks[i] = np.ceil(
            rebin(lowpass_masks[i], v_coeffs.highpasses[2].shape) * (1 / step)
        )
        lowpass_masks[i] *= 1.0 / max(12.0, np.amax(lowpass_masks[i]))

    lv1_h = wm_coeffs.highpasses[0][:, :, i].shape[0]
    lv1_w = wm_coeffs.highpasses[0][:, :, i].shape[1]
    for i in range(4):
        coeff = wm_coeffs.lowpass
        # print(coeff.shape) (68, 120)
        h, w = coeff.shape
        coeffs = np.zeros(lowpass_masks[i].shape)
        coeffs[
            random_positions[0][0] : random_positions[0][1],
            random_positions[0][2] : random_positions[0][3],
        ] = coeff
        v_coeffs.highpasses[2][:, :, i] += lowpass_str * (lowpass_masks[i] * coeffs)

    img[:, :, 1] = img_transform.inverse(img_coeffs)
    img[:, :, 2] = img_transform.inverse(v_coeffs)
    ### embed watermark lowpass into V channel's highpass[2] end

    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    img = np.clip(img, a_min=0, a_max=255)
    img = np.around(img).astype(np.uint8)

    return img


# Code Start here ------------------------------------------------------------------------
# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4
video_writer = cv2.VideoWriter(
    output_path, fourcc, fps, (frame_width, frame_height)
)  # note: width height position seem to be oppsite of shape


# calculate the wm size that this cover image can afford
wm_w = (((((frame_width + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2
wm_h = (((((frame_height + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2
if wm_w % 2 == 1:
    wm_w += 1
if wm_h % 2 == 1:
    wm_h += 1

if (wm_w // bit_to_pixel) * (wm_h // bit_to_pixel) < code_length:
    print("code length is too long")
    quit()

# generate wm
random_binary_string = generate_random_binary_string(code_length, key)
wm = generate_image(random_binary_string, wm_w, wm_h, bit_to_pixel)
wm_transform = dtcwt.Transform2d()
# If nlevel is 1, then len(highpass) == 1
wm_coeffs = wm_transform.forward(wm, nlevels=wm_level)
# print(random_binary_string)

frame_idx = 0

while True:
    input_args = []
    for i in range(threads):
        ret, frame = video_capture.read()
        if not ret:
            break
        else:
            input_args.append((frame, wm_coeffs))

    if len(input_args) == 0:
        break

    with multiprocessing.Pool(processes=threads) as pool:
        output_frames = pool.starmap(embed_frame, input_args)

    for frame in output_frames:
        video_writer.write(frame)

# Release the video capture object
video_capture.release()
video_writer.release()
