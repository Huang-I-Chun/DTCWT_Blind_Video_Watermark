import cv2
import os
import random
import numpy as np
import dtcwt
import multiprocessing
import heapq
import json
import argparse

# global parameter
video_path = "video/life_300_wm.mp4"
output_path = "result/life_300_wm.json"
key = 9999
code_length = 500
bit_to_pixel = 2
step = 5
highpass_str = 5
lowpass_str = 4
threads = 16
wm_level = 4
random_placement_key = 9260

# python extract.py -i video/life_300_wm.mp4 -o result/life_300_wm.json -k 66666 -cl 60 -t 16 -wl 4
# python extract.py -i /mnt/ssd1/H264_dirty_detect/experiment_diff_crf_NCC_key/video/life_300_wm_crf29.mp4 -o result/life_300_wm_crf29.json -k 66666 -cl 60 -t 16
# python extract.py -i /mnt/ssd1/H264_dirty_detect/experiment_diff_crf_NCC_key/video/speed_bag_300_wm_crf33.mp4 -o result/speed_bag_300_wm_crf33.json -k 66666 -cl 60 -t 16
# python extract.py -i /mnt/ssd1/H264_dirty_detect/experiment_diff_crf_NCC_key/video/life_300_wm_crf1.mp4 -o result/life_300_wm_crf1.json -k 66666 -cl 60 -t 16


# python extract.py -i video/speed_bag_300_wm_wl4.mp4 -o result/speed_bag_300_wm_wl4.json -k 66666 -cl 60 -t 16 -wl 4
# python extract.py -i video/speed_bag_300_wm_wl4_crf35.mp4 -o result/speed_bag_300_wm_wl4_crf35.json -k 66666 -cl 60 -t 16 -wl 4


# python extract.py -i video/life_300_wm_sk-1.mp4 -o result/life_300_wm_sk-1.json -k 66666 -cl 60 -t 16 -wl 4 -sk -1
# python extract.py -i video/life_300_wm_rpk1984.mp4 -o result/life_300_wm_rpk1984.json -k 66666 -cl 60 -t 16 -wl 4 -rpk 1984
# python extract.py -i video/life_300_wm_rpk1984_lowu.mp4 -o result/life_300_wm_rpk1984_lowu.json -k 66666 -cl 60 -t 16 -wl 4 -rpk 1984
# python extract.py -i video/life_300_wm_rpk1984_lowu_diff.mp4 -o result/life_300_wm_rpk1984_lowu_diff.json -k 66666 -cl 60 -t 16 -wl 4 -rpk 1984


parser = argparse.ArgumentParser(description="Blind Video Watermarking in DTCWT Domain")
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
    "-bp",
    dest="bit_to_pixel",
    type=int,
    default=2,
    help="Set 1 bit equal to how many pixels",
)

parser.add_argument(
    "-rpk",
    dest="random_placement_key",
    type=int,
    default=9260,
    help="Set random_placement_key",
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


wm_coeffs = 0


# utility functions ----------------------------------------------------------------------
def rebin(a, shape):
    if a.shape[0] % 2 == 1:
        a = np.vstack((a, np.zeros((1, a.shape[1]))))
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


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
            col = 0
            row += n

    # cv2.imwrite("wm.png", image)
    # Save the generated image using OpenCV
    return image


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


def decode_frame(wmed_img):
    wmed_img = cv2.cvtColor(wmed_img.astype(np.float32), cv2.COLOR_BGR2YUV)

    wmed_transform = dtcwt.Transform2d()
    wmed_coeffs = wmed_transform.forward(wmed_img[:, :, 1], nlevels=3)

    y_transform = dtcwt.Transform2d()
    y_coeffs = y_transform.forward(wmed_img[:, :, 0], nlevels=3)

    v_transform = dtcwt.Transform2d()
    v_coeffs = v_transform.forward(wmed_img[:, :, 2], nlevels=3)

    masks3 = [0 for i in range(6)]
    inv_masks3 = [0 for i in range(6)]
    shape3 = y_coeffs.highpasses[2][:, :, 0].shape
    for i in range(6):
        masks3[i] = cv2.filter2D(
            np.abs(y_coeffs.highpasses[1][:, :, i]),
            -1,
            np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]]),
        )
        masks3[i] = np.ceil(rebin(masks3[i], shape3) * (1.0 / step))
        masks3[i][masks3[i] == 0] = 0.01
        masks3[i] *= 1.0 / max(12.0, np.amax(masks3[i]))
        inv_masks3[i] = 1.0 / masks3[i]

    shape = wmed_coeffs.highpasses[2][:, :, i].shape

    my_highpass = []
    h, w = (((shape[0] + 1) // 2 + 1) // 2 + 1) // 2, (
        ((shape[1] + 1) // 2 + 1) // 2 + 1
    ) // 2

    for i in range(wm_level):
        my_highpass.append(np.zeros((h, w, 6), dtype="complex_"))
        h = (h + 1) // 2
        w = (w + 1) // 2
        # print(my_highpass[i].shape)

    lowpass = np.zeros(
        (my_highpass[-1].shape[0] * 2, my_highpass[-1].shape[1] * 2), dtype="complex_"
    )

    small_matrixs = []
    small_matrixs.append(lowpass)
    small_matrixs.append(lowpass)
    small_matrixs.append(lowpass)
    small_matrixs.append(lowpass)
    for lv in range(wm_level):
        # 4 times redudant in each level
        small_matrixs.append(my_highpass[lv][:, :, 0])
        small_matrixs.append(my_highpass[lv][:, :, 0])
        small_matrixs.append(my_highpass[lv][:, :, 0])
        small_matrixs.append(my_highpass[lv][:, :, 0])

    random_positions = get_random_pos(
        inv_masks3[0].shape, small_matrixs, random_placement_key
    )

    for i in range(6):
        coeff = (wmed_coeffs.highpasses[2][:, :, i]) * inv_masks3[i] * 1 / highpass_str

        for level in range(wm_level):
            lv = level + 1  # to avoid first 4 position for lowpass
            # print(random_positions[lv * 4])
            # print(random_positions[lv * 4 + 1])
            # print(random_positions[lv * 4 + 2])
            # print(random_positions[lv * 4 + 3])
            # print(my_highpass[level][:, :, i].shape)

            my_highpass[level][:, :, i] = (
                coeff[
                    random_positions[lv * 4][0] : random_positions[lv * 4][1],
                    random_positions[lv * 4][2] : random_positions[lv * 4][3],
                ]
                + coeff[
                    random_positions[lv * 4 + 1][0] : random_positions[lv * 4 + 1][1],
                    random_positions[lv * 4 + 1][2] : random_positions[lv * 4 + 1][3],
                ]
                + coeff[
                    random_positions[lv * 4 + 2][0] : random_positions[lv * 4 + 2][1],
                    random_positions[lv * 4 + 2][2] : random_positions[lv * 4 + 2][3],
                ]
                + coeff[
                    random_positions[lv * 4 + 3][0] : random_positions[lv * 4 + 3][1],
                    random_positions[lv * 4 + 3][2] : random_positions[lv * 4 + 3][3],
                ]
            )

    highpasses = tuple(my_highpass)

    ### extract watermark lowpass into V channel's highpass[2] (highpass as mask)
    lowpass_masks = [0 for i in range(6)]
    inv_lowpass_masks = [0 for i in range(6)]

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
        lowpass_masks[i][lowpass_masks[i] == 0] = 0.01
        inv_lowpass_masks[i] = 1.0 / lowpass_masks[i]

    for i in range(4):
        coeff = (v_coeffs.highpasses[2][:, :, i]) * inv_masks3[i] * 1 / lowpass_str
        lowpass[:, :] += coeff[
            random_positions[0][0] : random_positions[0][1],
            random_positions[0][2] : random_positions[0][3],
        ]

    lowpass = lowpass.real.astype(np.float32)

    t = dtcwt.Transform2d()
    wm = t.inverse(dtcwt.Pyramid(lowpass, highpasses))

    recovered_string = recover_string_from_image(bit_to_pixel, code_length, wm)
    return recovered_string, wm


# Code Start here ------------------------------------------------------------------------
# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


# calculate the wm size that this cover image can afford
wm_w = (((((frame_width + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2
wm_h = (((((frame_height + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2
if wm_w % 2 == 1:
    wm_w += 1
if wm_h % 2 == 1:
    wm_h += 1

if (wm_w / bit_to_pixel) * (wm_h / bit_to_pixel) < code_length:
    print(
        f" {(wm_w / bit_to_pixel) * (wm_h / bit_to_pixel)} < {code_length}, code length is too long"
    )
    quit()

# generate wm
random_binary_string = generate_random_binary_string(code_length, key)
original_wm = generate_image(random_binary_string, wm_w, wm_h, bit_to_pixel)

wm_transform = dtcwt.Transform2d()
# If nlevel is 1, then len(highpass) == 1
wm_coeffs = wm_transform.forward(original_wm, nlevels=wm_level)

keys = []
frame_idx = 0

overlap_wm = np.zeros((wm_h, wm_w))

while True:
    input_args = []
    for i in range(threads):
        ret, frame = video_capture.read()
        if not ret:
            break
        else:
            input_args.append(frame)

    if len(input_args) == 0:
        break

    with multiprocessing.Pool(processes=threads) as pool:
        pool_return = pool.map(decode_frame, input_args)

    for key, wm in pool_return:
        keys.append(key)
        overlap_wm += wm
        # cv2.imwrite(f"wm/{frame_idx}.png", wm)
        frame_idx += 1


overlap_wm /= frame_idx + 1
overlap_key = recover_string_from_image(bit_to_pixel, code_length, overlap_wm)

# cv2.imwrite(f"wm/overlap.png", overlap_wm)

overlap_filename = os.path.basename(video_path)
overlap_filename = os.path.splitext(overlap_filename)[0]
# print(f"/mnt/ssd1/H264_dirty_detect/Experiment/wm/{overlap_filename}.png")
cv2.imwrite(
    f"/mnt/ssd1/H264_dirty_detect/Experiment/wm/{overlap_filename}.png",
    overlap_wm,
)

# output_dict = {"keys": keys, "ans": random_binary_string}
output_dict = {"keys": [overlap_key], "ans": random_binary_string}


with open(output_path, "w") as json_file:
    json.dump(output_dict, json_file)

# Release the video capture object
video_capture.release()
