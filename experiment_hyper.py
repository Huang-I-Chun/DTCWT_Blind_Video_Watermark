import subprocess
import multiprocessing
import argparse
import cv2
import numpy as np
import random
import json

result_folder_path = "/mnt/ssd1/H264_dirty_detect/result"


def recover_string_from_image(
    n,
    original_length,
    gray_image,
    consider_black_threshold=30,
    black_appear_threshold=0,
    avg_color_threshold=127.5,
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
            for i in range(n):
                for j in range(n):
                    if block[i, j] < consider_black_threshold:
                        black_appear += 1

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


def normalized_cross_correlation_zero_lag(str1, str2):
    # Convert binary strings to numpy arrays of integers
    arr1 = np.array([int(bit) for bit in str1])
    arr2 = np.array([int(bit) for bit in str2])

    # Calculate the means of the arrays
    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)

    # Calculate the numerator and denominator
    numerator = np.sum((arr1 - mean1) * (arr2 - mean2))
    denominator = np.sqrt(np.sum((arr1 - mean1) ** 2) * np.sum((arr2 - mean2) ** 2))

    if denominator == 0:
        return 0

    # Calculate the normalized correlation coefficient
    corr_coefficient = numerator / denominator

    return corr_coefficient


def generate_random_binary_string(length, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    binary_string = "".join(random.choice("10") for _ in range(length))
    return binary_string


datasets = ["life", "park_joy", "pedestrian_area", "speed_bag"]
frame_length = 300
keys = [2486, 1298, 5237, 6492, 9341, 7884, 5426, 8123, 6142, 4389]
code_length = 60
wm_level = 4
methods = ["wm"]  # "wm", "SVD"
crfs = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37]
consider_black_thresholds = [5, 10, 15, 20]
black_appear_thresholds = [-1, 0, 1]
avg_color_thresholds = [10, 30, 50, 70, 90, 110]


result_dict = dict()

for key in keys:
    random_binary_string = generate_random_binary_string(code_length, key)

    for dataset in datasets:
        for method in methods:
            for crf in crfs:
                for consider_black in consider_black_thresholds:
                    for black_appear in black_appear_thresholds:
                        for avg_color in avg_color_thresholds:
                            image_path = f"/mnt/ssd1/H264_dirty_detect/Experiment/wm/{dataset}_{frame_length}_{method}_k{key}_cl{code_length}_wl{4}_crf{crf}.png"
                            img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)

                            overlap_key = recover_string_from_image(
                                2,
                                code_length,
                                img,
                                consider_black,
                                black_appear,
                                avg_color,
                            )
                            ncc = normalized_cross_correlation_zero_lag(
                                overlap_key, random_binary_string
                            )
                            if (
                                not f"{consider_black}_{black_appear}_{avg_color}"
                                in result_dict.keys()
                            ):
                                result_dict[
                                    f"{consider_black}_{black_appear}_{avg_color}"
                                ] = [ncc]
                            else:
                                result_dict[
                                    f"{consider_black}_{black_appear}_{avg_color}"
                                ].append(ncc)


max_consider_black = 0
max_black_appear = 0
max_avg_color = 0
max_key = ""
max_ncc = 0
for consider_black in consider_black_thresholds:
    for black_appear in black_appear_thresholds:
        for avg_color in avg_color_thresholds:
            my_key = f"{consider_black}_{black_appear}_{avg_color}"
            avg_ncc = np.mean(result_dict[my_key])
            print(f"{my_key}: {avg_ncc}")
            print("-----------------------")
            if max_ncc < avg_ncc:
                max_consider_black = consider_black
                max_black_appear = black_appear
                max_avg_color = avg_color
                max_ncc = avg_ncc
                max_key = my_key


# Reproduce json using max combination
for key in keys:
    random_binary_string = generate_random_binary_string(code_length, key)

    for dataset in datasets:
        for method in methods:
            for crf in crfs:
                image_path = f"/mnt/ssd1/H264_dirty_detect/Experiment/wm/{dataset}_{frame_length}_{method}_k{key}_cl{code_length}_wl{4}_crf{crf}.png"
                img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)

                overlap_key = recover_string_from_image(
                    2,
                    code_length,
                    img,
                    max_consider_black,
                    max_black_appear,
                    max_avg_color,
                )

                # output_dict = {"keys": keys, "ans": random_binary_string}
                output_dict = {"keys": [overlap_key], "ans": random_binary_string}

                with open(
                    f"{result_folder_path}/{dataset}_{frame_length}_{method}_k{key}_cl{code_length}_wl{wm_level}_crf{crf}.json",
                    "w",
                ) as json_file:
                    json.dump(output_dict, json_file)

print(f"{max_key} {max_ncc}")
print(f"{result_dict[max_key]}")
