import numpy as np
import cv2
import argparse
import os
from moviepy.editor import VideoFileClip
import functools
from train import train
import pickle
from sliding_window import WindowSlider
from scipy.stats import threshold
from scipy.ndimage.measurements import label
import time
from collections import deque
from featuresextractor import FeaturesExtractor

frame_id=0

def find_vehicles(image, window_slider, features_extractor, scaler, model, debug_text=False,
                  save_boxes_in_x_range=(0, 600)):
    start_time = time.time()
    luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    features, windows = features_extractor.extract_image_features_multiscale(luv)

    end_time = time.time()
    if debug_text:
        print('Time to extract features from %d windows: %0.3f'%(len(windows), end_time - start_time))

    start_time = end_time
    scaled_features = scaler.transform(features)

    predictions = model.predict_proba(scaled_features)

    end_time = time.time()
    if debug_text:
        print('Time to predict %d features: %0.3f'%(len(scaled_features), end_time - start_time))

    car_windows = []
    car_score = []
    min_prob = .85
    score_mul = 1 / ((1 - min_prob) / 4)
    for i, window in enumerate(windows):
        if predictions[i][1] > min_prob:
            car_windows.append(window)
            car_score.append((predictions[i][1] - min_prob) * score_mul + 1)

            if save_boxes_in_x_range is not None and \
                save_boxes_in_x_range[0] < window[0][0] < save_boxes_in_x_range[1]:
                global frame_id
                frame_id += 1
                cv2.imwrite('training_data/non-vehicles/negative-samples/%dx'%frame_id + 'x'.join(
                    str(e) for e in np.array(window).flatten()) + '.png',
                    cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0],:], (64, 64)))


    return car_windows, car_score


def add_heat(heatmap, windows, scores):
    for i, window in enumerate(windows):
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += scores[i]

    return heatmap


def calculate_bounding_boxes(thres_heatmap):
    nonzero_vals = thres_heatmap[thres_heatmap.nonzero()]
    if len(nonzero_vals) == 0:
        return []

    median = np.percentile(nonzero_vals, 10)
    thres_heatmap = threshold(thres_heatmap, median)
    labels = label(thres_heatmap)
    cars_bboxes = []

    hw_ratio = 2/3

    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        if hw_ratio < (box[1][0] - box[0][0]) / (box[1][1] - box[0][1]):
            cars_bboxes.append(box)

    return cars_bboxes


def draw_debug_windows(image, windows, thickness=1):
    debug_image = np.copy(image)
    for window in windows:
        cv2.rectangle(debug_image, tuple(window[0]), tuple(window[1]), color=(0, 0, 255), thickness=thickness)
    return debug_image


def normalize_heat_to_image(heatmap):
    return (heatmap * 255.0 / heatmap.max()) / 255.0


def process_image(image, window_slider, features_extractor, model, scaler,
                  debug_windows=True, debug_heatmap=True, debug_id=0):
    windows, scores = find_vehicles(image, window_slider, features_extractor, scaler, model, debug_text=True)
    if debug_windows:
        cv2.imshow('Debug windows %s'%debug_id, draw_debug_windows(image, windows))

    heatmap = add_heat(np.zeros(image.shape[0:2], dtype=np.float64), windows, scores)
    print(heatmap.max(), heatmap.mean(), np.median(heatmap))
    thresholded = threshold(heatmap, 2)
    thresholded = threshold(thresholded, threshmax=6, newval=6)
    if debug_heatmap:
        scaled = normalize_heat_to_image(heatmap)
        cv2.imshow('Heatmap', np.dstack((scaled, scaled, scaled)))
        scaled = normalize_heat_to_image(thresholded)
        cv2.imshow('Heatmap thresholded', np.dstack((scaled, scaled, scaled)))

    car_bboxes = calculate_bounding_boxes(thresholded)
    for bbox in car_bboxes:
        cv2.rectangle(image, bbox[0], bbox[1], color=(255, 0, 0), thickness=2)

    return image


def process_test_images(path='./test_images', show_images=True):
    '''
    Runs the whole pipeline on test images found on a given path (be it an image or a directory).
    '''
    images = {}
    if os.path.isdir(path):
        for f in os.listdir(path):
            images[f] = cv2.imread(os.path.join(path, f))
    else:
        images[os.path.basename(path)] = cv2.imread(path)

    window_slider = WindowSlider()
    features_extractor = FeaturesExtractor()
    model, scaler = load_model()

    for name, img in images.items():
        if img is not None:
            print('Processing image: ', name)
            final = process_image(img, window_slider, features_extractor, model, scaler, debug_id=name)

            if show_images:
                cv2.imshow(name, final)
        else:
            print('Unable to process image: ', name)


    if show_images:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video_image(window_slider, features_extractor, model, scaler, heatmap_history, image):
    '''
    Processes an image from the video. First it converts the image to BGR from RGB, because that's what the pipeline
    uses.
    :return: The processed image.
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    windows, scores = find_vehicles(image, window_slider, features_extractor, scaler, model)

    raw_windows = cv2.resize(draw_debug_windows(image, windows, thickness=4), (0, 0), fx=.3, fy=.3)

    heatmap = np.int32(add_heat(np.zeros(image.shape[0:2], dtype=np.float64), windows, scores))
    heatmap_history.append(heatmap)
    if len(heatmap_history) > 15:
        heatmap_history.popleft()

    full_heatmap = np.zeros(image.shape[0:2], dtype=np.int32)
    for i, prev_heatmap in enumerate(heatmap_history):
        full_heatmap = np.add(full_heatmap, prev_heatmap * ((i + 1) / len(heatmap_history)))

    scaled_heatmap = full_heatmap * 255.0 / full_heatmap.max()
    heatmap_image = cv2.resize(np.dstack((scaled_heatmap, scaled_heatmap, scaled_heatmap)), (0, 0), fx=.3, fy=.3)

    thresholded = threshold(full_heatmap, min(len(heatmap_history) * 2, 10))
    thresholded = threshold(thresholded, threshmax=len(heatmap_history) * 6, newval=len(heatmap_history) * 6)

    scaled_heatmap = thresholded * 255.0 / thresholded.max()
    thresheat_image = cv2.resize(np.dstack((scaled_heatmap, scaled_heatmap, scaled_heatmap)), (0, 0), fx=.3, fy=.3)

    car_bboxes = calculate_bounding_boxes(thresholded)
    for bbox in car_bboxes:
        cv2.rectangle(image, bbox[0], bbox[1], color=(255, 0, 0), thickness=2)

    offs = 20
    third = image.shape[1] // 3
    image[offs:raw_windows.shape[0] + offs, offs:raw_windows.shape[1] + offs, :] = raw_windows
    image[offs:heatmap_image.shape[0] + offs, third + offs:heatmap_image.shape[1] + third + offs, :] = \
        heatmap_image
    image[offs:heatmap_image.shape[0] + offs, 2 * third + offs:heatmap_image.shape[1] + 2 * third + offs, :] = \
        thresheat_image

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def process_video(input_path, output_path, start, end):
    '''
    Runs the pipeline on every frame of a video and then combines it back into a video.
    '''

    window_slider = WindowSlider()
    features_extractor = FeaturesExtractor()
    model, scaler = load_model()
    heatmap_history = deque()

    clip1 = VideoFileClip(input_path).subclip(start, end)
    white_clip = clip1.fl_image(functools.partial(
        process_video_image, window_slider, features_extractor, model, scaler, heatmap_history))
    white_clip.write_videofile(output_path, audio=False)


def save_debug_video_frames(input_path, output_path, start, end, fps):
    '''
    Saves the video frame for debugging purposes.
    :param input_path: the input video path
    :param output_path: the output directory to save the frames to
    :param start: the start position in the video, can be float
    :param end: the end position in the video, can be float
    :param fps: the fps to use, i.e. save 10 images per second, etc.
    '''
    clip1 = VideoFileClip(input_path)
    if end is None: end = clip1.end
    for frame in range(int(start * fps), int(end * fps)):
        clip1.save_frame(os.path.join(output_path, 'frame%d.png'%frame), frame / fps)


def train_classifier(outpath='model.pkl'):
    model, scaler = train()
    with open(outpath, 'wb') as outfile:
        pickle.dump((model, scaler), outfile)


def load_model(path='model.pkl'):
    with open(path, 'rb') as infile:
        return pickle.load(infile)


def main(args):

    if bool(args.train):
        train_classifier()

    if bool(args.images):
        process_test_images(path=args.images, show_images=True)

    if bool(args.video):
        if args.debug:
            save_debug_video_frames(
                args.video[0], args.video[1], args.start, args.end, args.fps)
        else:
            process_video(args.video[0], args.video[1], args.start, args.end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the car/not car classifier.')
    parser.add_argument('--images', type=str, help='Runs the pipeline on the images found at the given path')
    parser.add_argument('--video', type=str, nargs=2,
                        help='Runs the pipeline on the given video and saves it to the output path. If debug is'
                                      ' set, instead of running the pipeline, it saves the frames of the video.')
    parser.add_argument('--start', type=float, default=0, help='Start position in the video.')
    parser.add_argument('--end', type=float, help='End position in the video.')
    parser.add_argument('--fps', type=int, default=10, help='The fps to save frames from the video in.')
    parser.add_argument('--debug', action='store_true', help='Whether to debug the current process or not.')
    args = parser.parse_args()

    main(args)