import numpy as np
import cv2


class WindowSlider:
    '''
    Windows slider to slide windows of given sizes with given overlapping percentages. This is not currently used,
    because hog subsampling requires a different approach.
    '''

    def __init__(self, window_sizes=(50, 100, 200, 300),
                 ranges_of_interest=(((300, 980), (400, 460)), ((0, 1280), (380, 520)),
                                     ((0, 1280), (350, 600)), ((0, 1280), (340, 660)))):
        self.__window_sizes = window_sizes
        self.__ranges_of_interest = ranges_of_interest

    def __slide_window(self, x_range, y_range, window_size, window_overlap_percentage):
        # Compute the span of the region to be searched
        xspan = x_range[1] - x_range[0]
        yspan = y_range[1] - y_range[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(np.ceil(window_size[0] * (1 - window_overlap_percentage[0])))
        ny_pix_per_step = np.int(np.ceil(window_size[1] * (1 - window_overlap_percentage[1])))
        # Compute the number of windows in x/y
        nx_buffer = np.floor(window_size[0] * (window_overlap_percentage[0]))
        ny_buffer = np.floor(window_size[1] * (window_overlap_percentage[1]))
        nx_windows = np.int(np.ceil((xspan - nx_buffer) / nx_pix_per_step))
        ny_windows = np.int(np.ceil((yspan - ny_buffer) / ny_pix_per_step))
        # Initialize a list to append window positions to
        window_list = []

        for ys in range(ny_windows):
            endy = min(y_range[0] + ys * ny_pix_per_step + window_size[1], y_range[1])
            starty = max(endy - window_size[1], 0)
            for xs in range(nx_windows):
                endx = min(x_range[0] + xs * nx_pix_per_step + window_size[0], x_range[1])
                startx = max(endx - window_size[0], 0)
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def slide_windows(self, debug_window_size_indices=()):
        windows_size_indices = list(debug_window_size_indices) \
            if len(debug_window_size_indices) > 0 \
            else list(range(len(self.__window_sizes)))
        windows = []
        for i in windows_size_indices:
            size = self.__window_sizes[i]
            x_range = self.__ranges_of_interest[i][0]
            y_range = self.__ranges_of_interest[i][1]
            windows.extend(self.__slide_window(x_range=x_range, y_range=y_range,
                                               window_size=(size, size), window_overlap_percentage=(.9, .9)))

        return windows

    def debug_range_of_interests(self):
        windows = []
        for range in self.__ranges_of_interest:
            windows.append(((range[0][0], range[1][0]), (range[0][1], range[1][1])))

        return windows

if __name__ == '__main__':
    image = cv2.imread('test_images/test6.jpg')

    # Here is your draw_boxes function from the previous exercise
    def draw_boxes(img, bbox, color=(255, 0, 0), thick=2):
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)

    window_slider = WindowSlider()
    windows = window_slider.slide_windows()
    # windows = window_slider.debug_range_of_interests()
    print(len(windows))
    for window in windows:
        draw_boxes(image, window)

    cv2.imshow('Boxes', image)
    cv2.waitKey(0)