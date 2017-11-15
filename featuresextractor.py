import numpy as np
import cv2
from skimage.feature import hog

class FeaturesExtractor:

    def __init__(self, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9,
                 pix_per_cell=8, cell_per_block=2, window_size_pixels=64):
        self.__spatial_size = spatial_size
        self.__hist_bins = hist_bins
        self.__hist_range = hist_range
        self.__orient = orient
        self.__pix_per_cell = pix_per_cell
        self.__cell_per_block = cell_per_block
        self.__window_size_pixels = window_size_pixels

    def bin_spatial(self, image):
        return cv2.resize(image, self.__spatial_size).ravel()

    def color_hist(self, image):
        channel1_hist = np.histogram(image[:, :, 0], bins=self.__hist_bins, range=self.__hist_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=self.__hist_bins, range=self.__hist_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=self.__hist_bins, range=self.__hist_range)
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

        return hist_features

    def get_hog_features(self, image, feature_vec=True, visualise=False):
        return hog(image, orientations=self.__orient, pixels_per_cell=(self.__pix_per_cell, self.__pix_per_cell),
                   cells_per_block=(self.__cell_per_block, self.__cell_per_block), transform_sqrt=True,
                   visualise=visualise, feature_vector=feature_vec)

    def hog_features(self, scaled_image, feature_vec=False):
        hog_features = []
        for channel in range(scaled_image.shape[2]):
            hog_features.append(self.get_hog_features(scaled_image[:, :, channel], feature_vec=feature_vec))

        return np.array(hog_features)

    def extract_image_features_with_windows(self, image, windows):
        window_sizes = [50, 100, 200, 300]
        hog_features = []
        scaled_images = []
        y_search = 340
        image_to_search = image[y_search:, :, :]
        for window_size in window_sizes:
            scale = self.__window_size_pixels / window_size
            scaled_image = cv2.resize(image_to_search, (0, 0), fx=scale, fy=scale)
            scaled_images.append(scaled_image)
            hog_features.append(self.hog_features(scaled_image))

        features = []
        for window in windows:
            window_size = window[1][0] - window[0][0]
            scale = self.__window_size_pixels / window_size
            scale_index = window_sizes.index(window_size)

            y_start = np.int(np.round((window[0][1] - y_search) * scale / self.__pix_per_cell))
            y_end = np.int(np.round((window[1][1] - y_search) * scale / self.__pix_per_cell))
            x_start = np.int(np.round(window[0][0] * scale / self.__pix_per_cell))
            x_end = np.int(np.round(window[1][0] * scale / self.__pix_per_cell))

            cropped = scaled_images[scale_index][
                      y_start * self.__pix_per_cell:y_end * self.__pix_per_cell,
                      x_start * self.__pix_per_cell:x_end * self.__pix_per_cell, :]

            spatial_features = self.bin_spatial(cropped)
            hist_features = self.color_hist(cropped)

            cropped_hog = hog_features[scale_index][:, y_start:y_end - 1, x_start:x_end - 1].ravel()

            features.append(np.concatenate((spatial_features, hist_features, cropped_hog)))

        return features

    def extract_features(self, image, hog_features, cells_per_step=1):
        step_size_pixels = cells_per_step * self.__pix_per_cell
        window_size_hog = self.__window_size_pixels // self.__pix_per_cell - 1
        num_y_steps = (image.shape[0] - self.__window_size_pixels) // step_size_pixels + 1
        num_x_steps = (image.shape[1] - self.__window_size_pixels) // step_size_pixels + 1
        features = []
        windows = []
        for y_step in range(num_y_steps):
            y = y_step * step_size_pixels
            y_hog = y_step * cells_per_step
            for x_step in range(num_x_steps):
                x = x_step * step_size_pixels
                x_hog = x_step * cells_per_step
                cropped = image[y:y + self.__window_size_pixels, x:x + self.__window_size_pixels, :]

                spatial_features = self.bin_spatial(cropped)
                hist_features = self.color_hist(cropped)
                cropped_hog_features = \
                    hog_features[:, y_hog:y_hog + window_size_hog, x_hog:x_hog + window_size_hog].ravel()
                # cropped_hog_features = hog_for_scale(cropped).ravel()

                features.append(np.concatenate((spatial_features, hist_features, cropped_hog_features)))
                windows.append(((x, y), (x + self.__window_size_pixels, y + self.__window_size_pixels)))

        return features, windows

    def extract_image_features_multiscale(self, image):
        scales = (1/1.35, 1.0, 1/.65, 1/.65, 1.75, 2.0, 1/.35)
        ranges_of_interest = (
                              ((300, 980), (400, 460)),  # .75
                              ((300, 980), (400, 460)),  # 1.0
                              ((240, 1280), (378, 520)),  # 1.5
                              ((300, 1280), (392, 520)),  # 1.5
                              ((240, 1280), (378, 520)),  # 1.75
                              ((220, 1280), (360, 520)),  # 2.0
                              ((0, 1280), (350, 600))  # 3.0
                              )
        hog_features = []
        scaled_images = []
        for i, scale in enumerate(scales):
            image_to_search = image[ranges_of_interest[i][1][0]:ranges_of_interest[i][1][1],
                                    ranges_of_interest[i][0][0]:ranges_of_interest[i][0][1], :]
            # We want the "inverse" scale, i.e. instead of .5 we want to scale to 2, so that the area covered by the 64x64
            # pixels is originally only 32x32, thus scale of .5
            scaled_image = cv2.resize(image_to_search, (0, 0), fx=1/scale, fy=1/scale)
            scaled_images.append(scaled_image)
            hog_features.append(self.hog_features(scaled_image))

        all_features = []
        all_windows = []
        for i in range(len(scales)):
            features, windows = self.extract_features(scaled_images[i], hog_features[i])
            all_features.extend(features)
            for window in np.int32(np.round(np.array(windows) * scales[i])):
                all_windows.append(
                    ((ranges_of_interest[i][0][0] + window[0][0], ranges_of_interest[i][1][0] + window[0][1]),
                     (ranges_of_interest[i][0][0] + window[1][0], ranges_of_interest[i][1][0] + window[1][1])))

        return all_features, all_windows

    def extract_image_features(self, image):
        spatial_features = self.bin_spatial(image)
        hist_features = self.color_hist(image)
        hog_features = np.ravel(self.hog_features(image, feature_vec=True))

        return np.concatenate((spatial_features, hist_features, hog_features))

    def extract_image_features_with_windows_slow(self, image, windows):
        features = []
        for window in windows:
            cropped = image[window[0][1]:window[1][1], window[0][0]:window[1][0], :]
            cropped = cv2.resize(cropped, (self.__window_size_pixels, self.__window_size_pixels))

            features.append(np.float64(self.extract_image_features(cropped)))

        return features

if __name__ == '__main__':
    car = cv2.cvtColor(cv2.imread('./training_data/vehicles/KITTI_extracted/348.png'), cv2.COLOR_BGR2LUV)
    cv2.imshow('Car', cv2.cvtColor(cv2.resize(car, (0, 0), fx=4, fy=4), cv2.COLOR_LUV2BGR))
    features, hog_image_car = FeaturesExtractor().get_hog_features(car[:,:,0], visualise=True)
    cv2.imshow('Hog Car - Channel L', cv2.resize(hog_image_car, (0, 0), fx=4, fy=4))
    features, hog_image_car = FeaturesExtractor().get_hog_features(car[:,:,1], visualise=True)
    cv2.imshow('Hog Car - Channel U', cv2.resize(hog_image_car, (0, 0), fx=4, fy=4))
    features, hog_image_car = FeaturesExtractor().get_hog_features(car[:,:,2], visualise=True)
    cv2.imshow('Hog Car - Channel V', cv2.resize(hog_image_car, (0, 0), fx=4, fy=4))

    notcar = cv2.cvtColor(cv2.imread('./training_data/non-vehicles/Extras/extra177.png'), cv2.COLOR_BGR2LUV)
    cv2.imshow('Not Car', cv2.cvtColor(cv2.resize(notcar, (0, 0), fx=4, fy=4), cv2.COLOR_LUV2BGR))
    features, hog_image_notcar = FeaturesExtractor().get_hog_features(notcar[:,:,0], visualise=True)
    cv2.imshow('Hog Not Car - Channel L', cv2.resize(hog_image_notcar, (0, 0), fx=4, fy=4))
    features, hog_image_notcar = FeaturesExtractor().get_hog_features(notcar[:,:,1], visualise=True)
    cv2.imshow('Hog Not Car - Channel U', cv2.resize(hog_image_notcar, (0, 0), fx=4, fy=4))
    features, hog_image_notcar = FeaturesExtractor().get_hog_features(notcar[:,:,2], visualise=True)
    cv2.imshow('Hog Not Car - Channel V', cv2.resize(hog_image_notcar, (0, 0), fx=4, fy=4))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
