import numpy as np
import cv2
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from featuresextractor import FeaturesExtractor


def extract_features(imgs, features_extractor):
    features = []
    for file in imgs:
        image = cv2.imread(file)
        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        features.append(features_extractor.extract_image_features(feature_image))

    return features


def read_images(path):
    images = []
    for f in os.listdir(path):
        current = os.path.join(path, f)
        if os.path.isdir(current):
            images.extend(read_images(current))
        elif current.endswith('.png'):
            images.append(current)

    return images


def train(path='training_data', vehicles='vehicles', non_vehicles='non-vehicles'):
    cars = read_images(os.path.join(path, vehicles))
    not_cars = read_images(os.path.join(path, non_vehicles))

    features_extractor = FeaturesExtractor()

    print('Extracting car features...')
    car_features = extract_features(cars, features_extractor)
    print('Extracting not car features...')
    notcar_features = extract_features(not_cars, features_extractor)

    print('Scaling data...')
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    print('Training model...')

    model= LogisticRegression()
    t=time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train model...')
    # Check the score of the SVC
    print('Test Accuracy of model = ', round(model.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = len(X_test)
    print('My model predicts: ', model.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,' labels with model')

    return model, X_scaler


if __name__ == '__main__':
    train()