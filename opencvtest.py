import cv2
import numpy as np

minHessian = 400

minmatches = 10


def try_image(debug, template, source):
    # Detect features for template and source
    detector: cv2.xfeatures2d.DAISY_NRM_SIFT = cv2.xfeatures2d.SIFT_create()
    kp1, dsc1 = detector.detectAndCompute(template, None)
    kp2, dsc2 = detector.detectAndCompute(source, None)

    # Match both sets of features
    matcher: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher_create()
    knn_matches = matcher.knnMatch(dsc1, dsc2, 2)

    # Filter only matches > 50%
    ratio_tresh = 0.5
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_tresh * n.distance:
            good_matches.append(m)

    if debug:
        matches = cv2.drawMatches(template, kp1, source, kp2, good_matches, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Matches", matches)
        cv2.waitKey()

    # If not enough matches, this is not the right image
    if len(good_matches) < minmatches:
        return None, None

    # Reshape the arrays containing the matches
    src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dest = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimates the transformation matrix to get to source from template
    Matrix, mask = cv2.estimateAffine2D(src, dest)

    # Extract rotation from matrix
    angle = round(-np.degrees(np.arctan2(-Matrix[0, 1], Matrix[0, 0])))

    # Apply matrix to template center to get center in source
    h, w, ch = template.shape
    normal_center = np.array([w / 2, h / 2])
    points = np.array([normal_center])
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    transformed = Matrix.dot(points_ones.T).T

    final_center = (round(transformed[0][0]), round(transformed[0][1]))

    return angle, final_center


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def try_all(debug, folder: str, file_names: [str], base_image: np.array):
    angle, filename, translation = None, None, None
    for i in file_names:
        loaded = cv2.imread(folder + "/" + i)
        angle, translation = try_image(debug, loaded, base_image)
        if angle is None:
            continue
        else:
            filename = i
            break
    return filename, translation, angle
