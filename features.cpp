
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "features.h"

void list_detectors() {
#ifdef _XFEATURES
    cout << "Press: \n * f: FAST \n * b: BRISK \n * o: ORB \n * m: MSER \n * a: AKAZE \n";
#else
    cout << "Press: \n * s: SIFT \n * u: SURF \n * f: FAST \n * b: BRISK \n * o: ORB \n * m: MSER \n * a: AKAZE \n";
#endif
}

void list_descriptors() {
#ifdef _XFEATURES
    cout << "Press: \n * s: SIFT \n * u: SURF \n * o: ORB \n * a: AKAZE \n";
#else
    cout << "Press: \n * f: FAST \n * b: BRISK \n * o: ORB \n * m: MSER \n * a: AKAZE \n";
#endif
}

int decode_detector(int c) {
    switch(c) {
#ifdef _XFEATURES
    case 's':
        return FEATURES_SIFT;
    case 'u':
        return FEATURES_SURF;
#endif
    case 'f':
        return FEATURES_FAST;
    case 'b':
        return FEATURES_BRISK;
    case 'o':
        return FEATURES_ORB;
    case 'm':
        return FEATURES_MSER;
    case 'a':
        return FEATURES_AKAZE;
    default:
        return FEATURES_AKAZE;
    }
}

int decode_descriptor(int c) {
    switch(c) {
#ifdef _XFEATURES
    case 's':
        return FEATURES_SIFT;
    case 'u':
        return FEATURES_SURF;
#endif
    case 'o':
        return FEATURES_ORB;
    case 'a':
        return FEATURES_AKAZE;
    default:
        return FEATURES_AKAZE;
    }
}

Ptr<FeatureDetector> get_detector(int id) {
    Ptr<Feature2D> detector;

    switch (id) {
#ifdef _XFEATURES
    case FEATURES_SIFT: {
        detector = SIFT::create();
        break;
    }
    case FEATURES_SURF: {
        detector = SURF::create(400);
        break;
    }
#endif
    case FEATURES_FAST: {
        detector = FastFeatureDetector::create();
        break;
    }
    case FEATURES_BRISK: {
        detector = BRISK::create();
        break;
    }
    case FEATURES_ORB: {
        detector = ORB::create();
        break;
    }
    case FEATURES_MSER: {
        detector = MSER::create();
        break;
    }
    case FEATURES_AKAZE: {
        detector = AKAZE::create();
        break;
    }
    }

    return detector;
}

Ptr<Feature2D> get_descriptor(int id) {
    Ptr<Feature2D> descriptor;

    switch (id) {
#ifdef _XFEATURES
    case FEATURES_SIFT: {
        descriptor = SIFT::create();
        break;
    }
    case FEATURES_SURF: {
        descriptor = SURF::create(400);
        break;
    }
#endif
    case FEATURES_ORB: {
        descriptor = ORB::create();
        break;
    }
    case FEATURES_AKAZE: {
        descriptor = AKAZE::create();
        break;
    }
    }

    return descriptor;
}

bool detect_template_marker(Mat image, Mat pattern, float pattern_size, Mat intrinsics, Mat distortion, Mat& rotation, Mat& translation) {

    int threshold = 5;
    int block_size = 45;
    double confidence_threshold = 0.8;
    double best_confidence = 0;
    Point2f region_corners[4];
    Mat temp, normalized(pattern.size(), CV_8UC1);
    Point2f normalized_corners[4] = {Point2f(0, 0), Point2f(pattern.rows - 1, 0),
                                     Point2f(pattern.rows - 1, pattern.cols - 1), Point2f(0, pattern.cols - 1)
                                    };
    Mat rotated_pattern[4];

    adaptiveThreshold(image, temp, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, block_size, threshold);
    dilate(temp, temp, Mat());

    int avsize = (temp.rows + temp.cols) / 2;

    vector<vector<Point> > contours;
    vector<Point> polygon;

    findContours(temp, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    if(pattern.cols != pattern.rows)
        throw runtime_error("Not a square pattern");

    unsigned int i;
    Point p;
    int pMinX, pMinY, pMaxY, pMaxX;

    pattern.copyTo(rotated_pattern[0]);

    for(i = 1; i < 4; i++) {
        Mat temp;
        transpose(rotated_pattern[i-1], temp);
        flip(temp, rotated_pattern[i], 1);
    }

    for(i=0; i < contours.size(); i++) {

        Mat contour_matrix = Mat (contours[i]);
        const double per = arcLength(contour_matrix, true);

        if (per > (avsize / 4) && per < (4 * avsize)) {

            polygon.clear();
            approxPolyDP(contour_matrix, polygon, per * 0.02, true);

            //check rectangularity and convexity
            if (polygon.size() == 4 && isContourConvex(Mat(polygon))) {

                // determine the bounding box of contour
                p = polygon.at(0);
                pMinX = pMaxX = p.x;
                pMinY = pMaxY = p.y;
                int j;
                for(j=1; j<4; j++) {
                    p = polygon.at(j);
                    if (p.x < pMinX) pMinX = p.x;
                    if (p.x > pMaxX) pMaxX = p.x;
                    if (p.y < pMinY) pMinY = p.y;
                    if (p.y > pMaxY) pMaxY = p.y;
                }
                Rect bounding_box(pMinX, pMinY, pMaxX - pMinX + 1, pMaxY - pMinY + 1);

                //find the upper left vertex
                double dmin = (4 * avsize * avsize);
                int first_point = -1;
                for (j = 0; j < 4; j++) {
                    double d = norm(polygon.at(j));
                    if (d < dmin) {
                        dmin = d;
                        first_point = j;
                    }
                }

                vector<Point2f> refined(4);
                copy(polygon.begin(), polygon.end(), refined.begin());

                cornerSubPix(image, refined, Size(3,3), Size(-1,-1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

                for(j = 0; j < 4; j++)
                    region_corners[j] = Point2f(refined.at((4 + first_point - j) % 4).x - pMinX, refined.at((4 + first_point - j) % 4).y - pMinY);

                Mat H(3,3,CV_32F);
                H = getPerspectiveTransform(region_corners, normalized_corners);

                // warp the input based on the homography model to get the normalized interest region
                warpPerspective(image(Range(bounding_box.y, bounding_box.y + bounding_box.height),
                                      Range(bounding_box.x, bounding_box.x + bounding_box.width)),
                                normalized, H, Size(normalized.cols, normalized.rows));

                int orientation = 0;
                double confidence = 0;
                double N = (double)(pattern.cols * pattern.rows);

                Scalar mean_value, std_value;
                meanStdDev(normalized, mean_value, std_value);
                double normalized_squared = pow(norm(normalized), 2);

                for(j = 0; j < 4; j++) {

                    transpose(pattern, temp);
                    flip(temp, pattern, 1);

                    double const nnn = pow(norm(rotated_pattern[j]), 2);
                    double const mmm = mean(rotated_pattern[j]).val[0];

                    double nominator = normalized.dot(rotated_pattern[j]) - (N * mean_value.val[0] * mmm);
                    double denominator = sqrt((normalized_squared - (N * mean_value.val[0] * mean_value.val[0]) ) * (nnn - (N * mmm * mmm)));
                    double temp_confidence = nominator / denominator;

                    if(temp_confidence > confidence) {
                        confidence = temp_confidence;
                        orientation = j;
                    }
                }

                if (confidence > best_confidence) {

                    best_confidence = confidence;

                    vector<Point2f> image_points;
                    vector<Point3f> pattern_points;
                    for (j=0; j<4; j++)
                        image_points.push_back(refined.at((8 - orientation + first_point - j) % 4));

                    pattern_points.push_back(Point3f(0, 0, 0));
                    pattern_points.push_back(Point3f(pattern_size, 0, 0));
                    pattern_points.push_back(Point3f(pattern_size, pattern_size, 0));
                    pattern_points.push_back(Point3f(0, pattern_size, 0));

                    solvePnP(pattern_points, image_points, intrinsics, distortion, rotation, translation);
                }
            }
        }
    }

    return best_confidence >= confidence_threshold;
}



