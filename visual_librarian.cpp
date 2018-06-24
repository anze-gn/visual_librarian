#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"

#include <curl/curl.h>
#include "utilities.h"
#include "features.h"

using namespace std;
using namespace cv;

#define WINDOW_NAME "Visual Librarian"
#define WINDOW_HEIGHT 960.0

Mat scale_frame(Mat frame) {
	double scaled_frame_factor = WINDOW_HEIGHT / frame.rows;

	Mat scaled_frame;
	resize(frame, scaled_frame, Size(0,0), scaled_frame_factor, scaled_frame_factor, INTER_LINEAR);
	return scaled_frame;
}

bool compare_matches(DMatch first, DMatch second) {
	return (first.distance < second.distance);
}

int find_the_book(String books_folder, int num_of_books, Mat book_test_rgb, int actualIdx) { // DELETE ME {actualIdx}
	double stopwatch_start = (double)getTickCount(); // start the stopwatch

	Mat frame = scale_frame(book_test_rgb);
	imshow(WINDOW_NAME, frame);
	waitKey(30);

	Mat book_test_hsv, book_test_gs;
	cvtColor(book_test_rgb, book_test_hsv, COLOR_BGR2HSV); // convert to HSV
	cvtColor(book_test_rgb, book_test_gs, COLOR_BGR2GRAY); // convert to GRAYSCALE

	int h_bins = 50, s_bins = 60; // 50 bins for hue and 60 for saturation
	int histSize[] = { h_bins, s_bins };

	float h_ranges[] = { 0, 180 }; // hue varies from 0 to 179
	float s_ranges[] = { 0, 256 }; // saturation varies from 0 to 255
	const float* ranges[] = { h_ranges, s_ranges };
	
	int channels[] = { 0, 1 }; // use the 0-th and 1-st channel

	// calculate the histograms for the book_test_hsv
	Mat book_test_hist;
	calcHist(&book_test_hsv, 1, channels, Mat(), book_test_hist, 2, histSize, ranges, true, false);
	normalize(book_test_hist, book_test_hist, 0, 1, NORM_MINMAX, -1, Mat());

	// HOMOGRAPHY
	Mat book_test_descriptors;
	vector<KeyPoint> book_test_keypoints;
	Ptr<Feature2D> descriptor = get_descriptor(FEATURES_SIFT);
	descriptor->detectAndCompute(book_test_gs, Mat(), book_test_keypoints, book_test_descriptors);
	int max_matches_to_use = 100;

	int max_inliers = 0;
	int best_match_index = -1;
	int homography_count = 0; // number of books that were compared with homography
	for (size_t i = 1; i <= num_of_books; i++) {
		Mat book_compare_rgb = imread( join(books_folder, format("%03d.jpg", i)) );
		if (book_compare_rgb.empty()) {
			cerr << "book_compare_rgb is EMPTY\n";
			throw;
		}

		Mat book_compare_hsv, book_compare_gs;
		cvtColor(book_compare_rgb, book_compare_hsv, COLOR_BGR2HSV); // convert to HSV
		cvtColor(book_compare_rgb, book_compare_gs, COLOR_BGR2GRAY); // convert to GRAYSCALE

		// calculate the histograms for the book_compare_hsv
		Mat book_compare_hist;
		calcHist(&book_compare_hsv, 1, channels, Mat(), book_compare_hist, 2, histSize, ranges, true, false);
		normalize(book_compare_hist, book_compare_hist, 0, 1, NORM_MINMAX, -1, Mat());

		// compare histograms book_test_hist and book_compare_hist
		double hist_diff = compareHist(book_test_hist, book_compare_hist, CV_COMP_HELLINGER);
		
		// DELETE ME
		/*
		if (i == actualIdx)
			cout << hist_diff << endl;
		*/
		
		// only compare books with histogram difference 0.5 or less
		if (hist_diff > 0.5)
			continue;

		// HOMOGRAPHY
		homography_count++; // number of books that were compared with homography

		Mat frame_copy = frame.clone();
		putText(frame_copy, "analyzing"+String((homography_count%5)+1,'.'), Point(60, frame_copy.rows/2), FONT_HERSHEY_SIMPLEX, 2.5, Scalar(0, 255, 0, 0), 3);
		imshow(WINDOW_NAME, scale_frame(frame_copy));
		waitKey(30);

		Mat book_check_descriptors;
		vector<KeyPoint> book_check_keypoints;
		vector<DMatch> descriptor_matches;
		Ptr<DescriptorMatcher> descriptor_matcher = new BFMatcher();
		
		//descriptor->detectAndCompute(book_compare_gs, Mat(), book_check_keypoints, book_check_descriptors); // detect and compute book_check_keypoints and book_check_descriptors
		
		/*
		// save book_check_keypoints and book_check_descriptors to {i}.yaml
		// to use this code, you must comment out "if (hist_diff > 0.5)"
		FileStorage fs(join(books_folder, format("%03d.yaml", i)) , FileStorage::WRITE);
		fs << "book_check_keypoints" << book_check_keypoints;
		fs << "book_check_descriptors" << book_check_descriptors;
		fs.release();
		*/
		
		// use book_check_keypoints and book_check_descriptors from {i}.yaml
		FileStorage fs(join(books_folder, format("%03d.yaml", i)), FileStorage::READ);
		fs["book_check_keypoints"] >> book_check_keypoints;
		fs["book_check_descriptors"] >> book_check_descriptors;
		fs.release();
		

		descriptor_matcher->match(book_check_descriptors, book_test_descriptors, descriptor_matches); // save books' descriptor matches to descriptor_matches

		sort(descriptor_matches.begin(), descriptor_matches.end(), compare_matches);
		//double min_dist = descriptor_matches[0].distance;
		//double max_dist = descriptor_matches[descriptor_matches.size() - 1].distance;

		if (descriptor_matches.size() < 4) {
			cerr << "At least four descriptor_matches required to compute homography.\n";
			throw;
		}

		std::vector<Point2f> srcPoints, dstPoints;
		int num_of_matches_to_use = MIN(max_matches_to_use, (int)descriptor_matches.size());
		for (int i = 0; i < num_of_matches_to_use; i++) {
			srcPoints.push_back( book_check_keypoints[descriptor_matches[i].queryIdx].pt );
			dstPoints.push_back(  book_test_keypoints[descriptor_matches[i].trainIdx].pt );
		}

		vector<int> homography_inliers;
		double ransac_reproj_threshold = 0.5;
		Mat H = findHomography(srcPoints, dstPoints, CV_RANSAC, ransac_reproj_threshold, homography_inliers); // calculate the "homography_inliers" mask

		// count the number of inliers
		int homography_inliers_count = 0;
		for (int i = 0; i < num_of_matches_to_use; i++) {
			if (homography_inliers[i] == 1)
				homography_inliers_count++;
		}

		// check if it is a better match
		if (homography_inliers_count > max_inliers) {
			best_match_index = i;
			max_inliers = homography_inliers_count;
		}

		//cout << format("Book %02d: hist_diff:%6.5f   homography_inliers_count:%5d", i, hist_diff, homography_inliers_count) << endl;
	}

	double time_seconds = ((double)getTickCount() - stopwatch_start) / getTickFrequency();
	//cout << format("Best match: %d  (in %5.2f seconds)", best_match_index, time_seconds) << endl;

	// DELETE ME
	String result = (actualIdx == best_match_index) ? "----ok" : format("ER-%3d", actualIdx);
	cout << result << " in " << format("%5.2f seconds", time_seconds) << format("   compared: %2d images", homography_count) << endl;

	return best_match_index;
}


size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
	size_t written = fwrite(ptr, size, nmemb, stream);
	return written;
}

Mat capture_a_book(String phone_ip, String test_book_filename) {
	VideoCapture camera(phone_ip + "/video");

	if (!camera.isOpened()) {
		cerr << "Unable to access camera\n";
		throw;
	}

	float photo_height = 2880;
	float scaling_factor = WINDOW_HEIGHT / photo_height;
	Mat frame;

	Rect crop_area(600, 700, 1000, 1400);
	Rect crop_area_scaled(
		crop_area.x * scaling_factor,
		crop_area.y * scaling_factor,
		crop_area.width * scaling_factor,
		crop_area.height * scaling_factor
	);

	while (true) {
		camera.read(frame);
		frame = scale_frame(frame);
		putText(frame, "press space to capture", Point(10,50), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 0, 0, 0), 2);
		rectangle(frame, crop_area_scaled, Scalar(0, 255, 0, 0), 3); // draw a rectangle where the book has to be
		imshow(WINDOW_NAME, frame);

		char key = waitKey(30);
		if (key == ' ') {// save the photo as {test_book_filename}
			putText(frame, "saving...", Point(crop_area_scaled.x + 30, crop_area_scaled.y + crop_area_scaled.height/2), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255, 0), 2);
			imshow(WINDOW_NAME, scale_frame(frame));
			waitKey(30);

			CURL* curl = curl_easy_init();
			CURLcode res;
			char* url = (char*)malloc(50 * sizeof(char));
			sprintf(url, "%s/photoaf.jpg", phone_ip.c_str());

			curl_easy_setopt(curl, CURLOPT_URL, url);
			curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
			FILE* fp = fopen(test_book_filename.c_str(), "wb");
			curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
			curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);

			res = curl_easy_perform(curl); // res will get the return code
			if (res != CURLE_OK) {
				cerr << format("curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
				throw;
			}

			curl_easy_cleanup(curl);
			fclose(fp);

			break;
		}
	}

	Mat test_img = imread(test_book_filename);

	// rotate test_img right
	Mat temp;
	transpose(test_img, temp);
	flip(temp, test_img, 1);

	return test_img(crop_area);
}

vector<String> users = {
	"Ana Novak",
	"Marija Hrovat",
	"Janez Kranjc",
	"Marko Mlakar",
	"Maja Kos"
};
int select_user() {
	Mat background = imread(join("img","bkg.jpg"));
	Mat frame = scale_frame(background);
	putText(frame, "Visual Librarian", Point(20, 150), FONT_HERSHEY_SIMPLEX, 2.8, Scalar(0, 255, 0, 0), 5);
	rectangle(frame, Rect(40, 250, 400, 80 + users.size()*50), Scalar(255, 255, 255, 0), CV_FILLED);
	putText(frame, "Select user:", Point(50, 300), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 0, 0), 2);
	int offset_y = 300;

	int max_text_width = 0;
	for (size_t i = 0; i < users.size(); i++) {
		offset_y += 50;

		String text = format("%d - %s", i, users[i].c_str());
		int fontFace = FONT_HERSHEY_SIMPLEX;
		double fontScale = 1;
		int thickness = 2;
		int baseline = 0;

		putText(frame, text, Point(50, offset_y), fontFace, fontScale, Scalar(255, 0, 0, 0), thickness);
		Size textSize = getTextSize(text, fontFace,	fontScale, thickness, &baseline);
		if (max_text_width < textSize.width)
			max_text_width = textSize.width;
	}

	imshow(WINDOW_NAME, frame);

	int user_id = -1;
	while (char key = waitKey(0)) {
		user_id = key - '0';
		if (user_id >= 0 && user_id < users.size())
			break;
	}
	return user_id;
}


int main(int argc, char** argv) {
	if (argc < 4) {
		cerr << "** Error. Usage: ./visual_librarian <books_folder> <number_of_books>\n";
		throw;
	}

	while (true) {
		int user_id = select_user();

		
		String phone_ip = "http://192.168.43.1:8080";
		String test_book_filename = "test.jpg";

		Mat book_test_rgb = capture_a_book(phone_ip, test_book_filename);

		imshow(WINDOW_NAME, scale_frame(book_test_rgb));
		waitKey(30);
		
		/*
		for (size_t i = 1; i <= 81; i++) {
			Mat src_base = imread(format("C:\\Users\\Anze\\Box Sync\\FRI\\vid\\project\\zbirka1\\auto_all renamed+resized_only1_testing\\%03d.jpg", i));

			if (src_base.empty()) {
				cerr << "src_base is EMPTY\n";
				throw;
			}

			find_the_book(argv[1], stoi(argv[2]), src_base, i);
		}
		*/
		
		int book_detected_index = find_the_book(argv[1], stoi(argv[2]), book_test_rgb, -1);

		Mat frame = scale_frame(book_test_rgb);
		putText(frame, format("Detected book: %d", book_detected_index), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 0, 0, 0), 2);
		putText(frame, "o - OK",     Point(10, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 0), 2);
		putText(frame, "c - CANCEL", Point(10, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255, 0), 2);
		imshow(WINDOW_NAME, frame);
		char key = 0;
		while (key = waitKey(0)) {
			if (key == 'o' || key == 'c')
				break;
		}
		if (key == 'c')
			continue;
		
		
		frame = scale_frame(book_test_rgb);
		putText(frame, users[user_id],                     Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 0, 0, 0), 2);
		putText(frame, "borrowed",                        Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 0, 0, 0), 2);
		putText(frame, format("%d", book_detected_index), Point(10, 150), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 0, 0, 0), 2);

		putText(frame, "Do you want to scan another book?", Point(10, 250), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0, 0), 2);
		putText(frame, "y - YES", Point(10, 300), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 0), 2);
		putText(frame, "n - NO",  Point(10, 350), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255, 0), 2);
		imshow(WINDOW_NAME, frame);
		key = 0;
		while (key = waitKey(0)) {
			if (key == 'y' || key == 'n')
				break;
		}
		if (key == 'n')
			break;
				
	}
	return 0;
}