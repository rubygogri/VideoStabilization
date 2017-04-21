/*
This is an attempt to have video stabilization for videos with small and sudden jerks
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

// This is in reference to many papers - One of the very good papers was:
// Real - Time Optical flow - based Video Stabilization for Unmanned Aerial Vehicles and Ngiha Ho

//This is the stabilizing frame radius which should be low for sudden jerks
const int FRAME_RADIUS = 7;

// Size for border cropping when recreating the video
const int EDGE_REMOVAL = 35;

//A few major steps have been categorized as below:
// 1. Transformation - translation and rotation between two frames for the entire video
// 2. Motion estimation is found by considering all these transformation
// 3. Smoothening of these transformation is done by homography
// 4. New transformation is calculated by using affine model warping and Linear interpolation

//Structure for transformation details
struct TransVal
{
	TransVal() {}
	TransVal(int x, int y, int a) {
		dx = x;
		dy = y;
		da = a;
	}
	int dx; //translation
	int dy; //translation
	int da; //rotation
};

struct Homo
{
	Homo() {}
	Homo(int x1, int y1, int a1) {
		x = x1;
		y = y1;
		a = a1;
	}
	int x;
	int y;
	int a;
};

int main(int argc, char **argv)
{
	ofstream out1("rubyStatus.txt");

    string path = "C:\\Users\\rubyg\\Desktop\\Videos\\MyMovie12.mp4";
	VideoCapture vid(path);
	assert(vid.isOpened());

	Mat cur, cur1, cur2, cur_grey;
	Mat prev, prev1, prev_grey;

	//cv::VideoWriter outputVideo;
	//outputVideo.open("C:\\Users\\rubyg\\Desktop\\Videos\\compare.avi", CV_FOURCC('X', 'V', 'I', 'D'), 24, cvSize(cur.rows, cur.cols * 2 + 10), true);
	vid >> prev;

	//Convert to gray scale for faster execution
	cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

	// Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
	vector <TransVal> p2c_transform; 

	int k = 1, new_cnt = 0;
	int max_frames = vid.get(CV_CAP_PROP_FRAME_COUNT);
	Mat last_T;
	int cnt = 1;
	int counter = 0, counter1 = 0, mean=50;

	while (true) {
		vid >> cur;
		cur.copyTo(cur1);
		prev.copyTo(prev1);

		//Do until last frame
		if (cur.data == NULL) {
			break;
		}

		//Convert to grey scale for faster processing
		cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

		// vector from prev to cur
		vector <Point2f> prev_feat, cur_feat, discarded, corner_size;
		vector <Point2f> prev_feat2, cur_feat2;
		vector <uchar> status;
		vector <float> err;

		//Features are being detected by using Eigen Value algorithm with maximum corners as 150
		goodFeaturesToTrack(prev_grey, prev_feat, 150, 0.01, 30);

		//To display detected features for 2 frame comparison
		for (size_t i = 0; i < prev_feat.size(); i++)
		{
		cv::circle(prev1, prev_feat[i], 4, Scalar(CV_RGB(250, 0, 0)), 2);
		}

		/*status:output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features
		has been found, otherwise, it is set to 0. err output vector of errors; each element of the vector is set to an error for the corresponding
		feature, type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined */
		//Other algorithms are Horn–Schunck method, Shi and Tomasi algorithm
		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_feat, cur_feat, status, err);

		for (size_t i = 0; i < cur_feat.size(); i++)
		{
		cv::circle(cur1, cur_feat[i], 4, Scalar(250, 0, 0), 2);
		}

		Mat new_img;
		hconcat(prev1, cur1, new_img);

		std::string pth;

		pth.append(".\\output\\imgOut");
		pth.append(to_string(cnt));
		pth.append(".JPEG");

		//write the edited image
		imwrite(pth, new_img);
		//imwrite(pth, cur1);
		cnt = cnt + 1;

		// Remove weak matches
		for (size_t i = 0; i < status.size(); i++) {
			if (status[i]) {
				prev_feat2.push_back(prev_feat[i]);
				cur_feat2.push_back(cur_feat[i]);
			}
		}

		// Write to a file
		out1 << "prev_feat: " << prev_feat.size() << ":" << cnt - 1 << endl;
		out1 << "prev_feat2: " << prev_feat2.size() << ":" << cnt - 1 << endl;

		// Translation + rotation only
		if ((prev_feat.size() < prev_feat2.size() + 20) || (mean < prev_feat2.size() + 20)) { //prev_feat.size()/3) {
			new_cnt++;

			// Here false indicates a rigid transform, no scaling/shearing
			Mat T = estimateRigidTransform(prev_feat2, cur_feat2, false);

			if (T.data == NULL) {
				last_T.copyTo(T);
			}
			T.copyTo(last_T);

			// decompose T
			int dx = T.at<double>(0, 2);
			int dy = T.at<double>(1, 2);
			int da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));

			//if (prev_feat.size() < prev_feat2.size() + 25) {
			p2c_transform.push_back(TransVal(dx, dy, da));

			cur.copyTo(prev);
			cur_grey.copyTo(prev_grey);

			cout << " # of Features detected: " << prev_feat2.size() << " | Frame #: " << k << " of " << max_frames << endl;
		}
		else 
		{
			// counter for total number of skipped frames
			counter++;  

			//Total number of skipped frames for current video jerk which is no more than 40 - our assumption
			counter1++;

			//discarded.push_back = k;

			//Note all skipped frames occured together
			//corner_size.push_back = prev_feat2.size();

			if (counter1 > 30)
			{   
				//Reset counters
				//mean = 0;
				mean = (prev_feat2.size() + prev_feat.size()) / 2;
				counter1 = 0;
			}
		}
			
		k++;
		out1 << "new count:" << new_cnt;
	}
	cout << " # of frames skipped are :"<< counter << endl;

	// Step 2 - Motion estimation is found by considering all these transformation

	// Frame to frame transformation
	int a = 0;
	int x = 0;
	int y = 0;

	vector <Homo> homo;

	for (size_t i = 0; i < p2c_transform.size(); i++) {
		x += p2c_transform[i].dx;
		y += p2c_transform[i].dy;
		a += p2c_transform[i].da;

		homo.push_back(Homo(x, y, a));
	}

	// Step 3 - Smoothening of these transformation by homography
	vector <Homo> smoothed_Homo;

	for (size_t i = 0; i < homo.size(); i++) {
		int sum_x = 0;
		int sum_y = 0;
		int sum_a = 0;
		int count = 0;

		for (int j = -FRAME_RADIUS; j <= FRAME_RADIUS; j++) {
			if (i + j >= 0 && i + j < homo.size()) {
				sum_x += homo[i + j].x;
				sum_y += homo[i + j].y;
				sum_a += homo[i + j].a;
				count++;
			}
		}
		int avg_a = sum_a / count;
		int avg_x = sum_x / count;
		int avg_y = sum_y / count;

		smoothed_Homo.push_back(Homo(avg_x, avg_y, avg_a));
	}

	// Step 4 - New transformation is calculated by using affine model warping and Linear interpolation
	vector <TransVal> new_p2c_transform;
	a = 0;
	x = 0;
	y = 0;

	for (size_t i = 0; i < p2c_transform.size(); i++) {
		x += p2c_transform[i].dx;
		y += p2c_transform[i].dy;
		a += p2c_transform[i].da;

		int diff_x = smoothed_Homo[i].x - x;
		int diff_y = smoothed_Homo[i].y - y;
		int diff_a = smoothed_Homo[i].a - a;

		int dx = p2c_transform[i].dx + diff_x;
		int dy = p2c_transform[i].dy + diff_y;
		int da = p2c_transform[i].da + diff_a;

		new_p2c_transform.push_back(TransVal(dx, dy, da));
	}

	// To obtain the enhanced video
	vid.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat T(2, 3, CV_64F);

	//Aspect ratio - Movies have aspect ratio of 16:9 and TV is geverally 1.33:1. HDTV have 16:9
	int vert_border = EDGE_REMOVAL * prev.rows / prev.cols;

	k = 0;
	//while (k < max_frames - 1) { 
	while (k < new_cnt) {
		vid >> cur;

		if (cur.data == NULL) {
			break;
		}

		// To get the transformation matrix
		//  | cos -sin translation_x|
		//  | sin  cos translation_y|
		T.at<double>(0, 0) = cos(new_p2c_transform[k].da);
		T.at<double>(0, 1) = -sin(new_p2c_transform[k].da);
		T.at<double>(1, 0) = sin(new_p2c_transform[k].da);
		T.at<double>(1, 1) = cos(new_p2c_transform[k].da);

		T.at<double>(0, 2) = new_p2c_transform[k].dx;
		T.at<double>(1, 2) = new_p2c_transform[k].dy;

		//Applies Affine transformation to each frame of the video 
		warpAffine(cur, cur2, T, cur.size(), cv::INTER_LINEAR);

		cur2 = cur2(Range(vert_border, cur2.rows - vert_border), Range(EDGE_REMOVAL, cur2.cols - EDGE_REMOVAL));

		// Resize cur2 to cur size, for better side by side comparison
		resize(cur2, cur2, cur.size(), 0, 0, cv::INTER_LINEAR);

		// Enhance each frame to remove blur
		cv::GaussianBlur(cur2, prev, cv::Size(0, 0), 3);

		//decreasing 1.5 makes it brighter and increasing makes it lightier
		cv::addWeighted(prev, 1.5, prev, -0.5, 0, prev);

		cur2 = cur2*(2) + prev*(-1);
		//prev = cur2*(2) + prev*(-1);

		// Now draw the original and stablised side by side for coolness
		Mat canvas = Mat::zeros(cur.rows, cur.cols * 2 + 10, cur.type());

		cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
		cur2.copyTo(canvas(Range::all(), Range(cur2.cols + 10, cur2.cols * 2 + 10)));

		//Compare edited blur version to sharpened video of same video
		//cur2.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
		//prev.copyTo(canvas(Range::all(), Range(cur2.cols + 10, cur2.cols * 2 + 10)));

		// If too big to fit on the screen, then scale it down by 2
		if (canvas.cols > 1920) {
			resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));
		}

		//outputVideo.write(canvas);
		//outputVideo << canvas;
		imshow("before and after", canvas);
		waitKey(20);
		k++;
	}
	return 0;
}