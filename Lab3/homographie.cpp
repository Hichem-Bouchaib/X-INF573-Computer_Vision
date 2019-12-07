#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat I1 = imread("../IMG_0045.JPG", IMREAD_GRAYSCALE);
	Mat I2 = imread("../IMG_0046.JPG", IMREAD_GRAYSCALE);
	
	imshow("I1", I1);
	imshow("I2", I2);

	Ptr<ORB> D=ORB::create();
	//Ptr<AKAZE> D = AKAZE::create();

	vector<KeyPoint> m1, m2;
	Mat desc1, desc2;
	D->detectAndCompute(I1, Mat(), m1, desc1);
	D->detectAndCompute(I2, Mat(), m2, desc2);

	Mat J;
	drawKeypoints(I1, m1, J, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("I1", J);
	drawKeypoints(I2, m2, J, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("I2", J);
	waitKey();

	BFMatcher M(NORM_L2/*,true*/);
	vector<DMatch> matches;
	M.match(desc1, desc2, matches);
	cout << matches.size() << endl;
	drawMatches(I1, m1, I2, m2, matches, J);
	resize(J, J, Size(), .5, .5);
	imshow("Matches", J);
	waitKey();

	vector<Point2f> matches1, matches2;
	for (int i = 0; i<matches.size(); i++) {
		matches1.push_back(m1[matches[i].queryIdx].pt);
		matches2.push_back(m2[matches[i].trainIdx].pt);
	}
	Mat mask; // Inliers?
	Mat H = findHomography(matches1, matches2, RANSAC, 3, mask);
	cout << H << endl;
	vector<DMatch> inliers;
	for (int i = 0; i<matches.size(); i++)
		if (mask.at<uchar>(i, 0) != 0)
			inliers.push_back(matches[i]);

	drawMatches(I1, m1, I2, m2, inliers, J);
	resize(J, J, Size(), .5, .5);
	imshow("Inliers", J);
	cout << matches.size() << " matches" << " -> " << inliers.size() << " inliers" << endl;
	waitKey();

	
	Mat K(2 * I1.cols, I1.rows, CV_8U);
	warpPerspective(I1, K, Mat::eye(Size(3, 3), CV_32F), Size(2 * I1.cols, I1.rows));
	warpPerspective(I2, K, H, Size(2 * I1.cols, I1.rows), INTER_LINEAR + WARP_INVERSE_MAP, BORDER_TRANSPARENT);
	imshow("I1+I2", K);
	waitKey();

	return 0;
}
