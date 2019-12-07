#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

// Raw gradient. No denoising
void gradient(const Mat&Ic, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float ix, iy;
			if (i == 0 || i == m - 1)
				iy = 0;
			else
				iy = (float(I.at<uchar>(i - 1, j)) - float(I.at<uchar>(i + 1, j))) / 2;
			if (j == 0 || j == n - 1)
				ix = 0;
			else
				ix = (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1))) / 2;
			G2.at<float>(i, j) = (ix*ix + iy * iy);
		}
	}
}

// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float ix, iy;
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
				ix = 0;
				iy = 0;
			}
			else {
				// Sobel
				ix = ((float(I.at<uchar>(i - 1, j + 1)) - float(I.at<uchar>(i - 1, j - 1)))
					+ 2 * (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1)))
					+ (float(I.at<uchar>(i + 1, j + 1)) - float(I.at<uchar>(i + 1, j - 1)))) / 8;
				iy = ((float(I.at<uchar>(i + 1, j - 1)) - float(I.at<uchar>(i - 1, j - 1)))
					+ 2 * (float(I.at<uchar>(i + 1, j)) - float(I.at<uchar>(i - 1, j)))
					+ (float(I.at<uchar>(i + 1, j - 1)) - float(I.at<uchar>(i - 1, j - 1)))) / 8;
			}
			Ix.at<float>(i, j) = ix;
			Iy.at<float>(i, j) = iy;
			G2.at<float>(i, j) = ( ix * ix + iy * iy);
		}
	}
}

// Gradient thresholding, default = do not denoise
Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
	Mat Ix, Iy, G2;
	if (denoise)
		sobel(Ic, Ix, Iy, G2);
	else
		gradient(Ic, G2);
	s *= s; // squared gradient
	int m = Ic.rows, n = Ic.cols;
	Mat C(m, n, CV_8U);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			C.at<uchar>(i, j) = (G2.at<float>(i, j) > s ? 255 : 0);
	return C;
}

// Canny edge detector
Mat canny(const Mat& Ic, float s1)
{
	Mat Ix, Iy, G2;
	sobel(Ic, Ix, Iy, G2);

	float s2 = 3 * s1;
	s1 *= s1; s2 *= s2; // gradient is actually squared gradient

	int m = Ic.rows, n = Ic.cols;
	Mat Max(m, n, CV_8U);	// Max pixels ( G2 > s1 && max in the direction of the gradient )
	queue<Point> Q;			// Enqueue seeds ( Max pixels for which G2 > s2 )
	const float tan22 = .414f, tan67 = 2.414f;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			uchar mx = 0;
			float G0 = G2.at<float>(i, j);
			if (i > 0 && j > 0 && i < m - 1 && j<n - 1 && G0>s1) {
				float ix = Ix.at<float>(i, j), iy = Iy.at<float>(i, j);
				// if ix < 0, take symmetric (to get 4 cases only)
				if (ix < 0) {
					ix = -ix;
					iy = -iy;
				}
				float Ga, Gb;
				// The 4 cases
				if (iy > tan67*ix || iy < -tan67 * ix) {
					Ga = G2.at<float>(i + 1, j); Gb = G2.at<float>(i - 1, j);
				}
				else if (iy > .414*ix) {
					Ga = G2.at<float>(i + 1, j + 1); Gb = G2.at<float>(i - 1, j - 1);
				}
				else if (iy > -.414*ix) {
					Ga = G2.at<float>(i, j + 1); Gb = G2.at<float>(i, j - 1);
				}
				else {
					Ga = G2.at<float>(i + 1, j - 1); Gb = G2.at<float>(i - 1, j + 1);
				}
				if (G0 > Ga && G0 > Gb) {
					mx = 255;
					if (G0 > s2)
						Q.push(Point(j, i));
				}
			}
			Max.at<uchar>(i, j) = mx;
		}
	}
	
	// For testing purpose
	// imshow("Max", Max);

	// Propagate seeds
	Mat C(m, n, CV_8U);
	C.setTo(0);
	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
		Q.pop();
		C.at<uchar>(i, j) = 255;
		for (int a = max(0, i - 1); a < min(i + 2, m); a++) {
			for (int b = max(0, j - 1); b < min(j + 2, n); b++) {
				if (!C.at<uchar>(a, b) && Max.at<uchar>(a, b))
					Q.push(Point(b, a));
			}
		}
	}

	return C;
}

int main()
{
	Mat I = imread("../road.jpg");

	imshow("Input", I);
	imshow("Threshold", threshold(I, 15));
	imshow("Threshold + denoising", threshold(I, 15, true));
	imshow("Canny", canny(I, 15));

	waitKey();

	return 0;
}
