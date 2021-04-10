#include "ExperimentalFunctions.h"
#include "FileUtilities.h"
#include "OcvUtilities.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/video.hpp>


using namespace cv;
using namespace experimental;
using namespace std;
using namespace OcvUtility;

//Testing trackbar
int g_slider; // slider pos value
int gslider_max; // slider max value


namespace experimental
{
	//////////////////////////////////////////////////////////////////////////////////
	// computeRowWithMaximumBlackPixels()
	//
	// Compute the row of the specified image with the maximum number of black pixels.
	//////////////////////////////////////////////////////////////////////////////////
	int computeRowWithMaximumBlackPixels(cv::Mat image)
	{
		int rowPositionWithMaximumBlackPixels = 0;
		int maxBlackPixelsInAnyRow = 0;

		for (int y = 0; y < static_cast<int>(image.size().height * 0.1); ++y)
		{
			int numberOfBlackPixelsInRow = 0;

			for (int x = 0; x < image.size().width; ++x)
			{
				Point currentPoint = Point(x, y);
				if (image.at<uchar>(currentPoint) == 0)
				{
					numberOfBlackPixelsInRow++;
				}
			}

			if (numberOfBlackPixelsInRow > maxBlackPixelsInAnyRow)
			{
				maxBlackPixelsInAnyRow = numberOfBlackPixelsInRow;
				rowPositionWithMaximumBlackPixels = y;
			}
		}

		return rowPositionWithMaximumBlackPixels;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// computeMaximumRootExtents()
	//
	// Compute the maximum root extents below a specified startingY position.
	// TODO: Generalize this function.
	//////////////////////////////////////////////////////////////////////////////////
	Rect computeMaximumRootExtents(cv::Mat image, const int startingY)
	{
		int b = 0;
		int l = image.size().width;
		int r = 0;
		for (int y = startingY; y < image.size().height; ++y)
		{
			for (int x = 0; x < image.size().width; ++x)
			{
				Point currentPoint = Point(x, y);
				//containerImage.at<uchar>(currentPoint) = 255;	//TODO: Testing where the % line is drawn.
				if (image.at<uchar>(currentPoint) != 0)
				{
					if (currentPoint.x < l)
					{
						l = currentPoint.x;
					}
					if (currentPoint.x > r)
					{
						r = currentPoint.x;
					}
					if (currentPoint.y > b)
					{
						b = currentPoint.y;
					}
				}
			}
		}

		Rect rootRectangle = Rect(l, 0, r - l, b);

		return rootRectangle;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// computeAverageImage()
	//
	// Computes the average of several images.
	//////////////////////////////////////////////////////////////////////////////////
	Mat computeAverageImage(const vector<Mat>& images)
	{
		Mat averageImage = Mat::zeros(images.at(0).size().height, images.at(0).size().width, CV_32FC1);

		for (Mat image : images)
		{
			for (int i = 0; i < image.size().height; ++i)
			{
				for (int j = 0; j < image.size().width; ++j)
				{
					Point currentPoint = Point(j, i);
					averageImage.at<float>(currentPoint) += image.at<uchar>(currentPoint);
				}
			}
		}

		for (int i = 0; i < averageImage.size().height; ++i)
		{
			for (int j = 0; j < averageImage.size().width; ++j)
			{
				Point currentPoint = Point(j, i);
				averageImage.at<float>(currentPoint) /= images.size();
			}
		}

		averageImage.convertTo(averageImage, CV_8UC1);

		return averageImage;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// computeGradientImage()
	//
	// Compute the gradient of the specified image and return it.
	// Code from:
	// http://docs.opencv.org/2.4.10/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
	//////////////////////////////////////////////////////////////////////////////////
	Mat computeGradientImage(Mat image)
	{
		int scale = 1;
		int delta = 0;
		int ddepth = CV_8UC1;
		Mat grad;

		/// Generate grad_x and grad_y
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel(image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		return grad;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// drawRedRectOnImage()
	//
	// Draw a red rectangle on the image.
	//////////////////////////////////////////////////////////////////////////////////
	Mat drawRedRectOnImage(Mat image, Rect rect, int thickness)
	{
		Mat convertedImage = image.clone();

		cv::cvtColor(image.clone(), convertedImage, IMREAD_COLOR);
		cv::rectangle(convertedImage, rect, Scalar(0, 0, 255), thickness);
		
		return convertedImage;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// computeInnermostRectangle()
	//
	// Compute the innermost rectangle that can be defined based on the specified
	// image.
	//////////////////////////////////////////////////////////////////////////////////
	Rect computeInnermostRectangle(Mat image)	//TODO: This function name is terrible.
	{
		Point center = Point(image.size().width / 2, image.size().height / 2);
		int u = 0, d = image.size().height, l = 0, r = image.size().width;

		// Find the nearest nonwhite pixel above.
		for (int i = center.y; i >= 0; --i)
		{
			Point currentPoint = Point(center.x, i);
			int valueAtCurrentPoint = image.at<uchar>(currentPoint);
			if (valueAtCurrentPoint != 0)
			{
				u = i;
				break;
			}
		}

		// Find the nearest nonwhite pixel below.Unhandled exception at 0x00007FFF4D503B29 in autocropper.exe: Microsoft C++ exception: cv::Exception at memory location 0x0000008930DAEB50.
		for (int i = center.y; i < image.size().height; ++i)
		{
			Point currentPoint = Point(center.x, i);
			if (image.at<uchar>(currentPoint) != 0)
			{
				d = i;
				break;
			}
		}

		// Find the nearest nonwhite pixel to the left.
		for (int i = center.x; i >= 0; --i)
		{
			Point currentPoint = Point(i, center.y);
			if (image.at<uchar>(currentPoint) != 0)
			{
				l = i;
				break;
			}
		}

		// Find the nearest nonwhite pixel to the right.
		for (int i = center.x; i < image.size().width; ++i)
		{
			Point currentPoint = Point(i, center.y);
			if (image.at<uchar>(currentPoint) != 0)
			{
				r = i;
				break;
			}
		}

		return Rect(l, u, r - l, d - u);
	}

	//TODO: Write a function to compute the first full row of nonwhite pixels from the center.

	//////////////////////////////////////////////////////////////////////////////////
	// computeOutermostRectangle()
	//
	// Compute the outermost rectangle that can be formed by the specified image.
	// For now let's only check the top and bottom since this will only be used to
	// find the horizontal lines in the container image.
	//////////////////////////////////////////////////////////////////////////////////
	Rect computeOutermostRectangle(Mat image)
	{
		Point center = Point(image.size().width / 2, image.size().height / 2);
		int u = 0, d = image.size().height, l = 0, r = image.size().width;

		// Start at the top middle point and go until you find the first non-black pixel.
		// Start at the bottom middle point and go until you find the first non-black pixel
		// Start at the middle left point and go until ....
		// Start at the middle right point ...

		for (int i = 0; i < image.size().height; ++i)
		{
			Point currentPoint = Point(center.x, i);
			if (image.at<uchar>(currentPoint) != 0)
			{
				u = i;
				break;
			}
		}

		//for (int i = image.size().height - 1; i >= 0; --i)
		//{
		//	Point currentPoint = Point(center.x, i);
		//	if (image.at<uchar>(currentPoint) != 0)
		//	{
		//		d = i;
		//		break;
		//	}
		//}

		return Rect(l, u, r - l, d - u);
	}

	//////////////////////////////////////////////////////////////////////////////////
	// findLargestHorizontalLines()
	//
	// Returns an image containing the largest horizontal lines found in the image.
	//////////////////////////////////////////////////////////////////////////////////
	Mat findLargestHorizontalLines(Mat image, const double percentOfWidth)
	{
		Mat horizontalLines;

		int minimumHorizontalLineSize = static_cast<int>(image.size().width * percentOfWidth);
		auto horizElem = getStructuringElement(MORPH_RECT, Size(minimumHorizontalLineSize, 1));
		morphologyEx(image, horizontalLines, MORPH_OPEN, horizElem, Point(-1,-1), 1, BORDER_CONSTANT, Scalar(0));

		return horizontalLines;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// findLargestVerticalLines()
	//
	// Returns an image containing the largest vertical lines found in the image.
	//////////////////////////////////////////////////////////////////////////////////
	Mat findLargestVerticalLines(Mat image, const double percentOfHeight)
	{
		Mat verticalLines;

		int minimumVerticalLineSize = static_cast<int>(image.size().height * percentOfHeight);
		auto vertElem = getStructuringElement(MORPH_RECT, Size(1, minimumVerticalLineSize));
		morphologyEx(image, verticalLines, MORPH_OPEN, vertElem);

		return verticalLines;
	}
	
	//////////////////////////////////////////////////////////////////////////////////
	// computeForegroundImage()
	//
	// Computes a foreground image based on some background subtraction method.
	//////////////////////////////////////////////////////////////////////////////////
	Mat computeForegroundImage(const vector<Mat>& images)
	{
		Mat foregroundMask, foregroundImage, backgroundImage;
		Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();

		vector<Mat> foregroundMasks;

		int i = 0;

		for (Mat image : images)
		{
			if (foregroundImage.empty())
				foregroundImage.create(image.size(), image.type());

			pMOG2->apply(image, foregroundMask);

			foregroundImage = Scalar::all(0);
			image.copyTo(foregroundImage, foregroundMask);

			//string filename = utility::FileUtilities::buildFilename("C:\\Temp\\images\\fg", ++i);
			//imwrite(filename, foregroundImage);
		}

		return foregroundImage;

		waitKey();
	}
	/////////////////////////////////////////////////////////////////////////////////
	// imagePreprocess()
	//
	// To get rid of all the noises
	//////////////////////////////////////////////////////////////////////////////////
	std::vector<cv::Mat> imagePreprocess(const std::vector<cv::Mat>& images) {
		// Return Vector
		vector <cv::Mat> preprocess_images;
		cv::Mat image = images[1];
		int rows = image.rows;
		int cols = image.cols;
		
		//Blurred image looks good 
		cv::Mat gaussian_blur, blur_image;
		blur(image, blur_image, Size(3, 3));
		GaussianBlur(image, gaussian_blur, Size(3, 3), 0, 0);

		// 13 & 37 is the best for the blur without interfereing
		// 40 is the best for the median blur without interfering
		// 15 & 43 is the best for the gausian blur
		

		Mat result_gausian, result_blur;
		Canny(blur_image, result_blur, 12, 36, 3);
		Canny(gaussian_blur, result_gausian, 15, 43, 3);
		
		vector<vector<cv::Point>> contours_g;
		vector<vector<cv::Point>> contours_b;
		Mat mask1(image.size(), CV_8U, cv::Scalar(255));
		Mat mask2(image.size(), CV_8U, cv::Scalar(255));
		Mat element3 = getStructuringElement(cv::MORPH_RECT, Size(3, 3), Point(1, 1));
		/******************************** Blur **********************************/
		// Find the contour on blur image
		cv::morphologyEx(result_blur.clone(), result_blur, cv::MORPH_DILATE, element3);
		cv::findContours(result_blur.clone(), contours_b, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, Point(0, 0));
		
		// Filtering
		int cmin = 59;  
		int cmax = 5650;
		std::vector<std::vector<cv::Point>>::const_iterator itc = contours_b.begin();
		while (itc != contours_b.end()) {
			if (itc->size() < cmin || itc->size() > cmax)
				itc = contours_b.erase(itc);
			else
				++itc;
		}
		drawContours(mask1, contours_b, -1, cv::Scalar(0), cv::FILLED);
		
		/*************************** Gaussian Blur ****************************/
		cv::morphologyEx(result_gausian.clone(), result_gausian, cv::MORPH_DILATE, element3);
		cv::findContours(result_gausian.clone(), contours_g, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		
		// Filtering
		int cmin_g = 59;
		int cmax_g = 5650;
		std::vector<std::vector<cv::Point>>::const_iterator itc_g = contours_g.begin();
		while (itc_g != contours_g.end()) {
			if (itc_g->size() < cmin_g || itc_g->size() > cmax_g)
				itc_g = contours_g.erase(itc_g);
			else
				++itc_g;
		}
		drawContours(mask2, contours_g, -1, cv::Scalar(0), cv::FILLED);

		// Find the bounding Rectangle
		
		return preprocess_images;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// computeForegroundImages()
	//
	// Computes a foreground images based on some background subtraction method.
	//////////////////////////////////////////////////////////////////////////////////
	vector<Mat> computeForegroundImages(const vector<Mat>& images)
	{
		Mat foregroundMask, foregroundImage, backgroundImage;
		Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();

		vector<Mat> foregroundImages;

		int i = 0;

		for (Mat image : images)
		{
			if (foregroundImage.empty())
				foregroundImage.create(image.size(), image.type());

			pMOG2->apply(image, foregroundMask);

			foregroundImage = Scalar::all(0);
			image.copyTo(foregroundImage, foregroundMask);	//TODO: Does using the mask rather than the image improve the results?

			string filename = utility::FileUtilities::buildFilename("TestImages/DEBUG/foreground/", ++i);
			if (i > 1)	//TODO_DR: Deal with the first file.
			{
				imwrite(filename, foregroundImage);
				foregroundImages.push_back(foregroundImage.clone());
			}
		}

		return foregroundImages;

		waitKey();
	}

	//////////////////////////////////////////////////////////////////////////////////
	// computeHistogram()
	//
	// Compute the histogram of the specified image and return it.
	//////////////////////////////////////////////////////////////////////////////////
	cv::Mat computeHistogram(cv::Mat image)
	{
		Mat histogram;

		int numBins = 256;

		float range[] = { 0, 256 };
		const float* histogramRange = { range };
		bool uniform = true;

		calcHist(&image, 1, 0, Mat(), histogram, 1, &numBins, &histogramRange);

		// Draw histogram.
		int windowWidth = 1024; int windowHeight = 800;
		int bin_w = cvRound((double)windowWidth / numBins);

		Mat histImage(windowHeight, windowWidth, CV_8UC1, Scalar(0, 0, 0));

		/// Normalize the result to [ 0, histImage.rows ]
		normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		for (int i = 1; i < numBins; i++)
		{
			line(histImage, Point(bin_w*(i - 1), windowHeight - cvRound(histogram.at<float>(i - 1))),
				Point(bin_w*(i), windowHeight - cvRound(histogram.at<float>(i))),
				Scalar(255, 255, 255));
		}

		return histImage;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// plotHistogram()
	//
	// Another method to compute the histogram.
	//////////////////////////////////////////////////////////////////////////////////
	Mat plotHistogram(Mat image)
	{
		const unsigned int NUMBER_OF_BINS = 256;
		const unsigned int WINDOW_HEIGHT = NUMBER_OF_BINS;
		const unsigned int WINDOW_WIDTH = NUMBER_OF_BINS;
		Mat histogramImage = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);

		double hist[NUMBER_OF_BINS] = { 0 };

		// Let's compute the histogram.
		MatIterator_<uchar> it, end;
		for (it = image.begin<uchar>(), end = image.end<uchar>();
			it != end;
			++it)
		{
			hist[*it]++;
		}

		// Let's find the max bin amount in the histogram, so that we can scale the rest of the histogram accordingly.
		double max = 0;
		for (unsigned int bin = 0; bin < NUMBER_OF_BINS; ++bin)
		{
			const double binValue = hist[bin];
			if (binValue > max)
				max = binValue;
		}

		// Let's plot the histogram.
		for (unsigned int bin = 0; bin < NUMBER_OF_BINS; ++bin)
		{
			const int binHeight = static_cast<int>(hist[bin] * WINDOW_HEIGHT / max);

			line(histogramImage, Point(bin, WINDOW_HEIGHT - binHeight), Point(bin, WINDOW_HEIGHT), Scalar(255, 255, 255));
		}

		return histogramImage;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// generateEnhancedCenterMask()
	//
	// Generate a mask which has a value of 1.0 in the center, and 0.0 along the
	// edges, with a smooth gradient between. This mask may be sensitive to 
	// variations between length and width of the image.
	//////////////////////////////////////////////////////////////////////////////////
	Mat generateEnhancedCenterMask(Size size)
	{
		Mat image = Mat::ones(size, CV_8UC1);
		padImage(image, image);
		distanceTransform(image, image, CV_DIST_C, 3);
		double maxVal;
		minMaxLoc(image, NULL, &maxVal);
		image *= 1/maxVal;

		removePadding(image, image);

		return image;
	}

	void on_trackbar(int, void*) {
		printf("%d\n", g_slider);
	}
}
