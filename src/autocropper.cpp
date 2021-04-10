#include "ExperimentalFunctions.h"
#include "FileUtilities.h"
#include "ImageReader.h"
#include "OcvUtilities.h"
#include "TrackbarWindow.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace autocropper;
using namespace cv;
using namespace experimental;
using namespace std;
using namespace utility;
using namespace OcvUtility;

Rect computeVerticalContainerBoundaries(Mat originalImage)
{
	Mat verticalContainerBoundaries = findLargestVerticalLines(originalImage, 0.65);
	imwrite("TestImages/DEBUG/VerticalContainerLines.png", verticalContainerBoundaries);
	
	// If the container is not centered perfectly in front of the camera (which is likely),
	// then the vertical edges of the container will appear as thick lines in the background subtracted image.
	Rect verticalContainerRegion = computeInnermostRectangle(verticalContainerBoundaries);
	
	return verticalContainerRegion;
}

Rect computeHorizontalContainerBoundaries(Mat verticalContainerImage)
{
	Mat horizontalContainerBoundaries = findLargestHorizontalLines(verticalContainerImage, 0.9);
	imwrite("TestImages/DEBUG/HorizontalContainerLines.png", horizontalContainerBoundaries);
	Rect horizontalContainerRegion = computeInnermostRectangle(horizontalContainerBoundaries);

	return horizontalContainerRegion;
}

//TODO: Elaborate via comments how this is really the gel and not the container.
Rect computeGelRegion(Mat originalImage)
{
	Rect verticalContainerLines = computeVerticalContainerBoundaries(originalImage);
	Mat verticalContainerImage = originalImage(verticalContainerLines);
	imwrite("TestImages/DEBUG/VerticalContainerImage.png", verticalContainerImage);

	auto elem = getStructuringElement(MORPH_RECT, Size(11, 9));
	Mat tmpVerticalContainerImage;
	morphologyEx(verticalContainerImage, tmpVerticalContainerImage, MORPH_CLOSE, elem);

	Rect horizontalContainerLines = computeHorizontalContainerBoundaries(tmpVerticalContainerImage);
	Mat containerImage = verticalContainerImage(horizontalContainerLines);
	imwrite("TestImages/DEBUG/ContainerImage.png", containerImage);

	Rect gelRegionWRToriginal = Rect(verticalContainerLines.x + horizontalContainerLines.x, verticalContainerLines.y + horizontalContainerLines.y, horizontalContainerLines.width, horizontalContainerLines.height);

	return gelRegionWRToriginal;
}

Rect computeCropRegion(Mat img)
{
	// Draw original image.
	Mat orig = img.clone();
	imwrite("TestImages/1foregroundORImage.png", orig);

	// Compute image cropped to the gel boundary.
	Rect gelRegion = computeGelRegion(img);
	imwrite("TestImages/2highlightedGel.png", drawRedRectOnImage(orig, gelRegion, 3));
	Mat containerImage = img(gelRegion);

	// Find the root system within the gel.
	keepOnlyLargestContour(containerImage);
	Mat rootSystem = containerImage;
	imwrite("TestImages/DEBUG/PossibleRootSystem.png", containerImage);

	int rowPositionWithMaximumBlackPixels = computeRowWithMaximumBlackPixels(containerImage);
	Rect rootRectangle = computeMaximumRootExtents(containerImage, rowPositionWithMaximumBlackPixels);
	Rect rootRectangleWRToriginal = Rect(rootRectangle.x + gelRegion.x, rootRectangle.y + gelRegion.y, rootRectangle.width, rootRectangle.height);
	Mat rootImage = containerImage(rootRectangle);
	imwrite("TestImages/DEBUG/AA_FINALTEST_ROOT.png", rootImage);
	imwrite("TestImages/3highlightedRoots.png", drawRedRectOnImage(orig, rootRectangleWRToriginal, 3));

	return rootRectangleWRToriginal;
}

void cropOriginalImages(vector<Mat> originalImages, Rect cropRegion)
{
	const string outputDirectory = "TestImages/CroppedImages/";
	const string outputExtension = ".png";
	int i = 1;

	for (auto &image : originalImages)
	{
		stringstream ss;
		ss << outputDirectory << i++ << outputExtension;
		string str = ss.str();
		imwrite(str, image(cropRegion));
		ss.clear();
	}
}

int main(int argc, char** argv)
{
	if (argv[1] == 0)
	{
		cerr << "No starting file specified." << endl;
		cerr << "Exiting..." << endl;
		return EXIT_FAILURE;
	}

	if (!FileUtilities::fileExists(argv[1]))
	{
		cerr << "Specified starting file doesn't exist: " << argv[1] << endl;
		cerr << "Exiting..." << endl;
		return EXIT_FAILURE;
	}

	vector<Mat> originalImages = ImageReader::readDataset(argv[1]);

	vector<Mat> preprocess_Images = imagePreprocess(originalImages);
	vector<Mat> foregroundImages = computeForegroundImages(originalImages);
	
	Mat orImage = or_op(foregroundImages);

	Rect cropRegion = computeCropRegion(orImage);

	cropOriginalImages(originalImages, cropRegion);

	return EXIT_SUCCESS;
}
