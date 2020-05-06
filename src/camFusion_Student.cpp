
#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor,
                         cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx,
                         cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto itrLidar = lidarPoints.begin(); itrLidar != lidarPoints.end(); ++itrLidar)
    {
        // assemble vector for matrix-vector-multiplication
        // Set Lidar point onto the homogeneous coordinates
        X.at<double>(0, 0) = itrLidar->x;
        X.at<double>(1, 0) = itrLidar->y;
        X.at<double>(2, 0) = itrLidar->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        // pointers to all bounding boxes which enclose the current Lidar point
        vector<vector<BoundingBox>::iterator> enclosingBoxes;
        for (vector<BoundingBox>::iterator itrBox = boundingBoxes.begin(); itrBox != boundingBoxes.end(); ++itrBox)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*itrBox).roi.x + shrinkFactor * (*itrBox).roi.width / 2.0;
            smallerBox.y = (*itrBox).roi.y + shrinkFactor * (*itrBox).roi.height / 2.0;
            smallerBox.width = (*itrBox).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*itrBox).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(itrBox);
            }

        } // eof loop over all bounding boxes

        // check whether the point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*itrLidar);
            boundingBoxes[0].lidarPoints.push_back(*itrLidar);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto itrBox = boundingBoxes.begin(); itrBox != boundingBoxes.end(); ++itrBox)
    {
        // create randomized color for current 3D object
        cv::RNG rng(itrBox->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto itrLidar = itrBox->lidarPoints.begin(); itrLidar != itrBox->lidarPoints.end(); ++itrLidar)
        {
            // world coordinates
            float xw = (*itrLidar).x; // world position in m with x facing forward from sensor
            float yw = (*itrLidar).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", itrBox->boxID, (int)itrBox->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 0.5, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 0.5, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

cv::Rect shrinkBoundingBox(const BoundingBox &originalBoundingBox, const float &shrinkFactor)
{
    cv::Rect smallerBox;
    smallerBox.x = originalBoundingBox.roi.x + shrinkFactor * originalBoundingBox.roi.width / 2.0;
    smallerBox.y = originalBoundingBox.roi.y + shrinkFactor * originalBoundingBox.roi.height / 2.0;
    smallerBox.width = originalBoundingBox.roi.width * (1 - shrinkFactor);
    smallerBox.height = originalBoundingBox.roi.height * (1 - shrinkFactor);
    return smallerBox;
}

// associate a given bounding box with the keypoints boundingBoxPrev contains
void clusterKptMatchesWithROI(BoundingBox &boundingBoxCurr,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches)
{
    float shrinkFactor = 0.1; // Shrink bounding boxes 10% to reduce outliers
    cv::Rect smallerBoxCurr = shrinkBoundingBox(boundingBoxCurr, shrinkFactor);

    // Loop over each match to see if boudning box contains keypoints
    for (auto match : kptMatches)
    {
        int currKeyPointIdx = match.trainIdx;
        cv::Point currKeyPoint = kptsCurr.at(currKeyPointIdx).pt;
        if (smallerBoxCurr.contains(currKeyPoint))
        {
            boundingBoxCurr.kptMatches.push_back(match);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches,
                      double frameRate,
                      double &TTC)
{
    double dt = 1.0 / frameRate;
    // Compute distance ratios on every pair of keypoints, O(n^2) on the number of matches contained within the ROI
    vector<double> distRatios;
    for (auto itrFirstMatch = kptMatches.begin(); itrFirstMatch != kptMatches.end() - 1; ++itrFirstMatch)
    {
        // First points in previous and current frame
        cv::KeyPoint kptFirstCurr = kptsCurr.at(itrFirstMatch->trainIdx);
        cv::KeyPoint kptFirstPrev = kptsPrev.at(itrFirstMatch->queryIdx);

        for (auto itrSecondMatch = kptMatches.begin() + 1; itrSecondMatch != kptMatches.end(); ++itrSecondMatch)
        {
            cv::KeyPoint kptSecondCurr = kptsCurr.at(itrSecondMatch->trainIdx);
            cv::KeyPoint kptSecondPrev = kptsPrev.at(itrSecondMatch->queryIdx);

            // Compute distances for previous and current frame
            double distCurr = cv::norm(kptFirstCurr.pt - kptSecondCurr.pt);
            double distPrev = cv::norm(kptFirstPrev.pt - kptSecondPrev.pt);

            double minDist = 100.0; // Threshold the calculated distRatios by requiring a minimum current distance

            // Avoid division by zero and apply the threshold
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // Only continue if the vector of distRatios is not empty
    if (distRatios.size() == 0)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // Use the median to exclude outliers
    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = distRatios[distRatios.size() / 2];

    TTC = -dt / (1 - medianDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr,
                     double frameRate,
                     double &TTC)
{
    double dt = 1.0 / frameRate;
    double minXPrev = 1e9, minXCurr = 1e9;

    // Find closest distance to Lidar points
    for (auto itrPointsPrev = lidarPointsPrev.begin(); itrPointsPrev != lidarPointsPrev.end(); ++itrPointsPrev)
    {
        if (itrPointsPrev->x < minXPrev)
        {
            minXPrev = itrPointsPrev->x;
        }
    }

    for (auto itrPointsCurr = lidarPointsCurr.begin(); itrPointsCurr != lidarPointsCurr.end(); ++itrPointsCurr)
    {
        if (itrPointsCurr->x < minXCurr)
        {
            minXCurr = itrPointsCurr->x;
        }
    }

    // compute TTC from both measurements
    TTC = minXCurr * dt / (minXPrev - minXCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches,
                        DataFrame &prevFrame,
                        DataFrame &currFrame)
{
    // Loop over every bounding box in the previous frame
    for (auto boundingBoxPrev : prevFrame.boundingBoxes)
    {
        // Create a container for storing matches that are enclosed by this bounding box in previous frame
        vector<cv::DMatch> matchesEnclosedInPrev;

        // Check how many keypoints each bounding box encloses
        for (auto match : matches)
        {
            int prevKeyPointIdx = match.queryIdx;
            cv::Point prevKeyPoint = prevFrame.keypoints.at(prevKeyPointIdx).pt;

            if (boundingBoxPrev.roi.contains(prevKeyPoint))
            {
                matchesEnclosedInPrev.push_back(match);
            }
        }

        // Create a map that stores the IDs of bounding box in current frame which also enclose the same matches
        // as in previous frame.
        std::multimap<int, int> sameMatchesCurr;

        // Loop over each match that is found in this bounding box in previous frame,
        // check which bounding boxes in the current frame also enclose is.
        for (auto matchInPrev : matchesEnclosedInPrev)
        {
            int currKeyPointIdx = matchInPrev.trainIdx;
            for (auto boundingBoxCurr : currFrame.boundingBoxes)
            {
                cv::Point currKeyPoint = currFrame.keypoints.at(currKeyPointIdx).pt;
                if (boundingBoxCurr.roi.contains(currKeyPoint))
                {
                    sameMatchesCurr.insert({boundingBoxCurr.boxID, currKeyPointIdx});
                }
            }
        }

        // Check which bounding box in current frame has highest number of matches,
        // then it will assigned as the best bounding box.
        if (sameMatchesCurr.size() > 0)
        {
            int maxOccurance = 0;
            int boundingBoxBestMatchedIdx;

            for (auto sameMatchCurr : sameMatchesCurr)
            {
                int boxID = sameMatchCurr.first;
                size_t boxIdOccurance = sameMatchesCurr.count(boxID);
                if (boxIdOccurance > maxOccurance)
                {
                    maxOccurance = boxIdOccurance;
                    boundingBoxBestMatchedIdx = boxID;
                }
            }
            bbBestMatches.insert({boundingBoxPrev.boxID, boundingBoxBestMatchedIdx});
        }
    }
}
