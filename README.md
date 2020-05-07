# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Writeup for submission
<p align="center">
  <img  src="https://github.com/paulyehtw/SFND_3D_Object_Tracking/blob/master/results/animated_result.gif">
</p>

### FP.1 Match 3D Objects
in _camFusion_Student.cpp_ : implement `matchBoundingBoxes`
```
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

```

### FP.2 Compute Lidar-based TTC
in _camFusion_Student.cpp_ : implement `computeTTCLidar`
```
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
```

### FP.3 Associate Keypoint Correspondences with Bounding Boxes
in _camFusion_Student.cpp_ : implement `clusterKptMatchesWithROI`
```
void clusterKptMatchesWithROI(BoundingBox &boundingBoxCurr,
                              BoundingBox &boundingBoxPrev,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches)
{
    float shrinkFactor = 0.15; //Shrink bounding boxes 10 % to reduce outliers
    std::vector<cv::DMatch> kptMatches_roi;
    cv::Rect smallerBoxCurr = shrinkBoundingBox(boundingBoxCurr, shrinkFactor);
    cv::Rect smallerBoxPrev = shrinkBoundingBox(boundingBoxPrev, shrinkFactor);

    // Loop over each match to see if boudning box contains keypoints
    for (auto match : kptMatches)
    {
        cv::KeyPoint currKeypoint = kptsCurr.at(match.trainIdx);
        auto currCvPoint = cv::Point(currKeypoint.pt.x, currKeypoint.pt.y);

        cv::KeyPoint prevKeypoint = kptsPrev.at(match.queryIdx);
        auto prevCvPoint = cv::Point(prevKeypoint.pt.x, prevKeypoint.pt.y);

        if (smallerBoxCurr.contains(currCvPoint) && smallerBoxPrev.contains(prevCvPoint))
        {
            kptMatches_roi.push_back(match);
        }
    }

    double meanKeypointDist = 0.0;
    for (auto kptMatch : kptMatches_roi)
    {
        meanKeypointDist += cv::norm(kptsCurr.at(kptMatch.trainIdx).pt - kptsPrev.at(kptMatch.queryIdx).pt);
    }
    // Calculate mean match distance
    if (kptMatches_roi.size() > 0)
    {
        meanKeypointDist /= kptMatches_roi.size();
    }
    else
    {
        return;
    }

    // Keep the match distance < dist_mean * 1.5
    double distThreshold = meanKeypointDist * 1.5;
    for (auto kptMatch : kptMatches_roi)
    {
        float dist = cv::norm(kptsCurr.at(kptMatch.trainIdx).pt - kptsPrev.at(kptMatch.queryIdx).pt);
        if (dist < distThreshold)
        {
            boundingBoxCurr.kptMatches.push_back(kptMatch);
            boundingBoxPrev.kptMatches.push_back(kptMatch);
        }
    }
}
```

### FP.4 Compute Camera-based TTC
in _camFusion_Student.cpp_ : implement `computeTTCCamera`
```
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
```

### FP.5 Performance Evaluation 1
In some of the frames like(11,12) and (16,17), TTC of Lidar can lead to negative as the procceding car is actually accelerating, thus the method should be adapted to this case.

### FP.6 Performance Evaluation 2
From the performance evaluation results from Midterm Project, Top 3 suggestion works pretty well, I choose **FAST/BRIEF** as the combination as it's fast and it detects relatively more keypoints. The TTC from camera is quite stable and realistic, also it is quite consistent with TTC from Lidar.
As for combination of for excample HARRIS/ORB, the TTC output from camera becomes really unstable. Sometimes TTC is `NaN` or `inf` because there are too few keypoints detected or matches found.
