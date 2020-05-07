
#ifndef VISUALIZATIONHELPER_H_
#define VISUALIZATIONHELPER_H_

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "dataStructures.h"

void visualizeBoundingBox(cv::Mat &targetImg, BoundingBox &box, std::string windowName)
{
    cv::Mat visImg = targetImg.clone();
    cv::rectangle(visImg,
                  cv::Point(box.roi.x, box.roi.y),
                  cv::Point(box.roi.x + box.roi.width,
                            box.roi.y + box.roi.height),
                  cv::Scalar(0, 255, 0), 2);

    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, visImg);
    std::cout << "Press key to continue to next frame" << std::endl;
    cv::waitKey(0);
};

#endif // VISUALIZATIONHELPER_H_
