/*
 * Copyright (C) 2021 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 */
#include "Eigen/Core"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "dataConvert.hpp"
#include "logging.hpp"

#include "calibration/pnp_solver.hpp"

#include "apriltags/Tag36h11.h"
#include "apriltags/TagDetector.h"

/**
 * @brief An example of extrinsic parameter calibration
 * NOTE: If the translation of the extrinsic parameter is known and very
 * accurate, you can only optimize the rotation, so that the accuracy of the
 * solved external parameter is higher.
 *
 */

char usage[] = {"[Usage]: ./bin/run_ex image  \n"
                "eg: ./bin/run_ex data/chessboard.jpg \n"};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "usage:" << usage;
        return -1;
    }

    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path);

    // convert image to gray vector
    cv::Mat gray, gray_img;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray_img, CV_32F);

    bool display_img = true;

    AprilTags::TagCodes m_tagCodes(AprilTags::tagCodes36h11);
    AprilTags::TagDetector *m_tagDetector =
        new AprilTags::TagDetector(m_tagCodes);
    vector<AprilTags::TagDetection> detections =
        m_tagDetector->extractTags(gray);
    bool detection_success = false;
    if (detections.size() == 36)
        detection_success = true;

    std::vector<cv::Point3f> objPts; // 世界坐标系点
    std::vector<cv::Point2f> imgPts; // 像素坐标系点
    if (display_img && detection_success)
    {
        cv::imwrite(image_path.c_str(), image);
        float tag_dist = 0.0352 * 1.3f;
        for (size_t i = 0; i < detections.size(); i++)
        {
            // also highlight in the image
            detections[i].draw(image);
            std::cout << "tag" << i << ": " << detections[i].cxy.first << " " << detections[i].cxy.second << std::endl;
            imgPts.push_back(cv::Point2f(detections[i].cxy.first, detections[i].cxy.second));
            objPts.push_back(cv::Point3f(i % 6 * tag_dist, i / 6 * tag_dist, 0.0f));
        }
        // cv::imshow(image_path.c_str(), image);
        // cv::waitKey();
        cv::imwrite("apriltags_detection.png", image);
    }

    bool real_data = true;
    // Camera to Car extrinsic
    if (detection_success && real_data)
    {

        cv::Mat rvec, tvec;
        cv::Matx33f cameraMatrix(
                                397.22037047810716, 0, 312.2723218982026,
                                0, 397.3364019128068, 284.32034664388965,
                                0,  0,  1);
        cv::Vec4f distParam(-0.43411867618874456, 0.16386180338444847, -0.0014287457907101915, 0.00356140326695589); // all 0?
        cv::solvePnP(objPts, imgPts, cameraMatrix, distParam, rvec, tvec);
        cv::Matx33d r;
        cv::Rodrigues(rvec, r);
        Eigen::Matrix3d wRo;
        wRo << r(0,0), r(0,1), r(0,2), r(1,0), r(1,1), r(1,2), r(2,0), r(2,1), r(2,2);

        Eigen::Matrix4d T; 
        T.topLeftCorner(3,3) = wRo;
        T.col(3).head(3) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
        T.row(3) << 0,0,0,1;

        std::cout<<"T: "<<T<<std::endl;

    }

    return 0;
}
