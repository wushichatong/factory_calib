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

    std::vector<std::vector<float>> pts2d;                    // detected image points
    std::vector<std::vector<float>> obj_pts;                                              // coordinates in the car coordinate system

    if (display_img && detection_success)
    {
        cv::imwrite(image_path.c_str(), image);
        float tag_dist = 0.0352 * 1.3f;
        for (size_t i = 0; i < detections.size(); i++)
        {
            // also highlight in the image
            detections[i].draw(image);
            std::cout << "tag" << i << ": " << detections[i].cxy.first << " " << detections[i].cxy.second << std::endl;
            pts2d.push_back({detections[i].cxy.first, detections[i].cxy.second});
            obj_pts.push_back({i % 6 * tag_dist, i / 6 * tag_dist, 0.0f});
        }
        // cv::imshow(image_path.c_str(), image);
        // cv::waitKey();
        cv::imwrite("apriltags_detection.png", image);
    }

    bool real_data = true;
    // Camera to Car extrinsic
    if (detection_success && real_data)
    {
        std::vector<std::vector<double>> intrinsic; // Camera intrinsic
        intrinsic.push_back({397.22037047810716, 0, 312.2723218982026});
        intrinsic.push_back({0, 397.3364019128068, 284.32034664388965});
        intrinsic.push_back({0.0, 0.0, 1.0});
        std::vector<double> distortion = {-0.43411867618874456, 0.16386180338444847, -0.0014287457907101915, 0.00356140326695589};                   // Camera distortion
        // 轴角表示
        std::vector<float> rvec, tvec;                            // calibration result
        rvec = {0.01, 0.01, 0.01};
        tvec = {0.12, -0.0517, 0.348};
        solveCamPnP(obj_pts, pts2d, intrinsic, distortion, rvec, tvec); // solver
        printf("rvec");
        for(size_t i = 0; i < rvec.size(); i++){
            printf(" : %f", rvec[i]);
        }
        printf("\n");
        printf("tvec");
        for(size_t i = 0; i < tvec.size(); i++){
            printf(" : %f", tvec[i]);
        }
        printf("\n");
    }

    return 0;
}
