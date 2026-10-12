#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/bgsegm.hpp>
#include <iostream>
#include <iomanip>

int main(int argc, char **argv) {
    cv::Mat frame_ref, frame_mask, frame_mask_mod;

    //auto capture = cv::VideoCapture("highway.mp4");
    auto capture = cv::VideoCapture("https://1510162109.rsc.cdn77.org/live/EbW613.stream/chunklist.m3u8");
    auto subtractor = cv::createBackgroundSubtractorMOG2(100, 100, false);

    int roi_x = 175;
    int roi_y = 368;

    int roi_width = 150;
    int roi_height = 150;

    while (capture.read(frame_ref)) {
        if (cv::waitKey(40) & 0xFF == 27) {
            break;
        }

        subtractor->apply(frame_ref, frame_mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2,2));
        cv::morphologyEx(frame_mask, frame_mask_mod, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(frame_mask_mod, frame_mask_mod, cv::MORPH_CLOSE, kernel);
        // Przykład: cv::MORPH_ERODE

        cv::Mat frame_roi = frame_mask_mod(cv::Rect(roi_x, roi_y, roi_width, roi_height));



        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(frame_roi, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        cv::rectangle(frame_ref, cv::Point(roi_x, roi_y),cv::Point(roi_x+roi_width, roi_y+roi_height),cv::Scalar(0, 0, 255), 3);

        int car_number=0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > 100) {
                car_number++;
                cv::Rect rect = cv::boundingRect(contours[i])+cv::Point(roi_x, roi_y);
                cv::rectangle(frame_ref, rect, cv::Scalar(0, 255, 0), 3);
                cv::drawContours(frame_ref, contours, -1, cv::Scalar(255, 0, 0),
                     2, cv::LINE_AA, hierarchy, 3, cv::Point(roi_x, roi_y));
            }
        }
        cv::putText(frame_ref, "Ilosc pojazdow: " + std::to_string(car_number), cv::Point(800, 459), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        cv::imshow("original", frame_ref);
        cv::imshow("mask", frame_mask);
        cv::imshow("mask:Roi", frame_roi);
        cv::imshow("mask_modified", frame_mask_mod);
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}