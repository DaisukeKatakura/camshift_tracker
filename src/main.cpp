#include "camshift_tracker.h"
#include <iostream>

inline bool ObjectExists(int state)
{
	if (state != 0) { return  true; }
	else { return false;}
}

void MouseCb(int event, int x, int y, int flag, void *arg)
{
    TrackObject *data = (TrackObject*)arg;
 
    if (data->selectObject) { //while a mouse is being clicked (= user is selecting object)
        data->selection.x = MIN(x, data->origin.x);
        data->selection.y = MIN(y, data->origin.y);
        data->selection.width = std::abs(x - data->origin.x);
        data->selection.height = std::abs(y - data->origin.y);
    }
 
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            data->origin = cv::Point(x, y);
            data->selection = cv::Rect(x, y, 0, 0);
            data->selectObject = true;
            break;
        case cv::EVENT_LBUTTONUP:
            data->selectObject = false;
            if (data->selection.width > 0 && data->selection.height > 0){
			   	data->state = -1; 
			}
            break;
    }
}
 

int main(int argc, char *argv[])
{
	cv::RotatedRect trackBox;
	TrackObject trk_obj;
	trk_obj.state = 0;

	int vmin = 10, vmax = 256, smin=30;
	int hsize = 16;
	float hmin = 0, hmax = 180;
	
    cv::namedWindow("Histogram");
    cv::namedWindow("CamShift Demo");
	cv::namedWindow("Mask");
    cv::createTrackbar("Vmin", "CamShift Demo", &vmin, 256);
    cv::createTrackbar("Vmax", "CamShift Demo", &vmax, 256);
    cv::createTrackbar("Smin", "CamShift Demo", &smin, 256);
	cv::setMouseCallback("CamShift Demo", MouseCb, &trk_obj);

	cv::VideoCapture cap;
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
		std::exit(1);
    }

    while (1) {
		cv::Mat image;
		cap >> image;
        if (image.empty()){ break; }
 
		if (ObjectExists(trk_obj.state)) {
			cv::Mat hsv;
			cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV); 

			int ch[] = { 0, 0 };
			CamshiftTracker ct;
			ct.setColorParam(vmin, vmax, smin, hsize, hmin, hmax);
			ct.setImage(hsv);
			ct.extractChannel(ch);
			ct.masking();
			ct.tracking(trk_obj, trackBox);
			cv::Mat hist_img = ct.getHistogramImage();// for debug
			cv::Mat mask     = ct.getMask();

			cv::imshow("Histogram", hist_img);
			cv::imshow("Mask", mask);
			cv::ellipse(image, trackBox, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
		}
 
        cv::imshow("CamShift Demo", image);

		char c = cv::waitKey(10);
		if (c == 'q') { break; }
    }

 
    return 0;
}
