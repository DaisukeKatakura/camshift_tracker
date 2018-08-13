#include "camshift_tracker.h"

cv::Mat CamshiftTracker::Shist_;

void CamshiftTracker::setImage(const cv::Mat& hsv)
{
	hsv_ = hsv;
}

void CamshiftTracker::setColorParam(int vmin, int vmax, int smin, int hsize, float hmin, float hmax)
{
	vmin_       = vmin;
	vmax_       = vmax;
	smin_       = smin;
	hsize_      = hsize;
	hranges_[0] = hmin;
	hranges_[1] = hmax;
	phranges_   = hranges_;
}

void CamshiftTracker::masking()
{
	cv::Scalar lower(  0, smin_, MIN(vmin_, vmax_));
	cv::Scalar upper(180,  256, MAX(vmin_, vmax_));
	cv::inRange(hsv_, lower, upper, mask_);
}

inline bool CamshiftTracker::FirstTracking(int state)
{
	if (state == -1 ){ return true;}
	else { return false;}
}

cv::Mat CamshiftTracker::getMask()
{
	return mask_.clone(); 
}

void CamshiftTracker::extractChannel(const int ch[])
{
	hue_ = cv::Mat(hsv_.size(), hsv_.depth());
	cv::mixChannels(&hsv_, 1, &hue_, 1, ch, 1); 
}

cv::Mat CamshiftTracker::getHistogramImage()
{
	cv::Mat hist_img = cv::Mat::zeros(200, 320, CV_8UC3);
	int binW = hist_img.cols / hsize_;
	cv::Mat buf(1, hsize_, CV_8UC3);
	for (int i = 0; i < hsize_; i++){
	   	buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / hsize_), 255, 255);
	}
	cv::cvtColor(buf, buf, cv::COLOR_HSV2BGR);

	//generate histogram image
	for (int i = 0; i < hsize_; i++) {
		int val = cv::saturate_cast<int>(Shist_.at<float>(i)*hist_img.rows / 255);
		cv::rectangle(hist_img, cv::Point(i*binW, hist_img.rows), cv::Point((i + 1)*binW, hist_img.rows - val),
							cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
	}
	return hist_img.clone();
}

void CamshiftTracker::tracking(TrackObject& trk_obj, cv::RotatedRect& trackBox)
{
	cv::Rect trackWindow;
	if (FirstTracking(trk_obj.state)) {
		trackWindow = trk_obj.selection;

		cv::Mat roi(hue_, trackWindow);
		cv::Mat mask_roi(mask_, trackWindow);
		cv::calcHist(&roi, 1, 0, mask_roi, Shist_, 1, &hsize_, &phranges_);
		cv::normalize(Shist_, Shist_, 0, 255, cv::NORM_MINMAX);
		trk_obj.state = 1;
	}else {
		trackWindow = trackBox.boundingRect();
	}

	cv::Mat backproj;
	cv::calcBackProject(&hue_, 1, 0, Shist_, backproj, &phranges_);
	backproj &= mask_;

	trackBox = cv::CamShift(backproj, trackWindow, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
}
