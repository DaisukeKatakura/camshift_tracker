#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

//tracked object structure
struct TrackObject {
	int state;
	bool selectObject;   
	cv::Point origin; 
	cv::Rect selection; 
};

class CamshiftTracker
{
	private:
		static cv::Mat Shist_;
		int vmin_, vmax_, smin_;
		int hsize_;
		float hranges_[];
		const float* phranges_;
		TrackObject trk_obj_;
		cv::Mat hsv_, mask_, hue_;
	public:
		void setImage(const cv::Mat& hsv);
		void setColorParam(int vmin, int vmax, int smin, int hsize, float hmin, float hmax);
		void masking();
		void extractChannel(const int ch[]);
		void tracking(TrackObject& trk_obj, cv::RotatedRect& trackBox);
		cv::Mat getMask();
		cv::Mat getHistogramImage();
		inline bool FirstTracking(int state);
};
