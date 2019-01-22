/*----------------------------------------------------------------------------*/
/* Copyright (c) 2017-2018 FIRST. All Rights Reserved.                        */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include "frc/WPILib.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cameraserver/CameraServer.h>
#include <frc/TimedRobot.h>
#include <wpi/raw_ostream.h>
#include <cscore_oo.h>
#include <cscore_cpp.h>
#include <cameraserver/CameraServer.h>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cs;
using namespace cv;
using namespace std;

/**
 * Uses the CameraServer class to automatically capture video from a USB webcam
 * and send it to the FRC dashboard without doing any vision processing. This is
 * the easiest way to get camera images to the dashboard. Just add this to the
 * RobotInit() method in your program.
 */
class Robot : public frc::TimedRobot {
 public:
  vector<Point> imageComparation(Mat object, Mat scene, int match_method, float peek_percent) { 
    Mat img_display; 
    scene.copyTo(img_display);
    // Create the result matrix
    int result_cols = scene.cols - object.cols + 1;
    int result_rows = scene.rows - object.rows + 1;

    Mat result(result_cols, result_rows, CV_32FC1);

    // match scene with template
    matchTemplate(scene, object, result, match_method);
    imshow("matched_template", result);

    //normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    imshow("normalized", result);

    // Localizing the best match with minMaxLoc
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;

    // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
    {
        matchLoc = minLoc;
        //threshold(result, result, 0.1, 1, CV_THRESH_BINARY_INV);
        threshold(result, result, 0.1, 1, CV_THRESH_BINARY_INV);
        //imshow("threshold_1", result);
    }
    else
    {
        matchLoc = maxLoc;
        threshold(result, result, 0.9, 1, CV_THRESH_TOZERO);
        // imshow("threshold_2", result);
    }

    vector<Point> res;
    maxVal = 1.f;
    while (maxVal > peek_percent)
    {
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        if (maxVal > peek_percent)
        {
            rectangle(result, Point(maxLoc.x - object.cols / 2, maxLoc.y - object.rows / 2), Point(maxLoc.x + object.cols / 2, maxLoc.y + object.rows / 2), Scalar::all(0), -1);
            res.push_back(maxLoc);
            printf("\n%d:%d\n", maxLoc.x, maxLoc.y);
        }
    }

    return res;
  }

  static double angle( Point pt1, Point pt2, Point pt0 )
  {
      double dx1 = pt1.x - pt0.x;
      double dy1 = pt1.y - pt0.y;
      double dx2 = pt2.x - pt0.x;
      double dy2 = pt2.y - pt0.y;
      return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
  }

  void findSquares( const Mat& image, vector<vector<Point> >& squares )
  {
    int thresh = 50, N = 11;

      squares.clear();
      Mat pyr, timg, gray0(image.size(), CV_8U), gray;
      // down-scale and upscale the image to filter out the noise
      pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
      pyrUp(pyr, timg, image.size());
      vector<vector<Point> > contours;
      // find squares in every color plane of the image
      for( int c = 0; c < 1; c++ )
      {
          int ch[] = {c, 0};
          mixChannels(&timg, 1, &gray0, 1, ch, 1);
          // try several threshold levels
          for( int l = 0; l < N; l++ )
          {
              // hack: use Canny instead of zero threshold level.
              // Canny helps to catch squares with gradient shading
              if( l == 0 )
              {
                  // apply Canny. Take the upper threshold from slider
                  // and set the lower to 0 (which forces edges merging)
                  Canny(gray0, gray, 0, thresh, 5);
                  // dilate canny output to remove potential
                  // holes between edge segments
                  dilate(gray, gray, Mat(), Point(-1,-1));
              }
              else
              {
                  // apply threshold if l!=0:
                  //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                  gray = gray0 >= (l+1)*255/N;
              }
              // find contours and store them all as a list
              findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
              vector<Point> approx;
              // test each contour
              for( size_t i = 0; i < contours.size(); i++ )
              {
                  // approximate contour with accuracy proportional
                  // to the contour perimeter
                  approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
                  // square contours should have 4 vertices after approximation
                  // relatively large area (to filter out noisy contours)
                  // and be convex.
                  // Note: absolute value of an area is used because
                  // area may be positive or negative - in accordance with the
                  // contour orientation
                  if( approx.size() == 4 &&
                      fabs(contourArea(approx)) > 1000 &&
                      isContourConvex(approx) )
                  {
                      double maxCosine = 0;
                      for( int j = 2; j < 5; j++ )
                      {
                          // find the maximum cosine of the angle between joint edges
                          double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                          maxCosine = MAX(maxCosine, cosine);
                      }
                      // if cosines of all angles are small
                      // (all angles are ~90 degree) then write quandrange
                      // vertices to resultant sequence
                      if( maxCosine < 0.3 )
                          squares.push_back(approx);
                  }
              }
          }
      }
  }

  void drawSquares( Mat& image, const vector<vector<Point> >& squares )
  {
      for( size_t i = 0; i < squares.size(); i++ )
      {
          const Point* p = &squares[i][0];
          int n = (int)squares[i].size();
          polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
      }
  }


  static void findLines() {

      cv::Mat input = cv::imread("/Users/Basil/Downloads/main.png");

      // convert to HSV color space
      cv::Mat hsvImage;
      cv::cvtColor(input, hsvImage, CV_BGR2HSV);

      // split the channels
      std::vector<cv::Mat> hsvChannels;
      cv::split(hsvImage, hsvChannels);

      // hue channels tells you the color tone, if saturation and value aren't too low.

      // red color is a special case, because the hue space is circular and red is exactly at the beginning/end of the circle.
      // in literature, hue space goes from 0 to 360 degrees, but OpenCV rescales the range to 0 up to 180, because 360 does not fit in a single byte. Alternatively there is another mode where 0..360 is rescaled to 0..255 but this isn't as common.
      int hueValue = 0; // red color
      int hueRange = 15; // how much difference from the desired color we want to include to the result If you increase this value, for example a red color would detect some orange values, too.

      int minSaturation = 50; // I'm not sure which value is good here...
      int minValue = 50; // not sure whether 50 is a good min value here...

      cv::Mat hueImage = hsvChannels[0]; // [hue, saturation, value]

      // is the color within the lower hue range?
      cv::Mat hueMask;
      cv::inRange(hueImage, hueValue - hueRange, hueValue + hueRange, hueMask);

      // if the desired color is near the border of the hue space, check the other side too:
      // TODO: this won't work if "hueValue + hueRange > 180" - maybe use two different if-cases instead... with int lowerHueValue = hueValue - 180
      if (hueValue - hueRange < 0 || hueValue + hueRange > 180)
      {
          cv::Mat hueMaskUpper;
          int upperHueValue = hueValue + 180; // in reality this would be + 360 instead
          cv::inRange(hueImage, upperHueValue - hueRange, upperHueValue + hueRange, hueMaskUpper);

          // add this mask to the other one
          hueMask = hueMask | hueMaskUpper;
      }

      // now we have to filter out all the pixels where saturation and value do not fit the limits:
      cv::Mat saturationMask = hsvChannels[1] > minSaturation;
      cv::Mat valueMask = hsvChannels[2] > minValue;

      hueMask = (hueMask & saturationMask) & valueMask;

      cv::imshow("desired color", hueMask);

      // now perform the line detection
      std::vector<cv::Vec4i> lines;
      cv::HoughLinesP(hueMask, lines, 1, CV_PI / 360, 50, 50, 10);

      // draw the result as big green lines:
      for (unsigned int i = 0; i < lines.size(); ++i)
      {
          cv::line(input, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 255, 0), 5);
      }


      //cv::imwrite("C:/StackOverflow/Output/coloredLines_mask.png", hueMask);
      cv::imwrite("/Users/Basil/Downloads/result.jpeg", input);

      return;
  }

  void deskew(const char* filename, double angle)
  {
      cv::Mat img = cv::imread(filename, 0);

      Point2f src_center(img.cols/2.0F, img.rows/2.0F);
      Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
      Mat rotated;
      warpAffine(img, rotated, rot_mat, img.size(), cv::INTER_CUBIC);

      imwrite("/Users/Basil/Downloads/result.jpeg",rotated);

  }

static void findLines1(char *filename) {
  Mat src = imread(filename, 0);
  Mat dst, cdst;
  Canny(src, dst, 50, 200, 3);
  cvtColor(dst, cdst, CV_GRAY2BGR);

  vector<Vec4i> lines;
  HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
  }

  imwrite("/Users/Basil/Downloads/result.jpeg",cdst);
}

  void makeSquare(char *fileName, char *result) {
    cv::Mat img = cv::imread(fileName, 0);
    vector<vector<Point> > squares;
    findSquares(img, squares);
    drawSquares(img, squares);
    imwrite(result, img);
  }
  
  void makeGrid(Mat img) {
    int width= img.size().width;
    int height= img.size().height;

    int midWidth = width /2;
    int midHeight = height /2;

    int dist=50;
    
    for (int uHeight = midHeight; uHeight>0; uHeight -= dist)
      cv::line(img, Point(0, uHeight), Point(width, uHeight), cv::Scalar(255, 0, 0));

    for (int bHeigth = midHeight; bHeigth<height; bHeigth += dist)
      cv::line(img, Point(0, bHeigth), Point(width, bHeigth), cv::Scalar(255, 0, 0));

    cv::line(img, Point(0, midHeight), Point(width, midHeight), cv::Scalar(0, 0, 255));

    //for (int i = 0; i<height; i += dist)
    //  cv::line(img, Point(0, i), Point(width, i), cv::Scalar(255, 0, 0));

    for (int lWidth = midWidth; lWidth>0; lWidth -= dist) 
      cv::line(img, Point(lWidth, 0), Point(lWidth, height), cv::Scalar(255, 0, 0));

    for (int rWidth = midWidth; rWidth<width; rWidth += dist) 
      cv::line(img, Point(rWidth, 0), Point(rWidth, height), cv::Scalar(255, 0, 0));

    cv::line(img, Point(midWidth, 0), Point(midWidth, height), cv::Scalar(0, 0, 255));
    //for (int i = 0; i<width; i += dist) 
    //  cv::line(img, Point(i, 0), Point(i, height), cv::Scalar(255, 0, 0));
  }

  Mat makeGrid(char *fileName, char *result) {
    cv::Mat img = cv::imread(fileName, 0);
    makeGrid(img);
    imwrite(result, img);
  }

  void RobotInit() override {
    //VideoCapture cap(0); // open the default camera
    VideoCapture cap("http://192.168.1.136:8081/cgi-bin/mjpg/video.cgi?&subtype=1");
    if(!cap.isOpened())  // check if we succeeded
        return;

    Mat edges;
    namedWindow("edges",1);
    for(;;) {
      Mat frame;
      cap >> frame;
      makeGrid(frame);
      imshow("edges", frame);
      if(waitKey(30) >= 0) break;
    }
    //this->makeGrid((char *)"/Users/Basil/Downloads/main.png", (char *)"/Users/Basil/Downloads/grid.png");
    //this->makeSquare((char *)"/Users/Basil/Downloads/main.png", (char *)"/Users/Basil/Downloads/square.png");
      
    wpi::errs() << "Vision only available on Linux.\n";
    wpi::errs().flush();
//#endif
  }
};

#ifndef RUNNING_FRC_TESTS
int main() { return frc::StartRobot<Robot>(); }
#endif
