#include <opencv2/opencv.hpp>

int main() {
  // Load the input image
  cv::Mat image = cv::imread("i1.jpg");

  // Convert the image to grayscale
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  // Load the Haar cascade classifier for face detection
  cv::CascadeClassifier face_cascade;
  face_cascade.load("haarcascade_frontalface_default.xml");

  // Detect faces in the image
  std::vector<cv::Rect> faces;
  face_cascade.detectMultiScale(gray, faces, 1.3, 5);

  // Draw a rectangle around each face
  for (auto& face : faces) {
    cv::rectangle(image, face, cv::Scalar(255, 0, 0), 2);
  }

  // Display the resulting image
  cv::imshow("Faces", image);
  cv::waitKey(0);
  cv::destroyAllWindows();

  // Load the input image
  image = cv::imread("input.jpg");

  // Convert the image to grayscale
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  // Blur the image to reduce high frequency noise
  cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

  // Perform Canny edge detection
  cv::Mat edges;
  cv::Canny(gray, edges, 50, 150, 3);

  // Run Hough transform on the edge image
  std::vector<cv::Vec2f> lines;
  cv::HoughLines(edges, lines, 1, CV_PI / 180, 200);

  // Iterate over the output lines and draw them on the image
  for (auto& line : lines) {
    float rho = line[0], theta = line[1];
    cv::Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 2);
  }

  // Display the resulting image
  cv::imshow("Shapes", image);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}
