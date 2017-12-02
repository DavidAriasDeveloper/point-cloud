/*
 *
 *  Generacion de nube de puntos a traves de imagenes estereoscopicas version CPU
 *  Por: Luis David Arias Manjarrez
 *  Adaptacion de: Victor  Eruhimov
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

void writeMatToFile(cv::Mat& m, const char* filename)
{
    ofstream fout(filename);

    if(!fout)
    {
        std::cout<<"File Not Opened"<<std::endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<"\t";
        }
        fout<<std::endl;
    }

    fout.close();
}

void save(const Mat& image3D, const std::string& fileName)
{
	CV_Assert(image3D.type() == CV_32FC3 && !image3D.empty());
	CV_Assert(!fileName.empty());

	std::ofstream outFile(fileName);

	if (!outFile.is_open())
	{
		std::cout << "ERROR: Could not open " << fileName << std::endl;
		return;
	}

	for (int i = 0; i < image3D.rows; i++)
	{
		const cv::Vec3f* image3D_ptr = image3D.ptr<cv::Vec3f>(i);

		for (int j = 0; j < image3D.cols; j++)
		{
			outFile << image3D_ptr[j][0] << " " << image3D_ptr[j][1] << " " << image3D_ptr[j][2] << std::endl;
		}
	}

	outFile.close();
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    float value = mat.at<float>(100,100);

    printf("%fn", value);

    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
          Vec3f point = mat.at<Vec3f>(y, x);
          if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
          //fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

int main(int argc, char** argv)
{
    std::string imgLeft_filename = "";
    std::string imgRight_filename = "";

    std::string intrinsic_filename = "";
    std::string extrinsic_filename = "";
    std::string disparity_filename = "aloe_disparity.jpg";
    std::string point_cloud_filename = "point_cloud.ply";

    int width, height;
    double f;
    int windowSize, minDisparity, numberOfDisparities;
    bool no_display = false;

    cv::CommandLineParser parser(argc, argv,
        "{@arg1||}{@arg2||}");

    imgLeft_filename = parser.get<std::string>(0);
    imgRight_filename = parser.get<std::string>(1);

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    if( imgLeft_filename.empty() || imgRight_filename.empty() )
    {
        printf("Error de parámetro en linea de comandos: Se deben especificar las dos imagenes\n");
        return -1;
    }

    Mat img1 = imread(imgLeft_filename);
    Mat img2 = imread(imgRight_filename);

    Mat imgLeft,imgRight;
    pyrDown(img1,imgLeft);
    pyrDown(img2,imgRight);

    width = imgLeft.cols;
    height = imgLeft.rows;
    windowSize = 3;
    minDisparity = 16;
    numberOfDisparities = 112 - minDisparity;
    f= 0.8 * width;

    int cn = imgLeft.channels();
    Size img_size = imgLeft.size();

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(
      minDisparity,
      numberOfDisparities,
      16.0,
      8*cn*windowSize*windowSize,
      32*cn*windowSize*windowSize,
      0,
      1,
      10,
      100,
      32,
      StereoSGBM::MODE_HH
    );

    // sgbm->setMinDisparity(minDisparity);
    // sgbm->setNumDisparities(numberOfDisparities);
    // sgbm->setBlockSize(16);
    // sgbm->setP1(8*cn*windowSize*windowSize);
    // sgbm->setP2(32*cn*windowSize*windowSize);
    // sgbm->setDisp12MaxDiff(1);
    // sgbm->setUniquenessRatio(10);
    // sgbm->setSpeckleWindowSize(100);
    // sgbm->setSpeckleRange(32);

    Mat Q = cv::Mat(4,4,CV_32F);
    Q.at<double>(0,0)=1.0;
    Q.at<double>(0,1)=0.0;
    Q.at<double>(0,2)=0.0;
    Q.at<double>(0,3)=-0.5*width; //cx
    Q.at<double>(1,0)=0.0;
    Q.at<double>(1,1)=-1.0;
    Q.at<double>(1,2)=0.0;
    Q.at<double>(1,3)=0.5*height;  //cy
    Q.at<double>(2,0)=0.0;
    Q.at<double>(2,1)=0.0;
    Q.at<double>(2,2)=0.0;
    Q.at<double>(2,3)=-1.0*f;  //Focal
    Q.at<double>(3,0)=0.0;
    Q.at<double>(3,1)=0.0;
    Q.at<double>(3,2)=1.0;    //1.0/BaseLine
    Q.at<double>(3,3)=0.0;    //cx - cx'

    Mat disp, disp8;

    int64 t = getTickCount();
    sgbm->compute(imgLeft, imgRight, disp);

    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));

    if( !no_display )
    {
        namedWindow("left", 1);
        imshow("left", imgLeft);
        namedWindow("disparity", 0);
        imshow("disparity", disp8);
        printf("Presione una tecla para continuar...");
        fflush(stdout);
        waitKey();
        printf("\n");
    }

    if(!disparity_filename.empty()){
        imwrite(disparity_filename, disp8);
    }

    if(!point_cloud_filename.empty())
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        reprojectImageTo3D(disp8, xyz, Q);
        
        writeMatToFile(mask,"test.txt");
        //saveXYZ(point_cloud_filename.c_str(), xyz);

        printf("\n");
    }

    return 0;
}
