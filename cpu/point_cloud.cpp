/*
 *
 *  Generacion de nube de puntos a traves de imagenes estereoscopicas version CPU
 *  Por: Luis David Arias Manjarrez
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

//Funcion para escribir fichero .ply
static void saveXYZ(const char* filename, const Mat& points, const Mat& colors)
{
   const double max_z = 1.0e4;//Maxima profundidad
   int point_counter = 0;//Contador de puntos procesados
   ostringstream body;//Cuerpo del fichero .ply
   ostringstream header;//Encabezado del fichero .ply
   for(int y = 0; y < points.rows; y++)
   {
       for(int x = 0; x < points.cols; x++)
       {
         Vec3f point = points.at<Vec3f>(y, x);
         Vec3b color = colors.at<Vec3b>(y, x);
         if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;//Limites de alcance de la proyeccion
         body<<point[0]<<" "<<point[1]<<" "<<point[2]<<" "<<std::to_string(color[0])<<" "<<std::to_string(color[1])<<" "<<std::to_string(color[2])<<"\n";//Los 3 primeros valores son coordenadas, los 3 ultimos son identificadores de color
         point_counter++;
       }
   }
   FILE* fp = fopen(filename, "wb");
   header<<"ply\nformat ascii 1.0\nelement vertex "<<point_counter<<"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
   std::string header_string = header.str();
   std::string body_string = body.str();
   //Conversion de compatibilidad para fprintf
   char* header_char = new char[header_string.length() + 1];
   char* body_char = new char[body_string.length() + 1];

   copy(header_string.c_str(), header_string.c_str() + header_string.length() + 1, header_char);
   copy(body_string.c_str(), body_string.c_str() + body_string.length() + 1, body_char);

   fprintf(fp,"%s",header_char);
   fprintf(fp,"%s",body_char);
   fclose(fp);
}

int main(int argc, char** argv)
{
    std::string imgLeft_filename = "";
    std::string imgRight_filename = "";
    std::string disparity_filename = "aloe_disparity.jpg";
    std::string point_cloud_filename = "point_cloud.ply";

    int width, height, cn;
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

    //Lectura de imagenes
    Mat img1 = imread(imgLeft_filename);
    Mat img2 = imread(imgRight_filename);
    Mat imgLeft,imgRight;
    pyrDown(img1,imgLeft);
    pyrDown(img2,imgRight);

    //Definicion de parametros
    Size img_size = imgLeft.size();
    width = imgLeft.cols;
    height = imgLeft.rows;
    cn = imgLeft.channels();
    windowSize = 3;
    minDisparity = 16;
    numberOfDisparities = 112 - minDisparity;
    f= 0.8 * width;

    Matx44d Q = cv::Matx44d(
        1.0, 0.0, 0.0, -0.5*width,
        0.0, -1.0, 0.0, 0.5*height,
        0.0, 0.0, 0.0, -1.0*f,
        0.0, 0.0, 1.0, 0/*(CX - CX) / baseLine*/
    );

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
      StereoSGBM::MODE_SGBM
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

    Mat disp8(imgLeft.size(), CV_8U);
    Mat disparity(imgLeft.size(), CV_32F);

    int64 t = getTickCount();
    //Calculo de disparidad
    sgbm->compute(imgLeft, imgRight, disparity);
    t = getTickCount() - t;
    printf("CPU: Tiempo transcurrido: %fms\n", t*1000/getTickFrequency());

    disparity.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));


    if( !no_display )
    {
        //namedWindow("left", 1);
        //imshow("left", imgLeft);
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
        printf("Generando nube de puntos...");
        fflush(stdout);
        //Matrices de posiciones y colores
        Mat xyz,colors;
        //Generacion de nube de puntos a partir de mapa de disparidad
        reprojectImageTo3D(disp8, xyz, Q);
        cvtColor(imgLeft,colors, COLOR_BGR2RGB);
        //Exportacion de archivo ply
        saveXYZ(point_cloud_filename.c_str(), xyz, colors);

        printf("\n");
    }

    return 0;
}
