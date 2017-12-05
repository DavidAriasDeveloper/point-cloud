/*
*  Generacion de nube de puntos a traves de imagenes estereoscopicas version GPU
*  Por: Luis David Arias Manjarrez
*
*/

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/cudastereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <stdio.h>
#include <iostream>
#include <string>

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
   pyrDown(img1,img1);
   pyrDown(img2,img2);
   cuda::GpuMat d_left, d_right;
   Mat imgLeft,imgRight;
   cvtColor(img1, imgLeft, COLOR_BGR2GRAY);
   cvtColor(img2, imgRight, COLOR_BGR2GRAY);


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


   Ptr<cuda::StereoBM> bm;
   bm = cuda::createStereoBM(numberOfDisparities);
   bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);

   Mat disp8(imgLeft.size(), CV_8U);
   cuda::GpuMat d_disp8(imgLeft.size(), CV_8U);

   int64 t = getTickCount();
   d_left.upload(imgLeft);
   d_right.upload(imgRight);

   //Calculo de disparidad
   bm->compute(d_left, d_right, d_disp8);

   d_disp8.download(disp8);
   t = getTickCount() - t;
   printf("GPU: Tiempo transcurrido: %fms\n", t*1000/getTickFrequency());


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
       Mat xyz, colors;
       //Generacion de nube de puntos a partir de mapa de disparidad
       reprojectImageTo3D(disp8, xyz, Q);
       cvtColor(img1,colors, COLOR_BGR2RGB);
       //Exportacion de archivo ply
       saveXYZ(point_cloud_filename.c_str(), xyz, colors);

       printf("\n");
   }

   return 0;
}
