## 实验内容

1. 读取并加载灰度图像；
2. 利用高斯滤波器对图像进行平滑处理，使其更加模糊；
3. 计算梯度幅值和方向，以便后续边缘检测操作；
4. 进行非极大值抑制，去除非最大值梯度点；
5. 使用双阈值算法，将所有梯度值分为强边缘和弱边缘两部分，只保留强边缘并连接相邻的弱边缘；
6. 将结果输出到显示窗口中。

## Task

#### 1.code

```cpp
#include<iostream>
#include<opencv.hpp>
#include<math.h>

#define _USE_MATH_DEFINES
#define M_PI 3.14

using namespace std;
using namespace cv;

void mergeImg(Mat& dst, Mat& src1, Mat& src2)
{
	int rows = src1.rows;
	int cols = src1.cols + 5 + src2.cols;
	CV_Assert(src1.type() == src2.type());
	dst.create(rows, cols, src1.type());
	src1.copyTo(dst(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(dst(Rect(src1.cols + 5, 0, src2.cols, src2.rows)));
}

void gaussianConvolution(Mat& img, Mat& dst)
{
	int nr = img.rows;
	int nc = img.cols;
	int templates[3] = { 1, 2, 1 };
	//按行遍历除每行边缘点的所有点
	for (int j = 0; j < nr; j++)
	{
		uchar* data = img.ptr<uchar>(j);
		for (int i = 1; i < nc - 1; i++)
		{
			int sum = 0;
			for (int n = 0; n < 3; n++)
			{
				sum += data[i - 1 + n] * templates[n]; // 相称累加
			}
			sum /= 4;
			dst.ptr<uchar>(j)[i] = sum;
		}
	}
}

void gaussianFilter(Mat& img, Mat& dst)
{
	Mat dst1 = img.clone();
	gaussianConvolution(img, dst1);
	Mat dst2;
	transpose(dst1, dst2);
	Mat dst3 = dst2.clone();
	gaussianConvolution(dst2, dst3);
	transpose(dst3, dst);
}

void getGrandient(Mat& img, Mat& gradXY, Mat& theta)
{
	gradXY = Mat::zeros(img.size(), CV_8U);
	theta = Mat::zeros(img.size(), CV_8U);
	for (int j = 1; j < img.rows - 1; j++)
	{
		for (int i = 1; i < img.cols - 1; i++)
		{
			double gradY = double(img.ptr<uchar>(j)[i + 1] - img.ptr<uchar>(j)[i] + img.ptr<uchar>(j + 1)[i + 1]);
			double gradX = double(img.ptr<uchar>(j + 1)[i] - img.ptr<uchar>(j)[i] + img.ptr<uchar>(j + 1)[i + 1]);
			gradXY.ptr<uchar>(j)[i] = sqrt(gradX * gradX + gradY * gradY); //计算梯度
			theta.ptr<uchar>(j)[i] = atan(gradY / gradX); //计算梯度方向
		}
	}
}

void nonLocalMaxValue(Mat& gradXY, Mat& theta, Mat& dst)
{
	dst = gradXY.clone();
	for (int j = 1; j < gradXY.rows - 1; j++) {
		for (int i = 1; i < gradXY.cols - 1; i++)
		{
			double t = double(theta.ptr<uchar>(j)[i]);
			double g = double(dst.ptr<uchar>(j)[i]);
			if (g == 0.0)
			{
				continue;
			}
			double g0, g1;
			if ((t >= (3 * M_PI / 8)) && (t < -(M_PI / 8)))
			{
				g0 = double(dst.ptr<uchar>(j - 1)[i - 1]);
				g1 = double(dst.ptr<uchar>(j + 1)[i + 1]);
			}
			else if ((t >= -(M_PI / 8)) && (t < M_PI / 8))
			{
				g0 = double(dst.ptr<uchar>(j)[i - 1]);
				g1 = double(dst.ptr<uchar>(j)[i + 1]);
			}
			else if ((t >= M_PI / 8) && (t < 3 * M_PI / 8))
			{
				g0 = double(dst.ptr<uchar>(j - 1)[i + 1]);
				g1 = double(dst.ptr<uchar>(j + 1)[i - 1]);
			}
			else
			{
				g0 = double(dst.ptr<uchar>(j - 1)[i]);
				g1 = double(dst.ptr<uchar>(j + 1)[i]);
			}
			if (g <= g0 || g <= g1)
			{
				dst.ptr<uchar>(j)[i] = 0.0;
			}
		}
	}
}

void doubleThresholdLink(Mat& img)
{
	//循环找到强边缘点，把其领域内的弱边缘点变为强边缘点
	for (int j = 1; j < img.rows - 2; j++)
	{
		for (int i = 1; i < img.cols - 2; i++)
		{
			//如果该点是强边缘点
			if (img.ptr<uchar>(j)[i] == 255)
			{
				//遍历该强边缘点领域
				for (int m = -1; m < 1; m++)
				{
					for (int n = -1; n < 1; n++)
					{
						//该点为弱边缘点(不是强边缘点，也不是被抑制的0点)
						if (img.ptr<uchar>(j + m)[i + n] != 0 && img.ptr<uchar>(j + m)[i + n] != 255)
						{
							img.ptr<uchar>(j + m)[i + n] = 255; //该弱边缘点补充为强边缘
						}
					}
				}
			}
		}
	}
	for (int j = 0; j < img.rows - 1; j++)
	{
		for (int i = 0; i < img.cols - 1; i++)
		{
			//如果该点依旧是弱边缘点，及此点是孤立边缘点
			if (img.ptr<uchar>(j)[i] != 255 && img.ptr<uchar>(j)[i] != 255)
			{
				img.ptr<uchar>(j)[i] = 0; //该孤立弱边缘点抑制
			}
		}
	}
}

void doubleThreshold(double low, double high, Mat& img, Mat& dst)
{
	dst = img.clone();
	//区分出弱边缘点和强边缘点
	for (int j = 0; j < img.rows - 1; j++)
	{
		for (int i = 0; i < img.cols - 1; i++)
		{
			double x = double(dst.ptr<uchar>(j)[i]);
			//像素点为强边缘点，置255
			if (x > high)
			{
				dst.ptr<uchar>(j)[i] = 255;
				//像素点置0，被抑制掉
			}
			else if (x < low)
			{
				dst.ptr<uchar>(j)[i] = 0;
			}
		}
	}
	//弱边缘点补充连接强边缘点
	doubleThresholdLink(dst);
}

int main()
{
	Mat img = imread("D:/CV_figures/6.png", IMREAD_GRAYSCALE); //从文件中加载灰度图像

	//读取图片失败，则停止
	if (img.empty())
	{
		printf("读取图像文件失败");
		system("pause");
		return 0;
	}
	//高斯滤波
	Mat gauss_img;
	gaussianFilter(img, gauss_img); // 高斯虑波器
	//用一阶偏导有限差分计算梯度幅值和方向
	Mat gradXY, theta;
	getGrandient(gauss_img, gradXY, theta);
	//局部非极大值抑制
	Mat Local_img;
	nonLocalMaxValue(gradXY, theta, Local_img);
	//用双阅值算法检测和连接边缘
	Mat dst;

	doubleThreshold(250, 10, Local_img, dst);
	// 图像显示
	Mat outImg;
	Mat out;
	mergeImg(outImg, img, dst);
	namedWindow("img");
	imshow("img", outImg);
	waitKey();
	return 0;
}
```

#### 2.实验结果

![image-20230412102345942](https://raw.githubusercontent.com/kurisaW/picbed/main/img2023/202304121023705.png)