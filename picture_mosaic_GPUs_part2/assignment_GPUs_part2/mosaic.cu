#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acv18mg"		//replace with your user name
#define MIX_COLOUR 255;



void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

unsigned int c = 0;
MODE execution_mode = CPU;
char *ppmfilein;
char *ppmfileout;
__constant__ int d_c;
__constant__ int d_WIDTH;


//struct of colour RGB 
typedef struct {
	unsigned char red, green, blue;
} PPMrgb;

//w of ppm file
typedef struct {
	char *form;
	int high, width;
	PPMrgb *rgb;
} PPMImage;


void print_help();
int process_command_line(int argc, char *argv[]);
PPMImage *readPPM(const char *filename);
void writePPM(const char *filename, PPMImage *img);
void CPU_MODE(PPMImage *image);
void OPENMP_MODE(PPMImage *image);


//void CUDA_MODE(PPMImage *image);
//void CUDA_MODE2(PPMImage *image);
void CUDA_MODE3(PPMImage *image);
void CUDA_MODE3_struc(PPMImage *image);
//__global__ void aveMosaic(PPMImage *image);
//__global__ void aveMosaic2(int *d_rarr, int *d_garr, int *d_barr, int c, int width);
__global__ void aveMosaic3(int *d_rarr, int *d_garr, int *d_barr, int c, int width, int *d_sumr, int *d_sumg, int *d_sumb);
__global__ void aveMosaic3_struc(PPMrgb *d_rgb,  int *d_sumr, int *d_sumg, int *d_sumb);



/* Main entrance */

int main(int argc, char *argv[]) {
	float begin, end;
	float seconds;
	if (process_command_line(argc, argv) == FAILURE)
		return 1;

	//TODO: read input image file (either binary or plain text PPM) 
	PPMImage *image;
	image = readPPM(ppmfilein);

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		//TODO: starting timing here
		begin = clock();
		//TODO: calculate the average colour value

		//call CPU mode function
		CPU_MODE(image);



		//TODO: end timing here
		end = clock();
		seconds = (end - begin) / (float)CLOCKS_PER_SEC;
		int second = seconds;
		printf("CPU mode execution time took %d s and %f ms\n", second, 100 * (seconds - (float)second));
		break;
	}

	case (OPENMP): {//TODO: starting timing here
		begin = omp_get_wtime();
		//TODO: calculate the average colour value

		//call openmp mode function
		OPENMP_MODE(image);

		//TODO: end timing here
		end = omp_get_wtime();
		seconds = (end - begin);
		int second = seconds;
		printf("OPENMP mode execution time took %d s and %f ms\n", second, 100 * (seconds - (float)second));
		break;
	}
	case (CUDA): {
		//printf("CUDA Implementation not required for assignment part 1\n");


		//cudaDeviceSynchronize();
		begin = clock();

		CUDA_MODE3_struc(image);
		//CPU_MODE(image);
		//cudaDeviceSynchronize();
		end = clock();
		seconds = (float)(end - begin) / CLOCKS_PER_SEC;
		printf("GPUtime:%f", seconds);

		break;
	}
	case (ALL): {


		/*cpu--------------------------------------------------------------*/
		begin = clock();
		CPU_MODE(image);
		end = clock();
		seconds = (end - begin) / (float)CLOCKS_PER_SEC;
		int second = seconds;
		printf("CPU mode execution time took %d s and %f ms\n", second, 100 * (seconds - (float)second));


		/*openmp--------------------------------------------------------------*/
		begin = omp_get_wtime();
		OPENMP_MODE(image);
		end = omp_get_wtime();
		seconds = (end - begin);
		second = seconds;
		printf("OPENMP mode execution time took %d s and %f ms\n", second, 100 * (seconds - (float)second));


		/*gpu--------------------------------------------------------------*/
		cudaDeviceSynchronize();
		begin = clock();
		CUDA_MODE3_struc(image);
		end = clock();
		seconds = (float)(end - begin) / CLOCKS_PER_SEC;
		printf("GPUtime:%f", seconds);

		break;
	}
	}
	//save the output image file (from last executed mode)
	writePPM(ppmfileout, image);
	return 0;
}



/*Read ppm file*/
PPMImage *readPPM(const char *filename)
{
	FILE *fr;
	char buff[16];
	//char form = NULL;
	PPMImage *img;
	int color;
	int ch;

	fr = fopen(filename, "rb");
	if (filename == NULL) {
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fr)) {
		perror(filename);
		exit(1);
	}

	//malloc memory form image
	img = (PPMImage *)malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}
	//check the image format

	char *a = buff;
	if (strcmp(a, "P6\n") == 0)
	{
		img->form = "P6";
	}
	else if (strcmp(a, "P3\n") == 0)
	{
		img->form = "P3";
	}
	else {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	//check for comments
	ch = getc(fr);
	while (ch == '#') {
		while (getc(fr) != '\n');
		ch = getc(fr);
	}

	//read image high width
	ungetc(ch, fr);
	if (fscanf(fr, "%d %d", &img->high, &img->width) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	if (c > img->high || c > img->width) {
		fprintf(stderr, "C is bigger than img size\n", filename);
		exit(1);
	}

	//read rgb component
	if (fscanf(fr, "%d", &color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}


	while (fgetc(fr) != '\n');
	//while (!feof(fr))
	//memory allocation for pixel data
	img->rgb = (PPMrgb*)malloc((img->high) * (img->width) * (sizeof(PPMrgb)));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}
	//read pixel data from file
	if (strcmp(img->form, "P6") == 0) {
		fread(img->rgb, 1, img->width * img->width * 3, fr);
	}

	else if (strcmp(img->form, "P3") == 0) {
		int count = 0;
		for (int i = 0; i < img->high*img->width; i++) {
			fscanf(fr, "%d %d %d ", &img->rgb[count].red, &img->rgb[count].green, &img->rgb[count].blue);
			count++;
		}
	}
	fclose(fr);
	return img;
}

/*-----------------------------------------------------------------------*/
/* output ppm file after Image processing */


void writePPM(const char *filename, PPMImage *img)
{
	FILE *fw;
	//open file for output
	fw = fopen(filename, "wb");
	if (!fw) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//write the header file
	//image format
	fprintf(fw, "%s\n", img->form);

	//comments
	fprintf(fw, "# Created by %s\n", USER_NAME);

	//image size
	fprintf(fw, "%d %d\n", img->high, img->width);
	//printf("high:%d;width:%d\n", img->high, img->width);

	// rgb mix color
	fprintf(fw, "%d\n", 255);
	// pixel data
	if (strcmp(img->form, "P6") == 0) {
		fwrite(img->rgb, 3 * img->high, img->width, fw);
	}
	else if (strcmp(img->form, "P3") == 0) {
		int count = 0;
		for (int i = 0; i < img->high*img->width; i++) {
			fprintf(fw, "%d %d %d ", img->rgb[count].red, img->rgb[count].green, img->rgb[count].blue);
			count++;
		}
	}
	fclose(fw);
}



/*-----------------------------------------------------------------------*/
/*  the function of CPU mode*/

void CPU_MODE(PPMImage *image) {
	printf("-------------------");
	printf("Initialize CPU mode");
	printf("-------------------\n");


	//molloc space for 2d arr to store rgb
	PPMrgb **arr = (PPMrgb **)malloc(sizeof(PPMrgb *)*(image->width));
	//printf("cpu!!!!!!!!!!!");
	for (int count = 0; count < image->width; count++) {
		arr[count] = (PPMrgb *)malloc(sizeof(PPMrgb *)*(image->high));
	}
	//PPMrgb *p2 = image->rgb;

	//store rgb into array
	int count2 = 0;
	for (int i = 0; i < image->high; i++) {
		for (int j = 0; j < image->width; j++) {
			arr[i][j] = image->rgb[count2];
			//printf("arr[%d][%d]:%d\n",i,j,arr[i][j].green);
			count2++;
		}
	}


	//printf("cpu!!!!!!!!!!!");


	int sumRed = 0;
	int sumGreen = 0;
	int sumBlue = 0;
	int aveRed = 0;
	int aveGreen = 0;
	int aveBlue = 0;
	int SUMRED = 0;
	int SUMGREEN = 0;
	int SUMBLUE = 0;

	//RGB **arr2 = (RGB **)malloc(5 * (image->high));
	//for (int count = 0; count< image->width; count++)
	//	arr2[count] = (RGB *)malloc(5 * (image->width));

	for (int i = 0; i < image->high; i = i + c) {
		for (int j = 0; j < image->width; j = j + c) {

			for (int x = 0; x < c; x++) {
				for (int y = 0; y < c; y++) {
					sumRed = sumRed + arr[i + x][j + y].red;
					sumGreen = sumGreen + arr[i + x][j + y].green;
					sumBlue = sumBlue + arr[i + x][j + y].blue;
					//printf("red:%d\n", arr[i + x][j + y].red);
				}
			}
			aveRed = sumRed / (c*c);
			//printf("aveRed:%d;;", aveRed);

			aveGreen = sumGreen / (c*c);
			//printf("aveGreen:%d;;", aveGreen);

			aveBlue = sumBlue / (c*c);
			//printf("aveblue:%d\n",aveBlue);

			for (int x = 0; x < c; x++) {
				for (int y = 0; y < c; y++) {
					arr[i + x][j + y].red = aveRed;
					arr[i + x][j + y].green = aveGreen;
					arr[i + x][j + y].blue = aveBlue;
					//printf("red:%d\n", arr[i+x][j+y].red);
				}
			}
			SUMRED += sumRed;
			SUMGREEN += sumGreen;
			SUMBLUE += sumBlue;

			sumRed = 0;
			sumGreen = 0;
			sumBlue = 0;
		}
	}

	/*for (int m = 0; m < 16; m++) {
	for (int n = 0; n < 16; n++) {
	printf("red:%d\n", arr[m][n].red);
	}
	}*/

	aveRed = SUMRED / (image->high*image->width);
	aveGreen = SUMGREEN / (image->high*image->width);
	aveBlue = SUMBLUE / (image->high*image->width);

	/*for (int m = 0; m < 16; m++) {
	for (int n = 0; n < 16; n++) {
	printf("red:%d\n", arr[m][n].red);
	}
	}*/

	int count = 0;
	for (int i = 0; i < image->high; i++) {
		for (int j = 0; j < image->width; j++) {
			//printf("red%d:%d;;",count, arr[i][j].red);
			image->rgb[count].red = arr[i][j].red;
			//printf("red2:%d;;", image->rgb[count].red);

			//printf("green%d:%d;;", count, arr[i][j].green);
			image->rgb[count].green = arr[i][j].green;
			//printf("green2:%d;;", image->rgb[count].green);

			//printf("blue%d:%d\n", count, arr[i][j].blue);
			image->rgb[count].blue = arr[i][j].blue;
			//printf("blue2:%d\n", image->rgb[count].blue);

			//image->rgb[count] = arr[i][j];
			//printf("arr[%d][%d]:%d\n", i, j, image->rgb[count].red);
			count++;
		}
	}

	/*for (int m = 0; m < 256; m++) {

	printf("IMG red%d:%d\n",m, image->rgb[m].red);

	}*/


	// Output the average colour value for the image
	printf("CPU Average image colour red = %d, green = %d, blue = %d \n", aveRed, aveGreen, aveBlue);
	free(arr);

}


/*-----------------------------------------------------------------------*/
/*  the function of openmp mode*/

void OPENMP_MODE(PPMImage *image) {

	printf("-------------------");
	printf("Initialize OPENMP mode");
	printf("-------------------\n");
	//molloc space for 2d arr to store rgb
	PPMrgb **arr = (PPMrgb **)malloc(sizeof(PPMrgb *)*(image->width) * 2);
	for (int count = 0; count < image->width; count++)
		arr[count] = (PPMrgb *)malloc(sizeof(PPMrgb *)*(image->high) * 2);
	PPMrgb *p2 = image->rgb;

	//store rgb into array
	int count2 = 0;
	for (int i = 0; i < image->high; i++) {
		for (int j = 0; j < image->width; j++) {
			arr[i][j] = image->rgb[count2];
			//printf("arr[%d][%d]:%d\n",i,j,arr[i][j].green);
			count2++;
		}
	}


	int sumRed = 0;
	int sumGreen = 0;
	int sumBlue = 0;
	int aveRed = 0;
	int aveGreen = 0;
	int aveBlue = 0;
	int SUMRED = 0;
	int SUMGREEN = 0;
	int SUMBLUE = 0;
	int i;
	int j;
	int x;
	int y;

	//RGB **arr2 = (RGB **)malloc(5 * (image->high));
	//for (int count = 0; count< image->width; count++)
	//	arr2[count] = (RGB *)malloc(5 * (image->width));

	for (i = 0; i < image->high; i = i + c) {
		for (j = 0; j < image->width; j = j + c) {


#pragma omp for schedule(static, 2)
			for (x = 0; x < c; x++) {
				for (y = 0; y < c; y++) {
					sumRed = sumRed + arr[i + x][j + y].red;
					sumGreen = sumGreen + arr[i + x][j + y].green;
					sumBlue = sumBlue + arr[i + x][j + y].blue;
					//	printf("red:%d\n", arr[i + x][j + y].red);
					//arr2[0][0].red += arr[i + x][j + y].red;
					//arr2[0][0].green += arr[i + x][j + y].green;
					//arr2[0][0].blue += arr[i + x][j + y].blue;
				}
			}


			aveRed = sumRed / (c*c);
			aveGreen = sumGreen / (c*c);
			aveBlue = sumBlue / (c*c);



			for (int x = 0; x < c; x++) {
				for (int y = 0; y < c; y++) {
					arr[i + x][j + y].red = aveRed;
					arr[i + x][j + y].green = aveGreen;
					arr[i + x][j + y].blue = aveBlue;
					//printf("red:%d\n", arr[i+x][j+y].red);
				}
			}

			SUMRED += sumRed;
			SUMGREEN += sumGreen;
			SUMBLUE += sumBlue;

			sumBlue = 0;
			sumRed = 0;
			sumGreen = 0;
		}
	}
	aveRed = SUMRED / (image->high*image->width);
	aveGreen = SUMGREEN / (image->high*image->width);
	aveBlue = SUMBLUE / (image->high*image->width);

	int count = 0;

	for (int i = 0; i < image->high; i++) {
		for (int j = 0; j < image->width; j++) {
			image->rgb[count].red = arr[i][j].red;
			image->rgb[count].green = arr[i][j].green;
			image->rgb[count].blue = arr[i][j].blue;
			//image->rgb[count] = arr[i][j];
			//printf("arr[%d][%d]:%d\n", i, j, image->rgb[count].red);
			count++;
		}
	}


	// Output the average colour value for the image
	printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", aveRed, aveGreen, aveBlue);
	free(arr);


}


/*-----------------------------------------------------------------------*/
/*  the function of CUDA mode*/


/*
__global__ void aveMosaic(PPMImage *image, int c) {


	//printf("cuda initial.......");
	//int d_c = c;
	int width = image->width;
	int heigh = image->high;





	int sum = 0;
	__shared__ int sumRed;
	__shared__ int sumGreen;
	__shared__ int sumBlue;
	__shared__ int aveRed;
	__shared__ int aveGreen;
	__shared__ int aveBlue;

	//int index = threadIdx.x + threadIdx.y * blockDim.x * c  + blockIdx.x * c + blockIdx.y * c  * width;
	int index = threadIdx.x + threadIdx.y*width + blockIdx.x*c + blockIdx.y * c  * width;

	for (int i = 0; i < c; i++) {
		sumRed = sumRed + image->rgb[index + i].red;
		sumGreen = sumGreen + image->rgb[index + i].green;
		sumBlue = sumBlue + image->rgb[index + i].blue;
	}

	if (threadIdx.x == 0) {
		aveRed = sumRed / (c*c);
		aveGreen = sumGreen / (c*c);
		aveBlue = sumBlue / (c*c);
	}

	for (int i = 0; i < c; i++) {
		image->rgb[index + i].red = aveRed;
		image->rgb[index + i].green = aveGreen;
		image->rgb[index + i].blue = aveBlue;
	}

	//sumRed = sumRed + image->rgb[index].red;
	//sumGreen = sumGreen + image->rgb[index].green;
	//sumBlue = sumBlue + image->rgb[index].blue;

	//atomicAdd(&sumRed, image->rgb[index].red);
	//atomicAdd(&sumGreen, image->rgb[index].green);
	//atomicAdd(&sumBlue, image->rgb[index].blue);


	//if (threadIdx.x == 0) {
	//	aveRed = sumRed / (c*c);
	//	aveGreen = sumGreen / (c*c);
	//	aveBlue = sumBlue / (c*c);
	//}

	//image->rgb[index].red = aveRed;
	//image->rgb[index].green = aveGreen;
	//image->rgb[index].blue = aveBlue;


	//int index = threadIdx.x * width + blockIdx.x *c + blockIdx.y * c;

}*/

/*
void CUDA_MODE(PPMImage *image) {

	int high = image->high;
	int width = image->high;


	PPMImage *h_image;
	h_image = (PPMImage*)malloc(sizeof(PPMImage));
	h_image = image;

	PPMImage *d_image;
	cudaMalloc((void **)&d_image, sizeof(PPMImage));
	checkCUDAError("CUDA malloc");


	cudaMemcpy(d_image, h_image, sizeof(PPMImage), cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");


	dim3 dimGrid(width / c, high / c);
	dim3 dimBlock(c, c);
	aveMosaic << < dimGrid, dimBlock >> > (d_image, c);
	checkCUDAError("CUDA kernel");


	cudaMemcpy(h_image, d_image, sizeof(PPMImage), cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy2");



	// Output the average colour value for the image
	//printf("CPU Average image colour red = %d, green = %d, blue = %d \n", aveRed, aveGreen, aveBlue);
	cudaFree(d_image);
	checkCUDAError("CUDA cleanup");

}

*/


/*
__global__ void aveMosaic2(int *d_rarr, int *d_garr, int *d_barr, int c, int width) {

	int sum = 0;
	__shared__ int sumRed;
	__shared__ int sumGreen;
	__shared__ int sumBlue;
	__shared__ int aveRed;
	__shared__ int aveGreen;
	__shared__ int aveBlue;

	//int index = threadIdx.x + threadIdx.y * blockDim.x * c  + blockIdx.x * c + blockIdx.y * c  * width;
	int index = threadIdx.x + threadIdx.y*width + blockIdx.x*c + blockIdx.y * c  * width;

	atomicAdd(&sumRed, d_rarr[index]);
	atomicAdd(&sumGreen, d_garr[index]);
	atomicAdd(&sumBlue, d_barr[index]);

	__syncthreads();
		//atomicAdd(&d_sumr[0], sumRed);
		//printf("sumred:%d",sumRed);

		//atomicAdd(&d_sumg[0], sumGreen);
		//atomicAdd(&d_sumb[0], sumBlue);	

	if (threadIdx.x == 0) {
		aveRed = sumRed / (c*c);
		aveGreen = sumGreen / (c*c);
		aveBlue = sumBlue / (c*c);
	}

	__syncthreads();
	d_rarr[index] = aveRed;
	d_garr[index] = aveGreen;
	d_barr[index] = aveBlue;


}




void CUDA_MODE2(PPMImage *image) {

	int high = image->high;
	int width = image->high;
	int temp = (width * high) / (c * c);
	//int aver = 0;
	//int aveg = 0;
	//int aveb = 0;


	int *rarr = (int *)malloc(width*high * sizeof(int));
	int *garr = (int *)malloc(width*high * sizeof(int));
	int *barr = (int *)malloc(width*high * sizeof(int));
	//int *sumr = (int *)malloc(temp * sizeof(int));
	//int *sumg = (int *)malloc(temp * sizeof(int));
	//int *sumb = (int *)malloc(temp * sizeof(int));

	for (int count = 0; count < width*high; count++) {
		rarr[count] = image->rgb[count].red;
		garr[count] = image->rgb[count].green;
		barr[count] = image->rgb[count].blue;
	}
	int *d_rarr;
	int *d_garr;
	int *d_barr;
	cudaMalloc((void **)&d_rarr, width*high * sizeof(int));
	cudaMalloc((void **)&d_garr, width*high * sizeof(int));
	cudaMalloc((void **)&d_barr, width*high * sizeof(int));
	//int *d_sumr;
	//int *d_sumg;
	//int *d_sumb;
	//cudaMalloc((void **)&d_sumr, temp * sizeof(int));
	//cudaMalloc((void **)&d_sumg, temp * sizeof(int));
	//cudaMalloc((void **)&d_sumb, temp * sizeof(int));
	checkCUDAError("CUDA malloc");

	cudaMemcpy(d_rarr, rarr, width*high * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_garr, garr, width*high * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_barr, barr, width*high * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_sumr, sumr, temp * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_sumg, sumg, temp * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_sumb, sumb, temp * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");


	dim3 dimGrid(width / c, high / c);
	dim3 dimBlock(c, c);
	aveMosaic2 << < dimGrid, dimBlock >> > (d_rarr, d_garr, d_barr, c, width);
	checkCUDAError("CUDA kernel");


	cudaMemcpy(rarr, d_rarr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(garr, d_garr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(barr, d_barr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(sumr, d_sumr, temp * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(sumg, d_sumg, temp * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(sumb, d_sumb, temp * sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy2");


	//aver = sumr[0] / (width*high);
	//aveg = sumg[0] / (width*high);
	//aveb = sumb[0] / (width*high);

	for (int count = 0; count < width*high; count++) {
		image->rgb[count].red = rarr[count];
		image->rgb[count].green = garr[count];
		image->rgb[count].blue = barr[count];
	}

	//printf("GPU Average image colour red = %d, green = %d, blue = %d \n", aver, aveg, aveb);

	// Output the average colour value for the image
	//printf("CPU Average image colour red = %d, green = %d, blue = %d \n", aveRed, aveGreen, aveBlue);
	cudaFree(d_rarr);
	cudaFree(d_garr);
	cudaFree(d_barr);
	//cudaFree(d_sumr);
	//cudaFree(d_sumg);
	//cudaFree(d_sumb);
	checkCUDAError("CUDA cleanup");

}
*/

//, , int *d_block_results

__global__ void aveMosaic3(int *d_rarr, int *d_garr, int *d_barr, int c, int width, int *d_sumr, int *d_sumg, int *d_sumb) {

	//__shared__ int sumRed;
	//__shared__ int sumGreen;
	//__shared__ int sumBlue;
	//__shared__ int aveRed;
	//__shared__ int aveGreen;
	//__shared__ int aveBlue;

	int div = c*c;
	int sumRed = 0;
	int sumGreen = 0;
	int sumBlue = 0;
	int aveRed = 0;
	int aveGreen = 0;
	int aveBlue = 0;
	//extern __shared__ int red[];
	//int index = threadIdx.x + threadIdx.y * blockDim.x * c  + blockIdx.x * c + blockIdx.y * c  * width;
	int index = blockIdx.x*c + blockIdx.y * c  * width;

	__syncthreads();

	//threads index rgb and do add
	for (int i = 0; i < c; i++) {
		for (int j = 0; j < c; j++) {
			sumRed = sumRed + d_rarr[index + j + i*width];
			sumGreen = sumGreen + d_garr[index + j + i*width];
			sumBlue = sumBlue + d_barr[index + j + i*width];
		}
	}

	__syncthreads();

	//calculate the whole pic's rgb sum
	if (threadIdx.x == 0) {
		atomicAdd(&d_sumr[0], sumRed);
		atomicAdd(&d_sumg[0], sumGreen);
		atomicAdd(&d_sumb[0], sumBlue);

	}


	/*

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
	unsigned int strided_i = threadIdx.x * 2 * stride;
	if (strided_i < blockDim.x) {
	red[strided_i] += red[strided_i + stride];
	}
	__syncthreads();
	}
	if (threadIdx.x == 0) {
	//atomicAdd(block_results, red[0]);
	d_block_results[blockIdx.x] = red[0];

	}

	*/


	//printf("block:%d\n", block_results[0]);

	//AVERED = block_results[blockIdx.x] / (blockDim.x * blockDim.y);

	//calculate the average rgb of every mosaic
	aveRed = sumRed / div;
	aveGreen = sumGreen / div;
	aveBlue = sumBlue / div;

	//b->x = a->x / div;
	//b->y = a->y / div;
	//b->z = a->z / div;


	for (int i = 0; i < c; i++) {
		for (int j = 0; j < c; j++) {


			d_rarr[index + j + i*width] = aveRed;
			d_garr[index + j + i*width] = aveGreen;
			d_barr[index + j + i*width] = aveBlue;

			//d_rarr[index + j + i*width] = b->x;
			//d_garr[index + j + i*width] = b->y;
			//d_barr[index + j + i*width] = b->z;
		}
	}


}


void CUDA_MODE3(PPMImage *image) {

	printf("-------------------");
	printf("Initialize GPU mode");
	printf("-------------------\n");
	int high = image->high;;
	int width = image->width;
	int temp = (width * high) / (c * c);

	int aver = 0;
	int aveg = 0;
	int aveb = 0;


	//malloc cpu space
	int *rarr = (int *)malloc(width*high * sizeof(int));
	int *garr = (int *)malloc(width*high * sizeof(int));
	int *barr = (int *)malloc(width*high * sizeof(int));
	//int *block_results = (int *)malloc(temp * sizeof(int));
	int *sumr = (int *)malloc(temp * sizeof(int));
	int *sumg = (int *)malloc(temp * sizeof(int));
	int *sumb = (int *)malloc(temp * sizeof(int));

	for (int count = 0; count < width*high; count++) {
		rarr[count] = image->rgb[count].red;
		garr[count] = image->rgb[count].green;
		barr[count] = image->rgb[count].blue;
	}

	int *d_rarr;
	int *d_garr;
	int *d_barr;
	int *d_sumr;
	int *d_sumg;
	int *d_sumb;
	//int *d_block_results;
	//malloc gpu space
	cudaMalloc((void **)&d_rarr, width*high * sizeof(int));
	cudaMalloc((void **)&d_garr, width*high * sizeof(int));
	cudaMalloc((void **)&d_barr, width*high * sizeof(int));
	cudaMalloc((void **)&d_sumr, temp * sizeof(int));
	cudaMalloc((void **)&d_sumg, temp * sizeof(int));
	cudaMalloc((void **)&d_sumb, temp * sizeof(int));
	//cudaMalloc((void **)&d_block_results, temp * sizeof(int));
	checkCUDAError("CUDA malloc");

	//copy data from cpu to gpu,store in global memory
	cudaMemcpy(d_rarr, rarr, width*high * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_garr, garr, width*high * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_barr, barr, width*high * sizeof(int), cudaMemcpyHostToDevice);
	//daMemcpy(d_block_results, block_results, width*high * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sumr, sumr, temp * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sumg, sumg, temp * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sumb, sumb, temp * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");
	//call kernel
	dim3 Grid(width / c, high / c);
	dim3 Block(1);
	aveMosaic3 << < Grid, Block, temp >> > (d_rarr, d_garr, d_barr, c, width, d_sumr, d_sumg, d_sumb);
	checkCUDAError("CUDA kernel");

	//, sumr, d_block_results

	//copy data from gpu to cpu,
	cudaMemcpy(rarr, d_rarr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(garr, d_garr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(barr, d_barr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(block_results, d_block_results, temp * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sumr, d_sumr, temp * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sumg, d_sumg, temp * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sumb, d_sumb, temp * sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy2");

	//calculate average of picture's r,g,b
	aver = sumr[0] / (width*high);
	aveg = sumg[0] / (width*high);
	aveb = sumb[0] / (width*high);

	//store average rgb into pic
	for (int count = 0; count < width*high; count++) {
		image->rgb[count].red = rarr[count];
		image->rgb[count].green = garr[count];
		image->rgb[count].blue = barr[count];
	}


	// Output the average colour value for the image
	printf("GPU Average image colour red = %d, green = %d, blue = %d \n", aver, aveg, aveb);


	cudaFree(d_rarr);
	cudaFree(d_garr);
	cudaFree(d_barr);
	//cudaFree(d_block_results);
	cudaFree(d_sumr);
	cudaFree(d_sumg);
	cudaFree(d_sumb);
	checkCUDAError("CUDA cleanup");

}

__global__ void aveMosaic3_struc(PPMrgb *d_rgb,int *d_sumr, int *d_sumg, int *d_sumb) {



	int div = d_c*d_c;
	int sumRed = 0;
	int sumGreen = 0;
	int sumBlue = 0;
	int aveRed = 0;
	int aveGreen = 0;
	int aveBlue = 0;

	int index = blockIdx.x*d_c + blockIdx.y * d_c  * d_WIDTH;

	//threads index rgb and do add
	for (int i = 0; i < d_c; i++) {
		for (int j = 0; j < d_c; j++) {
			sumRed = sumRed + d_rgb[index + j + i*d_WIDTH].red;
			sumGreen = sumGreen + d_rgb[index + j + i*d_WIDTH].green;
			sumBlue = sumBlue + d_rgb[index + j + i*d_WIDTH].blue;
		}
	}

	__syncthreads();
	//calculate the whole pic's rgb sum
	if (threadIdx.x == 0) {
		atomicAdd(&d_sumr[0], sumRed);
		atomicAdd(&d_sumg[0], sumGreen);
		atomicAdd(&d_sumb[0], sumBlue);

	}

	__syncthreads();
	//calculate the average rgb of every mosaic
	aveRed = sumRed / div;
	aveGreen = sumGreen / div;
	aveBlue = sumBlue / div;

	//store the average rgb back into pic's rgb structure
	for (int i = 0; i < d_c; i++) {
		for (int j = 0; j < d_c; j++) {
			d_rgb[index + j + i*d_WIDTH].red = aveRed;
			d_rgb[index + j + i*d_WIDTH].green = aveGreen;
			d_rgb[index + j + i*d_WIDTH].blue = aveBlue;
		}
	}


}

void CUDA_MODE3_struc(PPMImage *image) {


	printf("-------------------");
	printf("Initialize GPU mode");
	printf("-------------------\n");
	int high = image->high;
	int width = image->width;
	int temp = (width * high) / (c * c);
	int size = width * high;

	int aver = 0;
	int aveg = 0;
	int aveb = 0;

	cudaMemcpyToSymbol(d_WIDTH, &width, sizeof(int));
	cudaMemcpyToSymbol(d_c, &c, sizeof(int));
	

	//malloc cpu space
	PPMrgb *rgb = (PPMrgb *)malloc(sizeof(PPMrgb)*size);
	rgb = image->rgb;
	int *sumr = (int *)malloc(temp * sizeof(int));
	int *sumg = (int *)malloc(temp * sizeof(int));
	int *sumb = (int *)malloc(temp * sizeof(int));

	int *d_sumr;
	int *d_sumg;
	int *d_sumb;
	PPMrgb *d_rgb;
	//malloc gpu space
	cudaMalloc((void **)&d_sumr, temp * sizeof(int));
	cudaMalloc((void **)&d_sumg, temp * sizeof(int));
	cudaMalloc((void **)&d_sumb, temp * sizeof(int));
	cudaMalloc((void **)&d_rgb, sizeof(PPMrgb)*size);
	checkCUDAError("CUDA malloc");

	//cpy from cpu to gpu
	cudaMemcpy(d_sumr, sumr, temp * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sumg, sumg, temp * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sumb, sumb, temp * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rgb, rgb, sizeof(PPMrgb)*size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	//call kernel 
	dim3 Grid(width / c, high / c);
	dim3 Block(1);
	aveMosaic3_struc << < Grid, Block, temp >> > (d_rgb,   d_sumr, d_sumg, d_sumb);
	checkCUDAError("CUDA kernel");


	cudaMemcpy(sumr, d_sumr, temp * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sumg, d_sumg, temp * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sumb, d_sumb, temp * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(rgb, d_rgb, sizeof(PPMrgb)*size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy2");

	aver = sumr[0] / (width*high);
	aveg = sumg[0] / (width*high);
	aveb = sumb[0] / (width*high);



	// Output the average colour value for the image
	printf("GPU Average image colour red = %d, green = %d, blue = %d \n", aver, aveg, aveb);

	cudaFree(d_sumr);
	cudaFree(d_sumg);
	cudaFree(d_sumb);
	cudaFree(d_rgb);
	checkCUDAError("CUDA cleanup");



}

/*
__global__ void aveMosaic4(int *d_rarr, int *d_garr, int *d_barr, int c, int width) {


//printf("cuda initial.......");
//int d_c = c;

__shared__ int d_c = c;
int sum = 0;
int sumRed;
int sumGreen;
int sumBlue;
int aveRed;
int aveGreen;
int aveBlue;

//int index = threadIdx.x + threadIdx.y * blockDim.x * c  + blockIdx.x * c + blockIdx.y * c  * width;
int index = threadIdx.x *width + blockIdx.x*d_c + blockIdx.y * d_c  * width;


for (int i = 0; i < c;i++) {
//sumRed[index] = sumRed[index] + d_rarr[index +i];
//sumGreen[index] = sumGreen[index] + d_garr[index +i];
//sumBlue[index] = sumBlue[index] + d_barr[index + i];



atomicAdd(&sumRed, d_rarr[index+i]);
atomicAdd(&sumGreen, d_garr[index+i]);
atomicAdd(&sumBlue, d_barr[index+i]);
}




__syncthreads();


if (threadIdx.x == 0) {
aveRed = sumRed / (c*c);
aveGreen = sumGreen / (c*c);
aveBlue = sumBlue / (c*c);
}

__syncthreads();

for (int i = 0; i < c; i++) {
for (int j = 0; j < c; j++) {

d_rarr[index + i*width] = aveRed;
d_garr[index + i*width] = aveGreen;
d_barr[index + i*width] = aveBlue;

}
}


}


void CUDA_MODE4(PPMImage *image) {

int high = image->high;
int width = image->high;


int *rarr = (int *)malloc(width*high * sizeof(int));
int *garr = (int *)malloc(width*high * sizeof(int));
int *barr = (int *)malloc(width*high * sizeof(int));

for (int count = 0; count < width*high; count++) {
rarr[count] = image->rgb[count].red;
garr[count] = image->rgb[count].green;
barr[count] = image->rgb[count].blue;
}

int *d_rarr;
int *d_garr;
int *d_barr;
cudaMalloc((void **)&d_rarr, width*high * sizeof(int));
cudaMalloc((void **)&d_garr, width*high * sizeof(int));
cudaMalloc((void **)&d_barr, width*high * sizeof(int));
checkCUDAError("CUDA malloc");




cudaMemcpy(d_rarr, rarr, width*high * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_garr, garr, width*high * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_barr, barr, width*high * sizeof(int), cudaMemcpyHostToDevice);
checkCUDAError("CUDA memcpy");


dim3 dimGrid(width / c, high / c);
dim3 dimBlock(c);
aveMosaic4 << < dimGrid, dimBlock >> > (d_rarr, d_garr, d_barr, c, width);
checkCUDAError("CUDA kernel");


cudaMemcpy(rarr, d_rarr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(garr, d_garr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(barr, d_barr, width*high * sizeof(int), cudaMemcpyDeviceToHost);
checkCUDAError("CUDA memcpy2");


for (int count = 0; count < width*high; count++) {
image->rgb[count].red = rarr[count];
//printf("%d;", rarr[count]);
image->rgb[count].green = garr[count];
image->rgb[count].blue = barr[count];
}



// Output the average colour value for the image
//printf("CPU Average image colour red = %d, green = %d, blue = %d \n", aveRed, aveGreen, aveBlue);
cudaFree(d_rarr);
cudaFree(d_garr);
cudaFree(d_barr);
checkCUDAError("CUDA cleanup");

}
*/




void print_help() {

	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}


int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name
	//argv[0] = "Assignment_GPUs_part2.exe";

	//read in the non optional command line arguments
	c = atoi(argv[1]);
	if (c <= 0 || c % 2 != 0) {
		fprintf(stderr, "Error: a wrong input of the second argument. Correct usage is...\n");
		//printf("the value of C should be positive");
	}
	//c = 2 ^ c;

	//TODO: read in the mode
	if (strcmp(argv[2], "CPU") == 0) { execution_mode = CPU; }
	else if (strcmp(argv[2], "OPENMP") == 0) { execution_mode = OPENMP; }
	else if (strcmp(argv[2], "CUDA") == 0) { execution_mode = CUDA; }
	else if (strcmp(argv[2], "ALL") == 0) { execution_mode = ALL; }
	else {
		fprintf(stderr, "Error: a wrong input of the third argument. Correct usage is...\n");
		//printf("Is the mode with a value of either CPU, OPENMP, CUDA or ALL\n");
		exit(1);
	}

	//TODO: read in the input image name
	if (strcmp(argv[3], "-i") != 0) {
		printf("error input,you should type -i befor the name of input file");
	}
	ppmfilein = argv[4];

	//TODO: read in the output image name
	if (strcmp(argv[5], "-o") != 0) {
		printf("error input,you should type -o befor the name of output file");
	}
	ppmfileout = argv[6];
	//TODO: read in any optional part 3 arguments
	return SUCCESS;
}