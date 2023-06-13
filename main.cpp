/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a preconditioned conjugate gradient solver on
 * the GPU using CUBLAS and CUSPARSE.  Relative to the conjugateGradient
 * SDK example, this demonstrates the use of cusparseScsrilu0() for
 * computing the incompute-LU preconditioner and cusparseScsrsv_solve()
 * for solving triangular systems.  Specifically, the preconditioned
 * conjugate gradient method with an incomplete LU preconditioner is
 * used to solve the Laplacian operator in 2D on a uniform mesh.
 *
 * Note that the code in this example and the specific matrices used here
 * were chosen to demonstrate the use of the CUSPARSE library as simply
 * and as clearly as possible.  This is not optimized code and the input
 * matrices have been chosen for simplicity rather than performance.
 * These should not be used either as a performance guide or for
 * benchmarking purposes.
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <iomanip>

using std::setw;
using namespace std;

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper for CUDA error checking

const char *sSDKname     = "conjugateGradientPrecond";

/* genLaplace: Generate a matrix representing a second order, regular, Laplacian operator on a 2d domain in Compressed Sparse Row format*/
double norm(double *x,double *rhs,int N);
void genLaplace(int *row_ptr, int *col_ind, double *val, int M, int N, int nz,double NY);

void CG(int argc, char **argv, int N1,double *x,double *rhs,double NY);
void CVS(double** T, FILE *fp,int NX, int NY,double delta_x, double delta_y);

/* Solve Ax=b using the conjugate gradient method a) without any preconditioning, b) using an Incomplete Cholesky preconditioner and c) using an ILU0 preconditioner. */
int main(int argc, char **argv)
{
    double **Un,**q1,**q2,**S1,**S2,**gradUn1,**gradUn2,**f,**f0;

    int k, M = 0, N = 0,kk;
    double dy,dx,NX,NY,L,phi,r;
    double *x,*a,*rhs;
    double rho,efe,absm,m1,m2,l;
    double B=0.1;//Bingham Number    
  
    FILE *fp;
    fp=fopen ("surface.csv","w");

    L=1.0; 
    NY=20.0;
    dy=1.0/NY;
    NX=L*NY;
    dx=L/NX;
    phi=dy/dx;
    r=1.0;
    rho=1.0;
   
    M = N = (NY-1)*(NY-1);
//it may be change for (NY+1)*(NY+1)?
//    cout<<"NY"<<NY<<"\n";
//    cout<<"NX"<<NX<<"\n";
//    cout<<"phi"<<phi<<"\n";



  Un=(double**)malloc((N)*sizeof(double));
    if(Un == NULL)
    {
      printf("Error! memory nor allocated");
      exit(0);
    }

    for(int i=0;i<NY+1;i++)
    {
        Un[i]=(double *)malloc((NY+1)*sizeof(double));
        if(Un[i]==NULL)
      {
         printf("Error! memory nor allocated");
         exit(0);
      }
    }


    for(int j=0;j<NY+1;j++)
      {
        for(int i=0;i<NY+1;i++)
         {
          Un[j][i]=0.0;
         }
      }


     q1=(double**)malloc((N)*sizeof(double));
      if(q1 == NULL)
       {
         printf("Error! memory nor allocated");
         exit(0);
        }

     for(int i=0;i<NY;i++)
      {
        q1[i]=(double *)malloc((NX)*sizeof(double));
        if(q1[i]==NULL)
      {
         printf("Error! memory nor allocated");
         exit(0);
      }
    }

    for(int j=0;j<NY;j++)
      {
        for(int i=0;i<NX;i++)
         {
          q1[j][i]=0.0;
         }
      }

    q2=(double**)malloc((N)*sizeof(double));
     if(q2 == NULL)
      {
       printf("Error! memory nor allocated");
       exit(0);
      }

    for(int i=0;i<NY;i++)
    {
        q2[i]=(double *)malloc((NX)*sizeof(double));
        if(q2[i]==NULL)
          {
            printf("Error! memory nor allocated");
            exit(0);
          }
    }

    for(int j=0;j<NY;j++)
      {
        for(int i=0;i<NX;i++)
         {
          q2[j][i]=0.0;
         }
      }

    S1=(double**)malloc((N)*sizeof(double));
    if(S1 == NULL)
    {
      printf("Error! memory nor allocated");
      exit(0);
    }

    for(int i=0;i<NY;i++)
    {
        S1[i]=(double *)malloc((NX)*sizeof(double));
        if(S1[i]==NULL)
      {
         printf("Error! memory nor allocated");
         exit(0);
      }
    }

    for(int j=0;j<NY;j++)
      {
        for(int i=0;i<NX;i++)
         {
          S1[j][i]=0.0;
         }
      }

    S2=(double**)malloc((N)*sizeof(double));
    if(S2 == NULL)
    {
      printf("Error! memory nor allocated");
      exit(0);
    }

    for(int i=0;i<NY;i++)
    {
        S2[i]=(double *)malloc((NX)*sizeof(double));
        if(S2[i]==NULL)
      {
         printf("Error! memory nor allocated");
         exit(0);
      }
    }

    for(int j=0;j<NY;j++)
      {
        for(int i=0;i<NX;i++)
         {
          S2[j][i]=0.0;
         }
      }

    gradUn1=(double**)malloc((N)*sizeof(double));
    if(gradUn1 == NULL)
    {
      printf("Error! memory nor allocated");
      exit(0);
    }

    for(int i=0;i<NY;i++)
    {
        gradUn1[i]=(double *)malloc((NX)*sizeof(double));
        if(gradUn1[i]==NULL)
      {
         printf("Error! memory nor allocated");
         exit(0);
      }
    }

    for(int j=0;j<NY;j++)
      {
        for(int i=0;i<NX;i++)
         {
          gradUn1[j][i]=0.0;
         }
      }

   gradUn2=(double**)malloc((N)*sizeof(double));
    if(gradUn2 == NULL)
    {
      printf("Error! memory nor allocated");
      exit(0);
    }

    for(int i=0;i<NY+1;i++)
    {
        gradUn2[i]=(double *)malloc((NX)*sizeof(double));
        if(gradUn2[i]==NULL)
      {
         printf("Error! memory nor allocated");
         exit(0);
      }
    }

    for(int j=0;j<NY;j++)
      {
        for(int i=0;i<NX;i++)
         {
          gradUn2[j][i]=0.0;
         }
      }


    f=(double**)malloc((N)*sizeof(double));
      if(f == NULL)
       {
         printf("Error! memory nor allocated");
         exit(0);
       }

    for(int i=0;i<NY+1;i++)
    {
        f[i]=(double *)malloc((NX+1)*sizeof(double));
        if(f[i]==NULL)
      {
         printf("Error! memory nor allocated");
         exit(0);
      }
    }


    for(int j=0;j<NY+1;j++)
      {
        for(int i=0;i<NX+1;i++)
         {
          f[j][i]=(dx*dy)/r;
         }
      }

     f0=(double**)malloc((N)*sizeof(double));
    if(f0 == NULL)
    {
      printf("Error! memory nor allocated");
      exit(0);
    }

    for(int i=0;i<NY+1;i++)
    {
        f0[i]=(double *)malloc((NX+1)*sizeof(double));
        if(f0[i]==NULL)
      {
         printf("Error! memory nor allocated");
         exit(0);
      }
    }


     for(int j=0;j<NY+1;j++)
      {
        for(int i=0;i<NX+1;i++)
         {
          f0[j][i]=1.0;
         }
      }
    
    
    x = (double *)malloc(sizeof(double)*N);
    a = (double *)malloc(sizeof(double)*N);
    rhs = (double *)malloc(sizeof(double)*N);

    for (int i = 0; i < N; i++)
   {
        rhs[i] = (1.0*dx*dy)/r;                            // Initialize RHS
        x[i] = 0.0;	// Initial approximation of solution
	a[i] = 0.0;
    }

 

 CG(argc,argv,N,x,rhs,NY);
/*
for(int i=0; i<N; i++){
    cout<<x[i]<<"\n"; }

 cout<<"\n";
*/
   k=0;

     for(int j=1;j<NY;j++)
      {
        for(int i=1;i<NX;i++)
         {
          Un[j][i]=x[k];
          k++;
         }

      }
/*
    for(int j=0;j<=NY;j++)
      {
        for(int i=0;i<=NX;i++)
         {
          cout<<Un[j][i]<<" ";
         }
	cout<<"\n";
      }
*/

double error=1.0;

int counter=0;

//for(int kk=0;kk<100;kk++){
while(error>1e-10) {

    for (int i = 0; i < N; i++)
    {
       a[i] = x[i];
    }

    for(int j=0;j<NY;j++)
      {
        for(int i=0;i<NX;i++)
         {
           gradUn1[j][i]= 0.5*(Un[j][i+1]-Un[j][i]+Un[j+1][i+1]-Un[j+1][i])/dx;
	   gradUn2[j][i]= 0.5*(Un[j+1][i]-Un[j][i]+Un[j+1][i+1]-Un[j][i+1])/dy;
           m1=gradUn1[j][i]+S1[j][i];
	   m2=gradUn2[j][i]+S2[j][i];
	   absm=sqrt(((m1)*(m1))+((m2)*(m2)));
            if(absm<=B){
                q1[j][i]=0.0;
	        q2[j][i]=0.0;
             }else {
		q1[j][i]=((1.0-(B/absm))/(1.0+r))*m1;   
                q2[j][i]=((1.0-(B/absm))/(1.0+r))*m2;
 	     }
	 }
      }

   std::cout<<"\n";

     for(int j=0;j<NY;j++)
      {
        for(int i=0;i<NX;i++)
         {
           S1[j][i]=S1[j][i]+rho*(gradUn1[j][i]-q1[j][i]);
	   S2[j][i]=S2[j][i]+rho*(gradUn2[j][i]-q2[j][i]);

	 }
      }

     for(int j=1;j<NY;j++)
      {
        for(int i=1;i<NX;i++)
         {
            f[j][i]= f0[j][i] - 0.5*(r*(q1[j][i]-q1[j][i-1]+q1[j-1][i]-q1[j-1][i-1]))/dx+0.5*(S1[j][i]-S1[j][i-1]+S1[j-1][i]-S1[j-1][i-1])/dx - 0.5*(r*(q2[j][i]-q2[j-1][i]+q2[j][i-1]-q2[j-1][i-1]))/dy+0.5*(S2[j][i]-S2[j-1][i]+S2[j][i-1]-S2[j-1][i-1])/dy;
           
         }
      }


    for(int j=0;j<=NY;j++)
      {
        for(int i=0;i<=NX;i++)
         {
		 f[j][i]=f[j][i]*(dx*dy/r);
         }
      }
k=0;

  for(int j=1;j<NY;j++)
      {
        for(int i=1;i<NX;i++)
         {
              rhs[k] = f[j][i];
              k++;
         }
      }

/*
 for (int i = 0; i < N; i++)
    {
      std::cout<<"rhs="<<rhs[i]<<"\n";
    }
*/
    CG(argc,argv,N,x,rhs,NY);
 

/*
  for(int i=0; i<N; i++){
	  cout<<x[i]<<"\n"; }
*/
    /*
   k=0;

  cout<<"\n";

  for(int j=1;j<NY;j++)
      {
        for(int i=1;i<NX;i++)
         {
          Un[j][i]=x[k];
	  cout<<Un[j][i]<<"\n";
	  //k=k++;
	  k++;
         }

      }
*/
/*

    for(int j=0;j<=NY;j++)
      {
        for(int i=0;i<=NX;i++)
         {
          cout<<Un[j][i]<<" ";
         }
        cout<<"\n";
      }
*/
    counter++;
    error=norm(x,a,N);
   // cout<<"error="<<error<<"\n";
}

cout<<"\n";
/*
for(int j=1;j<NY;j++)
      {
        for(int i=1;i<NX;i++)
         {
          cout<<Un[j][i]<<"\n";
         }

      }
*/
//   }while(tolerance<err);
cout<<"error="<<error<<"\n";
cout<<"counter="<<counter<<"\n";

CVS(Un,fp,NX,NY,dx,dy);

fclose(fp);

}

void CVS(double** T, FILE *fp,int NX, int NY,double delta_x, double delta_y)
{
/*
for(int j=1;j<NY;j++)
      {
        for(int i=1;i<NX;i++)
         {
          cout<<Un[j][i]<<"\n";
         }

      }
*/

 double x=0.0,y=0.0;

fprintf(fp, "\"x\", \"y\",\"U\"\n");
/*
for(int j=0;j<=NY;j++)
      {
        y=delta_y*(j-0.5);
        for(int i=0;i<=NX;i++)
         {
          x=delta_x*(i-0.5);
         // fprintf(fp, "%.10f","%.10f","%.10f",x,y,Un[j][i]);
          fprintf(fp, "%.10f,%.10f,%.10f \n",x,y,T[j][i]);
         }
       // fprintf(fp, "\n");
      }
*/
for(int j=1;j<NY;j++)
      {
        y=delta_y*(j-0.5);
        for(int i=1;i<NX;i++)
         {
          x=delta_x*(i-0.5);
         // fprintf(fp, "%.10f","%.10f","%.10f",x,y,Un[j][i]);
          fprintf(fp, "%.10f,%.10f,%.10f \n",x,y,T[j][i]);
         }
       // fprintf(fp, "\n");
      }


}


double norm(double *x,double *rhs,int N){

double error=0;

for(int i=1;i<N;i++)
         {
              error=error+((x[i]-rhs[i])*(x[i]-rhs[i])/(N*N)) ;

         }

  error=sqrt(error);

  return error;
}


void CG(int argc, char **argv, int N,double *x,double *rhs,double NY){

    const int max_iter = 1000;
    int k, M = 0, nz = 0, *I = NULL, *J = NULL;
    int *d_col, *d_row;
    int qatest = 0;
    const double tol = 1e-12f;
    double r0, r1, alpha, beta;
    double *d_val, *d_x;
    double *d_zm1, *d_zm2, *d_rm2;
    double *d_r, *d_p, *d_omega, *d_y;
    double *val = NULL;
    double *d_valsILU0;
    double *valsILU0;
    double rsum, diff, err = 0.0;
    double qaerr1, qaerr2 = 0.0;
    double dot, numerator, denominator, nalpha;
    const double doubleone = 1.0;
    const double doublezero = 0.0;
    int nErrors = 0;

    M=N;
  //  printf("conjugateGradientPrecond (in the CG function) starting...\n");

    /* QA testing mode */
    if (checkCmdLineFlag(argc, (const char **)argv, "qatest"))
    {
        qatest = 1;
    }

    /* This will pick the best possible CUDA capable device */
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);
    //printf("GPU selected Device ID = %d \n", devID);

    if (devID < 0)
    {
        printf("Invalid GPU device %d selected,  exiting...\n", devID);
        exit(EXIT_SUCCESS);
    }

    //checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    /* Statistics about the GPU device 
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
*/
    nz = 5*N-4*(int)sqrt((double)N);
    I = (int *)malloc(sizeof(int)*(N+1));                              // csr row pointers for matrix A
    J = (int *)malloc(sizeof(int)*nz);                                 // csr column indices for matrix A
    val = (double *)malloc(sizeof(double)*nz);                           // csr values for matrix A


    genLaplace(I, J, val, M, N, nz,NY);

/*
     cout<<"N="<<N<<"  nz="<<nz<<"\n";
     cout<<"val"<<"  column J"<<"\n";

     for(int i=0; i<nz; i++){
     cout<<val[i]<<"  "<<J[i]<<"\n";
    }
    cout<<"row I="<<"\n";
    cout<<"\n";

     for(int i=0; i<=N; i++){
     cout<<I[i]<<"\n";
    }


    for (int i = 0; i < N; i++)
    {
         std::cout<<"x="<<x[i]<<" rhs= "<<rhs[i]<<"\n";
    }

  */  

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    /* Description of the A matrix*/
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    /* Define the properties of the matrix */
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    /* Allocate required memory */
    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_y, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_omega, N*sizeof(double)));

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(double), cudaMemcpyHostToDevice);


   
    
    /* Preconditioned Conjugate Gradient using ILU.
       --------------------------------------------
       Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

  //  printf("\nConvergence of conjugate gradient using incomplete LU preconditioning: \n");

    int nzILU0 = 2*N-1;
    valsILU0 = (double *) malloc(nz*sizeof(double));

    checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nz*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_zm1, (N)*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_zm2, (N)*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_rm2, (N)*sizeof(double)));

    /* create the analysis info object for the A matrix */
    cusparseSolveAnalysisInfo_t infoA = 0;
    cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);

    checkCudaErrors(cusparseStatus);

    /* Perform the analysis for the Non-Transpose case */
    cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             N, nz, descr, d_val, d_row, d_col, infoA);

    checkCudaErrors(cusparseStatus);

    /* Copy A data to ILU0 vals as input*/
    cudaMemcpy(d_valsILU0, d_val, nz*sizeof(double), cudaMemcpyDeviceToDevice);

    /* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
    cusparseStatus = cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descr, d_valsILU0, d_row, d_col, infoA);

    checkCudaErrors(cusparseStatus);

    /* Create info objects for the ILU0 preconditioner */
    cusparseSolveAnalysisInfo_t info_u;
    cusparseCreateSolveAnalysisInfo(&info_u);

    cusparseMatDescr_t descrL = 0;
    cusparseStatus = cusparseCreateMatDescr(&descrL);
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseMatDescr_t descrU = 0;
    cusparseStatus = cusparseCreateMatDescr(&descrU);
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseStatus = cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val, d_row, d_col, info_u);

    /* reset the initial guess of the solution to zero */
  
    for (int i = 0; i < N; i++)
    {
        x[i] = 0.0;
    }

    checkCudaErrors(cudaMemcpy(d_r, rhs, N*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice));

    k = 0;
    cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    while (r1 > tol*tol && k <= max_iter)
    {
        // Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
        cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrL,
                                              d_valsILU0, d_row, d_col, infoA, d_r, d_y);
        checkCudaErrors(cusparseStatus);

        // Back Substitution
        cusparseStatus = cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &doubleone, descrU,
                                              d_valsILU0, d_row, d_col, info_u, d_y, d_zm1);
        checkCudaErrors(cusparseStatus);

        k++;

        if (k == 1)
        {
            cublasDcopy(cublasHandle, N, d_zm1, 1, d_p, 1);
        }
        else
        {
            cublasDdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator);
            cublasDdot(cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator);
            beta = numerator/denominator;
            cublasDscal(cublasHandle, N, &beta, d_p, 1);
            cublasDaxpy(cublasHandle, N, &doubleone, d_zm1, 1, d_p, 1) ;
        }

        cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nzILU0, &doubleone, descrU, d_val, d_row, d_col, d_p, &doublezero, d_omega);
        cublasDdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator);
        cublasDdot(cublasHandle, N, d_p, 1, d_omega, 1, &denominator);
        alpha = numerator / denominator;
        cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
        cublasDcopy(cublasHandle, N, d_r, 1, d_rm2, 1);
        cublasDcopy(cublasHandle, N, d_zm1, 1, d_zm2, 1);
        nalpha = -alpha;
        cublasDaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
        cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    }

  //  printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

    cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);

    //for(int i=0; i<N; i++){
    //cout<<x[i]<<"\n"; }


    /* check result */
    err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr2 = err;

    /* Destroy parameters */
    cusparseDestroySolveAnalysisInfo(infoA);
    cusparseDestroySolveAnalysisInfo(info_u);

    /* Destroy contexts */
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    /* Free device memory 
    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    free(valsILU0);
*/
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_omega);
    cudaFree(d_valsILU0);
    cudaFree(d_zm1);
    cudaFree(d_zm2);
    cudaFree(d_rm2);

  //  printf("  Test Summary:\n");
  //  printf("     Counted total of %d errors\n", nErrors);
 //   printf("     qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
 

  // exit((nErrors == 0 &&fabs(qaerr1)<1e-5 && fabs(qaerr2) < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE));


//return 0;

//for(int i=0; i<N; i++){
  //  cout<<x[i]<<"\n"; }


}

void genLaplace(int *row_ptr, int *col_ind, double *val, int M, int N, int nz,double NY)
{
    assert(M==N);
    int n=(int)sqrt((double)N);
    assert(n*n==N);
   // printf("laplace dimension = %d\n", n);
    int idx = 0;
    double dy,NX,dx,phi,L;

    dy=1.0/NY;
    NX=L*NY;
    dx=L/NX;
    phi=dy/dx;

    // loop over degrees of freedom
    for (int i=0; i<N; i++)
    {
        int ix = i%n;
        int iy = i/n;

        row_ptr[i] = idx;

        // up
        if (iy > 0)
        {
            val[idx] = -phi;
            col_ind[idx] = i-n;
            idx++;
        }
        else
        {
           // rhs[i] -= 1.0;
        }

        // left
        if (ix > 0)
        {
            val[idx] = -phi;
            col_ind[idx] = i-1;
            idx++;
        }
        else
        {
        //    rhs[i] -= 0.0;
        }

        // center
        val[idx] = 2.0*(phi + 1.0/phi);
        col_ind[idx]=i;
        idx++;

        //right
        if (ix  < n-1)
        {
            val[idx] = -phi;
            col_ind[idx] = i+1;
            idx++;
        }
        else
        {
          //  rhs[i] -= 0.0;
        }

        //down
        if (iy  < n-1)
        {
            val[idx] = -1.0;
            col_ind[idx] = i+n;
            idx++;
        }
        else
        {
            //rhs[i] -= 0.0;
        }

    }

    row_ptr[N] = idx;

}



