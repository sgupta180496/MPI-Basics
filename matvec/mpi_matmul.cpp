#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <mpi.h>
#include <math.h>

using namespace std;

float genA (int row, int col) 
{
  	if (row > col)
    		return 1.0;
  	else
   		return 0.0;
}

float genx0 (int i) 
{
  	return 1.0;
}

void checkx (int iter, long i, float xval) 
{
  	if (iter == 1) 
	{
    		float shouldbe=i;
    		if (fabs(xval/shouldbe) > 1.01 || fabs(xval/shouldbe) < .99 )
      			cout<<"incorrect : x["<<i<<"] at iteration "<<iter<<" should be "<<shouldbe<<" not "<<xval<<endl;
  	}

  	if (iter == 2) 
	{
    		float shouldbe =(i-1)*i/2;
    		if (fabs(xval/shouldbe) > 1.01 || fabs(xval/shouldbe) < .99)
      		cout<<"incorrect : x["<<i<<"] at iteration "<<iter<<" should be "<<shouldbe<<" not "<<xval<<endl;
  	}
}

void matmul(float*A, float*x, float*y, long n) 
{
  
  	for(long row=0;row<n;++row) 
	{
    		float sum=0;
    
    		for (long col=0;col<n;++col) 
		{
       			sum+=x[col]*A[row*n+col];
    		}
       		y[row]=sum;
  	}
}

int main(int argc, char* argv[]) 
{

  	if (argc < 3) 
	{
    		std::cout<<"usage: "<<argv[0]<<" <n> <iteration>"<<std::endl;
  	}
  
   	MPI_Init(&argc,&argv);
   	long n=atol(argv[1]);
   	int iter=atoi(argv[2]);
   	bool check=true;
   	int rank,size;
   	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   	MPI_Comm_size(MPI_COMM_WORLD,&size);
   	int p=sqrt(size);
   	long partition=n/p; 
   	int row_partition=rank/p,col_partition=rank%p;
   
   	MPI_Comm comm_r;
   	int rowrank;
   	MPI_Comm_split(MPI_COMM_WORLD,row_partition,rank,&comm_r);
   	MPI_Comm_rank(comm_r,&rowrank);

   	MPI_Comm comm_c;
   	int colrank;
   	MPI_Comm_split(MPI_COMM_WORLD,col_partition,rank,&comm_c);
   	MPI_Comm_rank(comm_c,&colrank);
	
	float* A = new float[partition*partition];
   	long rowstart=(row_partition*partition),colstart=(col_partition*partition);
   	long end_r=rowstart+partition,end_c=colstart+partition;

  	for(long row=rowstart,rset=0;row<end_r;row++,rset++) 
	{
    		for(long col=colstart,cset=0;col<end_c;col++,cset++) 
		{
      			A[(rset*partition)+cset]=genA(row, col);
    		}
  	}	
  	float* x=new float[partition];
  	for(long i=0;i<partition;i++)
    	{
     		x[i]=genx0(i);
    	}
  	float* y=new float[partition];
  	for(long i=0;i<partition;i++)
    		y[i]=0.0;
 
   	double start=MPI_Wtime(); 
   	for(int j=0;j<iter;j++) 
   	{
      		matmul(A,x,y,partition);
      		MPI_Reduce(y,x,partition,MPI_FLOAT,MPI_SUM,row_partition,comm_r);
      		MPI_Bcast(x,partition,MPI_FLOAT,col_partition, comm_c);
  		if(check)
		{
    			for(long i=colstart,p=0;i<end_c;++i,++p)
			{
        			checkx (j+1, i, x[p]);
             		}
          	}
  	}
  
  	if(rank == 0)
	{
        	double end=MPI_Wtime(); 
        	cerr<<end-start<<endl;
   	}
 
  	MPI_Comm_free(&comm_r);
  	MPI_Comm_free(&comm_c);

  	delete[] A;
  	delete[] x;
  	delete[] y;
  	MPI_Finalize();
 
  	return 0;
}
