#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <mpi.h>

using namespace std;

#ifdef __cplusplus
extern "C" 
{
	#endif
	float f1(float x, int intensity);
	float f2(float x, int intensity);
	float f3(float x, int intensity);
	float f4(float x, int intensity);
	#ifdef __cplusplus
}

#endif  
int main(int argc, char* argv[]) 
{
  
  	if (argc < 6) 
	{
    		std::cerr<<"usage: "<<argv[0]<<" <functionid> <a> <b> <n> <intensity>"<<std::endl;
    		return -1;
  	}
  
  	MPI_Init(&argc,&argv);
  	int functionid=atoi(argv[1]);
  	float a=atof(argv[2]);
  	float b=atof(argv[3]);
  	int n=atoi(argv[4]);
  	int intensity=atoi(argv[5]);
  	int rank,size;
  	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  	MPI_Comm_size(MPI_COMM_WORLD,&size);
  	float (*res)(float,int);
  	float div=(b-a)/n;
  	int first,last,partition;
  	double start = MPI_Wtime(); 
  	partition = n/size;
  	first = rank*partition; 
	last = (rank+1)*partition;
  	if(rank == size-1)
	last = n;
  	switch(functionid)
  	{
     		case 1 : res = &f1;
 	               	 break;

     		case 2 : res = &f2;
                	 break;

     		case 3 : res = &f3;
                	 break;

     		case 4 : res = &f4;
                	 break;

     		default : cerr<<"wrong functionid"<<endl; 
                return -1;                

   	};
   
	float par_compute=0.0,compute=0.0;
   	for(int i=first;i<last;i++)
        {
           	float x=a+((i+.5)*div);
	       	par_compute+=(*res)(x,intensity); 
        }
   	MPI_Reduce(&par_compute,&compute,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
   	if(rank == 0)
   	{
	   	compute=compute*div;
	   	cout<<compute<<endl;
           	double end = MPI_Wtime(); 
           	cerr<<end-start<<endl;
   	}
     
   	MPI_Finalize();
  	return 0;
}
