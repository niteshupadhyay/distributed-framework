#include <mpi.h> 
#include  "read_distributed_graph.h"
#include <omp.h> 
#include<sys/time.h>
#include<unistd.h>
#include <CL/cl.h> 
#include <iostream>
#include <fstream>
#include <string>
struct struct_hgraph { 
	double   *pr ;//has to given size of property type
	double   *sum ;//has to given size of property type
};
void alloc_extra_hgraph(HGraph &hgraph,int flag) {
	if(flag==0)hgraph.extra=(struct struct_hgraph  *)malloc(sizeof(struct struct_hgraph )) ;
	((struct struct_hgraph  *)hgraph.extra)->pr=(double  *)malloc(sizeof(double ) * hgraph.property_size);
	((struct struct_hgraph  *)hgraph.extra)->sum=(double  *)malloc(sizeof(double ) * hgraph.property_size);
}

void alloc_extra_graph(GGraph &graph){
	graph.properties = (cl_mem *)malloc(2 * sizeof(cl_mem));
	graph.properties[0] = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(double )* graph.property_size, NULL, NULL);
	graph.properties[1] = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(double )* graph.property_size, NULL, NULL);
}
