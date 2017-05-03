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
	bool  *changed ;//has to given size of property type
	int   *olddist ;//has to given size of property type
	bool  *updated ;//has to given size of property type
	int   *dist ;//has to given size of property type
	int   *old_dist ;//has to given size of property type
};
void alloc_extra_hgraph(HGraph &hgraph,int flag) {
	if(flag==0)hgraph.extra=(struct struct_hgraph  *)malloc(sizeof(struct struct_hgraph )) ;
	((struct struct_hgraph  *)hgraph.extra)->changed=(bool *)malloc(sizeof(bool) * 1) ;
	((struct struct_hgraph  *)hgraph.extra)->olddist=(int  *)malloc(sizeof(int ) * hgraph.property_size);
	((struct struct_hgraph  *)hgraph.extra)->updated=(bool *)malloc(sizeof(bool) * hgraph.property_size);
	((struct struct_hgraph  *)hgraph.extra)->dist=(int  *)malloc(sizeof(int ) * hgraph.property_size);
	((struct struct_hgraph  *)hgraph.extra)->old_dist=(int  *)malloc(sizeof(int ) * hgraph.property_size);
}

void alloc_extra_graph(GGraph &graph){
	graph.properties = (cl_mem *)malloc(4 * sizeof(cl_mem));
	graph.properties[0] = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(bool)* 1, NULL, NULL);
	graph.properties[1] = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(int )* graph.property_size, NULL, NULL);
	graph.properties[2] = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(bool)* graph.property_size, NULL, NULL);
	graph.properties[3] = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(int )* graph.property_size, NULL, NULL);
}
