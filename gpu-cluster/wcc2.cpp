#include<string.h>
#include <stdio.h> 
#include "wcc2.h"
#include <CL/cl.h> 
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h> 
#define GetCurrentDir getcwd 
#endif
char cCurrentPath[FILENAME_MAX];
char * option;
char *source_str;
size_t source_size;
cl_int error;
size_t global_size;
cl_device_id* devices;
cl_uint deviceCount;
cl_uint device_no = 0;
void WCC(char *edge,char *part) {
	HGraph  hgraph ;
	hgraph.read_graph(edge,part);  
	MPI_Barrier(MPI_COMM_WORLD);
	alloc_extra_hgraph(hgraph,0);

	GGraph graph(devices[0]);
	hgraph.copy_to_device(graph);
	graph.program = clCreateProgramWithSource(graph.context, 1, (const char **)&source_str,(const size_t *)&source_size, &error);
	if(error != CL_SUCCESS) printf("error in create program\n");
	error = clBuildProgram(graph.program, 1, &graph.device_id, option, NULL, NULL);
	if(error != CL_SUCCESS)printf(" error in build program \n");
	alloc_extra_graph(graph); 

	graph.kernel = clCreateKernel(graph.program, "init", &error);
	if(error != CL_SUCCESS)printf("error in create kernel\n");
	error = clSetKernelArg(graph.kernel, 0, sizeof(int), &graph.property_size);
	if(error!= CL_SUCCESS)printf("error in set arg 0\n");
	error = clSetKernelArg(graph.kernel, 1, sizeof(int), &graph.nedges);
	if(error!= CL_SUCCESS)printf("error in set arg 1\n");
	error = clSetKernelArg(graph.kernel, 2, sizeof(cl_mem), &graph.edges);
	if(error!= CL_SUCCESS)printf("error in set arg 2\n");
	error = clSetKernelArg(graph.kernel, 3, sizeof(cl_mem), &graph.points);
	if(error!= CL_SUCCESS)printf("error in set arg 3\n");
	error = clSetKernelArg(graph.kernel, 4, sizeof(cl_mem), &graph.nbrs_index);
	if(error!= CL_SUCCESS)printf("error in set arg 4\n");
	error = clSetKernelArg(graph.kernel, 5, sizeof(cl_mem), &graph.properties[0]);
	if(error!= CL_SUCCESS)printf("error in set arg 5\n");
	error = clSetKernelArg(graph.kernel, 6, sizeof(cl_mem), &graph.properties[1]);
	if(error!= CL_SUCCESS)printf("error in set arg 6\n");
	error = clSetKernelArg(graph.kernel, 7, sizeof(cl_mem), &graph.properties[2]);
	if(error!= CL_SUCCESS)printf("error in set arg 7\n");
	error = clSetKernelArg(graph.kernel, 8, sizeof(cl_mem), &graph.properties[3]);
	if(error!= CL_SUCCESS)printf("error in set arg 8\n");
	int init_offset = 0;
	error = clSetKernelArg(graph.kernel, 9, sizeof(int), &init_offset);
	if(error!= CL_SUCCESS)printf("error in set arg 9\n");
	global_size = hgraph.property_size;
	if(global_size){
		clEnqueueNDRangeKernel(graph.command_queue,graph.kernel,1,NULL,&global_size,NULL,0,NULL,NULL);
		clFinish(graph.command_queue);
	}

	bool  falcvt1;
	falcvt1=true;
	if(hgraph.partition_table[0] == rank){
		clEnqueueWriteBuffer(graph.command_queue,graph.properties[2],CL_TRUE,sizeof(bool)*(hgraph.partition_indices[0]+hgraph.offset[rank]),sizeof(bool),(void *)&(falcvt1),0,NULL,NULL);
	}
	int   falcvt2;
	falcvt2=0;
	if(hgraph.partition_table[0] == rank){
		clEnqueueWriteBuffer(graph.command_queue,graph.properties[3],CL_TRUE,sizeof(int )*(hgraph.partition_indices[0]+hgraph.offset[rank]),sizeof(int ),(void *)&(falcvt2),0,NULL,NULL);
	}

	while(1){ 
		bool  falcvt3;
		falcvt3=false;
		clEnqueueWriteBuffer(graph.command_queue,graph.properties[0],CL_TRUE,sizeof(bool)*0,sizeof(bool),(void *)&(falcvt3),0,NULL,NULL);
		graph.kernel = clCreateKernel(graph.program, "relaxgraph", &error);
		if(error != CL_SUCCESS)printf("error in create kernel\n");
		error = clSetKernelArg(graph.kernel, 0, sizeof(int), &graph.npoints);
		if(error!= CL_SUCCESS)printf("error in set arg 0\n");
		error = clSetKernelArg(graph.kernel, 1, sizeof(int), &graph.nedges);
		if(error!= CL_SUCCESS)printf("error in set arg 1\n");
		error = clSetKernelArg(graph.kernel, 2, sizeof(cl_mem), &graph.edges);
		if(error!= CL_SUCCESS)printf("error in set arg 2\n");
		error = clSetKernelArg(graph.kernel, 3, sizeof(cl_mem), &graph.points);
		if(error!= CL_SUCCESS)printf("error in set arg 3\n");
		error = clSetKernelArg(graph.kernel, 4, sizeof(cl_mem), &graph.nbrs_index);
		if(error!= CL_SUCCESS)printf("error in set arg 4\n");
		error = clSetKernelArg(graph.kernel, 5, sizeof(cl_mem), &graph.properties[0]);
		if(error!= CL_SUCCESS)printf("error in set arg 5\n");
		error = clSetKernelArg(graph.kernel, 6, sizeof(cl_mem), &graph.properties[1]);
		if(error!= CL_SUCCESS)printf("error in set arg 6\n");
		error = clSetKernelArg(graph.kernel, 7, sizeof(cl_mem), &graph.properties[2]);
		if(error!= CL_SUCCESS)printf("error in set arg 7\n");
		error = clSetKernelArg(graph.kernel, 8, sizeof(cl_mem), &graph.properties[3]);
		if(error!= CL_SUCCESS)printf("error in set arg 8\n");
		error = clSetKernelArg(graph.kernel, 9, sizeof(int), &hgraph.offset[rank]);
		if(error!= CL_SUCCESS)printf("error in set arg 9\n");
		global_size = hgraph.no_of_vertices;
		if(global_size){
			clEnqueueNDRangeKernel(graph.command_queue,graph.kernel,1,NULL,&global_size,NULL,0,NULL,NULL);
			clFinish(graph.command_queue);
		}
		//mpi-communication
		clEnqueueReadBuffer(graph.command_queue,graph.properties[3],CL_TRUE,0,sizeof(int )*graph.property_size,(void *)(((struct struct_hgraph  *)hgraph.extra)->dist),0,NULL,NULL);
		for(int i=0;i<graph.no_of_nodes;i++){
			if(i==rank){
				for(int j=0;j<graph.no_of_nodes;j++){
					if(j!=i){
						MPI_Send((void *)((((struct struct_hgraph  *)hgraph.extra)->dist)+hgraph.offset[j]),hgraph.no_of_outgoing_vertices[j],MPI_INT ,j,0,MPI_COMM_WORLD);
					}
				}
			}		
			else{
				int  *rcv = (int  *)malloc(sizeof(int )*hgraph.no_of_incoming_vertices[i]);
				MPI_Recv(rcv,hgraph.no_of_incoming_vertices[i],MPI_INT ,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				for(int k=0;k<hgraph.no_of_incoming_vertices[i];k++){
					int index = hgraph.partition_indices[hgraph.incoming_vertices[i][k]]+hgraph.offset[rank];
					if(((struct struct_hgraph  *)hgraph.extra)->dist[index] > rcv[k])
						((struct struct_hgraph  *)hgraph.extra)->dist[index] = rcv[k];
				}
			}
		}

		clEnqueueWriteBuffer(graph.command_queue,graph.properties[3],CL_TRUE,sizeof(int )*hgraph.offset[rank],sizeof(int )*graph.npoints,(void *)((((struct struct_hgraph  *)hgraph.extra)->dist)+hgraph.offset[rank]),0,NULL,NULL);

		bool  falcvt4;//temp
		falcvt4=false;
		clEnqueueReadBuffer(graph.command_queue,graph.properties[0],CL_TRUE,sizeof(bool)*0,sizeof(bool),(void *)&(falcvt4),0,NULL,NULL);
		bool  falcvt5 = falcvt4;//rcvtemp
		MPI_Bcast(&falcvt5,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		falcvt4 = falcvt4 || falcvt5;
		falcvt5 = falcvt4 || falcvt5;
		MPI_Bcast(&falcvt5,1,MPI_C_BOOL,1,MPI_COMM_WORLD);
		falcvt4 = falcvt4 || falcvt5;
		falcvt5 = falcvt4 || falcvt5;
		if(falcvt4==false)break;
		graph.kernel = clCreateKernel(graph.program, "reset", &error);
		if(error != CL_SUCCESS)printf("error in create kernel\n");
		error = clSetKernelArg(graph.kernel, 0, sizeof(int), &graph.npoints);
		if(error!= CL_SUCCESS)printf("error in set arg 0\n");
		error = clSetKernelArg(graph.kernel, 1, sizeof(int), &graph.nedges);
		if(error!= CL_SUCCESS)printf("error in set arg 1\n");
		error = clSetKernelArg(graph.kernel, 2, sizeof(cl_mem), &graph.edges);
		if(error!= CL_SUCCESS)printf("error in set arg 2\n");
		error = clSetKernelArg(graph.kernel, 3, sizeof(cl_mem), &graph.points);
		if(error!= CL_SUCCESS)printf("error in set arg 3\n");
		error = clSetKernelArg(graph.kernel, 4, sizeof(cl_mem), &graph.nbrs_index);
		if(error!= CL_SUCCESS)printf("error in set arg 4\n");
		error = clSetKernelArg(graph.kernel, 5, sizeof(cl_mem), &graph.properties[0]);
		if(error!= CL_SUCCESS)printf("error in set arg 5\n");
		error = clSetKernelArg(graph.kernel, 6, sizeof(cl_mem), &graph.properties[1]);
		if(error!= CL_SUCCESS)printf("error in set arg 6\n");
		error = clSetKernelArg(graph.kernel, 7, sizeof(cl_mem), &graph.properties[2]);
		if(error!= CL_SUCCESS)printf("error in set arg 7\n");
		error = clSetKernelArg(graph.kernel, 8, sizeof(cl_mem), &graph.properties[3]);
		if(error!= CL_SUCCESS)printf("error in set arg 8\n");
		error = clSetKernelArg(graph.kernel, 9, sizeof(int), &hgraph.offset[rank]);
		if(error!= CL_SUCCESS)printf("error in set arg 9\n");
		global_size = hgraph.no_of_vertices;
		if(global_size){
			clEnqueueNDRangeKernel(graph.command_queue,graph.kernel,1,NULL,&global_size,NULL,0,NULL,NULL);
			clFinish(graph.command_queue);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return ;
}


int main(int argc,char *argv[]) {
	MPI_Init (NULL,NULL);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
	is_all_selective = true;
	FILE *fp;
	char fileName[] = "wcc2.cl";
	fp = fopen(fileName, "r");
	fseek(fp, 0L, SEEK_END);
	source_size = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	source_str = (char*)malloc(source_size);
	source_size = fread(source_str, 1, source_size, fp);
	fclose(fp);
	cl_platform_id* platforms;
	cl_uint platformCount;
	clGetPlatformIDs(0, NULL, &platformCount);
	platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);
	cl_uint device;
	for (int i = 0; i < platformCount; i++) {
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device);
		deviceCount += device;}
	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
	cl_uint index = 0;
	if(rank < 0){
		for (int i = 0; i < platformCount; i++) {
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &device);
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, device, devices+index, NULL);
			index += device;
		}
	}
	else{
		for (int i = 0; i < platformCount; i++) {
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &device);
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, device, devices+index, NULL);
			index += device;
		}
	}
	GetCurrentDir(cCurrentPath, sizeof(cCurrentPath));
	cCurrentPath[sizeof(cCurrentPath) - 1] = '\0';
	option = (char *)malloc((FILENAME_MAX+3)*sizeof(char));
	sprintf(option,"-w -I/%s",cCurrentPath);

	WCC(argv[1],argv[2]);//rhs not null
	MPI_Finalize();
	return 0;
}
