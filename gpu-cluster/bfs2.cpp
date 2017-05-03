#include<string.h>
#include <stdio.h> 
#include "bfs2.h"
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

void BFS(char *edge,char *part){
	HGraph  hgraph ;
	int   temp =0;
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

	cl_mem coll1 = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(int)*graph.property_size, NULL, NULL);
	cl_mem coll1_size = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(int), NULL, NULL);
	cl_mem coll2 = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(int)*graph.property_size, NULL, NULL);
	cl_mem coll2_size = clCreateBuffer(graph.context,CL_MEM_READ_WRITE,sizeof(int), NULL, NULL);
	clEnqueueWriteBuffer(graph.command_queue,coll1_size,CL_TRUE,0,sizeof(int),(void *)&(temp),0,NULL,NULL);
	clEnqueueWriteBuffer(graph.command_queue,coll2_size,CL_TRUE,0,sizeof(int),(void *)&(temp),0,NULL,NULL);
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
	int init_offset = 0;
	error = clSetKernelArg(graph.kernel, 7, sizeof(int), &init_offset);
	if(error!= CL_SUCCESS)printf("error in set arg 7\n");
	global_size = hgraph.property_size;
	if(global_size){
		clEnqueueNDRangeKernel(graph.command_queue,graph.kernel,1,NULL,&global_size,NULL,0,NULL,NULL);
		clFinish(graph.command_queue);
	}

	int   falcvt1;
	falcvt1=0;
	if(hgraph.partition_table[0] == rank){
		clEnqueueWriteBuffer(graph.command_queue,graph.properties[1],CL_TRUE,sizeof(int )*(hgraph.partition_indices[0]+hgraph.offset[rank]),sizeof(int ),(void *)&(falcvt1),0,NULL,NULL);
	}
	if(hgraph.partition_table[0] == rank){
		int falcvt2;
		clEnqueueReadBuffer(graph.command_queue,coll1_size,CL_TRUE,0,sizeof(int),&falcvt2, 0, NULL, NULL);
		int falcvt3 =0;
		clEnqueueWriteBuffer(graph.command_queue,coll1,CL_TRUE,falcvt2,sizeof(int),&falcvt3, 0, NULL, NULL);
		falcvt2++;
		clEnqueueWriteBuffer(graph.command_queue,coll1_size,CL_TRUE,0,sizeof(int),&falcvt2, 0, NULL, NULL);
	}

	int   val =1;
	int falcvt4;
	clEnqueueReadBuffer(graph.command_queue,coll1_size,CL_TRUE,0,sizeof(int),(void *)&(falcvt4),0,NULL,NULL);
	global_size = falcvt4;
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
	error = clSetKernelArg(graph.kernel, 7, sizeof(int), &hgraph.offset[rank]);
	if(error!= CL_SUCCESS)printf("error in set arg 7\n");
	error = clSetKernelArg(graph.kernel, 8, sizeof(cl_mem), &coll1);
	if(error!= CL_SUCCESS)printf("error in set arg 8\n");
	error = clSetKernelArg(graph.kernel, 9, sizeof(cl_mem), &coll1_size);
	if(error!= CL_SUCCESS)printf("error in set arg 9\n");
	error = clSetKernelArg(graph.kernel, 10, sizeof(cl_mem), &coll2);
	if(error!= CL_SUCCESS)printf("error in set arg 10\n");
	error = clSetKernelArg(graph.kernel, 11, sizeof(cl_mem), &coll2_size);
	if(error!= CL_SUCCESS)printf("error in set arg 11\n");
	error = clSetKernelArg(graph.kernel, 12, sizeof(int ), &val);
	if(error!= CL_SUCCESS)printf("error in set arg 12\n");
	if(global_size){
		clEnqueueNDRangeKernel(graph.command_queue,graph.kernel,1,NULL,&global_size,NULL,0,NULL,NULL);
		clFinish(graph.command_queue);
	}

	//mpi-communication coll
	clEnqueueReadBuffer(graph.command_queue,graph.properties[1],CL_TRUE,0,sizeof(int )*graph.property_size,(void *)(((struct struct_hgraph  *)hgraph.extra)->dist),0,NULL,NULL);
	int coll_size;
	clEnqueueReadBuffer(graph.command_queue,coll2_size,CL_TRUE,0,sizeof(int),&coll_size,0,NULL,NULL);
	int host_coll2[hgraph.property_size];
	clEnqueueReadBuffer(graph.command_queue,coll2,CL_TRUE,0,sizeof(int)*coll_size,host_coll2,0,NULL,NULL);
	int send_coll_point[hgraph.no_of_nodes][coll_size];
	int send_prop[hgraph.no_of_nodes][coll_size];
	int send_size[hgraph.no_of_nodes];
	int recv_size[hgraph.no_of_nodes];
	for(int i=0;i<hgraph.no_of_nodes;i++){
		send_size[i] = recv_size[i] = 0;
	}
	//cook the data to be sent
	int tempid;
	for(int i=0;i<coll_size;i++){
		int j;
		for(j=0;j<hgraph.no_of_nodes;j++){
			if(hgraph.offset[j]>host_coll2[i])
				break;
		}
		j--;
		if(j!=rank){
			tempid=hgraph.outgoing_vertices[j][host_coll2[i]-hgraph.offset[j]];
			send_coll_point[j][send_size[j]]=tempid;
			send_prop[j][send_size[j]++]=((struct struct_hgraph  *)(hgraph.extra))->dist[host_coll2[i]];
		}
		else{
			tempid=host_coll2[i]-hgraph.offset[j];
			send_coll_point[j][send_size[j]++]=tempid;
		}
	}
	//now send and receive size value
	for(int i=0;i<hgraph.no_of_nodes;i++){
		if(i==rank){
			for(int j=0;j<hgraph.no_of_nodes;j++){
				if(j!=i)
					MPI_Send(&send_size[j],1,MPI_INT,j,0,MPI_COMM_WORLD);
			}
		}else {
			MPI_Recv(&recv_size[i],1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	//allocate the space for recv_coll_point
	int *recv_coll_point[hgraph.no_of_nodes];
	int *recv_prop[hgraph.no_of_nodes];
	for(int i=0;i<hgraph.no_of_nodes;i++){
		if(i!=rank){
			recv_coll_point[i]=(int *)malloc(sizeof(int)*recv_size[i]);
			recv_prop[i]=(int *)malloc(sizeof(int)*recv_size[i]);
		}
	}
	//send and receive actual collection point and property value
	for(int i=0;i<hgraph.no_of_nodes;i++){
		if(i==rank){
			for(int j=0;j<hgraph.no_of_nodes;j++){
				if(j!=i){
					MPI_Send(send_coll_point[j],send_size[j],MPI_INT,j,0,MPI_COMM_WORLD);
					MPI_Send(send_prop[j],send_size[j],MPI_INT,j,0,MPI_COMM_WORLD);
				}
			}
		}else {
			MPI_Recv(recv_coll_point[i],recv_size[i],MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			MPI_Recv(recv_prop[i],recv_size[i],MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
	}
	//copy back the actual collection
	bool *bitvalue=(bool *)malloc(sizeof(bool)*hgraph.no_of_vertices);
	memset(bitvalue,false,sizeof(bitvalue));
	int *host_temp_coll2=(int *)malloc(sizeof(int)*hgraph.no_of_vertices);
	int host_temp_coll2_size=0;
	for(int i=0;i<hgraph.no_of_nodes;i++){
		if(i==rank){
			for(int j=0;j<send_size[i];j++){
				if(!bitvalue[send_coll_point[i][j]]){
					bitvalue[send_coll_point[i][j]]=true;
					host_temp_coll2[host_temp_coll2_size++]=send_coll_point[i][j];
				}
			}
		}else {
			for(int j=0;j<recv_size[i];j++){
				int find_prop_index = hgraph.offset[hgraph.partition_table[recv_coll_point[i][j]]];
				find_prop_index = find_prop_index + hgraph.partition_indices[recv_coll_point[i][j]];
				if(((struct struct_hgraph  *)(hgraph.extra))->dist[find_prop_index] > recv_prop[i][j]){
					((struct struct_hgraph  *)(hgraph.extra))->dist[find_prop_index] = recv_prop[i][j];
					if(!bitvalue[hgraph.partition_indices[recv_coll_point[i][j]]]){
						bitvalue[hgraph.partition_indices[recv_coll_point[i][j]]] = true;
						host_temp_coll2[host_temp_coll2_size++] = hgraph.partition_indices[recv_coll_point[i][j]];
					}
				}
			}
		}
	}
	//copy the collection value to GPU
	clEnqueueWriteBuffer(graph.command_queue,coll2_size,CL_TRUE,0,sizeof(int),&host_temp_coll2_size,0,NULL,NULL);
	clEnqueueWriteBuffer(graph.command_queue,coll2,CL_TRUE,0,sizeof(int)*host_temp_coll2_size,host_temp_coll2,0,NULL,NULL);
	clEnqueueWriteBuffer(graph.command_queue,graph.properties[1],CL_TRUE,sizeof(int )*hgraph.offset[rank],sizeof(int )*graph.npoints,(void *)((((struct struct_hgraph  *)hgraph.extra)->dist)+hgraph.offset[rank]),0,NULL,NULL);

	while(1){ 
		bool  falcvt5;
		falcvt5=false;
		clEnqueueWriteBuffer(graph.command_queue,graph.properties[0],CL_TRUE,sizeof(bool)*0,sizeof(bool),(void *)&(falcvt5),0,NULL,NULL);

		int falcvt6;
		clEnqueueReadBuffer(graph.command_queue,coll2_size,CL_TRUE,0,sizeof(int),&falcvt6, 0, NULL, NULL);
		int * falcvt7 = (int *)malloc(sizeof(int)*graph.property_size);
		clEnqueueReadBuffer(graph.command_queue,coll2,CL_TRUE,0,sizeof(int)*falcvt6,(void *)(falcvt7),0,NULL,NULL);
		clEnqueueWriteBuffer(graph.command_queue,coll1,CL_TRUE,0,sizeof(int)*falcvt6,(void *)(falcvt7),0,NULL,NULL);
		clEnqueueWriteBuffer(graph.command_queue,coll1_size,CL_TRUE,0,sizeof(int),(void *)&(falcvt6),0,NULL,NULL);

		temp=0;
		clEnqueueWriteBuffer(graph.command_queue,coll2_size,CL_TRUE,0,sizeof(int),(void *)&(temp),0,NULL,NULL);
		val++;
		int falcvt8;
		clEnqueueReadBuffer(graph.command_queue,coll1_size,CL_TRUE,0,sizeof(int),(void *)&(falcvt8),0,NULL,NULL);
		global_size = falcvt8;
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
		error = clSetKernelArg(graph.kernel, 7, sizeof(int), &hgraph.offset[rank]);
		if(error!= CL_SUCCESS)printf("error in set arg 7\n");
		error = clSetKernelArg(graph.kernel, 8, sizeof(cl_mem), &coll1);
		if(error!= CL_SUCCESS)printf("error in set arg 8\n");
		error = clSetKernelArg(graph.kernel, 9, sizeof(cl_mem), &coll1_size);
		if(error!= CL_SUCCESS)printf("error in set arg 9\n");
		error = clSetKernelArg(graph.kernel, 10, sizeof(cl_mem), &coll2);
		if(error!= CL_SUCCESS)printf("error in set arg 10\n");
		error = clSetKernelArg(graph.kernel, 11, sizeof(cl_mem), &coll2_size);
		if(error!= CL_SUCCESS)printf("error in set arg 11\n");
		error = clSetKernelArg(graph.kernel, 12, sizeof(int ), &val);
		if(error!= CL_SUCCESS)printf("error in set arg 12\n");
		if(global_size){
			clEnqueueNDRangeKernel(graph.command_queue,graph.kernel,1,NULL,&global_size,NULL,0,NULL,NULL);
			clFinish(graph.command_queue);
		}
		//mpi-communication coll
		clEnqueueReadBuffer(graph.command_queue,graph.properties[1],CL_TRUE,0,sizeof(int )*graph.property_size,(void *)(((struct struct_hgraph  *)hgraph.extra)->dist),0,NULL,NULL);
		int coll_size;
		clEnqueueReadBuffer(graph.command_queue,coll2_size,CL_TRUE,0,sizeof(int),&coll_size,0,NULL,NULL);
		int host_coll2[hgraph.property_size];
		clEnqueueReadBuffer(graph.command_queue,coll2,CL_TRUE,0,sizeof(int)*coll_size,host_coll2,0,NULL,NULL);
		int send_coll_point[hgraph.no_of_nodes][coll_size];
		int send_prop[hgraph.no_of_nodes][coll_size];
		int send_size[hgraph.no_of_nodes];
		int recv_size[hgraph.no_of_nodes];
		for(int i=0;i<hgraph.no_of_nodes;i++){
			send_size[i] = recv_size[i] = 0;
		}
		//cook the data to be sent
		int tempid;
		for(int i=0;i<coll_size;i++){
			int j;
			for(j=0;j<hgraph.no_of_nodes;j++){
				if(hgraph.offset[j]>host_coll2[i])
					break;
			}
			j--;
			if(j!=rank){
				tempid=hgraph.outgoing_vertices[j][host_coll2[i]-hgraph.offset[j]];
				send_coll_point[j][send_size[j]]=tempid;
				send_prop[j][send_size[j]++]=((struct struct_hgraph  *)(hgraph.extra))->dist[host_coll2[i]];
			}
			else{
				tempid=host_coll2[i]-hgraph.offset[j];
				send_coll_point[j][send_size[j]++]=tempid;
			}
		}

		end = rtclock();
		compute += (end-start);
		start = rtclock();

		//now send and receive size value
		for(int i=0;i<hgraph.no_of_nodes;i++){
			if(i==rank){
				for(int j=0;j<hgraph.no_of_nodes;j++){
					if(j!=i)
						MPI_Send(&send_size[j],1,MPI_INT,j,0,MPI_COMM_WORLD);
				}
			}else {
				MPI_Recv(&recv_size[i],1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}

		//allocate the space for recv_coll_point
		int *recv_coll_point[hgraph.no_of_nodes];
		int *recv_prop[hgraph.no_of_nodes];
		for(int i=0;i<hgraph.no_of_nodes;i++){
			if(i!=rank){
				recv_coll_point[i]=(int *)malloc(sizeof(int)*recv_size[i]);
				recv_prop[i]=(int *)malloc(sizeof(int)*recv_size[i]);
			}
		}

		//send and receive actual collection point and property value
		for(int i=0;i<hgraph.no_of_nodes;i++){
			if(i==rank){
				for(int j=0;j<hgraph.no_of_nodes;j++){
					if(j!=i){
						MPI_Send(send_coll_point[j],send_size[j],MPI_INT,j,0,MPI_COMM_WORLD);
						MPI_Send(send_prop[j],send_size[j],MPI_INT,j,0,MPI_COMM_WORLD);
					}
				}
			}else {
				MPI_Recv(recv_coll_point[i],recv_size[i],MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(recv_prop[i],recv_size[i],MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			}
		}
		//copy back the actual collection
		bool *bitvalue=(bool *)malloc(sizeof(bool)*hgraph.no_of_vertices);
		memset(bitvalue,false,sizeof(bitvalue));
		int *host_temp_coll2=(int *)malloc(sizeof(int)*hgraph.no_of_vertices);
		int host_temp_coll2_size=0;
		for(int i=0;i<hgraph.no_of_nodes;i++){
			if(i==rank){
				for(int j=0;j<send_size[i];j++){
					if(!bitvalue[send_coll_point[i][j]]){
						bitvalue[send_coll_point[i][j]]=true;
						host_temp_coll2[host_temp_coll2_size++]=send_coll_point[i][j];
					}
				}
			}else {
				for(int j=0;j<recv_size[i];j++){
					int find_prop_index = hgraph.offset[hgraph.partition_table[recv_coll_point[i][j]]];
					find_prop_index = find_prop_index + hgraph.partition_indices[recv_coll_point[i][j]];
					if(((struct struct_hgraph  *)(hgraph.extra))->dist[find_prop_index] > recv_prop[i][j]){
						((struct struct_hgraph  *)(hgraph.extra))->dist[find_prop_index] = recv_prop[i][j];
						if(!bitvalue[hgraph.partition_indices[recv_coll_point[i][j]]]){
							bitvalue[hgraph.partition_indices[recv_coll_point[i][j]]] = true;
							host_temp_coll2[host_temp_coll2_size++] = hgraph.partition_indices[recv_coll_point[i][j]];
						}
					}
				}
			}
		}
		//copy the collection value to GPU
		clEnqueueWriteBuffer(graph.command_queue,coll2_size,CL_TRUE,0,sizeof(int),&host_temp_coll2_size,0,NULL,NULL);
		clEnqueueWriteBuffer(graph.command_queue,coll2,CL_TRUE,0,sizeof(int)*host_temp_coll2_size,host_temp_coll2,0,NULL,NULL);
		clEnqueueWriteBuffer(graph.command_queue,graph.properties[1],CL_TRUE,sizeof(int )*hgraph.offset[rank],sizeof(int )*graph.npoints,(void *)((((struct struct_hgraph  *)hgraph.extra)->dist)+hgraph.offset[rank]),0,NULL,NULL);


		bool  falcvt9;//temp
		falcvt9=false;
		clEnqueueReadBuffer(graph.command_queue,graph.properties[0],CL_TRUE,sizeof(bool)*0,sizeof(bool),(void *)&(falcvt9),0,NULL,NULL);
		bool  falcvt10 = falcvt9;//rcvtemp

		MPI_Bcast(&falcvt10,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		falcvt9 = falcvt9 || falcvt10;
		falcvt10 = falcvt9 || falcvt10;
		MPI_Bcast(&falcvt10,1,MPI_C_BOOL,1,MPI_COMM_WORLD);
		falcvt9 = falcvt9 || falcvt10;
		falcvt10 = falcvt9 || falcvt10;
		if(falcvt9==false)break;

	}//end of block
	MPI_Barrier(MPI_COMM_WORLD);
} 

int main(int argc,char *argv[]){
	MPI_Init (NULL,NULL);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
	is_all_selective = true;
	FILE *fp;
	char fileName[] = "bfs2.cl";
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

	BFS(argv[1],argv[2]);//rhs not null

	MPI_Finalize();
} 
