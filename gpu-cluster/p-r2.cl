#include "p-r2.global.h"
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
void AtomicAdd(__global double *val, double delta) {
	union {
		double f;
		ulong  i;
	} old;
	union {
		double f;
		ulong  i;
	} new;
	do {
		old.f = *val;
		new.f = old.f + delta;
	} while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);
}
union float_int 
{
	int ipe;
	float fpe;
};
__kernel  void   relaxgraph ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global double  * pr,__global double  * sum,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		int falcft0=nbrs_index[id+1]-nbrs_index[id];
		int falcft1=nbrs_index[id];
		for(int falcft2=0;falcft2<falcft0;falcft2++){
			int ut0=2*(falcft1+falcft2);
			int ut1=edges[ut0].ipe;
			int ut2=edges[ut0+1].ipe;
			AtomicAdd(&(sum[ut1]),pr[id+point_offset]/falcft0);//rhs not null
		}
	}
}

__kernel  void init( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global double  * pr,__global double  * sum,int point_offset){
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		pr[id+point_offset]=(double)1/(double)npoints; 
		sum[id+point_offset]=0.000000;
	}
}

__kernel  void   reset ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global double  * pr,__global double  * sum,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		pr[id+point_offset]=0.850000*sum[id+point_offset]+(double)0.150000/npoints; 
	}
}

