union float_int 
{
	int ipe;
	float fpe;
};
__kernel  void   relaxgraph ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global bool * changed,__global int  * olddist,__global bool * updated,__global int  * dist,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints && updated[id+point_offset] ){
		updated[id+point_offset]=false; 
		int falcft0=nbrs_index[id+1]-nbrs_index[id];
		int falcft1=nbrs_index[id];
		for(int falcft2=0;falcft2<falcft0;falcft2++){
			int ut0=2*(falcft1+falcft2);
			int ut1=edges[ut0].ipe;
			int ut2=edges[ut0+1].ipe;
			ch0= dist[ut1];
			atomic_min(&(dist[ut1]),dist[id+point_offset]+ut2);
			if(dist[ut1]<ch0 && changed[0]!=1) changed[0] = 1;
		}

	} 
}

__kernel  void   init ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global bool * changed,__global int  * olddist,__global bool * updated,__global int  * dist,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		dist[id+point_offset] = INT_MAX; 
		olddist[id+point_offset]= INT_MAX; 
		updated[id+point_offset]=false; 
	} 
}

__kernel  void   reset ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global bool * changed,__global int  * olddist,__global bool * updated,__global int  * dist,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		if( dist[id+point_offset]<olddist[id+point_offset] )
			updated[id+point_offset]=true; 
		olddist[id+point_offset]=dist[id+point_offset]; 
	}
}

