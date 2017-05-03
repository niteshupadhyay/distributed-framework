union float_int 
{
	int ipe;
	float fpe;
};
__kernel  void   init_left ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global int * count,__global bool * changed,__global int  * suit,__global int  * match,__global int  * left,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		int falcft0=nbrs_index[id+1]-nbrs_index[id];
		int falcft1=nbrs_index[id];
		for(int falcft2=0;falcft2<falcft0;falcft2++){
			int ut0=2*(falcft1+falcft2);
			int ut1=edges[ut0].ipe;
			int ut2=edges[ut0+1].ipe;

			left[id+point_offset]=1; 
			break;
		}
	} 

}


__kernel  void   init ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global int * count,__global bool * changed,__global int  * suit,__global int  * match,__global int  * left,int point_offset) {
	int ch0 ; 
	count[0] = 0;
	int id = get_global_id(0);
	if( id < npoints){
		left[id+point_offset]=0; 
		match[id+point_offset]=-1; 
		suit[id+point_offset]=-1; 
	}
}

__kernel  void   first_shake ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global int * count,__global bool * changed,__global int  * suit,__global int  * match,__global int  * left,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		if( left[id+point_offset]&&match[id+point_offset]==-1 )
		{
			int falcft3=nbrs_index[id+1]-nbrs_index[id];
			int falcft4=nbrs_index[id];
			for(int falcft5=0;falcft5<falcft3;falcft5++){
				int ut3=2*(falcft4+falcft5);
				int ut4=edges[ut3].ipe;
				int ut5=edges[ut3+1].ipe;

				if( match[ut4]==-1 && point_offset <= ut4 <= point_offset+npoints )
				{
					suit[ut4]=id+point_offset; 
					if( !changed[0] )
						changed[0]=1; 
				}
			}
		}
	}
}

__kernel  void   second_shake ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global int * count,__global bool * changed,__global int  * suit,__global int  * match,__global int  * left,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		if( !left[id+point_offset]&&match[id+point_offset]==-1&&suit[id+point_offset]!=-1 )
		{
			int n=suit[id+point_offset];
			suit[n]=id+point_offset; 
			suit[id+point_offset]=-1; 
		}
	} 
}

__kernel  void   third_shake ( int npoints,int nedges,__global union float_int * edges,__global union float_int * points, __global int * nbrs_index,__global int * count,__global bool * changed,__global int  * suit,__global int  * match,__global int  * left,int point_offset) {
	int ch0 ; 
	int id = get_global_id(0);
	if( id < npoints){
		if( left[id+point_offset]&&match[id+point_offset]==-1&&suit[id+point_offset]!=-1 )
		{
			int n=suit[id+point_offset];
			match[n]=id+point_offset; 
			match[id+point_offset]=n; 
			atomic_add(&(count[0]),1);

		}
	}
}

