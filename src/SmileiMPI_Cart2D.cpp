#include "SmileiMPI_Cart2D.h"

#include "Species.h"
#include "ParticleFactory.h"

#include "ElectroMagn.h"
#include "Field2D.h"

#include "Tools.h" 

#include <string>
#include <mpi.h>
#include <cmath>

using namespace std;

SmileiMPI_Cart2D::SmileiMPI_Cart2D( int* argc, char*** argv )
	: SmileiMPI( argc, argv )
{
}

SmileiMPI_Cart2D::SmileiMPI_Cart2D( SmileiMPI* smpi)
	: SmileiMPI( smpi )
{
	ndims_ = 2;
	number_of_procs = new int(ndims_);
	coords_  = new int(ndims_);
	periods_  = new int(ndims_);
	reorder_ = 0;

	nbNeighbors_ = 2; // per direction
	//neighbor_  = new int(nbNeighbors_);

	for (int i=0 ; i<ndims_ ; i++) periods_[i] = 0;
	for (int i=0 ; i<ndims_ ; i++) coords_[i] = 0;
	for (int i=0 ; i<ndims_ ; i++) number_of_procs[i] = 1;

	for (int iDim=0 ; iDim<ndims_ ; iDim++)
		for (int iNeighbors=0 ; iNeighbors<nbNeighbors_ ; iNeighbors++)
			neighbor_[iDim][iNeighbors] = MPI_PROC_NULL;

	for (int iDim=0 ; iDim<ndims_ ; iDim++) {
		for (int iNeighbor=0 ; iNeighbor<nbNeighbors_ ; iNeighbor++) {
			buff_send_[iDim][iNeighbor].resize(0);
			buff_recv_[iDim][iNeighbor].resize(0);
		}
	}

}

SmileiMPI_Cart2D::~SmileiMPI_Cart2D()
{
	for (int ix_isPrim=0 ; ix_isPrim<1 ; ix_isPrim++) {
		for (int iy_isPrim=0 ; iy_isPrim<1 ; iy_isPrim++) {
			MPI_Type_free( &ntype_[0][ix_isPrim][iy_isPrim]); //line
			MPI_Type_free( &ntype_[1][ix_isPrim][iy_isPrim]); // column

			MPI_Type_free( &ntypeSum_[0][ix_isPrim][iy_isPrim]); //line
			MPI_Type_free( &ntypeSum_[1][ix_isPrim][iy_isPrim]); // column
		}
	}

	delete number_of_procs;
	delete periods_;
	delete coords_;
	//delete neighbor_;

	delete [] buff_send;
	delete [] buff_recv;

	if ( SMILEI_COMM_2D != MPI_COMM_NULL) MPI_Comm_free(&SMILEI_COMM_2D);

}

void SmileiMPI_Cart2D::createTopology(PicParams& params)
{
	if (params.nDim_field == 2) {
		double tmp = params.res_space[0]*params.sim_length[0] / ( params.res_space[1]*params.sim_length[1] );
		number_of_procs[0] = 2;//min( smilei_sz, max(1, (int)sqrt ( (double)smilei_sz*tmp*tmp) ) );
		number_of_procs[1] = 2;//(int)(smilei_sz / number_of_procs[0]);
	}

	MPI_Cart_create( SMILEI_COMM_WORLD, ndims_, number_of_procs, periods_, reorder_, &SMILEI_COMM_2D );
	MPI_Cart_coords( SMILEI_COMM_2D, smilei_rk, ndims_, coords_ );

	//                  |                   |                  //
	//                  |  neighbor_[2][1]  |                  //
	//                  |                   |                  //

	//                  |  neighbor_[1][1]  |                  //
	// neighbor_[0][0]  |  Current process  |  neighbor_[0][1] //
	//                  |  neighbor_[1][0]  |                  //

	//                  |                   |                  //
	//                  |  neighbor_[2][0]  |                  //
	//                  |                   |                  //

	// ==========================================================
	// ==========================================================
	// ==========================================================

	// crossNei_[x][x]  | crossNei_[x][x]   | crossNei_[x][x]  //
	// crossNei_[x][x]  |                   | crossNei_[x][x]  //
	// crossNei_[x][x]  | crossNei_[x][x]   | crossNei_[x][x]  //

	// crossNei_[x][x]  |                   | crossNei_[x][x]  //
	//                  |  Current process  |                  //		-> Manage working direction per direction
	// crossNei_[x][x]  |                   | crossNei_[x][x]  //

	// crossNei_[x][x]  | crossNei_[x][x]   | crossNei_[x][x]  //
	// crossNei_[x][x]  |                   | crossNei_[x][x]  //
	// crossNei_[x][x]  | crossNei_[x][x]   | crossNei_[x][x]  //


	for (int iDim=0 ; iDim<ndims_ ; iDim++) {
		MPI_Cart_shift( SMILEI_COMM_2D, iDim, 1, &(neighbor_[iDim][0]), &(neighbor_[iDim][1]) );
		PMESSAGE ( 0, smilei_rk, "Neighbors of process in direction " << iDim << " : " << neighbor_[iDim][0] << " - " << neighbor_[iDim][1]  );
	}


	for (unsigned int i=0 ; i<params.nDim_field ; i++) {

		params.n_space[i] = params.n_space_global[i] / number_of_procs[i];
		if ( number_of_procs[i]*params.n_space[i] != params.n_space_global[i] ) {
			//WARNING( "Domain splitting does not match to the global domain" );
			if (coords_[i]==number_of_procs[i]-1) {
				params.n_space[i] = params.n_space_global[i] - params.n_space[i]*(number_of_procs[i]-1);
			}
		}

		n_space_global[i] = params.n_space_global[i];
		oversize[i] = params.oversize[i] = 2;
		//! \todo{replace cell_starting_global_index compute by a most sophisticated or input data}
		cell_starting_global_index[i] = coords_[i]*(params.n_space_global[i] / number_of_procs[i]);
		// min/max_local : describe local domain in which particles cat be moved
		//                 different from domain on which E, B, J are defined
		min_local[i] = (cell_starting_global_index[i]                  )*params.cell_length[i];
		max_local[i] = (cell_starting_global_index[i]+params.n_space[i])*params.cell_length[i];
		cell_starting_global_index[i] -= params.oversize[i];

	}
	MESSAGE( "n_space / rank " << smilei_rk << " = " << params.n_space[0] << " " << params.n_space[1] );


}

void SmileiMPI_Cart2D::exchangeParticles(Species* species, int ispec, PicParams* params)
{
	std::vector<Particle*>* cuParticles = &species->particles;

	//int n_particles = species->getNbrOfParticles();

	//DEBUG( 2, "\tProcess " << smilei_rk << " : " << species->getNbrOfParticles() << " Particles of species " << ispec );
	//MESSAGE( "xmin_local = " << min_local[0] << " - x_max_local = " << max_local[0] );

	/********************************************************************************/
	// Build list of particle to exchange
	// Arrays buff_send/buff_recv indexed as array neighbors_
	/********************************************************************************/
	int iPart = 0;
	int n_part_send = indexes_of_particles_to_exchange.size();
	int n_part_recv;
	for (int i=n_part_send-1 ; i>=0 ; i--) {
		iPart = indexes_of_particles_to_exchange[i];

		for (int iDim=0 ; iDim<ndims_ ; iDim++) {
			if ( (*cuParticles)[iPart]->position(iDim) < min_local[iDim]) {
				buff_send_[iDim][0].push_back( (*cuParticles)[iPart] );
				cuParticles->erase(cuParticles->begin()+iPart);
				break;
			}
			if ( (*cuParticles)[iPart]->position(iDim) >= max_local[iDim]) {
				buff_send_[iDim][1].push_back( (*cuParticles)[iPart] );
				cuParticles->erase(cuParticles->begin()+iPart);
				break;
			}
		}

	} // END for iPart


	/********************************************************************************/
	// Exchange particles
	/********************************************************************************/
	MPI_Status stat;
	for (int iDim=0 ; iDim<ndims_ ; iDim++) {

		for (int iNeighbor=0 ; iNeighbor<nbNeighbors_ ; iNeighbor++) {

			n_part_send = buff_send_[iDim][iNeighbor].size();

			if (neighbor_[iDim][iNeighbor]!=MPI_PROC_NULL) {
				MPI_Send( &n_part_send, 1, MPI_INT, neighbor_[iDim][iNeighbor], 0, SMILEI_COMM_2D );

				if (n_part_send!=0) {
					for (int iPart=0 ; iPart<n_part_send; iPart++ ) {
						MPI_Send( &(buff_send[iNeighbor][iPart]->position(0)), 6, MPI_DOUBLE, neighbor_[iDim][iNeighbor], 0, SMILEI_COMM_2D );
					}
				}
			} // END of Send

			if (neighbor_[iDim][(iNeighbor+1)%2]!=MPI_PROC_NULL) {
				MPI_Recv( &n_part_recv, 1, MPI_INT, neighbor_[iDim][(iNeighbor+1)%2], 0, SMILEI_COMM_2D, &stat );

				if (n_part_recv!=0) {
					buff_recv_[iDim][(iNeighbor+1)%2] = ParticleFactory::createVector(params, ispec, n_part_recv);
					for (int iPart=0 ; iPart<n_part_recv; iPart++ ) {
						MPI_Recv( &(buff_recv_[iDim][(iNeighbor+1)%2][iPart]->position(0)), 6, MPI_DOUBLE, neighbor_[iDim][(iNeighbor+1)%2], 0, SMILEI_COMM_2D, &stat );
						cuParticles->push_back(buff_recv_[iDim][(iNeighbor+1)%2][iPart]);
					}
				}
			} // END of Recv

		} // END for iNeighbor

	} // END for iDim

	/********************************************************************************/
	// delete Particles included in buff_send/buff_recv
	/********************************************************************************/
	for (int iDim=0 ; iDim<ndims_ ; iDim++) {
		for (int i=0 ; i<nbNeighbors_ ; i++) {
			// Particles must be deleted on process sender
			n_part_send =  buff_send_[iDim][i].size();
			/*for (unsigned int iPart=0 ; iPart<n_part_send; iPart++ ) {
				delete buff_send[i][iPart];
			}*/
			buff_send_[iDim][i].clear();

			// Not on process receiver, Particles are stored in species
			// Just clean the buffer
			buff_recv_[iDim][i].clear();
		} // END for iNeighbor
	} // END for iDim

} // END exchangeParticles


void SmileiMPI_Cart2D::createType( PicParams& params )
{
	int nx0 = params.n_space[0] + 1 + 2*params.oversize[0];
	int ny0 = params.n_space[1] + 1 + 2*params.oversize[1];

	// MPI_Datatype ntype_[nDim][primDual][primDual]
	int nx, ny;
	int nline, ncol;
	for (int ix_isPrim=0 ; ix_isPrim<2 ; ix_isPrim++) {
		nx = nx0 + ix_isPrim;
		for (int iy_isPrim=0 ; iy_isPrim<2 ; iy_isPrim++) {
			ny = ny0 + iy_isPrim;
			ntype_[0][ix_isPrim][iy_isPrim] = NULL;
			MPI_Type_contiguous(ny, MPI_DOUBLE, &(ntype_[0][ix_isPrim][iy_isPrim]));    //line
			MPI_Type_commit( &(ntype_[0][ix_isPrim][iy_isPrim]) );
			ntype_[1][ix_isPrim][iy_isPrim] = NULL;
			MPI_Type_vector(nx, 1, ny, MPI_DOUBLE, &(ntype_[1][ix_isPrim][iy_isPrim])); // column
			MPI_Type_commit( &(ntype_[1][ix_isPrim][iy_isPrim]) );

			ntypeSum_[0][ix_isPrim][iy_isPrim] = NULL;
			nline = 1 + 2*params.oversize[0] + ix_isPrim;
			MPI_Type_contiguous(nline, ntype_[0][ix_isPrim][iy_isPrim], &(ntypeSum_[0][ix_isPrim][iy_isPrim]));    //line
			MPI_Type_commit( &(ntypeSum_[0][ix_isPrim][iy_isPrim]) );
			ntypeSum_[1][ix_isPrim][iy_isPrim] = NULL;
			ncol  = 1 + 2*params.oversize[1] + iy_isPrim;
			MPI_Type_vector(nx, ncol, ny, MPI_DOUBLE, &(ntypeSum_[1][ix_isPrim][iy_isPrim])); // column
			MPI_Type_commit( &(ntypeSum_[1][ix_isPrim][iy_isPrim]) );

		}
	}

}


void SmileiMPI_Cart2D::sumField( Field* field )
{
	std::vector<unsigned int> n_elem = field->dims_;
	std::vector<unsigned int> isPrimal = field->isPrimal_;
	Field2D* f2D =  static_cast<Field2D*>(field);


	// Use a buffer per direction to exchange data before summing
	Field2D buf[ndims_][ nbNeighbors_ ];
	// Size buffer is 2 oversize (1 inside & 1 outside of the current subdomain)
	std::vector<unsigned int> oversize2 = oversize;
	oversize2[0] *= 2; oversize2[0] += 1 + f2D->isPrimal_[0];
	oversize2[1] *= 2; oversize2[1] += 1 + f2D->isPrimal_[1];

	for (int iDim=0 ; iDim<ndims_ ; iDim++) {
		for (int iNeighbor=0 ; iNeighbor<nbNeighbors_ ; iNeighbor++) {
			std::vector<unsigned int> tmp(ndims_,0);
			tmp[0] =    iDim  * n_elem[0] + (1-iDim) * oversize2[0];
			tmp[1] = (1-iDim) * n_elem[1] +    iDim  * oversize2[1];
			buf[iDim][iNeighbor].allocateDims( tmp );
		}
	}

	int istart, ix, iy;
	MPI_Status stat;

	/********************************************************************************/
	// Send/Recv in a buffer data to sum
	/********************************************************************************/
	for (int iDim=0 ; iDim<ndims_ ; iDim++) {

		MPI_Datatype ntype = ntypeSum_[iDim][isPrimal[0]][isPrimal[1]];

		for (int iNeighbor=0 ; iNeighbor<nbNeighbors_ ; iNeighbor++) {

			if (neighbor_[iDim][iNeighbor]!=MPI_PROC_NULL) {
				istart = iNeighbor * ( n_elem[iDim]- oversize2[iDim] ) + (1-iNeighbor) * ( 0 );
				ix = (1-iDim)*istart;
				iy =    iDim *istart;
				MPI_Send( &(f2D->data_[ix][iy]), 1, ntype, neighbor_[iDim][iNeighbor], 0, SMILEI_COMM_2D );
			} // END of Send

			if (neighbor_[iDim][(iNeighbor+1)%2]!=MPI_PROC_NULL) {
				int tmp_elem = (buf[iDim][(iNeighbor+1)%2]).dims_[0]*(buf[iDim][(iNeighbor+1)%2]).dims_[1];
				MPI_Recv( &( (buf[iDim][(iNeighbor+1)%2]).data_[0][0] ), tmp_elem, MPI_DOUBLE, neighbor_[iDim][(iNeighbor+1)%2], 0, SMILEI_COMM_2D, &stat );
			} // END of Recv

		} // END for iNeighbor



		// Synchro before summing, to not sum with data ever sum
		// Merge loops, Sum direction by direction permits to not communicate with diagonal neighbors
		barrier();
		/********************************************************************************/
		// Sum data on each process, same operation on both side
		/********************************************************************************/

		for (int iNeighbor=0 ; iNeighbor<nbNeighbors_ ; iNeighbor++) {
			istart = ( (iNeighbor+1)%2 ) * ( n_elem[iDim]- oversize2[iDim] ) + (1-(iNeighbor+1)%2) * ( 0 );
			int ix0 = (1-iDim)*istart;
			int iy0 =    iDim *istart;
			if (neighbor_[iDim][(iNeighbor+1)%2]!=MPI_PROC_NULL) {
				for (unsigned int ix=0 ; ix< (buf[iDim][(iNeighbor+1)%2]).dims_[0] ; ix++) {
					for (unsigned int iy=0 ; iy< (buf[iDim][(iNeighbor+1)%2]).dims_[1] ; iy++)
						f2D->data_[ix0+ix][iy0+iy] += (buf[iDim][(iNeighbor+1)%2])(ix,iy);
				}
			} // END if

		} // END for iNeighbor
		
		barrier();
		
	} // END for iDim

} // END sumField


void SmileiMPI_Cart2D::exchangeField( Field* field )
{
	std::vector<unsigned int> n_elem   = field->dims_;
	std::vector<unsigned int> isPrimal = field->isPrimal_;
	Field2D* f2D =  static_cast<Field2D*>(field);

	MPI_Status stat;

	int istart, ix, iy;

	// Loop over dimField
	for (int iDim=0 ; iDim<ndims_ ; iDim++) {

		MPI_Datatype ntype = ntype_[iDim][isPrimal[0]][isPrimal[1]];
		for (int iNeighbor=0 ; iNeighbor<nbNeighbors_ ; iNeighbor++) {

			if (neighbor_[iDim][iNeighbor]!=MPI_PROC_NULL) {

				istart = iNeighbor * ( n_elem[iDim]- (2*oversize[iDim]+1+isPrimal[iDim]) ) + (1-iNeighbor) * ( 2*oversize[iDim]+1-(1-isPrimal[iDim]) );
				ix = (1-iDim)*istart;
				iy =    iDim *istart;
				MPI_Send( &(f2D->data_[ix][iy]), 1, ntype, neighbor_[iDim][iNeighbor], 0, SMILEI_COMM_2D );

			} // END of Send

			if (neighbor_[iDim][(iNeighbor+1)%2]!=MPI_PROC_NULL) {

				istart = ( (iNeighbor+1)%2 ) * ( n_elem[iDim] - 1 ) + (1-(iNeighbor+1)%2) * ( 0 )  ;
				ix = (1-iDim)*istart;
				iy =    iDim *istart;
				MPI_Recv( &(f2D->data_[ix][iy]), 1, ntype, neighbor_[iDim][(iNeighbor+1)%2], 0, SMILEI_COMM_2D, &stat );

			} // END of Recv

		} // END for iNeighbor

	} // END for iDim


} // END exchangeField

#ifdef _HDF5
#include "hdf5.h"
#endif

void SmileiMPI_Cart2D::writeField( Field* field, string name )
{
#ifdef _HDF5
	MESSAGE( "to be implemented" );

	Field2D* f2D =  static_cast<Field2D*>(field);
	std::vector<unsigned int> n_elem = field->dims_;

	std::vector<unsigned int> istart = oversize;
	std::vector<unsigned int> bufsize = n_elem;

	for (int i=0 ; i<ndims_ ; i++) {

		if (coords_[i]!=0) istart[i]+=1;  							// A ajuster 2D -> coords_

		bufsize[i] = n_elem[i] - 2*oversize[i];

		if (number_of_procs[i] != 1) {

			if ( f2D->isPrimal_[i] == 0 ) {
				if (coords_[i]!=0) {
					bufsize[i]--;
				}
			}
			else {
				if ( (coords_[i]!=0) && (coords_[i]!=number_of_procs[i]-1) )
					bufsize[i] -= 2;
				else
					bufsize[i] -= 1;
			}

//			bufsize[i] -= f2D->isPrimal_[i];
//			if (coords_[i]!=0) {										// A ajuster 2D -> coords_
//				if (f2D->isPrimal_[i] == 0) bufsize[i]-=1;
//			}
//			else if (coords_[i]!=number_of_procs[i]-1)  bufsize[i]-=1;			// A ajuster 2D -> coords_


		}
	}

	MPI_Info info  = MPI_INFO_NULL;

    /*
     * Set up file access property list with parallel I/O access
     */
	hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, info);
   /*
    * Create a new file collectively and release property list identifier.
    */
   hid_t file_id = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
   //H5Pclose(plist_id);
   MESSAGE( "file created" );
   barrier();
   cout.flush();

   /*
    * Create the dataspace for the dataset.
    */
   hsize_t     chunk_dims[2];            /* chunk dimensions */
   hsize_t     offset[2];
   hsize_t     stride[2];
   hsize_t     count[2];
   hsize_t     block[2];

   chunk_dims[0] = n_elem[0];
   chunk_dims[1] = n_elem[1];
   for ( int ik=0 ; ik<getSize(); ik++) {
	   if (ik==getRank() ) {
		   cout << smilei_rk << " - coords = " << coords_[0] << ", " << coords_[1] << " -> chunk OK : " << chunk_dims[0] << ", " << chunk_dims[1] << endl;
		   sleep(1);
		   cout.flush();
	   }
	   barrier();
   }
   hid_t memspace  = H5Screate_simple(ndims_, chunk_dims, NULL);
   offset[0] = istart[0];
   offset[1] = istart[1];
   stride[0] = 1;
   stride[1] = 1;
   count[0] = bufsize[0];
   count[1] = bufsize[1];
   for ( int ik=0 ; ik<getSize(); ik++) {
	   if (ik==getRank() ) {
		   cout << smilei_rk << " - coords = " << coords_[0] << ", " << coords_[1] << " -> offset OK : " << offset[0] << ", " << offset[1] << endl;
		   cout << smilei_rk << " - coords = " << coords_[0] << ", " << coords_[1] << " -> count  OK : " << count[0] << ", " << count[1] << endl;
		   sleep(1);
		   cout.flush();
	   }
	   barrier();
   }
   herr_t status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset, stride, count, NULL);
   MESSAGE( "memspace created" );
   barrier();
   sleep(1);
   cout.flush();

   //
   // Each process defines dataset in memory and writes it to the hyperslab
   // in the file.
   //
   hsize_t     dimsf[2];
   dimsf[0] = n_space_global[0]+1+f2D->isPrimal_[0]; // +1	// A ajuster
   dimsf[1] = n_space_global[1]+1+f2D->isPrimal_[1]; // +1	// A ajuster

   cout << "\tfilespace OK : " << dimsf[0] << " " << dimsf[1] << endl;

   hid_t filespace = H5Screate_simple(ndims_, dimsf, NULL);
   hid_t plist_id2 = H5Pcreate(H5P_DATASET_CREATE);
   chunk_dims[0] = bufsize[0];
   chunk_dims[1] = bufsize[1];
   for ( int ik=0 ; ik<getSize(); ik++) {
	   if (ik==getRank() ) {
		   cout << smilei_rk << " - coords = " << coords_[0] << ", " << coords_[1] << " -> write OK " << bufsize[0] << ", " << bufsize[1] << endl;
		   //cout << endl << endl;
		   sleep(1);
		   cout.flush();
	   }
	   barrier();
   }

   //H5Pset_chunk(plist_id2, ndims_, chunk_dims);
   hid_t dset_id = H5Dcreate(file_id, "Field", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, plist_id2, H5P_DEFAULT);
   //H5Pclose(plist_id);

   //
   // Select hyperslab in the file.
   //
   offset[0] = cell_starting_global_index[0]+istart[0]; // istart = oversize (+1 if != coords_[i])
   offset[1] = cell_starting_global_index[1]+istart[1];

   cout << "\t\t" << smilei_rk << ", offset = " << offset[0] << " " << offset[1] << ", size = " << bufsize[0] << " " << bufsize[1] << endl;
   cout << "\t\t" << smilei_rk << ", global index = " << cell_starting_global_index[0] << " " << cell_starting_global_index[1] << ", " << istart[0] << " " << istart[1] << endl;
   stride[0] = 1;
   stride[1] = 1;
   count[0] = 1;
   count[1] = 1;
   block[0] = bufsize[0];
   block[1] = bufsize[1];
   status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, stride, count, block);
   MESSAGE( "filespace created" );

   //
   // Create property list for collective dataset write.
   //
   hid_t plist_id3 = H5Pcreate(H5P_DATASET_XFER);
   H5Pset_dxpl_mpio(plist_id3, H5FD_MPIO_INDEPENDENT);
   MESSAGE( "Start H5Dwrite" );
   //status = H5Dwrite( dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id3, &(f2D->data_[0][0]) );
   for ( int ik=0 ; ik<getSize(); ik++) {
	   if (ik==getRank() ) {
		   status = H5Dwrite( dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id3, &(f2D->data_[0][0]) );
	   }
	   barrier();
   }

   MESSAGE( "End H5Dwrite" );

   //
   // Close/release resources.
   //
   H5Pclose(plist_id3);
   H5Dclose(dset_id);
   H5Pclose(plist_id2);
   H5Sclose(filespace);
   H5Sclose(memspace);
   H5Fclose(file_id);
   H5Pclose(plist_id);
#endif
} // END writeField
