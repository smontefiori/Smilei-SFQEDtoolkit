#if defined( SMILEI_ACCELERATOR_GPU_OMP )

    //! Simple switch to jump between the reference (omp) implementation and the
    //! hip one.
    //! NOTE: If you wanna use the OMP version, you must rename this file to
    //! .cpp instead of .cu for the HIP. The preprocessor and the Smilei
    //! makefile will take care of the rest.
    //!
    #if defined( __HIP__ )
    // HIP compiler support enabled (for .cu files)
    #else
        #define PRIVATE_SMILEI_USE_OPENMP_PROJECTION_IMPLENTATION = 1
    #endif

    #if defined( PRIVATE_SMILEI_USE_OPENMP_PROJECTION_IMPLENTATION )
        #include <cmath>

        #include "Tools.h"
    #else
        #include <hip/hip_runtime.h>

        #include "Params.h"
        #include "gpu.h"
    #endif

    #if defined( PRIVATE_SMILEI_USE_OPENMP_PROJECTION_IMPLENTATION )

namespace naive {

    static inline void
    currentDepositionKernel( double *__restrict__ Jx,
                             double *__restrict__ Jy,
                             double *__restrict__ Jz,
                             int Jx_size,
                             int Jy_size,
                             int Jz_size,
                             const double *__restrict__ device_particle_position_x,
                             const double *__restrict__ device_particle_position_y,
                             const double *__restrict__ device_particle_momentum_z,
                             const short *__restrict__ device_particle_charge,
                             const double *__restrict__ device_particle_weight,
                             const int *__restrict__ host_bin_index,
                             unsigned int x_dimension_bin_count,
                             unsigned int y_dimension_bin_count,
                             const double *__restrict__ invgf_,
                             const int *__restrict__ iold_,
                             const double *__restrict__ deltaold_,
                             double inv_cell_volume,
                             double dx_inv,
                             double dy_inv,
                             double dx_ov_dt,
                             double dy_ov_dt,
                             int    i_domain_begin,
                             int    j_domain_begin,
                             int    nprimy,
                             int    pxr )
    {
        const unsigned int bin_count = x_dimension_bin_count * y_dimension_bin_count;

        SMILEI_ASSERT( bin_count > 0 );

        const int particle_count = host_bin_index[bin_count - 1];

            // // Arrays used for the Esirkepov projection method
            // static constexpr bool kAutoDeviceFree = true;
            // const std::size_t     kTmpArraySize   = particle_count * 5;

            // smilei::tools::gpu::NonInitializingVector<double, kAutoDeviceFree> Sx0_buffer{ kTmpArraySize };
            // smilei::tools::gpu::NonInitializingVector<double, kAutoDeviceFree> Sx1_buffer{ kTmpArraySize };
            // smilei::tools::gpu::NonInitializingVector<double, kAutoDeviceFree> Sy0_buffer{ kTmpArraySize };
            // smilei::tools::gpu::NonInitializingVector<double, kAutoDeviceFree> Sy1_buffer{ kTmpArraySize };
            // // smilei::tools::gpu::NonInitializingVector<double, kAutoDeviceFree> DSx_buffer{ kTmpArraySize };
            // // smilei::tools::gpu::NonInitializingVector<double, kAutoDeviceFree> DSy_buffer{ kTmpArraySize };

            // double *const __restrict__ Sx0_buffer_data = Sx0_buffer.data();
            // double *const __restrict__ Sx1_buffer_data = Sx1_buffer.data();
            // double *const __restrict__ Sy0_buffer_data = Sy0_buffer.data();
            // double *const __restrict__ Sy1_buffer_data = Sy1_buffer.data();
            // // double *const __restrict__ DSx_buffer_data = DSx_buffer.data();
            // // double *const __restrict__ DSy_buffer_data = DSy_buffer.data();

        #pragma omp target     is_device_ptr /* map */ ( /* to: */                                            \
                                                     device_particle_position_x /* [0:particle_count] */, \
                                                     device_particle_position_y /* [0:particle_count] */, \
                                                     device_particle_momentum_z /* [0:particle_count] */, \
                                                     device_particle_charge /* [0:particle_count] */,     \
                                                     device_particle_weight /* [0:particle_count] */ )
        #pragma omp teams thread_limit( 64 )
        #pragma omp distribute parallel for
        for( int particle_index = 0; particle_index < particle_count; ++particle_index ) {
            const double invgf                        = invgf_[particle_index];
            const int *const __restrict__ iold        = &iold_[particle_index];
            const double *const __restrict__ deltaold = &deltaold_[particle_index];

            double Sx0[5];
            double Sx1[5];
            double Sy0[5];
            double Sy1[5];
            // double DSx[5];
            // double DSy[5];

            // double *const __restrict__ Sx0 = Sx0_buffer_data + 5 * ( particle_index - 0 );
            // double *const __restrict__ Sx1 = Sx1_buffer_data + 5 * ( particle_index - 0 );
            // double *const __restrict__ Sy0 = Sy0_buffer_data + 5 * ( particle_index - 0 );
            // double *const __restrict__ Sy1 = Sy1_buffer_data + 5 * ( particle_index - 0 );
            // // double *const __restrict__ DSx = DSx_buffer_data + 5 * ( particle_index - 0 );
            // // double *const __restrict__ DSy = DSy_buffer_data + 5 * ( particle_index - 0 );

            // Variable declaration & initialization
            // Esirkepov's paper: https://arxiv.org/pdf/physics/9901047.pdf

            // Locate the particle on the primal grid at former time-step & calculate coeff. S0
            {
                const double delta  = deltaold[0 * particle_count];
                const double delta2 = delta * delta;
                Sx0[0]              = 0.0;
                Sx0[1]              = 0.5 * ( delta2 - delta + 0.25 );
                Sx0[2]              = 0.75 - delta2;
                Sx0[3]              = 0.5 * ( delta2 + delta + 0.25 );
                Sx0[4]              = 0.0;
            }
            {
                const double delta  = deltaold[1 * particle_count];
                const double delta2 = delta * delta;
                Sy0[0]              = 0.0;
                Sy0[1]              = 0.5 * ( delta2 - delta + 0.25 );
                Sy0[2]              = 0.75 - delta2;
                Sy0[3]              = 0.5 * ( delta2 + delta + 0.25 );
                Sy0[4]              = 0.0;
            }

            // Locate the particle on the primal grid at current time-step & calculate coeff. S1
            {
                const double xpn      = device_particle_position_x[particle_index] * dx_inv;
                const int    ip       = std::round( xpn );
                const int    ipo      = iold[0 * particle_count];
                const int    ip_m_ipo = ip - ipo - i_domain_begin;
                const double delta    = xpn - static_cast<double>( ip );
                const double delta2   = delta * delta;

                Sx1[0] = 0.0;
                Sx1[1] = 0.0;
                // Sx1[2] = 0.0; // Always set below
                Sx1[3] = 0.0;
                Sx1[4] = 0.0;

                Sx1[ip_m_ipo + 1] = 0.5 * ( delta2 - delta + 0.25 );
                Sx1[ip_m_ipo + 2] = 0.75 - delta2;
                Sx1[ip_m_ipo + 3] = 0.5 * ( delta2 + delta + 0.25 );
            }
            {
                const double ypn      = device_particle_position_y[particle_index] * dy_inv;
                const int    jp       = std::round( ypn );
                const int    jpo      = iold[1 * particle_count];
                const int    jp_m_jpo = jp - jpo - j_domain_begin;
                const double delta    = ypn - static_cast<double>( jp );
                const double delta2   = delta * delta;

                Sy1[0] = 0.0;
                Sy1[1] = 0.0;
                // Sy1[2] = 0.0; // Always set below
                Sy1[3] = 0.0;
                Sy1[4] = 0.0;

                Sy1[jp_m_jpo + 1] = 0.5 * ( delta2 - delta + 0.25 );
                Sy1[jp_m_jpo + 2] = 0.75 - delta2;
                Sy1[jp_m_jpo + 3] = 0.5 * ( delta2 + delta + 0.25 );
            }

            // DSx[0] = Sx1[0] - Sx0[0];
            // DSx[1] = Sx1[1] - Sx0[1];
            // DSx[2] = Sx1[2] - Sx0[2];
            // DSx[3] = Sx1[3] - Sx0[3];
            // DSx[4] = Sx1[4] - Sx0[4];

            // DSy[0] = Sy1[0] - Sy0[0];
            // DSy[1] = Sy1[1] - Sy0[1];
            // DSy[2] = Sy1[2] - Sy0[2];
            // DSy[3] = Sy1[3] - Sy0[3];
            // DSy[4] = Sy1[4] - Sy0[4];
            // }

            // // Charge deposition on the grid

            // for( int particle_index = 0; particle_index < particle_count; ++particle_index ) {
            //     const double invgf                        = invgf_[particle_index];
            //     const int *const __restrict__ iold        = &iold_[particle_index];
            //     const double *const __restrict__ deltaold = &deltaold_[particle_index];

            //     double *const __restrict__ Sx0 = Sx0_buffer_data + 5 * ( particle_index - 0 );
            //     double *const __restrict__ Sx1 = Sx1_buffer_data + 5 * ( particle_index - 0 );
            //     double *const __restrict__ Sy0 = Sy0_buffer_data + 5 * ( particle_index - 0 );
            //     double *const __restrict__ Sy1 = Sy1_buffer_data + 5 * ( particle_index - 0 );
            //     // double *const __restrict__ DSx = DSx_buffer_data + 5 * ( particle_index - 0 );
            //     // double *const __restrict__ DSy = DSy_buffer_data + 5 * ( particle_index - 0 );

            // (x,y,z) components of the current density for the macro-particle
            const double charge_weight = inv_cell_volume * static_cast<double>( device_particle_charge[particle_index] ) * device_particle_weight[particle_index];
            const double crx_p         = charge_weight * dx_ov_dt;
            const double cry_p         = charge_weight * dy_ov_dt;
            const double crz_p         = charge_weight * ( 1.0 / 3.0 ) * device_particle_momentum_z[particle_index] * invgf;

            // This is the particle position as grid index
            // This minus 2 come from the order 2 scheme, based on a 5 points stencil from -2 to +2.
            const int ipo = iold[0 * particle_count] - 2;
            const int jpo = iold[1 * particle_count] - 2;

            for( unsigned int i = 0; i < 1; ++i ) {
                const int iloc = ( i + ipo ) * nprimy + jpo;
                    /* Jx[iloc] += tmpJx[0]; */
        #pragma omp atomic update
                Jz[iloc] += crz_p * ( Sy1[0] * ( /* 0.5 * Sx0[i] + */ Sx1[i] ) );
                double tmp = 0.0;
                for( unsigned int j = 1; j < 5; j++ ) {
                    tmp -= cry_p * ( Sy1[j - 1] - Sy0[j - 1] ) * ( Sx0[i] + 0.5 * ( Sx1[i] - Sx0[i] ) );
        #pragma omp atomic update
                    Jy[iloc + j + pxr * ( /* i + */ ipo )] += tmp;
        #pragma omp atomic update
                    Jz[iloc + j] += crz_p * ( Sy0[j] * ( 0.5 * Sx1[i] /* + Sx0[i] */ ) +
                                              Sy1[j] * ( /* 0.5 * Sx0[i] + */ Sx1[i] ) );
                }
            }

            double tmpJx[5]{};

            for( unsigned int i = 1; i < 5; ++i ) {
                const int iloc = ( i + ipo ) * nprimy + jpo;
                tmpJx[0] -= crx_p * ( Sx1[i - 1] - Sx0[i - 1] ) * ( 0.5 * ( Sy1[0] - Sy0[0] ) );
        #pragma omp atomic update
                Jx[iloc] += tmpJx[0];
        #pragma omp atomic update
                Jz[iloc] += crz_p * ( Sy1[0] * ( 0.5 * Sx0[i] + Sx1[i] ) );
                double tmp = 0.0;
                for( unsigned int j = 1; j < 5; ++j ) {
                    tmpJx[j] -= crx_p * ( Sx1[i - 1] - Sx0[i - 1] ) * ( Sy0[j] + 0.5 * ( Sy1[j] - Sy0[j] ) );
        #pragma omp atomic update
                    Jx[iloc + j] += tmpJx[j];
                    tmp -= cry_p * ( Sy1[j - 1] - Sy0[j - 1] ) * ( Sx0[i] + 0.5 * ( Sx1[i] - Sx0[i] ) );
        #pragma omp atomic update
                    Jy[iloc + j + pxr * ( i + ipo )] += tmp;

        #pragma omp atomic update
                    Jz[iloc + j] += crz_p * ( Sy0[j] * ( 0.5 * Sx1[i] + Sx0[i] ) +
                                              Sy1[j] * ( 0.5 * Sx0[i] + Sx1[i] ) );
                }
            }
        }
    }
} // namespace naive

    #else

namespace hip {
    namespace detail {
        void checkErrors( ::hipError_t an_error_code,
                          const char  *file_name,
                          int          line )
        {
            if( an_error_code != ::hipError_t::hipSuccess ) {
                std::cout << "HIP error at " << file_name << ":" << line
                          << " -> " << ::hipGetErrorString( an_error_code );
                std::exit( EXIT_FAILURE );
            }
        }
    } // namespace detail

        #define checkHIPErrors( an_expression )                           \
            do {                                                          \
                detail::checkErrors( an_expression, __FILE__, __LINE__ ); \
            } while( 0 )

    namespace kernel {
        // TODO(Etienne M): Template on WGs low dimension parameters
        __global__ void
        depositeForAllCurrentDimensions( double *__restrict__ device_Jx,
                                         double *__restrict__ device_Jy,
                                         double *__restrict__ device_Jz,
                                         int Jx_size,
                                         int Jy_size,
                                         int Jz_size,
                                         const double *__restrict__ device_particle_position_x,
                                         const double *__restrict__ device_particle_position_y,
                                         const double *__restrict__ device_particle_momentum_z,
                                         const short *__restrict__ device_particle_charge,
                                         const double *__restrict__ device_particle_weight,
                                         const int *__restrict__ device_bin_index,
                                         const double *__restrict__ device_invgf_,
                                         const int *__restrict__ device_iold_,
                                         const double *__restrict__ device_deltaold_,
                                         double inv_cell_volume,
                                         double dx_inv,
                                         double dy_inv,
                                         double dx_ov_dt,
                                         double dy_ov_dt,
                                         int    i_domain_begin,
                                         int    j_domain_begin,
                                         int    nprimy,
                                         int    pxr )
        {
            // TODO(Etienne M): refactor this function. Break it into smaller
            // pieces (lds init/store, coeff computation, deposition etc..)
            const unsigned int workgroup_size = blockDim.x;
            const unsigned int bin_count      = gridDim.x * gridDim.y;
            const unsigned int loop_stride    = workgroup_size; // This stride should enable better memory access coalescing

            // It seems slightly faster to traverse the clusters the C indexing
            // order (compared to the Smilei order).
            const unsigned int x_cluster_coordinate          = blockIdx.x;
            const unsigned int y_cluster_coordinate          = blockIdx.y;
            const unsigned int workgroup_dedicated_bin_index = y_cluster_coordinate * gridDim.x + x_cluster_coordinate;
            const unsigned int thread_index_offset           = threadIdx.x;

            // NOTE: We gain from the particles not being sorted inside a
            // cluster because it reduces the bank conflicts one gets when
            // multiple threads access the same part of the shared memory. Such
            // "conflicted" accesses are serialized !
            // NOTE: We use a bit to much LDS. For Jx, the first row could be
            // discarded, for Jy we could remove the last column (know that the
            // access pattern is wierd (rhombus shaped)).

            static constexpr unsigned int kFieldScratchSpaceSize = Params::getGPUInterpolationClusterCellVolume( 2 /* 2D */, 2 /* 2nd order interpolation */ );

            __shared__ double Jx_field_scratch_space[kFieldScratchSpaceSize];
            __shared__ double Jy_field_scratch_space[kFieldScratchSpaceSize];
            __shared__ double Jz_field_scratch_space[kFieldScratchSpaceSize];

            // Init the shared memory

            for( unsigned int field_index = thread_index_offset;
                 field_index < kFieldScratchSpaceSize;
                 field_index += workgroup_size ) {
                // TODO(Etienne M): Should I try to remvoe the bank conflicts?
                Jx_field_scratch_space[field_index] = 0.0;
                Jy_field_scratch_space[field_index] = 0.0;
                Jz_field_scratch_space[field_index] = 0.0;
            }

            __syncthreads();

            const unsigned int particle_count = device_bin_index[bin_count - 1];

            // This workgroup has to process distance(last_particle,
            // first_particle) particles
            const unsigned int first_particle = workgroup_dedicated_bin_index == 0 ? 0 :
                                                                                     device_bin_index[workgroup_dedicated_bin_index - 1];
            const unsigned int last_particle  = device_bin_index[workgroup_dedicated_bin_index];

            for( unsigned int particle_index = first_particle + thread_index_offset;
                 particle_index < last_particle;
                 particle_index += loop_stride ) {
                const double invgf                        = device_invgf_[particle_index];
                const int *const __restrict__ iold        = &device_iold_[particle_index];
                const double *const __restrict__ deltaold = &device_deltaold_[particle_index];

                double Sx0[5];
                double Sx1[5];
                double Sy0[5];
                double Sy1[5];
                // double DSx[5];
                // double DSy[5];

                // Variable declaration & initialization
                // Esirkepov's paper: https://arxiv.org/pdf/physics/9901047.pdf

                // Locate the particle on the primal grid at former time-step & calculate coeff. S0
                {
                    const double delta  = deltaold[0 * particle_count];
                    const double delta2 = delta * delta;
                    Sx0[0]              = 0.0;
                    Sx0[1]              = 0.5 * ( delta2 - delta + 0.25 );
                    Sx0[2]              = 0.75 - delta2;
                    Sx0[3]              = 0.5 * ( delta2 + delta + 0.25 );
                    Sx0[4]              = 0.0;
                }
                {
                    const double delta  = deltaold[1 * particle_count];
                    const double delta2 = delta * delta;
                    Sy0[0]              = 0.0;
                    Sy0[1]              = 0.5 * ( delta2 - delta + 0.25 );
                    Sy0[2]              = 0.75 - delta2;
                    Sy0[3]              = 0.5 * ( delta2 + delta + 0.25 );
                    Sy0[4]              = 0.0;
                }

                // Locate the particle on the primal grid at current time-step & calculate coeff. S1
                {
                    const double xpn = device_particle_position_x[particle_index] * dx_inv;
                    const int    ip  = std::round( xpn );
                    // const int    ip       = static_cast<int>( xpn + 0.5 ); // std::round | rounding approximation which is correct enough and faster in this case
                    const int    ipo      = iold[0 * particle_count];
                    const int    ip_m_ipo = ip - ipo - i_domain_begin;
                    const double delta    = xpn - static_cast<double>( ip );
                    const double delta2   = delta * delta;

                    Sx1[0] = 0.0;
                    Sx1[1] = 0.0;
                    // Sx1[2] = 0.0; // Always set below
                    Sx1[3] = 0.0;
                    Sx1[4] = 0.0;

                    Sx1[ip_m_ipo + 1] = 0.5 * ( delta2 - delta + 0.25 );
                    Sx1[ip_m_ipo + 2] = 0.75 - delta2;
                    Sx1[ip_m_ipo + 3] = 0.5 * ( delta2 + delta + 0.25 );
                }
                {
                    const double ypn = device_particle_position_y[particle_index] * dy_inv;
                    const int    jp  = std::round( ypn );
                    // const int    jp       = static_cast<int>( ypn + 0.5 ); // std::round | rounding approximation which is correct enough and faster in this case
                    const int    jpo      = iold[1 * particle_count];
                    const int    jp_m_jpo = jp - jpo - j_domain_begin;
                    const double delta    = ypn - static_cast<double>( jp );
                    const double delta2   = delta * delta;

                    Sy1[0] = 0.0;
                    Sy1[1] = 0.0;
                    // Sy1[2] = 0.0; // Always set below
                    Sy1[3] = 0.0;
                    Sy1[4] = 0.0;

                    Sy1[jp_m_jpo + 1] = 0.5 * ( delta2 - delta + 0.25 );
                    Sy1[jp_m_jpo + 2] = 0.75 - delta2;
                    Sy1[jp_m_jpo + 3] = 0.5 * ( delta2 + delta + 0.25 );
                }

                // DSx[0] = Sx1[0] - Sx0[0];
                // DSx[1] = Sx1[1] - Sx0[1];
                // DSx[2] = Sx1[2] - Sx0[2];
                // DSx[3] = Sx1[3] - Sx0[3];
                // DSx[4] = Sx1[4] - Sx0[4];

                // DSy[0] = Sy1[0] - Sy0[0];
                // DSy[1] = Sy1[1] - Sy0[1];
                // DSy[2] = Sy1[2] - Sy0[2];
                // DSy[3] = Sy1[3] - Sy0[3];
                // DSy[4] = Sy1[4] - Sy0[4];

                // (x,y,z) components of the current density for the macro-particle
                const double charge_weight = inv_cell_volume * static_cast<double>( device_particle_charge[particle_index] ) * device_particle_weight[particle_index];
                const double crx_p         = charge_weight * dx_ov_dt;
                const double cry_p         = charge_weight * dy_ov_dt;
                const double crz_p         = charge_weight * ( 1.0 / 3.0 ) * device_particle_momentum_z[particle_index] * invgf;

                // This is the particle position as grid index
                // This minus 2 come from the order 2 scheme, based on a 5 points stencil from -2 to +2.
                const int ipo = iold[0 * particle_count] - 2;
                const int jpo = iold[1 * particle_count] - 2;

                // Jx

                double tmpJx[5]{};

                for( unsigned int i = 1; i < 5; ++i ) {
                    const int iloc = ( i + ipo ) * nprimy + jpo;
                    tmpJx[0] -= crx_p * ( Sx1[i - 1] - Sx0[i - 1] ) * ( 0.5 * ( Sy1[0] - Sy0[0] ) );
                    ::atomicAdd( &device_Jx[iloc], tmpJx[0] );
                    for( unsigned int j = 1; j < 5; ++j ) {
                        tmpJx[j] -= crx_p * ( Sx1[i - 1] - Sx0[i - 1] ) * ( Sy0[j] + 0.5 * ( Sy1[j] - Sy0[j] ) );
                        ::atomicAdd( &device_Jx[iloc + j], tmpJx[j] );
                    }
                }

                // Jy

                for( unsigned int i = 0; i < 1; ++i ) {
                    const int iloc = ( i + ipo ) * nprimy + jpo;
                    double    tmp  = 0.0;
                    for( unsigned int j = 1; j < 5; j++ ) {
                        tmp -= cry_p * ( Sy1[j - 1] - Sy0[j - 1] ) * ( Sx0[i] + 0.5 * ( Sx1[i] - Sx0[i] ) );
                        ::atomicAdd( &device_Jy[iloc + j + pxr * ( /* i + */ ipo )], tmp );
                    }
                }

                for( unsigned int i = 1; i < 5; ++i ) {
                    const int iloc = ( i + ipo ) * nprimy + jpo;
                    double    tmp  = 0.0;
                    for( unsigned int j = 1; j < 5; ++j ) {
                        tmp -= cry_p * ( Sy1[j - 1] - Sy0[j - 1] ) * ( Sx0[i] + 0.5 * ( Sx1[i] - Sx0[i] ) );
                        ::atomicAdd( &device_Jy[iloc + j + pxr * ( i + ipo )], tmp );
                    }
                }

                // Jz

                for( unsigned int i = 0; i < 1; ++i ) {
                    const int iloc = ( i + ipo ) * nprimy + jpo;
                    /* Jx[iloc] += tmpJx[0]; */
                    ::atomicAdd( &device_Jz[iloc], crz_p * ( Sy1[0] * ( /* 0.5 * Sx0[i] + */ Sx1[i] ) ) );
                    for( unsigned int j = 1; j < 5; j++ ) {
                        ::atomicAdd( &device_Jz[iloc + j], crz_p * ( Sy0[j] * ( 0.5 * Sx1[i] /* + Sx0[i] */ ) +
                                                                     Sy1[j] * ( /* 0.5 * Sx0[i] + */ Sx1[i] ) ) );
                    }
                }

                for( unsigned int i = 1; i < 5; ++i ) {
                    const int iloc = ( i + ipo ) * nprimy + jpo;
                    ::atomicAdd( &device_Jz[iloc], crz_p * ( Sy1[0] * ( 0.5 * Sx0[i] + Sx1[i] ) ) );
                    double tmp = 0.0;
                    for( unsigned int j = 1; j < 5; ++j ) {
                        ::atomicAdd( &device_Jz[iloc + j], crz_p * ( Sy0[j] * ( 0.5 * Sx1[i] + Sx0[i] ) +
                                                                     Sy1[j] * ( 0.5 * Sx0[i] + Sx1[i] ) ) );
                    }
                }
            }

            __syncthreads();

            for( unsigned int field_index = thread_index_offset;
                 field_index < kFieldScratchSpaceSize;
                 field_index += workgroup_size ) {
                // TODO(Etienne M): Should I try to remove the bank conflicts?
                // This could prevent coalescing the access to global memory!

                ::atomicAdd( &device_Jx[/* TODO */ field_index], Jx_field_scratch_space[field_index] );
                ::atomicAdd( &device_Jy[/* TODO */ field_index], Jy_field_scratch_space[field_index] );
                ::atomicAdd( &device_Jz[/* TODO */ field_index], Jz_field_scratch_space[field_index] );
            }
        }
    } // namespace kernel

    static inline void
    currentDepositionKernel( double *__restrict__ host_Jx,
                             double *__restrict__ host_Jy,
                             double *__restrict__ host_Jz,
                             int Jx_size,
                             int Jy_size,
                             int Jz_size,
                             const double *__restrict__ device_particle_position_x,
                             const double *__restrict__ device_particle_position_y,
                             const double *__restrict__ device_particle_momentum_z,
                             const short *__restrict__ device_particle_charge,
                             const double *__restrict__ device_particle_weight,
                             const int *__restrict__ host_bin_index,
                             unsigned int x_dimension_bin_count,
                             unsigned int y_dimension_bin_count,
                             const double *__restrict__ host_invgf_,
                             const int *__restrict__ host_iold_,
                             const double *__restrict__ host_deltaold_,
                             double inv_cell_volume,
                             double dx_inv,
                             double dy_inv,
                             double dx_ov_dt,
                             double dy_ov_dt,
                             int    i_domain_begin,
                             int    j_domain_begin,
                             int    nprimy,
                             int    pxr )
    {
        int device_count;
        checkHIPErrors( ::hipGetDeviceCount( &device_count ) );
        SMILEI_ASSERT( device_count == 1 );

        // NOTE:
        // Doc at: https://github.com/RadeonOpenCompute/ROCm/tree/rocm-4.5.2
        // 1 or 3 streams (Jx Jy Jz) ?
        // hipOccupancyMaxPotentialBlockSize
        // __ldg | non coherent cache | this is sometimes generated implicitly when using restricted ptrs
        //

        const ::dim3 kGridDimensionInBlock{ static_cast<uint32_t>( x_dimension_bin_count ), static_cast<uint32_t>( y_dimension_bin_count ), 1 };
        const ::dim3 kBlockDimensionInWorkItem{ 128, 1, 1 };

        hipLaunchKernelGGL( kernel::depositeForAllCurrentDimensions,
                            kGridDimensionInBlock,
                            kBlockDimensionInWorkItem,
                            0, // Shared memory
                            0, // Stream
                            // Kernel arguments
                            smilei::tools::gpu::HostDeviceMemoryManagment::GetDevicePointer( host_Jx ),
                            smilei::tools::gpu::HostDeviceMemoryManagment::GetDevicePointer( host_Jy ),
                            smilei::tools::gpu::HostDeviceMemoryManagment::GetDevicePointer( host_Jz ),
                            Jx_size, Jy_size, Jz_size,
                            device_particle_position_x,
                            device_particle_position_y,
                            device_particle_momentum_z,
                            device_particle_charge,
                            device_particle_weight,
                            smilei::tools::gpu::HostDeviceMemoryManagment::GetDevicePointer( host_bin_index ),
                            smilei::tools::gpu::HostDeviceMemoryManagment::GetDevicePointer( host_invgf_ ),
                            smilei::tools::gpu::HostDeviceMemoryManagment::GetDevicePointer( host_iold_ ),
                            smilei::tools::gpu::HostDeviceMemoryManagment::GetDevicePointer( host_deltaold_ ),
                            inv_cell_volume,
                            dx_inv, dy_inv,
                            dx_ov_dt, dy_ov_dt,
                            i_domain_begin, j_domain_begin,
                            nprimy,
                            pxr );

        checkHIPErrors( ::hipDeviceSynchronize() );
    }

} // namespace hip

    #endif

//! Project global current densities (EMfields->Jx_/Jy_/Jz_)
//!
extern "C" void
currentDepositionKernel( double *__restrict__ host_Jx,
                         double *__restrict__ host_Jy,
                         double *__restrict__ host_Jz,
                         int Jx_size,
                         int Jy_size,
                         int Jz_size,
                         const double *__restrict__ device_particle_position_x,
                         const double *__restrict__ device_particle_position_y,
                         const double *__restrict__ device_particle_momentum_z,
                         const short *__restrict__ device_particle_charge,
                         const double *__restrict__ device_particle_weight,
                         const int *__restrict__ host_bin_index,
                         unsigned int x_dimension_bin_count,
                         unsigned int y_dimension_bin_count,
                         const double *__restrict__ host_invgf_,
                         const int *__restrict__ host_iold_,
                         const double *__restrict__ host_deltaold_,
                         double inv_cell_volume,
                         double dx_inv,
                         double dy_inv,
                         double dx_ov_dt,
                         double dy_ov_dt,
                         int    i_domain_begin,
                         int    j_domain_begin,
                         int    nprimy,
                         int    pxr )
{
    #if defined( PRIVATE_SMILEI_USE_OPENMP_PROJECTION_IMPLENTATION )
    naive::
    #else
    hip::
    #endif
        currentDepositionKernel( host_Jx, host_Jy, host_Jz,
                                 Jx_size, Jy_size, Jz_size,
                                 device_particle_position_x, device_particle_position_y,
                                 device_particle_momentum_z,
                                 device_particle_charge,
                                 device_particle_weight,
                                 host_bin_index,
                                 x_dimension_bin_count,
                                 y_dimension_bin_count,
                                 host_invgf_,
                                 host_iold_, host_deltaold_,
                                 inv_cell_volume,
                                 dx_inv, dy_inv,
                                 dx_ov_dt, dy_ov_dt,
                                 i_domain_begin, j_domain_begin,
                                 nprimy,
                                 pxr );
}

#endif
