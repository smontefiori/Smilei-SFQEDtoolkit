/*! @file Pusher.h

 @brief Pusher.h  generic class for the particle pusher

 @date 2013-02-15
 */

#ifndef PUSHERBORISBY_H
#define PUSHERBORISBY_H

#include "Pusher.h"

#ifdef SMILEI_SFQEDTOOLKIT
#include "SFQEDtoolkit_Interface.hpp"
#endif

//  --------------------------------------------------------------------------------------------------------------------
//! Class PusherBorisV
//  --------------------------------------------------------------------------------------------------------------------
class PusherBorisBeyond : public Pusher
{
public:
    //! Creator for Pusher
    PusherBorisBeyond( Params &params, Species *species );
    ~PusherBorisBeyond();
    //! Overloading of () operator
    void operator()( Particles &particles, SmileiMPI *smpi, int istart, int iend, int ithread, int ipart_buffer_offset = 0 ) override;
};

#endif
