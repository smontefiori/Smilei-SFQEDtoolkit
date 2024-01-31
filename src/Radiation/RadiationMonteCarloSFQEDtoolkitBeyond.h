// ----------------------------------------------------------------------------
//! \file RadiationMonteCarloSFQEDtoolkit.h
//
//! \brief This class performs the Nonlinear Inverse Compton Scattering
//! on particles using the SFQEDtoolkit library available at
//! https://github.com/QuantumPlasma/SFQEDtoolkit.
//
//! \details This header contains the definition of the class RadiationMonteCarlo.
//! The implementation is adapted from the thesis results of M. Lobet
//! See http://www.theses.fr/2015BORD0361
// ----------------------------------------------------------------------------

#ifndef RADIATIONMONTECARLOSFQEDTOOLKITBY_H
#define RADIATIONMONTECARLOSFQEDTOOLKITBY_H

#include "RadiationTables.h"
#include "Radiation.h"
#include "userFunctions.h"

#ifdef SMILEI_OPENACC_MODE
#include <openacc.h>
// This is wrong. Dont include nvidiaParticles, it may cause problem!
// See particle factory.
#include "nvidiaParticles.h"
#endif

#ifdef SMILEI_SFQEDTOOLKIT
#include "SFQEDtoolkit_Interface.hpp"
#endif

//----------------------------------------------------------------------------------------------------------------------
//! RadiationMonteCarlo class: holds parameters and functions to apply the
//! nonlinear inverse Compton scattering on Particles.
//----------------------------------------------------------------------------------------------------------------------
class RadiationMonteCarloSFQEDtoolkitBeyond : public Radiation
{

public:

    //! Constructor for RadiationMonteCarlo
    RadiationMonteCarloSFQEDtoolkitBeyond( Params &params, Species *species, Random * rand  );

    //! Destructor for RadiationMonteCarlo
    ~RadiationMonteCarloSFQEDtoolkitBeyond();

    // ---------------------------------------------------------------------
    //! Overloading of () operator: perform the Discontinuous radiation
    //! reaction induced by the nonlinear inverse Compton scattering
    //! \param particles   particle object containing the particles
    //! \param photon_species species that will receive emitted photons
    //!                    properties of the current species
    //! \param smpi        MPI properties
    //! \param radiation_tables Cross-section data tables and useful functions
    //                     for nonlinear inverse Compton scattering
    //! \param istart      Index of the first particle
    //! \param iend        Index of the last particle
    //! \param ithread     Thread index
    //! \param radiated_energy     overall energy radiated during the call to this method
    // ---------------------------------------------------------------------
    virtual void operator()(
        Particles &particles,
        Particles *photons,
        SmileiMPI *smpi,
        RadiationTables &radiation_tables,
        double          &radiated_energy,
        int             istart,
        int             iend,
        int             ithread,
        int             ibin,
        int             ipart_ref = 0
       );

protected:

    // ________________________________________
    // General parameters

    //! Number of photons emitted per event for statisctics purposes
    int radiation_photon_sampling_;

    // Maximum number of emission per particle per iteration
    int max_photon_emissions_;

    //! Threshold on the photon Lorentz factor under which the macro-photon
    //! is not generated but directly added to the energy scalar diags
    //! This enable to limit emission of useless low-energy photons
    double radiation_photon_gamma_threshold_;

    //! Inverse number of photons emitted per event for statisctics purposes
    double inv_radiation_photon_sampling_;

    //! Max number of Monte-Carlo iteration
    const int max_monte_carlo_iterations_ = 100;

    //! Espilon to check when tau is near 0
    const double epsilon_tau_ = 1e-100;

    //! normalized compton time
    double norm_Compton_time;

private:

};

#endif

