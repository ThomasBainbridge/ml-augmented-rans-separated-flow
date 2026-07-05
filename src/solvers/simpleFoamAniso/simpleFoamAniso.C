/*---------------------------------------------------------------------------*\
  simpleFoamAniso

  A fork of simpleFoam for a-posteriori testing of a data-driven Reynolds-stress
  ANISOTROPY correction. It reads a fixed, symmetric, deviatoric stress-
  correction field aDelta (predicted offline) and adds its divergence to the
  momentum equation, on top of the eddy-viscosity stress from the turbulence
  model:

      div(phi,U) + divDevReff(U) + div(aDelta) == -grad(p)

  With aDelta = 2 k (b_DNS - b_RANS), the total deviatoric Reynolds stress
  becomes 2 k b_DNS: the modelled anisotropy is upgraded to the DNS anisotropy
  (with the RANS turbulent energy scale). Turbulence still updates each
  iteration, so the correction is coupled, not frozen.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state incompressible solver with a data-driven Reynolds-stress"
        " anisotropy correction (coupled)."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        {
            #include "UEqn.H"
            #include "pEqn.H"
        }

        laminarTransport.correct();
        turbulence->correct();

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
