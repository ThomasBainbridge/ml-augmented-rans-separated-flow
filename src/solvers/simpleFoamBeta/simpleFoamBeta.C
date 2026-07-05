/*---------------------------------------------------------------------------*\
  simpleFoamBeta

  A minimal fork of OpenFOAM's simpleFoam for a-posteriori, closed-loop testing
  of a data-driven eddy-viscosity correction. It reads a fixed corrective field
  betaNut (predicted offline by the ML model) and solves the momentum equation
  with an effective viscosity

      nuEff = nu + betaNut * nut

  where nut is the LIVE turbulent viscosity: turbulence->correct() still runs
  every SIMPLE iteration, so k, omega, nut and the mean flow all co-adapt to the
  correction. This is genuine coupling (field-inversion style), not the frozen
  propagation of analysis/06 and analysis/10.

  Only createFields.H (reads betaNut) and UEqn.H (uses nuEff) differ from the
  stock solver.
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
        "Steady-state incompressible solver with a data-driven betaNut"
        " eddy-viscosity correction (coupled)."
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

        // --- Pressure-velocity SIMPLE corrector
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
