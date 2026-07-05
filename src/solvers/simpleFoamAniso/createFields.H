Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject("p", runTime.timeName(), mesh,
             IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject("U", runTime.timeName(), mesh,
             IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

#include "createPhi.H"

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"

// Data-driven deviatoric Reynolds-stress anisotropy correction
// aDelta = 2 k (b_DNS - b_RANS), predicted offline. Dimensions of a kinematic
// stress [m^2/s^2]. Fixed during the solve; turbulence still updates nut.
Info<< "Reading field aDelta (data-driven anisotropy stress correction)\n" << endl;
volSymmTensorField aDelta
(
    IOobject("aDelta", runTime.timeName(), mesh,
             IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);
