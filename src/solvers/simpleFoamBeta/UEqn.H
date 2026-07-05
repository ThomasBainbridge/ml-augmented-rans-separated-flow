    // Momentum predictor

    MRF.correctBoundaryVelocity(U);

    // Corrected effective viscosity: nu + betaNut * nut. nut is the live value
    // from the turbulence model (updated each iteration), so the correction is
    // fully coupled with k, omega and the mean flow. This reproduces
    // turbulence->divDevReff(U) exactly, with nuEff -> nu + betaNut*nut.
    volScalarField nuEffb
    (
        "nuEffb",
        laminarTransport.nu() + betaNut*turbulence->nut()
    );

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(U)
      - fvm::laplacian(nuEffb, U)
      - fvc::div(nuEffb*dev2(T(fvc::grad(U))))
     ==
        fvOptions(U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (simple.momentumPredictor())
    {
        solve(UEqn == -fvc::grad(p));

        fvOptions.correct(U);
    }
