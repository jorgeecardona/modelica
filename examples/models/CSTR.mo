model CSTR "An exothermic reaction in a cooled stirred tank: heat and concentration feed back"
  parameter Real q(unit = "L/min") = 100 "Volumetric feed and outflow rate";
  parameter Real V(unit = "L") = 100 "Reactor volume";
  parameter Real Caf(unit = "mol/L") = 1 "Feed concentration of A";
  parameter Real Tf(unit = "K") = 350 "Feed temperature";
  parameter Real Tc(unit = "K") = 300 "Coolant temperature";
  parameter Real k0(unit = "1/min") = 72000000000.0 "Arrhenius pre-exponential factor";
  parameter Real EoR(unit = "K") = 8750 "Activation energy over the gas constant";
  parameter Real H_rxn(unit = "J/mol") = -50000.0 "Heat of reaction, negative is exothermic";
  parameter Real rho(unit = "g/L") = 1000 "Mixture density";
  parameter Real Cp(unit = "J/(g.K)") = 0.239 "Mixture heat capacity";
  parameter Real UA(unit = "J/(min.K)") = 50000.0 "Cooling-jacket heat-transfer coefficient";
  Real Ca(unit = "mol/L", start = 1, fixed = true) "Concentration of A in the reactor";
  Real T(unit = "K", start = 350, fixed = true) "Reactor temperature";
  Real r(unit = "mol/(L.min)") "Reaction rate";
equation
  r = k0 * exp(-EoR / T) * Ca;
  der(Ca) = q / V * (Caf - Ca) - r;
  rho * Cp * der(T) = rho * Cp * q / V * (Tf - T) - H_rxn * r + UA / V * (Tc - T);
end CSTR;
