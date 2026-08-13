model HIRES "High Irradiance RESponse of photomorphogenesis -- eight reactants, mildly stiff"
  Real y1(start = 1, fixed = true) "P_r, the red-absorbing form of phytochrome";
  Real y2(start = 0, fixed = true) "P_fr, the far-red-absorbing form";
  Real y3(start = 0, fixed = true) "P_rX, P_r bound to receptor X";
  Real y4(start = 0, fixed = true) "P_frX, P_fr bound to receptor X";
  Real y5(start = 0, fixed = true) "P_rX', P_r bound to receptor X'";
  Real y6(start = 0, fixed = true) "P_frX', P_fr bound to receptor X'";
  Real y7(start = 0, fixed = true) "P_frX'E, the enzyme complex";
  Real y8(start = 0.0057, fixed = true) "E, the free enzyme";
equation
  der(y1) = -1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007;
  der(y2) = 1.71 * y1 - 8.75 * y2;
  der(y3) = -10.03 * y3 + 0.43 * y4 + 0.035 * y5;
  der(y4) = 8.32 * y2 + 1.71 * y3 - 1.12 * y4;
  der(y5) = -1.745 * y5 + 0.43 * y6 + 0.43 * y7;
  der(y6) = -280 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7;
  der(y7) = 280 * y6 * y8 - 1.81 * y7;
  der(y8) = -280 * y6 * y8 + 1.81 * y7;
end HIRES;
