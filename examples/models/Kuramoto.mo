model Kuramoto "The Kuramoto model: four coupled phase oscillators pulling into synchrony"
  parameter Integer n = 4 "Number of oscillators";
  parameter Real k(unit = "rad/s") = 1.5 "Coupling strength (above the synchronising threshold)";
  parameter Real omega[n](unit = "rad/s") = {-0.6, -0.2, 0.2, 0.6} "Natural frequency of each oscillator (symmetric, mean zero)";
  Real theta[n](unit = "rad", start = {0, 1.5, 3, 4.5}, fixed = true) "Phase of each oscillator";
  Real r "Order parameter: 0 fully incoherent, 1 fully synchronised";
equation
  der(theta[1]) = omega[1] + k / n * (sin(theta[1] - theta[1]) + sin(theta[2] - theta[1]) + sin(theta[3] - theta[1]) + sin(theta[4] - theta[1]));
  der(theta[2]) = omega[2] + k / n * (sin(theta[1] - theta[2]) + sin(theta[2] - theta[2]) + sin(theta[3] - theta[2]) + sin(theta[4] - theta[2]));
  der(theta[3]) = omega[3] + k / n * (sin(theta[1] - theta[3]) + sin(theta[2] - theta[3]) + sin(theta[3] - theta[3]) + sin(theta[4] - theta[3]));
  der(theta[4]) = omega[4] + k / n * (sin(theta[1] - theta[4]) + sin(theta[2] - theta[4]) + sin(theta[3] - theta[4]) + sin(theta[4] - theta[4]));
  r = sqrt(((cos(theta[1]) + cos(theta[2]) + cos(theta[3]) + cos(theta[4])) / n) ^ 2 + ((sin(theta[1]) + sin(theta[2]) + sin(theta[3]) + sin(theta[4])) / n) ^ 2);
end Kuramoto;
