model CoupledOscillators "Two masses coupled by a spring, in state-space form der(z) = A*z: the energy beats between them"
  parameter Real a[4, 4] = {{0, 0, 1, 0}, {0, 0, 0, 1}, {-1.5, 0.5, 0, 0}, {0.5, -1.5, 0, 0}} "State matrix: unit masses, wall springs 1, coupling 0.5";
  Real z[4](start = {1, 0, 0, 0}, fixed = true) "State [x1, x2, v1, v2]";
equation
  der(z) = a * z;
end CoupledOscillators;
