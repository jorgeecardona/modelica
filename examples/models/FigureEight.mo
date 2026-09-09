model FigureEight "The figure-eight three-body choreography: three equal masses on one closed curve"
  parameter Real gm(unit = "m3/s2") = 1 "G times the common mass, with the mass taken as 1";
  Real x1(unit = "m", start = 0.97000436, fixed = true) "Body 1 position, x";
  Real y1(unit = "m", start = -0.24308753, fixed = true) "Body 1 position, y";
  Real x2(unit = "m", start = -0.97000436, fixed = true) "Body 2 position, x";
  Real y2(unit = "m", start = 0.24308753, fixed = true) "Body 2 position, y";
  Real x3(unit = "m", start = 0, fixed = true) "Body 3 position, x";
  Real y3(unit = "m", start = 0, fixed = true) "Body 3 position, y";
  Real vx1(unit = "m/s", start = 0.46620369, fixed = true) "Body 1 velocity, x";
  Real vy1(unit = "m/s", start = 0.43236573, fixed = true) "Body 1 velocity, y";
  Real vx2(unit = "m/s", start = 0.46620369, fixed = true) "Body 2 velocity, x";
  Real vy2(unit = "m/s", start = 0.43236573, fixed = true) "Body 2 velocity, y";
  Real vx3(unit = "m/s", start = -0.93240737, fixed = true) "Body 3 velocity, x";
  Real vy3(unit = "m/s", start = -0.86473146, fixed = true) "Body 3 velocity, y";
equation
  der(x1) = vx1;
  der(y1) = vy1;
  der(x2) = vx2;
  der(y2) = vy2;
  der(x3) = vx3;
  der(y3) = vy3;
  der(vx1) = gm * ((x2 - x1) / ((x1 - x2) ^ 2 + (y1 - y2) ^ 2) ^ 1.5 + (x3 - x1) / ((x1 - x3) ^ 2 + (y1 - y3) ^ 2) ^ 1.5);
  der(vy1) = gm * ((y2 - y1) / ((x1 - x2) ^ 2 + (y1 - y2) ^ 2) ^ 1.5 + (y3 - y1) / ((x1 - x3) ^ 2 + (y1 - y3) ^ 2) ^ 1.5);
  der(vx2) = gm * ((x1 - x2) / ((x1 - x2) ^ 2 + (y1 - y2) ^ 2) ^ 1.5 + (x3 - x2) / ((x2 - x3) ^ 2 + (y2 - y3) ^ 2) ^ 1.5);
  der(vy2) = gm * ((y1 - y2) / ((x1 - x2) ^ 2 + (y1 - y2) ^ 2) ^ 1.5 + (y3 - y2) / ((x2 - x3) ^ 2 + (y2 - y3) ^ 2) ^ 1.5);
  der(vx3) = gm * ((x1 - x3) / ((x1 - x3) ^ 2 + (y1 - y3) ^ 2) ^ 1.5 + (x2 - x3) / ((x2 - x3) ^ 2 + (y2 - y3) ^ 2) ^ 1.5);
  der(vy3) = gm * ((y1 - y3) / ((x1 - x3) ^ 2 + (y1 - y3) ^ 2) ^ 1.5 + (y2 - y3) / ((x2 - x3) ^ 2 + (y2 - y3) ^ 2) ^ 1.5);
end FigureEight;
