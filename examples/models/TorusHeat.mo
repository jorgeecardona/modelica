model TorusHeat "The heat equation on a flat torus: a 2-D periodic grid by the method of lines"
  parameter Integer n = 8 "Cells along each side of the square grid";
  parameter Real a(unit = "1/s") = 1 "Diffusivity over the square of the cell width";
  constant Real pi = 3.141592653589793 "The ratio of a circle's circumference to diameter";
  Real T[n, n] "Cell temperatures over the torus";
initial equation
  for i in 1:n loop
    for j in 1:n loop
      T[i, j] = cos(2 * pi * (i - 1) / n) * cos(2 * pi * (j - 1) / n);
    end for;
  end for;
equation
  for i in 1:n loop
    for j in 1:n loop
      der(T[i, j]) = a * (T[mod(i - 2, n) + 1, j] + T[mod(i, n) + 1, j] + T[i, mod(j - 2, n) + 1] + T[i, mod(j, n) + 1] - 4 * T[i, j]);
    end for;
  end for;
end TorusHeat;
