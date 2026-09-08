model HeatExchanger "Two fluid streams trading heat through a conducting wall, a lumped exchanger with wall inertia"
  parameter Real c_hot(unit = "J/K") = 4000 "Heat capacity of the hot control volume";
  parameter Real c_wall(unit = "J/K") = 8000 "Heat capacity of the separating wall";
  parameter Real c_cold(unit = "J/K") = 4000 "Heat capacity of the cold control volume";
  parameter Real q_hot(unit = "W/K") = 200 "Hot-stream flow heat-capacity rate, mdot*cp";
  parameter Real q_cold(unit = "W/K") = 150 "Cold-stream flow heat-capacity rate, mdot*cp";
  parameter Real ua_hot(unit = "W/K") = 500 "Conductance between the hot stream and the wall";
  parameter Real ua_cold(unit = "W/K") = 500 "Conductance between the wall and the cold stream";
  parameter Real th_in(unit = "K") = 360 "Hot-stream inlet temperature";
  parameter Real tc_in(unit = "K") = 300 "Cold-stream inlet temperature";
  Real t_hot(unit = "K", start = 300, fixed = true) "Hot-stream outlet temperature";
  Real t_wall(unit = "K", start = 300, fixed = true) "Wall temperature";
  Real t_cold(unit = "K", start = 300, fixed = true) "Cold-stream outlet temperature";
equation
  c_hot * der(t_hot) = q_hot * (th_in - t_hot) - ua_hot * (t_hot - t_wall);
  c_wall * der(t_wall) = ua_hot * (t_hot - t_wall) - ua_cold * (t_wall - t_cold);
  c_cold * der(t_cold) = q_cold * (tc_in - t_cold) + ua_cold * (t_wall - t_cold);
end HeatExchanger;
