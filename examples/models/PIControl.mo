model PIControl "A first-order plant under PI control, brought from 0 to a setpoint of 1"
  Blocks.Constant setpoint(k = 1);
  Blocks.Feedback error;
  Blocks.PI controller(k = 2, T = 0.5, x(start = 0, fixed = true));
  Blocks.FirstOrder plant(k = 1, T = 1, y(signal(start = 0, fixed = true)));
equation
  connect(setpoint.y, error.u1);
  connect(plant.y, error.u2);
  connect(error.y, controller.u);
  connect(controller.y, plant.u);
end PIControl;
