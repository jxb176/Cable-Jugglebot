# Cable-Jugglebot
## Path Planner (Draft Notes)
### Path primative notes
* Cubic/Quintics are difficult to manage envelope constraints and to optimize.  Perhaps worth revisiting later, but for now this will be deferred.
* Jerk limited S-curve trajectory profiles (similar to Ruckig) hold more promise with careful implementation to allow clean integration with optimizer.
* Monotonic acceleration line motion moves in straight line, doesn't allow acceleration reversal (aka velocity sign change)
* Option to scale acceleration/jerk to end segment at desired position/velocity end point.  Acceleration and jerk are scaled together with a linear k value together if both are applied, though optionally you can only scale one or the other. IF you scale neither, then the profile uses the exact acceleration and jerk.
* TODO: Flag for success (position/velocity tolerance met)
* TODO: Option to constrain start or end point when segment length can't be met
### Path Planner
* Juggling Pattern Generator - Catch/Throw sequences, positions, timing
* Feasibility and optimiziation will happen at this layer
