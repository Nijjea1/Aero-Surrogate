// What a drag and lift prediction means on the road.  OWNER: Paramveer.
//
// Standard aerodynamic loads at sea-level air density: dynamic pressure
// q = 1/2 rho v^2, force = q * coefficient * frontal area. Nothing here is
// modelled; every number follows from the prediction and the car's size.

export const AIR_DENSITY = 1.225;   // kg/m^3, sea level, 15 C

export function aeroLoads({ cd, cl, sigma = 0, area, speedKmh }) {
  const v = speedKmh / 3.6;                         // m/s
  const q = 0.5 * AIR_DENSITY * v * v;              // Pa
  const drag = q * cd * area;                       // N
  return {
    speed: v,
    dynamicPressure: q,
    cdA: cd * area,                                 // m^2
    drag,
    dragSigma: q * sigma * area,                    // N, one sigma of the Cd prediction
    lift: q * cl * area,                            // N, negative is downforce
    power: (drag * v) / 1000,                       // kW to push through the air
    energyPer100km: (drag * 100_000) / 3.6e6,       // kWh, drag alone
  };
}
