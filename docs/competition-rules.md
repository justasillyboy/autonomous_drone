# Competition rules (summary)

A summary of the parts of the "UAV Cứu Hỏa & Cứu Hộ" rulebook (version of 27/02/2026) that
the autonomous round depends on. The organisers' PDF is not redistributed here.

## Field (§IV, figure 1)

- 24 x 12 m. A 3.5 m deep **Safe Zone** at one end holds the helipad (1.5 m square, 1 m from
  the end line, on the centre line), which is also the start position.
- Five fire circles on 0.1 m platforms: **F1, F2 red (1.0 m)**, **F3, F4 yellow (0.6 m)**,
  **F5 blue (0.4 m)**. In the qualifier they sit at fixed positions (F2 at the centre of a
  2.5 m circle carrying the other four; F1 8 m from the helipad); in the semi-final and final
  they are drawn from a matrix of positions before the match.
- Four TZ gates (rings of 1.6 m inner diameter, centre 1.7 m high, 2.6 m tall) used by the
  manual round; in the autonomous round they are obstacles.
- Autonomous flight height: 3.0 to 5.0 m.

## Autonomous round (round 2)

1. **Takeoff:** from the start position, without pilot input, to above 3.0 m.
2. **Fire fighting:** drop tennis balls (at most 5 carried) into the circles; the order is
   the team's choice. Each circle scores once; a ball outside the circle or a second ball in
   a done circle scores nothing. The lowest point of the drone must be at least 3.0 m above
   the floor at release.
3. **Landing:** return to the Safe Zone and land on the start position, fully inside the
   helipad.

In the **final**, F2 is on a moving platform: 10 cm/s, back and forth along a straight line.

## Scoring (§VI)

| Item | Qualifier / semi-final | Final |
|---|---|---|
| Automatic takeoff (>= 3.0 m) | 20 | 10 |
| F1 red | 20 | 10 |
| F2 red | 20 | 50 (moving) |
| F3, F4 yellow | 30 each | 30 each |
| F5 blue | 50 | 50 |
| Automatic landing inside the helipad | 30 | 20 |
| **Total** | **200** | **200** |

## Penalties and limits (§VII, §VIII)

- Touching a TZ gate: -5 per contact.
- Leaving the field for more than 5 s: -5 per occurrence.
- Autonomous flight below 3.0 m for more than 5 s: -5 per occurrence.
- One retry per round; the clock keeps running and earlier points are cancelled.
- Fail-safe required: return to the Safe Zone or land when the link is lost for 3 s; the
  ground station needs an emergency stop.
- Match time: 300 s.

## How the code maps them

`uav_core.config.RulebookConfig` holds the scoring tables, field size and penalties;
`DropConfig` enforces the 3.0-5.0 m band (with a 5 cm margin) at startup. The simulation's
independent referee (`docs/simulation.md`) applies the rules from ground truth and states
where it had to interpret them (for instance, "-5 per occurrence" is counted per full 5 s,
as the mission does).
