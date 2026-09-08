# Behavioural State Segmentation

This MoveApp annotates movement fixes with behavioural states using the current
[`hmmcma`](https://github.com/Omar-Chatila/cma_hmm) interfaces. It offers a
data-driven Hidden Markov Model and the rule-based criteria segmentation backed
by the bundled C++ engine.

## Input and output

Input and output are `movingpandas.TrajectoryCollection` objects. Input fixes
must have a datetime index, a trajectory-ID column exposed by MovingPandas, and
a defined coordinate reference system. Before segmentation, the App selects an
appropriate WGS84 UTM zone independently for each animal and adds `utm_x`,
`utm_y`, and `utm_crs`; the original geometry and CRS remain unchanged.

The output retains the input fixes and adds the common hmmcma annotation
columns:

- `state`: numeric behavioural state; `-1` means unmodelled
- `cluster`: mirrors `state` for downstream compatibility
- `segment_id`: temporal segment identifier (`-1` where unavailable)
- `change_point`: whether the fix starts a new criteria segment
- `segment_name`: semantic rule label (criteria method only)

The App also creates `state_plot.png` as an App artifact. It contains a spatial
state plot and a state timeline per trajectory.

## Settings

### Segmentation method (`method`)

- `Hidden Markov Model`: fits a Gaussian HMM to the selected movement features.
- `Criteria-based segmentation`: applies labelled rules through the hmmcma C++
  backend.

### Number of states (`num_states`)

For HMM, this is the number of latent states. For criteria segmentation, this
must equal the number of semicolon-separated rules in `criteria_rules`.

### HMM features (`features`)

Enter comma- or space-separated numbers. Repeated selections are ignored.

1. SPEED
2. DISTANCE
3. TURN ANGLE
4. DIRECTION
5. ANGULAR DIFFERENCE
6. PERSISTENCE VELOCITY
7. TERRAIN

For example, `1,3` selects speed and turn angle. `TERRAIN` additionally requires
a numeric input column named `terrain`.

### Criteria rules (`criteria_rules`)

Enter one rule per state, separated by semicolons:

```text
resting:disk=100; travelling:minimum_hops=2&heading=35; foraging:maximum_hops=6
```

The text before `:` is the state label. If omitted, the App uses `state_1`,
`state_2`, and so on. Within one rule, combine criteria with `&` (all criteria)
or `|` (any criterion); prefix a criterion with `!` to negate it. Do not mix `&`
and `|` within one rule.

Available criteria are:

- `disk=RADIUS_METRES` (alias: `disc`)
- `minimum_hops=COUNT`, `maximum_hops=COUNT`
- `minimum_duration=SECONDS`
- `speed=METRES_PER_SECOND`
- `turning_angle=DEGREES` (alias: `turn_angle`)
- `heading=DEGREES`
- `speed_with_outliers=THRESHOLD,COUNT`
- `turning_angle_with_outliers=THRESHOLD,COUNT`
- `heading_with_outliers=THRESHOLD,COUNT`

Speed, turning-angle, and heading thresholds constrain the within-segment range;
they are not absolute observed values. Rule order matters when more than one rule
can match a segment.

### Fill with last state (`lfill`)

If disabled, fixes not classified by the selected method have `state=-1`. If
enabled, the App carries the most recent modelled state forward, independently
inside every trajectory. Leading unmodelled fixes stay `-1`, so states never
leak between animals.

## Errors and null handling

The App reports an error when the input CRS or valid geometry is absent, no
trajectory sequence has enough valid fixes, feature numbers are unknown,
criteria syntax is invalid, or the number of rules differs from `num_states`.
Invalid fixes and too-short trajectory fragments are retained as unmodelled
rather than removed.
