The goal is to train a profile predictor on a small set of high-beta DIII-D shots (~100)
and use this predictor to optimize a shape trajectory to avoid onset of instabilities

This is scuffed because what we really need is a predictor of NTM onset, since that's what
usually kills these high-beta discharges on DIII-D
That's exactly what my thesis will be on, and requires an NTM dataset (WORKING ON IT)

Still taking this opportunity to go through the motions and see what problems are
encountered when applying these methods to an actual experiment.

Brief description:

Inputs are Ip, edge density, beta, and shaping <- all of these in real-time feedback, quite good
- Can't touch Ip, current ramp is locked in
- Don't want to mess with edge density since that greatly changes the scenario
- Can't modify beta because the whole point is running in high beta
- Must ensure changes in shape are within acceptable boundaries

So, we can only change the shaping. There are some restrictions related to the DIII-D PCS though
- Cannot program in directly aminor, kappa, triangularity
- Operator will need to see what shape we request, and program it in manually
- Our 'trajectory' then is one new shape every 0.5 seconds

Our trajectory will aim to minimize a cost function that is a simple proxy for stability
- Minimize peaking factor (core temp/density vs edge temp/density) <- informed by https://arxiv.org/html/2502.20294v3
- some collisionality metric? Do that later, starting simple

