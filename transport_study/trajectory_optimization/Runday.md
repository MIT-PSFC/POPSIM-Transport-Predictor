# Checklist before anything can be run:

1. Ensure the PCS is set up to have the proper shape, density, and beta targets

From looking at 201927, our inputs are as follows:
- Plasma current, match iptipp and cpasma
- Toroidal field, match bttbt and bt
- Pedestal density (on PEDESTAL control), match dstdenp and dssneped
- Normalized beta (on NORMALIZED BETA control), match bmtpwrtar and betanf (or betan from EFIT)
- Major radius (on RSURF control), match idtrp and rsurf from EFIT
- Bottom X-point location, match idtzxbot with zxpt1 and idtrxbot with rxpt1
inner gap might have to be vibes, I'm not seeing a clear mapping to the outer segments between shots


# Order of operations and the scripts that make them happen:
1. characterize_dataset.sh
- Absolute ranges of our control variables for trajectory optimization
- Typical control errors for each signal

2. Trajectory optimization
- For our reference shot 201927, take 1000 random samples of its trajectory according to the characterization
- Optimize a trajectory that minimizes the cost function with ONE shape input at the start of the current ramp
- Output trajectory in terms of the variables we want and what that *should* be in the DIII-D PCS
- Things like a_minor_desired, etc.

3. predict_first.py (predict_first.sh)
given several model paths, 
make predict-first dataset
- predict-first only valid after the programmed beta actually works

1. Show a gif of predicted profiles error bars before the shot runs
2. Pull zipfits and plot them as a comparison (a fast one and a slow one)