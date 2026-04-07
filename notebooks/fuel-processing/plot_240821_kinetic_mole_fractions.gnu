set loadpath '/home/meck/diss-code'; load "fststyle.cfg"
set datafile sep ','

set terminal pngcairo enhanced font 'Verdana,10'
set output "plots/plot_240821_kinetic_mole_fractions.png"

set xlabel "Conversion X_{CH_4}"
set ylabel "Mole fraction x_i"

# set xrange [0:1]
set key center below
set key opaque box lc rgb "white" lw 1

# Set the arrow for equilibrium line
set arrow from eqConversion, graph 0 to eqConversion, graph 1 nohead lt 1 lw .5 lc rgb 'black'

# Define the filled box with a pattern fill 5 for hatching
# set object 1 rect from eqConversion, graph 0.95 to eqConversion+0.0125, graph 1 \
# fc rgb "black" fs pattern 7 noborder behind  

# Add a label "Equilibrium" right-aligned, rotated 90 degrees, with padding
set label "X_{CH_4,eq.}" at first eqConversion + 0.025, graph 0.05 left front font ',10' textcolor rgb 'black'

plot for [j=2:7] filename using 1:(column(j)) notitle w l ls (j-1) lc rgb 'black',\
    for [i=1:5] '+' using (eqConversion):(eqMoleFractions[i]) title idxCompound[i] w p ls i lc rgb 'black'