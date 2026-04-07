set loadpath '/home/meck/diss-code'; load "fststyle.cfg"
set datafile sep ','

set terminal pngcairo enhanced font 'Verdana,10'
set output "plots/plot_240405_chemical_equilibrium.png"

set xlabel "Temperature in K"
set ylabel "Mole fraction"

set xrange [600:1600]

plot for [j=2:7] filename using 1:(column(j)) title sprintf('%s',columnhead(j)) with linespoints ls (j-1) lc 'black'