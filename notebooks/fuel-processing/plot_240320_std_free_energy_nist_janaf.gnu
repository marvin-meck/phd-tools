set loadpath '/home/meck/diss-code'; load "fststyle.cfg"
set datafile sep ','

set key bottom left
set xrange [500:2000]

# notebook version
set terminal pngcairo enhanced font 'Verdana,10'
set output "plots/240701_std_free_energy_nist_janaf.png"

set xlabel 'Temperature in K'
set ylabel 'standard Gibbs free energy in kJ/mol'

plot for [j=1:6] filename using 1:(column(j+1)) with lp title sprintf('%s',columnhead(j+1)) ls (2*j-1) lc 'black'