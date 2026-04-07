set loadpath '/Users/marvinmeck/Documents/git/phd-tools-dev'; load "fststyle.cfg"
set datafile sep ','

set terminal pngcairo enhanced font 'Verdana,10'
set output "plots/250629_SampleAvgProb.png"

# set xrange [1:3];
set yrange [0:1];

set xlabel 'Investment cost in Euro'
set ylabel 'Sample average choice probability'

set arrow from graph 0, first 0.5 to graph 1, first 0.5 nohead lc rgb "gray" lw 1 dt 2

plot for [j=2:4] filename u (1000*$1):j title sprintf('%s',columnhead(j)) with lp ls (j-1) lc 'black'