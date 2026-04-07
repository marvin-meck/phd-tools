set loadpath '/home/meck/diss-code'; load "fststyle.cfg"
set datafile sep ','

set terminal pngcairo enhanced font 'Verdana,10'
set output "plots/plot_240830_eq_conversion.png"

set xlabel "Temperature in K"
set ylabel "Equilibrium conversion X_{CH_4,eq.}"
set y2label "Equilibrium production" textcolor rgb "gray"

set ytics nomirror
set y2tics

# set key top left

set title TITLE
set xrange [600:1600]
set yrange [0:1]
set y2range [0:4]

set border 11
set y2tics textcolor rgb "gray"

plot \
    conversion using 1:2 notitle with lp ls 1 lc rgb "black" axes x1y1,\
    production using 1:(column(2)) notitle with lp ls 1 lc rgb "gray" axes x1y2