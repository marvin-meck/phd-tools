set loadpath '/home/meck/diss-code'; load "fststyle.cfg"
set datafile sep ','

# set key top right
# set xrange [500:2000]

# notebook version
set terminal pngcairo enhanced font 'Verdana,10'
set output "plots/plot_240820_percent_error.png"

set multiplot layout 3,1 rowsfirst

set xlabel 'Temperature in K'
set ylabel 'Approx. error in %'

set yrange [0:0.25]


plot \
    for [j=1:5] filename u 1:(column(j+1))\
        skip 1 w lp notitle ls (2*j-1) lc 'black'

plot \
    for [j=6:10] filename u 1:(column(j+1))\
        skip 1 w lp notitle ls (2*(j-5)-1) lc 'black'

plot \
    for [j=11:15] filename u 1:(column(j+1))\
        skip 1 w lp notitle ls (2*(j-10)-1) lc 'black'