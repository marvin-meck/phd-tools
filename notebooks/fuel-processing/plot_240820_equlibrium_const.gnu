set loadpath '/home/meck/diss-code'; load "fststyle.cfg"
set datafile sep ','

set key top left
# set xrange [500:2000]

# notebook version
set terminal pngcairo enhanced font 'Verdana,10'
set output "plots/plot_240820_equlibrium_const.png"

set xlabel '1000/T in 1/K'
set ylabel '-log K_i'


set label 1 'K_1' at 1.55,14
set label 2 'K_2' at 1.55,-5
set label 3 'K_3' at 1.55,6


plot \
    for [j=1:3] filename1 u (1000/$1):(-1*column(j+1)) \
        w l lc rgb "black" notitle,\
    for [j=1:5] filename2 u (1000/$1):(-1*column(j+1))\
        skip 1 w p notitle ls (2*j-1) lc 'black',\
    for [j=6:10] filename2 u (1000/$1):(-1*column(j+1))\
        skip 1 w p notitle ls (2*(j-5)-1) lc 'black',\
    for [j=11:15] filename2 u (1000/$1):(-1*column(j+1)) \
        skip 1 w p title sprintf('-log Q_i* (p = %s)',columnhead(j+1)) ls (2*(j-10)-1) lc 'black'\
    ;

