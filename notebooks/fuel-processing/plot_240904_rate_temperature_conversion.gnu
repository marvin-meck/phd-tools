 # Gnuplot script to plot contour data with labels on the lines

set term pngcairo size 800,600
set output 'plots/contour_plot.png'

set xlabel "Temperature"
set ylabel "Conversion"

# Set contour settings
set contour
unset surface  # Only show contour lines, not the 3D surface
set view map   # 2D view for contour

# Enable contour labels
set cntrparam levels discrete -10000,-2500,-1250,-750,-500,-250,-100,-50,-25,-10,-1  # Contour levels, flipped
set clabel '%8.2f'   # Format for contour labels; here it's set to two decimal places

# Plot the contour
splot 'results/contour_data.dat' using 1:2:3 with lines
