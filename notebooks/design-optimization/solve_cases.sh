cd $TMPDIR

no_files=$(find *.yaml -type f | wc -l )

declare -i i=1

echo "solving..."
for filename in *.yaml; do
    echo "$i/$no_files"
    pyomo solve $filename > /dev/null
    i+=1
done
echo "...done!"

cd ..