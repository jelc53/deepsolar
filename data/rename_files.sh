for file in bar/*.png
do
  mv "$file" "${file/.png/_true_seg.png}"
done
