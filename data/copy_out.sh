#!/usr/bin/env bash

## The destination folder where your files will
## be copied to.
dest="bar";

## For each file path in your input file
while read path; do 
    ## $target is the name of the file, removing the path. 
    ## For example, given /foo/bar.txt, the $target will be bar.txt.
    target=$(basename "$path"); 
    ## Counter for duplicate files
    c=""; 
    ## Since $c is empty, this will check if the
    ## file exists in target.
    while [[ -e "$dest"/"$target"$c ]]; do
        echo "$target exists"; 
        ## If the target exists, add 1 to the value of $c
        ## and check if a file called $target$c (for example, bar.txt1)
        ## exists. This loop will continue until $c has a value
        ## such that there is no file called $target$c in the directory.
        let c++; 
        target="$target"$c; 
    done; 
    ## We now have everything we need, so lets copy.
    cp "$path" "$dest"/"$target"; 
done
