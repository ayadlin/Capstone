#!/bin/bash

for file in *.pdf
  do
    if [ ! -e "$file.txt" ] ; then
      pdftotext "$file" "$file.txt"
    fi
  done 
