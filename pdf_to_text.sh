#!/bin/bash

for file in *.pdf
  do
    if [ ! -e "../txt/$file.txt" ] ; then
      pdftotext "$file" "../txt/$file.txt"
    fi
  done
