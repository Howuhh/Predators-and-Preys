#! /usr/bin/env bash

os_name=$(uname)

cd ./predators_and_preys_env/engine
if [ $os_name = "Linux" ]
then
    make
else
    clang -dynamiclib -undefined dynamic_lookup -o game.dylib game.c
fi
 
cd ./physics
if [ $os_name = "Linux" ]
then
    make
else
    clang -dynamiclib -undefined dynamic_lookup -o entity.dylib entity.c
fi

cd ../../..