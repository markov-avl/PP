mkdir cmake-build-debug
cd cmake-build-debug
cmake ..
cmake --build .

echo "Running lab1"
./Integration > ../lab1.txt

echo "Running lab2-3"
./Matrix > ../lab2-3.txt

echo "Running lab4"
./ParallelMod > ../lab4.txt

echo "Running lab5"
./FFT > ../lab5.txt