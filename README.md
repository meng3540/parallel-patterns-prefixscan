# MENG 3540 Assignment - Prefix Sum

This is a research/lab assignment created by Ian Cameron, Alexander Colatosti, and Amelia Soon for MENG3540 - Parallel Programming. 
The assignment involves creating various implementations of a prefix sum and comparing their efficiency.

### Introduction to Prefix sums
A prefix sum, otherwise known as prefix scan or parallel scan, is an operation where, given a sequence of numbers, each element of the output array is equal to the sum of the previous elements in the input array. For example, a prefix sum applied to the array [1 2 3 4 5] would produce the array [1 3 6 10 15]. Prefix sums have many applications, such as image processing, but in this assignment were applied to simulated LiDAR data in order to count the number of obstacles detected by the sensor.

Further information, including information on the code and various algorithms used, is available in the included pdf file.

### References
Chang, L.-W., & Gomez-Luna, J. (2016). Parallel patterns: prefix sum. In *Programming Massively Parallel Processors* (3rd ed., pp. 176–196). Morgan Kaufmann. 
