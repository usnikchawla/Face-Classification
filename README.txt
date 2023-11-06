Requirements:

1. Python 3.9.6 (GCC 7.5.0 Anaconda on Linux)
2. NumPy == 1.20.3
3. SciPy == 1.7.1
4. Seaborn == 0.11.2
5. Matplotlib == 3.4.2

Data: codes/data/data.mat for task 2, codes/data/pose.mat for task 1.

Scripts (in codes/ folder):

1. data.py -- code for reading, processing, and partitioning data.(Modify the path according to local machine in get data)
2. funtions.py -- code for data preprocessing: MDA and PCA.

3. bayes-task1.py, bayes-task2.py -- Baye's classifier code for task 1 and 2. Running this code will produce output.

4. knn-task1.py, knn-task2.py -- k-NN classifier code for task 1 and 2. Running this code will produce output.

5. kernelsvm-task2.py -- Kernel SVM classifier code for task 2. Running this code will produce output.

6. adaboost-task2.py -- Boosted linear SVM classifier code for task 2. Running this code will produce output.

Other files/folders:
There are several files assosiated with each classifier that contains its functionality.

To run scripts,
1. Use Python and dependencies as mentioned above.
2. Go to the codes/ folder location.
3. Run "python bayes-task1.py" in the terminal for running script 4.
4. Once the script is exceuted, the output plots will be displayed.

@author-usnikchawla(uchawla@umd.edu)