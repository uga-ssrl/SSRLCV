import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../../histogram.csv', sep=',',header=None, index_col =0)

data.plot(kind='hist')
plt.ylabel('occurance')
plt.xlabel('E. error')
plt.title('Yee')

plt.show()
