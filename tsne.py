import pandas as pd
import numpy as np
from sklearn.manifold import TSNE



run="120"
allTestLogits = pd.DataFrame.from_csv ("/home/ahmed/Dropbox/DFCI/11_AWS_output/output/"+ run + "_dataFrame.csv")


allTestLogits = allTestLogits[ (allTestLogits.patient != 53283) 
                              & (allTestLogits.patient != 52912 )
                             & (allTestLogits.patient != 45272 )
                               & (allTestLogits.patient != 64831 )
                             ]
allTestLogits = allTestLogits.reset_index(drop=True)


features256 = pd.DataFrame()
features256['patient'] = allTestLogits.patient
for i in range(256):
    features256[ "feature_" + str(i) ] = allTestLogits[ "dense1_" + str(i) ] 



model = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
datamtrx = features256.as_matrix(columns= [ "feature_" + str(i) for i in range(256) ] )
twoDimensional = model.fit_transform(datamtrx) 
print (twoDimensional)