import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from Algorithms import TreeEmbedding
from Preprocessing import DataPreprocessing
import numpy as np

def CustomEvaluation(tree, y_test, y_pred):
  ro.r['source']('./R/Algorithms.R')
  ro.r['source']('./R/Evaluation.R')

  df = TreeEmbedding(tree).df
  K = tree.height
  edges = df[['parent', 'id']].values[1:, :]
  y_test = DataPreprocessing(tree).y_to_path(y_test)
  y_pred = np.array(y_pred)

  numpy2ri.activate()

  nr, nc = edges.shape
  edges_r = ro.r.matrix(edges, nr, nc)
  ro.r.assign("edges", edges_r)

  nr, nc = y_test.shape
  y_test_r = ro.r.matrix(y_test, nr, nc)
  ro.r.assign("y_test", y_test_r)

  nr, nc = y_pred.shape
  y_pred_r = ro.r.matrix(y_pred, nr, nc)
  ro.r.assign("y_pred", y_pred_r)

  return ro.r.evaluation(K, edges_r, y_test_r, y_pred_r)