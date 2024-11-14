# Functions implementation

## [Tree] HierTree
## [Algorithm 1] MultiEmb0(~3.centering), MultiEmb(~4.scaling)
## [Algorithm 2] HierEmb
## [TD Estimation] TD
## [TD Prediction] pred



# Tree
HierTree = function(tree){
  require(data.tree)
  
  tree$Set(id=0:(tree$totalCount-1), traversal='level')
  
  trees = Traverse(tree, 'level')
  tree_df = ToDataFrameNetwork(trees[[1]], 'level', 'id', 'isLeaf')
  tree_df$parents = sapply(trees[-1], function(node) node$parent$id)
  
  num_sib = numeric()
  for (i in 2:tree$totalCount){
    num_sib[i-1] = length(trees[[i]]$siblings)
  }
  
  return (list(trees = trees,
               tree_df = tree_df,
               num_sib = num_sib))
}



# Algorithm 1: Label embedding in multicategory classification
MultiEmb0 = function(c, q){
  
  # c: equal pairwise distance (positive constant)
  # q: number of leaf nodes
    
  # 1. Initialization
  Xi = matrix(0, q-1, q) # 2. Iteration (1)
  Xi[1,1] = -c/2; Xi[1,2] = c/2
  
  # 2. Iteration (2)
  if (q > 2){
    for (m in 2:(q-1)){
      if (m==2){ d_in = sum(Xi[1, 1:m])/m-Xi[1, m] }
      else{ d_in = rowSums(Xi[1:(m-1), 1:m])/m-Xi[1:(m-1), m] }
      d = sqrt(sum(d_in^2))
      a = sqrt(c^2 - d^2)
      e = numeric(m); e[m] = 1
    
      Xi[1:m, m+1] = rowSums(Xi[1:m, 1:m])/m + a*e
  }
  
  # 3. Centralization
  Xi_rowsum = rowSums(Xi)
  
  for (i in 1:q){
    Xi[, i] = Xi[, i] - Xi_rowsum/q
    }
  }

  return(Xi)
}

MultiEmb = function(t, q){
  
  # t: same l2 norm T
  # q: number of leaf nodes
  
  # 4. scaling
  cq = t*sqrt((2*q)/(q-1))
  Xi = MultiEmb0(cq, q)
  return(Xi)
}



# Algorithm 2: Label embedding in hierarchical classification
HierEmb = function(tree, t=1, delta=sqrt(5)){
  hiertree = HierTree(tree)
  
  trees = hiertree$trees
  tree_df = hiertree$tree_df
  parents = tree_df$parents
  
  n_nodes = tree$totalCount # = length(trees)
  n_leafs = tree$leafCount
  

  Xi = matrix(0, n_leafs-1, n_nodes-1)
  level = 0
  startrow = startcol = 1
  
  for (i in 1:n_nodes){
    curr_node = trees[[i]]
    
    if (!curr_node$isLeaf){ # nonleaf node
      m = curr_node$level
      n_leaf = curr_node$count
      Xi[startrow:(startrow+n_leaf-2), startcol:(startcol+n_leaf-1)] = MultiEmb(t/delta^(m-1), trees[[i]]$count)
      startrow = startrow+n_leaf-1; startcol = startcol+n_leaf
    }
    
    if (parents[i]!=0 & i<n_nodes){ # 부모가 루트 노드가 아님
      Xi[, i] = Xi[, i] + Xi[, parents[i]] # 첫번째 항이 Eta
    }
  }
  return (Xi)
}



# Linear Loss under Topdown (Estimation)
TD = function(tree, train_X, train_Y_path, t=1, delta=sqrt(5), lambda=1){
  # train_Y: Y_path
  # lambda: the tuning parameter
  hiertree = HierTree(tree)
  
  trees = hiertree$trees
  tree_df = hiertree$tree_df
  parents = tree_df$parents
  num_sib = hiertree$num_sib
  
  K = tree$height # n_layers
  n_nodes = tree$totalCount  # = length(trees)
  
  
  Xi = HierEmb(tree, t, delta); Eta = Xi
  
  for (i in 1:(n_nodes-1)){
    if (parents[i]!=0){
      Eta[, i] = Xi[, i] - Xi[, parents[i]]
    } 
  }
  n = nrow(train_X); p = ncol(train_X); q = nrow(Xi)
  
  
  train_Y_ls = lapply(seq_len(nrow(train_Y_path)),
                     function(r){train_Y_path[r,]})
  train_Y_ls = lapply(train_Y_ls,
                     function(node){node[node!=0]})
  
  temp = lapply(train_Y_ls,
                function(i){
                  rowSums(t( (num_sib[i]+1)*t(matrix(Eta[,i], nrow=q)) ))})
  
  B = t(matrix(unlist(temp), ncol=q, byrow=T)) %*% train_X
  A = lambda * B / n
  
return(A)#(list(A=A))
}



# Prediction (TD)
pred = function(tree, test_X, A){
  # X: feature vector, 
  # A: estimator

  hiertree = HierTree(tree)
  
  trees = hiertree$trees
  tree_df = hiertree$tree_df

  K = tree$height # n_layers
  n_nodes = tree$totalCount  # = length(trees)
  
  Xi = HierEmb(tree)
  
  pred_y = matrix(0, nrow=nrow(test_X), ncol=n_layers)
  l1 = tree_df[tree_df$level==2, 'id']
  leaf = tree_df[!tree_df$isLeaf, 'id']
  
  for (s in 1:nrow(test_X)){
    f = A %*% test_X[s,]        ## 여기를 NN으로?
    g = apply(Xi[,l1], 2, function(xi){f %*% xi})
    pred_y[s,2] = l1[which.min(g)]
    m = 2
    while (!pred_y[s, m] %in% leaf){
      lm = tree_df[tree_df$parents==pred_y[s, m], 'id']
      g = apply(Xi[,lm], 2, function(xi){f %*% xi})
      pred_y[s,(m+1)] = lm[which.min(g)]
      m = m+1
    }
  }
  return(pred_y)
}
