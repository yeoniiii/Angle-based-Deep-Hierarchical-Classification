# Evaluation implementation

ances<-function(mat,k) {
  anc.ind<-k
  s<-1
  while (s<=length(anc.ind)) { anc.ind<-c(anc.ind,which(mat[,anc.ind[s]]!=0)); s<-s+1 }
  if(length(anc.ind)==1) return(NULL) else return(anc.ind[-1])
} 

desce<-function(mat,k) {
  des.ind<-k
  s<-1
  while (s<=length(des.ind)) { des.ind<-c(des.ind,which(mat[des.ind[s],]!=0)); s<-s+1 }
  return(des.ind)
} 

sibnodes<-function(edges,y){
  parent=edges[edges[,2]==y,1]
  siblings=edges[edges[,1]==parent,2]
  siblings=siblings[siblings!=y]
  return(siblings)
}

#transform treatments into a binary vector without the root node
transform_y<-function(edges,y){
  newy=rep(0,max(edges))
  newy[y]=1
  return(newy)
}

#sibling weights to calculate hierarchical loss
omega1<-function(edges){
  omega=rep(0,max(edges))
  l1=edges[edges[,1]==0,2] #nodes at the second layer
  omega[1:length(l1)]=1/length(l1)
  if(max(edges)>length(l1)){
    for (i in (length(l1)+1):nrow(edges)){
      parent=edges[i,1]
      siblings=edges[edges[,1]==parent,2]
      omega[i]=omega[edges[i,1]]/length(siblings)
    }
  }
  return(omega)
}

#convert the hierarchy "edges" into another structure
transform_mat<-function(edges){
  mat=matrix(0,max(edges),max(edges))
  mat[edges]=1
  return(mat)
}

#subtree weights to calculate hierarchical loss
omega2<-function(edges){
  hier.mat=transform_mat(edges)
  omega=rep(0,max(edges))
  for (i in 1:max(edges)){
    omega[i]=length(desce(hier.mat,i))/max(edges)
  }
  return(omega)
}

evaluation<-function(K,edges,test_Y,pred_Y){
  n=nrow(test_Y)
  
  test_Y=t(apply(test_Y,1,function(x){transform_y(edges,x)}))
  pred_Y=t(apply(pred_Y,1,function(x){transform_y(edges,x)}))
  
  test_Y[test_Y==-1]=0; pred_Y[pred_Y==-1]=0
  myomega1=omega1(edges); myomega2=omega2(edges)
  
  accuracy=mean(apply(test_Y-pred_Y,1,function(x){sum(abs(x))>0}))
  hloss1=sum(apply(test_Y-pred_Y,1,function(x){myomega1[which(x!=0)[1]]}),na.rm=T)/n
  hloss2=sum(apply(test_Y-pred_Y,1,function(x){myomega2[which(x!=0)[1]]}),na.rm=T)/n
  
  hier.mat=transform_mat(edges)
  symloss=0
  hp=0; hr=0; c=0
  for (i in 1:n){
    a=which(test_Y[i,]==1); b=which(pred_Y[i,]==1)
    a_anc=c(); b_anc=c()
    for (j in 1:length(a)){
      a_anc=c(a_anc,ances(hier.mat,a[j]))}
    for (j in 1:length(b)){
      b_anc=c(b_anc,ances(hier.mat,b[j]))}
    a=unique(c(a,a_anc)); b=unique(c(b,b_anc))
    symloss=symloss+length(c(setdiff(a,b),setdiff(b,a)))
    c=c+length(intersect(a,b))
    hp=hp+length(b)
    hr=hr+length(a)
  }
  symloss=symloss/n
  hp=c/hp; hr=c/hr
  hf=2*hp*hr/(hp+hr)
  return(c(accuracy,symloss,hloss1,hloss2,hp,hr,hf))
}