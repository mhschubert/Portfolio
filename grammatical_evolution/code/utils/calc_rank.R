'calculates rankvote of all internal CVIs'

# extract CVIs
expintCVI<-comparison[["results"]][["partitional"]]

cvis<-c("Sil","D","COP","DB","DBstar","CH","SF")
expintCVI[cvis] <- lapply(expintCVI[,cvis],
                           function(x){(x-min(x))/(max(x)-min(x))}) #standardize int CVIs between 0 and 1

# recode indices to be minimized
expintCVI$COP <- 1 - expintCVI$COP
expintCVI$DBstar <- 1 - expintCVI$DBstar
expintCVI$DB <- 1 - expintCVI$DB

expintCVI<-expintCVI[expintCVI$gamma_distance==0.001,]

#long 
CVI_long <- gather(expintCVI[, c("k",cvis)], CVIs, CVI_val, Sil:SF, factor_key=TRUE)

#Rankvote: find number of clusters with highest rank
expintCVI$rankSil <- rank(expintCVI$Sil)
expintCVI$rankD <- rank(expintCVI$D)
expintCVI$rankDB <- rank(expintCVI$DB)
expintCVI$rankDBstar <- rank(expintCVI$DBstar)
expintCVI$rankSF <- rank(expintCVI$SF)
expintCVI$rankCH <- rank(expintCVI$CH)
expintCVI$rankCOP <- rank(expintCVI$COP)

expintCVI$rankvote <- NA
for (i in 1:dim(expintCVI)[1]) {
  expintCVI$rankvote[i] <- expintCVI$rankSil[i] + expintCVI$rankD[i] +  expintCVI$rankDB[i] + expintCVI$rankDBstar[i] +  expintCVI$rankSF[i] + expintCVI$rankCH[i] +  expintCVI$rankCOP[i]
}

int<-expintCVI$k[expintCVI$rankvote == max(expintCVI$rankvote)]
exp_id<-expintCVI$config_id[expintCVI$rankvote == max(expintCVI$rankvote)]