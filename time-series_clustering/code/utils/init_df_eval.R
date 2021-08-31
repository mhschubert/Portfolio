#@Author Marcel H. Schubert
#initializes a blank dataframe with needed columns to store configuration testing in

##accuracy per cluster
acc_c <- c(sprintf("accuracy_cluster_%d",seq(1:length(types))), "avg_accuracy_cluster")
##recall per cluster
rec_c <- c(sprintf("recall_cluster_%d",seq(1:length(types))),"avg_recall_cluster")
##precision per cluster
prec_c <- c(sprintf("precision_cluster_%d",seq(1:length(types))),"avg_precision_cluster")
##f1 score per cluster
f1_c <- c(sprintf("f1_cluster_%d",seq(1:length(types))),"avg_f1_cluster")

##accuracy per type
acc_t <- c(sprintf("accuracy_type_%d",seq(1:length(types))),"avg_accuracy_type")
##recall per type
rec_t <- c(sprintf("recall_type_%d",seq(1:length(types))),"avg_recall_type")
##precision per type
prec_t <- c(sprintf("precision_type_%d",seq(1:length(types))),"avg_precision_type")
##f1 score per type
f1_t <- c(sprintf("f1_type_%d",seq(1:length(types))),"avg_f1_type")

#cluster matching
matching <- sprintf("pred_clust_type_%d",seq(1:length(types)))

##cluster_sizes
cluster_sizes <- sprintf("size_cluster_%d",seq(1:length(types)))

#cluster
combis <- expand.grid(types, types)
cluster_n <- sprintf("n_clust_%d_type_%d",unname(as.vector(combis[,1])),  unname(as.vector(combis[,2])))

#freq
cluster_freq <- sprintf("freq_clust_%d_type_%d",unname(as.vector(combis[,1])),  unname(as.vector(combis[,2])))

df_colnames <- c('pick_success', "grp_comp", "types", "measure", "ratio", 'avg_max_share', 'median_share', "config", "sigma", "eta",
                 "grpsize", "numrounds", "endowment", "windowsize", "n", "identifier",
                 #hierarchical
                 "config_id", "k", "method", "symmetric", "preproc", "center_preproc",
                 "distance", "window.size_distance", "norm_distance", "gamma_distance", "centroid", "znorm_distance",
                 "norm_centroid", "znorm_centroid", 
                 #partitional
                 "config_id", "rep",  "k", "pam.precompute", "iter.max", "symmetric",
                 "version", "preproc", "center_preproc", "distance", "window.size_distance", "norm_distance",
                 "gamma_distance", "centroid", "znorm_centroid", "norm_centroid",
                 #tadpole
                 "config_id", "k", "dc", "window.size", "lb", "preproc", "new.length_preproc",
                 "centroid", "znorm_centroid", "norm_centroid",
                 #fuzzy
                 "config_id", "k", "fuzziness", "iter.max", "delta", "symmetric",
                 "version", "preproc", "distance", "window.size_distance", "norm_distance", "gamma_distance",
                 "centroid", "znorm_centroid",
                 #CVI
                 "ARI", "RI", "J", "FM",  "VI",
                 "Sil", "D", "COP", "DB", "DBstar", "CH", "SF",
                 ##fuzzy
                 "MPC", "K", "T", "SC", "PBMF",
                 acc_c, prec_c, rec_c, f1_c, acc_t, prec_t, rec_t, f1_t,
                 matching,
                 cluster_sizes,
                 cluster_n,
                 cluster_freq)
df_colnames <- unique(df_colnames)

dim2 <- length(df_colnames)
df_eval <- data.frame(matrix(vector(), 0, dim2,
                             dimnames=list(c(), df_colnames)),
                      stringsAsFactors=F)
empty_frame<-df_eval
